// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>

#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang-tidy/utils/IncludeInserter.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Lex/Lexer.h>
#include <common/names.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>

namespace clang::tidy::cccl
{
namespace
{
using namespace ast_matchers; // NOLINT(google-build-using-namespace)

[[nodiscard]] ClassTemplateDecl* unwrap(const Type& ty)
{
  if (const auto* spec = dyn_cast<TemplateSpecializationType>(&ty))
  {
    return dyn_cast_if_present<ClassTemplateDecl>(spec->getTemplateName().getAsTemplateDecl());
  }

  if (const auto* injected = dyn_cast<InjectedClassNameType>(&ty))
  {
    return injected->getDecl()->getDescribedClassTemplate();
  }

  if (const auto* record = dyn_cast<RecordType>(&ty))
  {
    if (const auto* spec = dyn_cast<ClassTemplateSpecializationDecl>(record->getDecl()))
    {
      return spec->getSpecializedTemplate();
    }
  }

  return nullptr;
}

// determine whether decl points to exactly "thrust::reference", "thrust::device_reference", or
// "thrust::tagged_reference".
[[nodiscard]] bool is_known_thrust_reference_template(const NamedDecl& decl)
{
  const StringRef name = decl.getName();

  if (name != "reference" && name != "device_reference" && name != "tagged_reference")
  {
    return false;
  }

  const DeclContext* ctx = decl.getDeclContext();

  while (const auto* ns = dyn_cast<NamespaceDecl>(ctx))
  {
    if (!ns->isInline())
    {
      return ns->getName() == "thrust";
    }

    ctx = ns->getParent();
  }
  return false;
}

AST_MATCHER(Type, isKnownThrustProxyReference) // NOLINT
{
  const ClassTemplateDecl* decl = unwrap(Node);

  return decl && is_known_thrust_reference_template(*decl);
}

AST_MATCHER(Type, isReservedClassTemplateType) // NOLINT
{
  if (const ClassTemplateDecl* const decl = unwrap(Node))
  {
    const IdentifierInfo* const ident = decl->getIdentifier();

    return ident && util::is_ugly_name(ident->getName());
  }
  return false;
}

AST_POLYMORPHIC_MATCHER(refersOnlyToFunctions, // NOLINT
                        AST_POLYMORPHIC_SUPPORTED_TYPES(UnresolvedLookupExpr, UnresolvedMemberExpr))
{
  return std::all_of(Node.decls_begin(), Node.decls_end(), [](const NamedDecl* decl) {
    return isa<FunctionDecl, FunctionTemplateDecl>(decl->getUnderlyingDecl());
  });
}

class PreferAddressofCheck final : public ClangTidyCheck
{
  static constexpr StringRef BIND{""};
  static constexpr StringRef OPERAND_BIND{"o"};

public:
  PreferAddressofCheck(StringRef name, ClangTidyContext* context)
      : ClangTidyCheck{name, context}
      , inserter_{utils::IncludeSorter::IS_LLVM, areDiagsSelfContained()}
  {}

  [[nodiscard]] std::optional<TraversalKind> getCheckTraversalKind() const override
  {
    return TK_IgnoreUnlessSpelledInSource;
  }

  void registerPPCallbacks(const SourceManager&, Preprocessor* pp, Preprocessor*) override
  {
    inserter_.registerPreprocessor(pp);
  }

  void registerMatchers(MatchFinder* finder) override
  {
    const auto operand =
      expr(unless(anyOf(
             dependentScopeDeclRefExpr(),
             // Known pointer and array types cannot overload address-of, even when their
             // element type is dependent.
             hasType(qualType(hasCanonicalType(anyOf(
               pointerType(),
               memberPointerType(),
               arrayType(),
               functionType(),
               // This is a heuristic. There are many cases in libcu++ where we do things like
               //
               // template <class T>
               // struct __wrapper { };
               //
               // auto *p = &__wrapper<T>{foo};
               //
               // We control the definition of `__wrapper`, and we definitely have not added an
               // operator&() overload for it.
               //
               // Now *technically* we should be using addressof() here (because some code
               // somewhere *could* define a specialization for some __wrapper<T> which adds
               // operator&()). But such specializations are probably UB anyways, so this
               // scenario is vanishingly unlikely and we can save a bunch of NOLINTs this way.
               isReservedClassTemplateType(),
               // Special-case for thrust proxy references, which we know to have valid
               // operator&() overloads that we should honor.
               isKnownThrustProxyReference())))),
             // We need this extra condition here because hasType(functionType())
             // does not necessarily fire for type-dependent function template
             // lookups (known in AST parlance as "unresolved lookup
             // expressions"). In that case we need to match all such lookup
             // expressions where all potential matches are function
             // types. For example:
             //
             // template <class T>
             // void foo() { }
             //
             // template <class T>
             // using fn_type = decltype(&foo<T>)
             //
             // Here `&foo<T>` is fully type-dependent, and is not yet considered a
             // function type *but* we know that all instantiations of `foo<T>` must
             // refer to the function template `foo`.
             //
             // The clang AST is weird sometimes.
             mapAnyOf(unresolvedLookupExpr, unresolvedMemberExpr).with(refersOnlyToFunctions()))))
        .bind(OPERAND_BIND);

    finder->addMatcher(
      mapAnyOf(unaryOperator, cxxOperatorCallExpr)
        .with(isTypeDependent(),
              hasOperatorName("&"),
              hasUnaryOperand(operand),
              // This is actually wrong (if you think about it, passing result of arbitrary
              // operator&() to asm is exactly the kind of scenario we *should* be warning
              // about), but we use this as a heuristic to avoid adding a bunch of NOLINT to
              // the generated PTX wrappers.
              //
              // Those are templated, but due to a series of prior type unwrapping and checking
              // we are pretty confident that the types passed to PTX are OK.
              unless(hasAncestor(asmStmt())))
        .bind(BIND),
      this);
  }

  void check(const MatchFinder::MatchResult& result) override
  {
    const auto& nodes         = result.Nodes;
    const auto& SM            = *result.SourceManager;
    const auto* const call    = nodes.getNodeAs<Expr>(BIND);
    const auto* const operand = nodes.getNodeAs<Expr>(OPERAND_BIND);

    auto d = diag(call->getBeginLoc(),
                  "do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead.");

    bool invalid   = false;
    const auto arg = Lexer::getSourceText(
      CharSourceRange::getTokenRange(operand->getSourceRange()), SM, result.Context->getLangOpts(), &invalid);

    if (!invalid)
    {
      d << FixItHint::CreateReplacement(call->getSourceRange(), ("::cuda::std::addressof(" + arg + ")").str());
      d << inserter_.createIncludeInsertion(SM.getFileID(call->getBeginLoc()), "<cuda/std/__memory/addressof.h>");
    }
  }

private:
  utils::IncludeInserter inserter_;
};

class PreferAddressofCheckModule final : public ClangTidyModule
{
public:
  static constexpr StringRef CHECK_NAME{"cccl-prefer-addressof"};

  void addCheckFactories(ClangTidyCheckFactories& check_factories) override
  {
    check_factories.registerCheck<PreferAddressofCheck>(CHECK_NAME);
  }
};

// NOLINTNEXTLINE(cert-err58-cpp, bugprone-throwing-static-initialization)
ClangTidyModuleRegistry::Add<PreferAddressofCheckModule> _{
  PreferAddressofCheckModule::CHECK_NAME,
  "Catch all cases of type-dependent use of operator& which should be addressof() instead."};
} // namespace
} // namespace clang::tidy::cccl
