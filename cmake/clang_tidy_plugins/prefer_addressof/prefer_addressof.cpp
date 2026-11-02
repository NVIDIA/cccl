// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>

#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang-tidy/utils/IncludeInserter.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Lex/Lexer.h>
#include <clang/Sema/HeuristicResolver.h>
#include <common/names.h>
#include <common/type_traits.h>
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

AST_MATCHER(Type, isKnownThrustTemplate) // NOLINT
{
  if (const ClassTemplateDecl* const decl = unwrap(Node))
  {
    const auto name = util::fully_qualified_name(*decl);

    return llvm::StringSwitch<bool>(name)
      .Cases(
        {
          "thrust::reference",
          "thrust::device_reference",
          "thrust::tagged_reference",
          "thrust::iterator_facade",
          "thrust::mr::memory_resource",
          "thrust::detail::vector_base",
        },
        /*Value=*/true)
      .Default(false);
  }

  return false;
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

// Known scalar (builtin, enum, bit-field, pointer), array, or function types cannot overload
// address-of, even when their (possible) element type is dependent.
[[nodiscard]] bool is_unoverloadable_type(const Type& ty)
{
  return ty.isScalarType() || ty.isArrayType() || ty.isFunctionType();
}

AST_MATCHER(Type, isUnOverloadableType) // NOLINT
{
  return is_unoverloadable_type(Node);
}

// This is an extension of HeuristicResolver::resolveExprToType(), which does not handle
// array subscript expressions (see https://github.com/clangd/clangd/issues/2723) when
// resolving the type.
//
// The whole point of this function is a heuristic to avoid emitting diagnostics on decl-ref
// expressions where the outer type a type-dependent template, but inner members of the
// template have concrete types. For example:
//
// template <class T>
// struct Foo {
//   int x;
//   T value;
// };
//
// template <class T>
// void foo(Foo<T> &f)
// {
//   &f.x; // we probably should not emit a diagnostic for this, Foo is probably not
//         // specialized
// }
//
// The entire heuristic rests on the assumption that the outer template is not expected to be
// specialized (which would manifestly change any member types anyways). So even though the
// expression is *technically* type dependent we can make predictions on the member types based
// on the text of the primary template.
[[nodiscard]] QualType resolve_expr_to_type(const Expr* expr, const ASTContext& ctx, const HeuristicResolver& resolver)
{
  expr = expr->IgnoreParenImpCasts();

  const QualType type = resolver.resolveExprToType(expr);

  if (!type.isNull() && !type->isDependentType())
  {
    return type;
  }

  if (const auto* const subscript = dyn_cast<ArraySubscriptExpr>(expr))
  {
    const QualType base = resolve_expr_to_type(subscript->getBase(), ctx, resolver).getNonReferenceType();

    if (!base.isNull())
    {
      if (const auto* const array = ctx.getAsArrayType(base))
      {
        return array->getElementType();
      }
      if (base->isPointerType())
      {
        return base->getPointeeType();
      }
    }
  }

  return type;
}

AST_MATCHER_P(Expr, hasHeuristicType, TypeMatcher, inner) // NOLINT
{
  auto& ctx           = Finder->getASTContext();
  const auto resolver = HeuristicResolver{ctx};
  const auto type     = resolve_expr_to_type(&Node, ctx, resolver);

  return !type.isNull() && inner.matches(type, Finder, Builder);
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
    const auto exempt_type = qualType(hasCanonicalType(
      anyOf(isUnOverloadableType(),
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
            // Special-case for known thrust templates like proxy references, which we know to
            // have valid operator&() overloads that we should honor.
            isKnownThrustTemplate())));

    const auto operand =
      expr(unless(anyOf(dependentScopeDeclRefExpr(),
                        hasType(exempt_type),
                        hasHeuristicType(exempt_type),
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
    const auto* const operand = nodes.getNodeAs<Expr>(OPERAND_BIND);
    const auto* const call    = nodes.getNodeAs<Expr>(BIND);

    auto d = diag(call->getBeginLoc(),
                  "do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead.");

    bool invalid = false;
    const auto arg =
      Lexer::getSourceText(CharSourceRange::getTokenRange(operand->getSourceRange()), SM, getLangOpts(), &invalid);

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
