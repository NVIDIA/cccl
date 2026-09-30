// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
// reserved.
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <clang-tidy/ClangTidy.h>
#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang/AST/QualTypeNames.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Lex/Lexer.h>
#include <clang/Tooling/Refactoring/Lookup.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Config/llvm-config.h>

#if LLVM_VERSION_MAJOR < 23
#  include <algorithm>
#endif

namespace clang::tidy::cccl
{
namespace
{
using namespace ast_matchers; // NOLINT(google-build-using-namespace)

AST_MATCHER(Type, isDependentType) // NOLINT
{
  return Node.isDependentType();
}

#if LLVM_VERSION_MAJOR < 23
// LLVM 22's hasAnyTemplateArgumentLoc does not support UnresolvedLookupExpr.
AST_MATCHER(UnresolvedLookupExpr, hasDependentTypeArgument) // NOLINT
{
  auto&& arguments = Node.template_arguments();

  return std::any_of(arguments.begin(), arguments.end(), [](const auto& argument_loc) {
    const auto& argument = argument_loc.getArgument();
    return argument.getKind() == TemplateArgument::Type && argument.getAsType()->isDependentType();
  });
}
#endif // LLVM_VERSION_MAJOR < 23

class PreferCUDATraitsCheck final : public ClangTidyCheck
{
  static constexpr StringRef TRAITS_OPTION{"Traits"};
  static constexpr StringRef TRAITS_OPTION_DEFAULT{
    "::cuda::std::is_trivially_copyable,::cuda::is_trivially_copyable;"
    "::cuda::std::is_floating_point,::cuda::is_floating_point"};

  static constexpr StringRef DECLARATION_BIND{"d"};
  static constexpr StringRef USE_BIND{"u"};
  static constexpr StringRef CONTEXT_BIND{"c"};

public:
  PreferCUDATraitsCheck(StringRef name, ClangTidyContext* context)
      : ClangTidyCheck{name, context}
      , traits_option_{Options.get(TRAITS_OPTION, TRAITS_OPTION_DEFAULT)}
  {
    llvm::SmallVector<StringRef> entries;
    traits_option_.split(entries, /*Separator=*/';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    sources_.reserve(entries.size() * 3);

    for (auto&& entry : entries)
    {
      auto&& [from, to] = entry.split(',');

      from = from.trim();
      to   = to.trim();

      if (from.empty() || to.empty() || to.contains(','))
      {
        configurationDiag("invalid %0 entry '%1': expected from,to", DiagnosticIDs::Level::Error)
          << TRAITS_OPTION << entry;
        continue;
      }

      if (!from.starts_with("::"))
      {
        configurationDiag("invalid %0 from entry '%1': must be a fully qualified symbol (starting from ::)",
                          DiagnosticIDs::Level::Error)
          << TRAITS_OPTION << from;
        continue;
      }

      if (!to.starts_with("::"))
      {
        configurationDiag("invalid %0 to entry '%1': must be a fully qualified symbol (starting from ::)",
                          DiagnosticIDs::Level::Error)
          << TRAITS_OPTION << to;
        continue;
      }

      add_replacement_(from, to.str());
      if (!from.ends_with("_v") && !from.ends_with("_t"))
      {
        add_replacement_((from + "_v").str(), (to + "_v").str());
        add_replacement_((from + "_t").str(), (to + "_t").str());
      }
    }

    if (sources_.empty())
    {
      configurationDiag("invalid %0 entry '%1': expected at least 1 entry", DiagnosticIDs::Level::Error)
        << TRAITS_OPTION << traits_option_;
    }
  }

  void storeOptions(ClangTidyOptions::OptionMap& options) override
  {
    Options.store(options, TRAITS_OPTION, traits_option_);
  }

  void registerMatchers(MatchFinder* finder) override
  {
    const auto declaration =
      namedDecl(hasAnyName(std::vector<StringRef>{sources_.begin(), sources_.end()})).bind(DECLARATION_BIND);
    const auto use_context = hasAncestor(decl().bind(CONTEXT_BIND));

    // Only check generic types. For example:
    //
    // is_trivially_copyable<T>       // Matches.
    // is_trivially_copyable<T*>      // Matches.
    // is_trivially_copyable<int>     // Does not match.
    //
    // A concrete type does not have the same unknown CUDA special-member behavior.
    const auto generic_argument = hasAnyTemplateArgumentLoc(hasTypeLoc(loc(isDependentType())));
#if LLVM_VERSION_MAJOR < 23
    const auto generic_lookup_argument = hasDependentTypeArgument();
#else
    const auto& generic_lookup_argument = generic_argument;
#endif

    // Reject concrete arguments before resolving declarations or traversing ancestors.
    // Each node kind gets one matcher, independent of the number of configured traits.
    //
    // Given:
    //
    // cuda::std::is_trivially_copyable<T> x;
    // static_assert(cuda::std::is_trivially_copyable<T>::value);
    // cuda::std::is_trivially_copyable<int> y;
    // cuda::is_trivially_copyable<T> z;
    //
    // Matches both occurrences of "cuda::std::is_trivially_copyable<T>".
    // Does not match "cuda::std::is_trivially_copyable<int>" or "cuda::is_trivially_copyable<T>".
    // TypeLoc provides the written type's source location for the replacement.
    // Ignore compiler-generated nodes, including copies from template instantiation.
    finder->addMatcher(
      traverse(TK_IgnoreUnlessSpelledInSource,
               templateSpecializationTypeLoc(
                 generic_argument, loc(templateSpecializationType(hasDeclaration(declaration))), use_context)
                 .bind(USE_BIND)),
      this);

    // Variable templates require expression matchers, unlike the type in "is_trivially_copyable<T>::value".
    // Given:
    //
    // static_assert(cuda::std::is_trivially_copyable_v<T>);
    // using cuda::std::is_trivially_copyable_v;
    // static_assert(is_trivially_copyable_v<T>);
    // static_assert(cuda::std::is_trivially_copyable_v<int>);
    // static_assert(cuda::is_trivially_copyable_v<T>);
    //
    // The expression matchers below match the first two assertions, but neither concrete arguments nor the replacement
    // trait. Clang represents resolved references as DeclRefExpr nodes pointing to the specialization.
    finder->addMatcher(traverse(TK_IgnoreUnlessSpelledInSource,
                                declRefExpr(generic_argument, to(varDecl(declaration)), use_context).bind(USE_BIND)),
                       this);

    // Separate registrations let MatchFinder reject unrelated expression kinds before evaluating these matchers.
    // A dependent lookup retains candidate declarations instead. For example:
    //
    // using cuda::std::is_trivially_copyable_v;
    // is_trivially_copyable_v<T>
    //
    // Match the use, not the using declaration. Follow using declarations to the original trait declaration.
    finder->addMatcher(
      traverse(TK_IgnoreUnlessSpelledInSource,
               unresolvedLookupExpr(
                 generic_lookup_argument, hasAnyDeclaration(namedDecl(hasUnderlyingDecl(declaration))), use_context)
                 .bind(USE_BIND)),
      this);
  }

  void check(const MatchFinder::MatchResult& result) override
  {
    auto&& nodes = result.Nodes;
    auto&& ctx   = *result.Context;

    const auto* const declaration = nodes.getNodeAs<NamedDecl>(DECLARATION_BIND);
    const auto* const use_context = nodes.getNodeAs<Decl>(CONTEXT_BIND)->getDeclContext();
    auto qualified_name           = declaration->getQualifiedNameAsString();
    const auto& replacement       = replacements_.at(qualified_name);

    NestedNameSpecifierLoc qualifier;
    SourceLocation name_loc;
    std::string original = std::move(qualified_name);

    if (const auto* const type = nodes.getNodeAs<TemplateSpecializationTypeLoc>(USE_BIND))
    {
      qualifier = type->getQualifierLoc();
      name_loc  = type->getTemplateNameLoc();
      original  = TypeName::getFullyQualifiedName(type->getType().getDesugaredType(ctx), ctx, ctx.getPrintingPolicy());
    }
    else if (const auto* const reference = nodes.getNodeAs<DeclRefExpr>(USE_BIND))
    {
      qualifier = reference->getQualifierLoc();
      name_loc  = reference->getLocation();
    }
    else if (const auto* const lookup = nodes.getNodeAs<UnresolvedLookupExpr>(USE_BIND))
    {
      qualifier = lookup->getQualifierLoc();
      name_loc  = lookup->getNameLoc();
    }
    else
    {
      llvm::reportFatalUsageError("cccl-prefer-cuda-traits: unhandled node kind, this is a bug in the check");
    }

    // Editing a macro definition affects every expansion:
    //
    // #define TRAIT cuda::std::foo // Should become cuda::foo.
    // TRAIT<T>   generic;
    // TRAIT<int> concrete; // Also becomes cuda::foo<int>.
    //
    // We accept this side effect for macro fixes, although the check otherwise excludes concrete arguments.
    const auto begin_loc = qualifier ? qualifier.getBeginLoc() : name_loc;

    // Given cuda::foo -> cuda::std::foo, preserve the template arguments and member access:
    //
    // cuda::foo<T>::value
    // ^-------^
    //   range
    const auto range = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(begin_loc, name_loc), *result.SourceManager, ctx.getLangOpts());
    const auto target = make_replacement_(ctx, declaration, use_context, qualifier, name_loc, replacement);
    auto diagnostic   = diag(name_loc, "use '%0' instead of '%1' for generic types") << target << original;

    // A partial macro expansion can have no corresponding file range:
    //
    // #define VALUE(T) cuda::foo_v<T>
    // static_assert(VALUE(T));
    //
    // The name range covers cuda::foo_v, but VALUE(T) expands to cuda::foo_v<T>.
    // Replacing the whole invocation discards <T>, so makeFileCharRange returns an invalid range.
    // Keep the warning even when no safe replacement exists.
    if (range.isValid())
    {
      diagnostic << FixItHint::CreateReplacement(range, target);
    }
  }

private:
  void add_replacement_(StringRef from, std::string target)
  {
    replacements_[from.drop_front(2 /* "::" */)] = std::move(target);
    sources_.push_back(from.str());
  }

  [[nodiscard]] static std::string make_replacement_(
    ASTContext& ctx,
    const NamedDecl* declaration,
    const DeclContext* use_context,
    NestedNameSpecifierLoc qualifier,
    SourceLocation name_loc,
    StringRef target)
  {
    auto scope = qualifier.getNestedNameSpecifier();

    if (scope.isFullyQualified())
    {
      // Given the mapping cuda::foo -> cuda::std::foo:
      //
      // namespace cuda {
      //   ::cuda::foo<T> v;  // Must become ::cuda::std::foo<T> v;
      // }
      //
      // replaceNestedName() returns "std::foo" here, so use the target directly to preserve the leading ::.
      return target.str();
    }

    if (!scope)
    {
      // Given the mapping cuda::foo -> cuda::std::foo:
      //
      // using cuda::foo;
      // foo<T> v;  // Must become cuda::std::foo<T> v;
      //
      // With an empty scope, replaceNestedName() returns "foo", which still names the original trait.
      // It assumes another fix updates the using declaration:
      //
      // https://github.com/llvm/llvm-project/blob/main/clang/lib/Tooling/Refactoring/Lookup.cpp
      //
      // We leave using declarations unchanged, so supply the trait's namespace instead.
      if (const auto* const ns = dyn_cast<NamespaceDecl>(declaration->getDeclContext()->getEnclosingNamespaceContext()))
      {
        // namespace cuda {
        //   foo<T> v;  // Must become std::foo<T> v;
        // }
        //
        // Supply cuda:: without a leading :: so the helper can omit the enclosing namespace.
        scope = NestedNameSpecifier{ctx, ns, /*Prefix=*/std::nullopt};
      }
      // A trait at global scope has no NamespaceDecl. replaceNestedName() handles its empty scope directly.
    }

    return tooling::replaceNestedName(scope, name_loc, use_context, declaration, target);
  }

  StringRef traits_option_{};
  std::vector<std::string> sources_{};
  llvm::StringMap<std::string> replacements_{};
};

// ==========================================================================================

class PreferCUDATraitsCheckModule final : public ClangTidyModule
{
public:
  static constexpr StringRef CHECK_NAME{"cccl-prefer-cuda-traits"};

  void addCheckFactories(ClangTidyCheckFactories& check_factories) override
  {
    check_factories.registerCheck<PreferCUDATraitsCheck>(CHECK_NAME);
  }
};

// Register the module using this statically initialized variable.
// NOLINTNEXTLINE(cert-err58-cpp, bugprone-throwing-static-initialization)
ClangTidyModuleRegistry::Add<PreferCUDATraitsCheckModule> _{
  PreferCUDATraitsCheckModule::CHECK_NAME, "Checks for use of cuda::std::meow which is superseded by cuda::meow"};
} // namespace
} // namespace clang::tidy::cccl
