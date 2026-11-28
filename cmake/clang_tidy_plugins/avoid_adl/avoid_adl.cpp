// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang-tidy/utils/OptionsUtils.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Sema/HeuristicResolver.h>
#include <common/matchers.h>
#include <common/type_traits.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Regex.h>

namespace clang::tidy::cccl
{
namespace
{
using namespace ast_matchers; // NOLINT(google-build-using-namespace)

AST_POLYMORPHIC_MATCHER( // NOLINT
  isFullyQualified,
  AST_POLYMORPHIC_SUPPORTED_TYPES(DeclRefExpr, UnresolvedLookupExpr, DependentScopeDeclRefExpr))
{
  return Node.getQualifier().isFullyQualified();
}

AST_MATCHER(CallExpr, isBuiltin) // NOLINT
{
  return Node.getBuiltinCallee() != 0;
}

AST_MATCHER(Decl, isLocalExternDecl) // NOLINT
{
  return Node.isLocalExternDecl();
}

AST_MATCHER(FunctionDecl, isHiddenFriend) // NOLINT
{
  return Node.getFriendObjectKind() == Decl::FriendObjectKind::FOK_Undeclared;
}

// We need the parameter named in the source, not the type we substituted for it.
AST_MATCHER_P(SubstTemplateTypeParmType, hasReplacedParameter, DeclarationMatcher, InnerMatcher) // NOLINT
{
  return InnerMatcher.matches(*Node.getReplacedParameter(), Finder, Builder);
}

// Return the enclosing non-inline namespace, or the translation unit.
//
// ctx: Starting declaration context.
//
// namespace cub { inline namespace abi { struct traits {}; } }
//                                               ^----^ <-- ctx.
// get_non_inline_namespace_context(ctx) -> cub.
[[nodiscard]] const DeclContext* get_non_inline_namespace_context(const DeclContext& ctx)
{
  auto ret = ctx.getEnclosingNamespaceContext();

  while (ret->isInlineNamespace())
  {
    ret = ret->getParent()->getEnclosingNamespaceContext();
  }
  return ret;
}

AST_MATCHER_P(NamespaceDecl, ignoringInlineNamespaces, DeclarationMatcher, InnerMatcher) // NOLINT
{
  const auto* const ns = dyn_cast<NamespaceDecl>(get_non_inline_namespace_context(Node));

  return ns && InnerMatcher.matches(*ns, Finder, Builder);
}

// Find the first written component and type in a qualifier.
//
// qualifier: Qualifier written before the callee.
//
// other_ns::Box<T>::template rebind<int>::call();
// ^------^ <-- returned root.
// ^--------------^ <-- returned type.
//
// For the fixit, we need the first component that was actually written.
//
//    Critical: don't desugar here!
//
//    E.g. if we have a namespace-scope alias called traits inside cuda::std, we want
//    ::cuda::std::traits::construct(...), even if the underlying type lives in
//    other_ns. There is no ::other_ns::traits alias. We must not search template
//    arguments, because they don't determine the namespace:
//
// other_ns::Box<typename T::nested_type>::call(); // Still needs ::other_ns::Box<...>::call().
[[nodiscard]] std::pair<NestedNameSpecifier, const Type*> find_written_qualifiers(NestedNameSpecifier qualifier)
{
  auto root        = qualifier;
  const Type* type = nullptr;

  while (qualifier)
  {
    root = qualifier;
    if (qualifier.getKind() == NestedNameSpecifier::Kind::Type)
    {
      type      = qualifier.getAsType();
      qualifier = type->getPrefix();
    }
    else if (qualifier.getKind() == NestedNameSpecifier::Kind::Namespace)
    {
      qualifier = qualifier.getAsNamespaceAndPrefix().Prefix;
    }
    else
    {
      break;
    }
  }
  return {std::move(root), type};
}

// type: Written type whose declaration we need
// ctx: AST context that owns the type.
[[nodiscard]] const NamedDecl* find_type_declaration(const Type& type, ASTContext& ctx)
{
  constexpr StringRef BIND{""};
  const auto declaration = namedDecl().bind(BIND);

  return selectFirst<NamedDecl>(
    BIND,
    match(qualType(anyOf(substTemplateTypeParmType(hasReplacedParameter(declaration)), hasDeclaration(declaration))),
          QualType{&type, /*Quals=*/0},
          ctx));
}

// Types from our immediate namespace don't need qualification. Neither do template parameters,
// since we can't qualify those anyway. Namespace identity also handles reopened namespace blocks.
//
// type: Declaration of the first written type.
// caller: Namespace that contains the call, with inline namespaces removed.
[[nodiscard]] bool is_exempt_type(const NamedDecl& type, const NamespaceDecl& caller)
{
  if (type.isTemplateParameter())
  {
    return true;
  }

  const auto* const ctx = type.getNonTransparentDeclContext();
  // Class and function-local type names don't need qualification either.
  return ctx->isRecord() || ctx->isFunctionOrMethod() || caller.Equals(get_non_inline_namespace_context(*ctx));
}

// Find the declaration context for an unqualified call's fixit prefix, or nullptr if unknown.
//
// call: Call expression whose declaration we need.
// callee: Call target without parentheses or implicit casts.
// ctx: AST context that owns the call.
[[nodiscard]] const DeclContext*
find_function_qualification_context(const CallExpr& call, const Expr& callee, ASTContext& ctx)
{
  // Here, we have an ordinary function call. But we still can't just use the function's
  // declaration, because the call can come through a using declaration:
  //
  // namespace other_ns { void foo(long); }
  // namespace cub { using other_ns::foo; }
  // namespace other_ns { void foo(int); }
  // namespace cub { void example() { foo(0); } }
  //
  // This calls foo(long). If we suggest ::other_ns::foo(0), we silently change it to foo(int).
  // getFoundDecl() gives us the using declaration, so we can suggest ::cub::foo(0) instead.
  if (const auto* const reference = dyn_cast<DeclRefExpr>(&callee))
  {
    return reference->getFoundDecl()->getNonTransparentDeclContext();
  }

  // We can have an UnresolvedLookupExpr even when ordinary lookup finds the function template:
  //
  // template<class T> constexpr int bits();
  // template<class T> inline constexpr int value = bits<T>();
  //                                               ^-----^ <-- UnresolvedLookupExpr.
  //
  // It's complicated to deduce this, but luckily Clang can do it for us with the power of
  // heuristics.
  const auto resolver = HeuristicResolver{ctx};

  for (const auto* const decl : resolver.resolveCalleeOfCallExpr(&call))
  {
    if (const auto* const ctx = decl->getNonTransparentDeclContext())
    {
      return ctx;
    }
  }
  return nullptr;
}

// Build a qualification fix from the supplied declaration context, if possible.
//
// callee: Call target without parentheses or implicit casts.
// context: Declaration context to print as the prefix. Must be non-transparent if not null.
//
// namespace cub {
//           ^-^ <-- context.
//   using traits = other_ns::external;
//
//   namespace detail {
//     void example() { (traits::call)(); }
//                       ^----------^ <-- callee, without the parentheses.
//   }
// }
//
// The fixit produces (::cub::traits::call)().
[[nodiscard]] std::optional<FixItHint>
try_make_fixit(const Expr& callee, const SourceManager& SM, const DeclContext* context)
{
  if (!context)
  {
    return std::nullopt;
  }

  const auto location = [&] {
    // We should not try to fix up the macro itself. We can use the same macro from different
    // namespaces, so adding a prefix to its body could break the other callers.
    //
    // But we *can* fixup macro arguments like:
    //
    // CubDebug(cudaFoo())
    // ^        ^------^
    // macro     we can fix this
    //
    // Note this is not strictly 100% correct if the macro does extreme token concatenation
    // gymnastics. But 99% of the time, it is The Right Thing To Do, so we accept the cost of a
    // few false positives here.
    auto loc = callee.getBeginLoc();

    while (loc.isMacroID() && SM.isMacroArgExpansion(loc))
    {
      loc = SM.getImmediateMacroCallerLoc(loc);
    }
    return loc;
  }();

  if (location.isInvalid() || location.isMacroID())
  {
    return std::nullopt;
  }

  constexpr auto next_scope = [](const DeclContext* scope) {
    return scope->getParent()->getNonTransparentContext();
  };

  for (const auto* scope = context; !scope->isTranslationUnit(); scope = next_scope(scope))
  {
    // If any part of the declcontext stack is not a namespace, then making a qualification
    // fixit is too complicated and we just give up.
    //
    // Note we already excluded class declcontexts in the original matcher, and
    // is_exempt_type() excludes class-scoped type names. Also namespaces themselves cannot
    // have a class/struct as their enclosing declcontext, so the only thing this guards
    // against is function-local namespace aliases (in which case scope is the function).
    if (!scope->isNamespace())
    {
      return std::nullopt;
    }

    if (scope == context)
    {
      // We don't want ABI namespaces in the fixit. E.g. __string_view_detail::foo() inside
      // cuda::std::__4 should become ::cuda::std::__string_view_detail::foo(). But the printer
      // only suppresses inline namespaces in a declaration's context, so we also need to skip
      // an inline namespace if it is the outermost declcontext ourselves.
      //
      // Likewise, we should skip all top-level anon namespaces, since these are printed as
      // "::cub::(anonymous)::" if they are the bottom of the declcontext stack
      if (const auto& ns = cast<NamespaceDecl>(*scope); ns.isInline() || ns.isAnonymousNamespace())
      {
        context = next_scope(scope);
      }
    }
  }

  auto prefix = SmallString<util::SMOL_LEN>{"::"};

  if (!context->isTranslationUnit())
  {
    // The only scenario in which the declcontext of the call expression is unnamed at this
    // point is if it's in the global namespace (recall we handle inline/anon namespaces, and
    // unnamed classes above already).
    prefix += util::fully_qualified_name(cast<NamedDecl>(*context));
    prefix += "::";
  }

  return FixItHint::CreateInsertion(location, prefix);
}

class AvoidADLCheck final : public ClangTidyCheck
{
  static constexpr StringRef NAMESPACES_OPTION{"Namespaces"};
  static constexpr StringRef NAMESPACE_MACROS_OPTION{"NamespaceMacros"};

  static constexpr StringRef QUALIFIER_BIND{"q"};
  static constexpr StringRef CALL_BIND{""};
  static constexpr StringRef CALLER_NS_BIND{"n"};

public:
  AvoidADLCheck(StringRef name, ClangTidyContext* context)
      : ClangTidyCheck{name, context}
      , namespaces_option_{Options.get(NAMESPACES_OPTION, "cub;cuda")}
      , namespace_macros_option_{Options.get(NAMESPACE_MACROS_OPTION, "cub=_CUB::")}
  {
    for (const auto ns : utils::options::parseStringList(namespaces_option_))
    {
      if (!namespace_regex_.empty())
      {
        namespace_regex_ += '|';
      }
      namespace_regex_ += llvm::Regex::escape(ns);
    }

    for (const auto entry : utils::options::parseStringList(namespace_macros_option_))
    {
      auto [ns, macro] = entry.split('=');

      ns    = ns.trim();
      macro = macro.trim();

      if (ns.empty() || macro.empty() || macro.contains('='))
      {
        configurationDiag("invalid %0 entry '%1': expected namespace=macro", DiagnosticIDs::Level::Error)
          << NAMESPACE_MACROS_OPTION << entry;
        continue;
      }

      constexpr auto qualify = [](StringRef src, bool full) {
        std::string ret;

        ret.reserve(src.size() + (full ? 2 * StringRef{"::"}.size() : 0));
        if (full && !src.starts_with("::"))
        {
          ret += "::";
        }
        ret += src;
        if (full && !src.ends_with("::"))
        {
          ret += "::";
        }
        return ret;
      };

      namespace_macros_.emplace_back(
        qualify(ns, /*full=*/true),
        // Macros, being macros, should have the full qualification necessary to just be
        // replaced wholesale. We do not attempt to add additional "::" to it.
        qualify(macro, /*full=*/false));
    }
  }

  void storeOptions(ClangTidyOptions::OptionMap& options) override
  {
    Options.store(options, NAMESPACES_OPTION, namespaces_option_);
    Options.store(options, NAMESPACE_MACROS_OPTION, namespace_macros_option_);
  }

  [[nodiscard]] std::optional<TraversalKind> getCheckTraversalKind() const override
  {
    return TK_IgnoreUnlessSpelledInSource;
  }

  void registerMatchers(MatchFinder* finder) override
  {
    if (namespace_regex_.empty())
    {
      return;
    }

    const auto local_callee = namedDecl(anyOf(
      // Is a class member call
      hasDeclContext(cxxRecordDecl()),
      // Is a function-local function definition:
      //
      // void foo()
      // {
      //   void local_fn()
      //   {
      //     ...
      //   }
      //
      //   local_fn();
      // }
      isLocalExternDecl()));

    const auto calls =
      callExpr(
        // Builtins and overloaded operator calls like [], ==, etc. don't need qualification.
        unless(anyOf(isBuiltin(), cxxOperatorCallExpr())),
        callee(expr(ignoringParenImpCasts(
          // We only handle name references here. Member access expressions need a different rewrite.
          mapAnyOf(declRefExpr, unresolvedLookupExpr, dependentScopeDeclRefExpr)
            .with(unless(anyOf(
                    isFullyQualified(),
                    // Calls through...
                    declRefExpr(to(anyOf(
                      // ...hidden friends...
                      functionDecl(isHiddenFriend()),
                      // ...named function arguments...
                      parmVarDecl(),
                      // ...template params...
                      nonTypeTemplateParmDecl(),
                      // ...or local variable callables (which may not match
                      // cxxOperatorCallExpr() if they are heavily templated)...
                      varDecl(hasLocalStorage())))),
                    // ...aren't qualifiable calls.
                    expr(
                      // Unqualified calls...
                      unless(has(nestedNameSpecifier())),
                      // ...that are declcontext-local (see above) don't need qualification.
                      anyOf(declRefExpr(to(local_callee)), unresolvedLookupExpr(hasAnyDeclaration(local_callee)))))),
                  optionally(has(nestedNameSpecifier().bind(QUALIFIER_BIND))))))))
        .bind(CALL_BIND);

    finder->addMatcher(
      util::namespaceMatches(
        "^::(" + namespace_regex_ + ")($|::)",
        ignoringInlineNamespaces(namespaceDecl().bind(CALLER_NS_BIND)),
        forEach(decl(
          // We must match all calls exactly in the current namespace and must not recursively
          // descend into nested namespaces so that calls get the correct caller namespace
          // binding:
          //
          // namespace cub {
          //           ^-^ <-- wrong caller namespace.
          //   namespace detail {
          //             ^----^ <-- correct caller namespace.
          //
          //     struct traits { static void call(); };
          //     void example() { traits::call(); }
          //
          //   }
          // }
          //
          // Here, traits and example both belong to cub::detail, so we don't require
          // qualification. If we descend from cub into detail, we bind cub as caller
          // namespace. traits then appears to come from a different namespace and we
          // incorrectly warn.
          unless(namespaceDecl()),
          forEachDescendant(calls)))),
      this);
  }

  void check(const MatchFinder::MatchResult& result) override
  {
    auto&& nodes                = result.Nodes;
    auto&& ctx                  = *result.Context;
    const auto& call            = *nodes.getNodeAs<CallExpr>(CALL_BIND);
    const auto& callee          = *call.getCallee()->IgnoreParenImpCasts();
    const auto* const qualifier = nodes.getNodeAs<NestedNameSpecifier>(QUALIFIER_BIND);

    if (!qualifier)
    {
      diagnose_(result, callee, find_function_qualification_context(call, callee, ctx));
      return;
    }

    // We need to qualify ordinary function calls. But for calls through a type, we also need
    // to check which namespace the type belongs to. For example, the allocator_traits
    // interface:
    //
    // // a type ("allocator_traits", not "T") dependent function call
    // allocator_traits<T>::construct(...);
    //
    // In this case we should only require qualification if the type comes from outside our
    // immediate namespace. E.g. if we are:
    //
    // namespace cuda::std::ranges {
    //
    // // Outside of our immediate namespace, because it is "above". This should be
    // // "::cuda::std::allocator_traits<T>::allocate(...)".
    // std::allocator_traits<T>::allocate(...);
    //
    // namespace nested {
    // struct foo { static void g(); };
    //        ^-^ <-- declaration.
    // }
    //
    // // Outside of our immediate namespace, because it is "below". This should be
    // // "::cuda::std::ranges::nested::foo::g()".
    // nested::foo::g();
    // ^----^ <-- root.
    // ^---------^ <-- type.
    //
    // }
    const auto [root, type]       = find_written_qualifiers(*qualifier);
    const auto* const declaration = type ? find_type_declaration(*type, ctx) : nullptr;

    if (declaration && is_exempt_type(*declaration, *nodes.getNodeAs<NamespaceDecl>(CALLER_NS_BIND)))
    {
      return;
    }

    // We can have namespace aliases here as well. E.g. cuda::std::views is an alias for
    // cuda::std::ranges::views. In that case we want ::cuda::std::views::foo(), so we need the
    // alias declaration itself. getUnderlyingDecl() would give us ranges::views instead.
    if (root.getKind() == NestedNameSpecifier::Kind::Namespace)
    {
      diagnose_(result, callee, root.getAsNamespaceAndPrefix().Namespace->getNonTransparentDeclContext());
    }
    else
    {
      diagnose_(result, callee, declaration ? declaration->getNonTransparentDeclContext() : nullptr);
    }
  }

private:
  void diagnose_(const MatchFinder::MatchResult& result, const Expr& callee, const DeclContext* ctx)
  {
    auto d   = diag(callee.getBeginLoc(), "Fully qualify all call expressions to avoid accidental ADL.");
    auto fix = try_make_fixit(callee, *result.SourceManager, ctx);

    if (!fix.has_value())
    {
      return;
    }

    for (const auto& [prefix, replacement] : namespace_macros_)
    {
      if (StringRef{fix->CodeToInsert}.starts_with(prefix))
      {
        fix->CodeToInsert.replace(0, prefix.size(), replacement);
        break;
      }
    }
    d << *std::move(fix);
  }

  std::string namespaces_option_;
  std::string namespace_macros_option_;
  std::string namespace_regex_;
  std::vector<std::pair<std::string, std::string>> namespace_macros_;
};

class AvoidADLCheckModule final : public ClangTidyModule
{
public:
  static constexpr StringRef CHECK_NAME{"cccl-avoid-adl"};

  void addCheckFactories(ClangTidyCheckFactories& check_factories) override
  {
    check_factories.registerCheck<AvoidADLCheck>(CHECK_NAME);
  }
};

// NOLINTNEXTLINE(cert-err58-cpp, bugprone-throwing-static-initialization)
ClangTidyModuleRegistry::Add<AvoidADLCheckModule> _{
  AvoidADLCheckModule::CHECK_NAME, "Diagnose all unqualified call expressions"};
} // namespace
} // namespace clang::tidy::cccl
