// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>

#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>

namespace clang::tidy::cccl
{
namespace
{
using namespace ast_matchers; // NOLINT(google-build-using-namespace)

AST_POLYMORPHIC_MATCHER( // NOLINT
  isFullyQualified,
  AST_POLYMORPHIC_SUPPORTED_TYPES(
    DeclRefExpr, UnresolvedLookupExpr, DependentScopeDeclRefExpr, MemberExpr, CXXDependentScopeMemberExpr))
{
  return Node.getQualifier().isFullyQualified();
}

AST_MATCHER(CallExpr, isBuiltin) // NOLINT
{
  return Node.getBuiltinCallee() != 0;
}

[[nodiscard]] const DeclContext* getNonInlineNamespaceContext(const DeclContext* ctx)
{
  ctx = ctx->getEnclosingNamespaceContext();

  while (ctx->isInlineNamespace())
  {
    ctx = ctx->getParent()->getEnclosingNamespaceContext();
  }
  return ctx;
}

AST_MATCHER_P(NamespaceDecl, ignoringInlineNamespaces, DeclarationMatcher, InnerMatcher) // NOLINT
{
  const auto* const ns = dyn_cast<NamespaceDecl>(getNonInlineNamespaceContext(&Node));

  return ns && InnerMatcher.matches(*ns, Finder, Builder);
}

// The builtin hasDeclContext() matcher returns the outer class for nested class declarations,
// but we want the namespace that the outer class was defined in.
AST_MATCHER_P(NamedDecl, hasEnclosingNamespace, DeclarationMatcher, InnerMatcher) // NOLINT
{
  const auto* const ns = dyn_cast<NamespaceDecl>(getNonInlineNamespaceContext(Node.getDeclContext()));

  return ns && InnerMatcher.matches(*ns, Finder, Builder);
}

AST_MATCHER_P( // NOLINT
  NestedNameSpecifier,
  ignoringDependentQualifiers,
  NestedNameSpecifierMatcher,
  InnerMatcher)
{
  auto qualifier = Node;

  // We need to look through dependent qualifiers before checking where the type comes from.
  // E.g. if we have:
  //
  // template <class T>
  // void example()
  // {
  //   T::nested_type::template rebind<int>::call();
  // }
  //
  // We don't know what nested_type or rebind<int> refer to until we substitute T.
  // But we can still check T, so we need to get past both of them.
  while (qualifier.getKind() == NestedNameSpecifier::Kind::Type)
  {
    // This can also come through an alias, e.g. using type = typename T::nested_type.
    const auto* const type           = qualifier.getAsType()->getUnqualifiedDesugaredType();
    const auto* const specialization = dyn_cast<TemplateSpecializationType>(type);

    // Here, nested_type is a DependentNameType, and rebind<int> is a
    // TemplateSpecializationType with a dependent template name. A type like Box<T> is also
    // dependent, but we already know it refers to Box. In that case we should stop and let
    // InnerMatcher check Box.
    if (!isa<DependentNameType>(type)
        && !(specialization && specialization->getTemplateName().getAsDependentTemplateName()))
    {
      break;
    }
    qualifier = type->getPrefix();
  }
  return qualifier && InnerMatcher.matches(qualifier, Finder, Builder);
}

class QualifiedCallCheck final : public ClangTidyCheck
{
  static constexpr StringRef CALL_BIND{""};
  static constexpr StringRef NS_BIND{"n"};

public:
  using ClangTidyCheck::ClangTidyCheck;

  [[nodiscard]] std::optional<TraversalKind> getCheckTraversalKind() const override
  {
    return TK_IgnoreUnlessSpelledInSource;
  }

  void registerMatchers(MatchFinder* finder) override
  {
    // The matcher here is rather complicated, because we the rules of when we can omit
    // namespaces is rather complicated.
    //
    // We need to qualify ordinary function calls no matter what. But we can also have
    // type-dependent calls which call overloaded operator() or static functions on a type
    // functor. For example, the allocator_traits interface:
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
    // // "::cuda::std::allocator_traits::allocate()".
    // std::allocator_traits<T>::allocate(...);
    //
    // namespace nested {
    // struct foo { static void g(); };
    // }
    //
    // // Outside of our immediate namespace, because it is "below". This should be
    // // "::cuda::std::ranges::nested::foo::g()".
    // nested::foo::g();
    //
    // }

    // 1. Match declarations whose immediate namespace equals the bound caller
    //    namespace. Compare namespace identity so reopened namespace blocks also match.
    const auto sameNamespaceDecl = namedDecl(hasEnclosingNamespace(declaresSameEntityAsBoundNode(NS_BIND.str())));

    // 2. Match a type qualifier whose underlying type belongs to that namespace or refers
    //    to a template parameter. Desugaring follows aliases to their underlying types.
    //
    // We can also have a dependent type qualified by one of these types. For example:
    //
    // template <class T>
    // void example()
    // {
    //   T::nested_type::call();
    //   Box<T>::template rebind<int>::call();
    // }
    //
    // Here, nested_type and rebind have no resolved declarations, but their qualifiers
    // refer to T and Box<T>. Follow unresolved qualifiers until we reach a known type.
    // We must not search template arguments, because they don't determine the namespace:
    //
    // other_ns::Box<typename T::nested_type>::call(); // Still requires ::other_ns::Box<...>::call().
    const auto exemptType =
      nestedNameSpecifier(ignoringDependentQualifiers(nestedNameSpecifier(specifiesType(hasUnqualifiedDesugaredType(
        mapAnyOf(recordType, templateSpecializationType, templateTypeParmType)
          .with(hasDeclaration(namedDecl(anyOf(
            // RecordType covers ordinary classes and concrete template specializations after
            // desugaring:
            //
            // struct S
            // {
            //   static void call();
            // };
            //
            // template <class T>
            // struct Box
            // {
            //   static void call();
            // };
            //
            // void example()
            // {
            //   S::call();        // RecordType refers to S.
            //   Box<int>::call(); // RecordType refers to the concrete specialization Box<int>.
            // }
            //
            // TemplateSpecializationType also covers dependent specializations without a concrete
            // record declaration:
            //
            // template <class T>
            // void example()
            // {
            //   Box<T>::call();
            // }
            //
            // Here, even though Box<T> remains dependent, its template declaration lives in a
            // namespace we can reason about.
            sameNamespaceDecl,
            // TemplateTypeParmType covers type parameters. TemplateSpecializationType also covers
            // specializations of template template parameters:
            //
            // template <class T, template <class> class Container>
            // void example()
            // {
            //   T::call();            // TemplateTypeParmType refers to T.
            //   Container<T>::call(); // TemplateSpecializationType refers to Container.
            // }
            //
            // In these cases we obviously don't require qualification.
            templateTypeParmDecl(),
            templateTemplateParmDecl())))))))));

    // 3. And finally, match the actual call expressions.
    const auto calls =
      callExpr(
        unless(anyOf(
          // builtins are special, they don't need ::
          isBuiltin(),
          // Ignore overloaded operator calls like [], ==, !=, <, etc. as these cannot
          // (and should not) be qualified.
          cxxOperatorCallExpr())),
        callee(expr(ignoringParenImpCasts(
          mapAnyOf(declRefExpr, unresolvedLookupExpr, dependentScopeDeclRefExpr)
            .with(unless(isFullyQualified()),
                  // Calls through variables or template parameters aren't free function calls
                  // and cannot be qualified
                  unless(declRefExpr(to(unless(functionDecl())))),
                  // Calls to a classes own static member function should not need qualification.
                  unless(declRefExpr(to(cxxMethodDecl()), unless(has(nestedNameSpecifier())))),
                  // static calls through direct template parameters (or aliases to them) also
                  // don't need qualification:
                  //
                  // template <class T>
                  // void f()
                  // {
                  //   T::foo(); // do not qualify (can't anywaway)
                  // }
                  // Ignore type-qualified calls when the type belongs to the caller's
                  // namespace. Follow unresolved type qualifiers. TK_AsIs permits
                  // matching implicit template specialization declarations.
                  unless(has(traverse(TK_AsIs, exemptType))))))))
        .bind(CALL_BIND);

    finder->addMatcher(
      namespaceDecl(
        matchesName("^::(cub|cuda)($|::)"),
        // Bind the normalized namespace before evaluating descendants.
        ignoringInlineNamespaces(namespaceDecl().bind(NS_BIND)),
        forEach(decl(
          // Need to avoid matching the namespaces above again
          unless(namespaceDecl()),
          forEachDescendant(calls)))),
      this);
  }

  void check(const MatchFinder::MatchResult& result) override
  {
    const auto* const expr = result.Nodes.getNodeAs<CallExpr>(CALL_BIND);

    diag(expr->getBeginLoc(), "Bad!") << expr->getSourceRange();
  }
};

class QualifiedCallCheckModule final : public ClangTidyModule
{
public:
  static constexpr StringRef CHECK_NAME{"cccl-qualified-call"};

  void addCheckFactories(ClangTidyCheckFactories& check_factories) override
  {
    check_factories.registerCheck<QualifiedCallCheck>(CHECK_NAME);
  }
};

// NOLINTNEXTLINE(cert-err58-cpp, bugprone-throwing-static-initialization)
ClangTidyModuleRegistry::Add<QualifiedCallCheckModule> _{
  QualifiedCallCheckModule::CHECK_NAME, "Diagnose all unqualified call expressions"};
} // namespace
} // namespace clang::tidy::cccl
