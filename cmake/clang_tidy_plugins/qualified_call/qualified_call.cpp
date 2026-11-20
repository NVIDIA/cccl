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

class QualifiedCallCheck final : public ClangTidyCheck
{
  static constexpr StringRef CALL_BIND{""};

public:
  using ClangTidyCheck::ClangTidyCheck;

  [[nodiscard]] std::optional<TraversalKind> getCheckTraversalKind() const override
  {
    return TK_IgnoreUnlessSpelledInSource;
  }

  void registerMatchers(MatchFinder* finder) override
  {
    finder->addMatcher(
      namespaceDecl(
        hasAnyName("::cub", "::cuda::std"),
        forEachDescendant(
          callExpr(
            unless(anyOf(
              // builtins are special, they don't need ::
              isBuiltin(),
              // Ignore overloaded operator calls like [], ==, !=, <, etc. as these cannot (and
              // should not) be qualified.
              cxxOperatorCallExpr())),
            callee(expr(ignoringParenImpCasts(
              mapAnyOf(declRefExpr, unresolvedLookupExpr, dependentScopeDeclRefExpr).with(unless(isFullyQualified()))))))
            .bind(CALL_BIND))),
      this);
  }

  void check(const MatchFinder::MatchResult& result) override
  {
    const auto* const expr = result.Nodes.getNodeAs<CallExpr>(CALL_BIND);

    diag(expr->getCallee()->getBeginLoc(), "Bad!");
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
