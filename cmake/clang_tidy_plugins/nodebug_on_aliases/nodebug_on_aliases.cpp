// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>

#include <clang-tidy/ClangTidy.h>
#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang/AST/Decl.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Lex/Lexer.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>

namespace clang::tidy::cccl
{
namespace
{
using namespace ast_matchers; // NOLINT(google-build-using-namespace)

[[nodiscard]] bool is_ugly_name(StringRef str)
{
  if (str.size() < 2)
  {
    return false;
  }

  if (str[0] == '_' && str[1] >= 'A' && str[1] <= 'Z')
  {
    // _Foo
    return true;
  }

  // __foo
  return str.starts_with("__");
}

AST_MATCHER(NamedDecl, isUgly) // NOLINT
{
  return is_ugly_name(Node.getName());
}

AST_MATCHER(NamedDecl, isPretty) // NOLINT
{
  return !is_ugly_name(Node.getName());
}

class NoDebugOnAliasesCheck final : public ClangTidyCheck
{
  static constexpr StringRef INTERNAL_BIND{"i"};
  static constexpr StringRef TYPE_BIND{"t"};

public:
  using ClangTidyCheck::ClangTidyCheck;

  [[nodiscard]] std::optional<TraversalKind> getCheckTraversalKind() const override
  {
    return TK_IgnoreUnlessSpelledInSource;
  }

  void registerMatchers(MatchFinder* finder) override
  {
    finder->addMatcher(
      typeAliasDecl(unless(anyOf(hasAttr(attr::NoDebug), isPretty(), hasAncestor(functionDecl())))).bind(INTERNAL_BIND),
      this);

    finder->addMatcher(
      typeAliasDecl(
        unless(hasAttr(attr::NoDebug)),
        hasName("type"),
        hasAncestor(cxxRecordDecl(unless(has(namedDecl(unless(anyOf(
          typeAliasDecl(hasName("type")),
          typeAliasDecl(isUgly()),
          recordDecl(isImplicit()),
          templateTypeParmDecl(),
          nonTypeTemplateParmDecl(),
          templateTemplateParmDecl()))))))))
        .bind(TYPE_BIND),
      this);
  }

  void check(const MatchFinder::MatchResult& result) override
  {
    auto&& nodes = result.Nodes;

    if (const auto* alias = nodes.getNodeAs<TypeAliasDecl>(INTERNAL_BIND))
    {
      emit("Internal aliases should always be marked _CCCL_NODEBUG", *alias, result);
      return;
    }

    if (const auto* alias = nodes.getNodeAs<TypeAliasDecl>(TYPE_BIND))
    {
      emit("The alias of type traits should always be marked _CCCL_NODEBUG", *alias, result);
      return;
    }

    llvm::reportFatalUsageError("cccl-nodebug-on-aliases: unhandled node kind, this is a bug in the check");
  }

private:
  void emit(StringRef base_msg, const TypeAliasDecl& alias, const MatchFinder::MatchResult& result)
  {
    auto d = diag(alias.getLocation(), base_msg);

    // Editing a macro expansion can change other declarations that use the same macro.
    if (alias.getBeginLoc().isMacroID() || alias.getLocation().isMacroID())
    {
      return;
    }

    const auto loc =
      Lexer::getLocForEndOfToken(alias.getLocation(), /*Offset=*/0, *result.SourceManager, getLangOpts());

    if (loc.isValid())
    {
      d << FixItHint::CreateInsertion(loc, /*Code=*/" _CCCL_NODEBUG");
    }
  }
};

class NoDebugOnAliasesCheckModule final : public ClangTidyModule
{
public:
  static constexpr StringRef CHECK_NAME{"cccl-nodebug-on-aliases"};

  void addCheckFactories(ClangTidyCheckFactories& check_factories) override
  {
    check_factories.registerCheck<NoDebugOnAliasesCheck>(CHECK_NAME);
  }
};

// NOLINTNEXTLINE(cert-err58-cpp, bugprone-throwing-static-initialization)
ClangTidyModuleRegistry::Add<NoDebugOnAliasesCheckModule> _{
  NoDebugOnAliasesCheckModule::CHECK_NAME, "Checks for missing _CCCL_NODEBUG on internal type aliases"};
} // namespace
} // namespace clang::tidy::cccl
