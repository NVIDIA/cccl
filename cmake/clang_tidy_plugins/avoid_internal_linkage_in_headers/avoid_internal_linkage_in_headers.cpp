// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>

#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang-tidy/FileExtensionsSet.h>
#include <clang-tidy/utils/FileExtensionsUtils.h>
#include <clang-tidy/utils/LexerUtils.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Lexer.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Path.h>

namespace clang::tidy::cccl
{
namespace
{
using namespace ast_matchers; // NOLINT(google-build-using-namespace)

AST_MATCHER_P(Decl, isInHeaderFile, const FileExtensionsSet*, extensions) // NOLINT
{
  const auto& source_manager = Finder->getASTContext().getSourceManager();
  const auto filename        = source_manager.getFilename(source_manager.getExpansionLoc(Node.getBeginLoc()));
  // LLVM's header utility rejects extensionless files even when the empty extension is configured.
  return utils::isExpansionLocInHeaderFile(Node.getBeginLoc(), source_manager, *extensions)
      || (extensions->contains("") && !filename.empty() && llvm::sys::path::extension(filename).empty());
}

AST_MATCHER(VarDecl, isFileVarDecl) // NOLINT
{
  return Node.isFileVarDecl();
}

class AvoidInternalLinkageInHeadersCheck final : public ClangTidyCheck
{
  static constexpr StringRef FUNCTION_BIND{"f"};
  static constexpr StringRef VARIABLE_BIND{"v"};

public:
  using ClangTidyCheck::ClangTidyCheck;

  [[nodiscard]] std::optional<TraversalKind> getCheckTraversalKind() const override
  {
    return TK_IgnoreUnlessSpelledInSource;
  }

  void registerMatchers(MatchFinder* finder) override
  {
    // misc-anonymous-namespace-in-header already diagnoses anonymous namespaces.
    const auto header_decl =
      allOf(unless(isExpansionInSystemHeader()),
            isInHeaderFile(&getHeaderFileExtensions()),
            unless(isInAnonymousNamespace()));

    finder->addMatcher(functionDecl(isStaticStorageClass(), unless(cxxMethodDecl()), header_decl).bind(FUNCTION_BIND),
                       this);

    finder->addMatcher(
      varDecl(isFileVarDecl(), isConstexpr(), unless(anyOf(isInline(), isTemplateInstantiation())), header_decl)
        .bind(VARIABLE_BIND),
      this);
  }

  void check(const MatchFinder::MatchResult& result) override
  {
    if (const auto* fn = result.Nodes.getNodeAs<FunctionDecl>(FUNCTION_BIND))
    {
      auto d = diag(fn->getLocation(), "static function %0 in a header has internal linkage") << fn;

      if (const auto range = keyword_range_(*fn, "static", *result.Context); range.isValid())
      {
        d << FixItHint::CreateReplacement(range, fn->isInlined() ? "" : "inline");
      }
      return;
    }

    if (const auto* var = result.Nodes.getNodeAs<VarDecl>(VARIABLE_BIND))
    {
      auto d = diag(var->getLocation(), "namespace-scope constexpr variable %0 in a header is not inline") << var;

      if (var->getStorageClass() == SC_Static)
      {
        if (const auto range = keyword_range_(*var, "static", *result.Context); range.isValid())
        {
          d << FixItHint::CreateReplacement(range, "inline");
        }
      }
      else
      {
        if (const auto range = keyword_range_(*var, "constexpr", *result.Context); range.isValid())
        {
          d << FixItHint::CreateInsertion(range.getBegin(), "inline ");
        }
      }
      return;
    }

    llvm::reportFatalUsageError("cccl-avoid-internal-linkage-in-headers: unhandled node kind, this is a bug in the "
                                "check");
  }

private:
  // Keywords like "static" and "constexpr" aren't actually real nodes in the clang AST, so we
  // need to find them lexically in the source.
  [[nodiscard]] static CharSourceRange
  keyword_range_(const NamedDecl& declaration, StringRef keyword, const ASTContext& context)
  {
    const auto& sm        = context.getSourceManager();
    const auto& lang_opts = context.getLangOpts();
    const auto range      = utils::lexer::findTokenTextInRange(
      CharSourceRange::getTokenRange(declaration.getBeginLoc(), declaration.getLocation()),
      sm,
      lang_opts,
      [&](const Token& token) {
        return Lexer::getSpelling(token, sm, lang_opts) == keyword;
      });

    return CharSourceRange::getTokenRange(range.getBegin(), range.getBegin());
  }
};

class AvoidInternalLinkageInHeadersCheckModule final : public ClangTidyModule
{
public:
  static constexpr StringRef CHECK_NAME{"cccl-avoid-internal-linkage-in-headers"};

  void addCheckFactories(ClangTidyCheckFactories& check_factories) override
  {
    check_factories.registerCheck<AvoidInternalLinkageInHeadersCheck>(CHECK_NAME);
  }
};

// NOLINTNEXTLINE(cert-err58-cpp, bugprone-throwing-static-initialization)
ClangTidyModuleRegistry::Add<AvoidInternalLinkageInHeadersCheckModule> _{
  AvoidInternalLinkageInHeadersCheckModule::CHECK_NAME,
  "Checks for static free functions and non-inline constexpr variables in headers"};
} // namespace
} // namespace clang::tidy::cccl
