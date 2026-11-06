// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Adapted from https://github.com/llvm/llvm-project/blob/main/libcxx/test/tools/clang_tidy_checks/hide_from_abi.cpp.

#include <memory>
#include <optional>
#include <string>

#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang-tidy/FileExtensionsSet.h>
#include <clang/AST/ASTTypeTraits.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/Expr.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Lexer.h>
#include <clang/Lex/Token.h>
#include <common/lexer.h>
#include <common/matchers.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>

namespace clang::tidy::cccl
{
namespace
{
using namespace ast_matchers; // NOLINT(google-build-using-namespace)

enum class [[clang::flag_enum]] execution_space : std::uint8_t
{
  none   = 0,
  host   = 1 << 0,
  device = 1 << 1,
  tile   = 1 << 2,
};

[[nodiscard]] constexpr execution_space operator|(execution_space lhs, execution_space rhs)
{
  return static_cast<execution_space>(llvm::to_underlying(lhs) | llvm::to_underlying(rhs));
}

constexpr execution_space& operator|=(execution_space& lhs, execution_space rhs)
{
  lhs = lhs | rhs;
  return lhs;
}

class HideFromABICheck final : public ClangTidyCheck
{
  static constexpr StringRef FUNCTION_BIND{""};
  static constexpr StringRef SPECIAL_FUNCTION_BIND{"s"};

public:
  HideFromABICheck(StringRef name, ClangTidyContext* context)
      : ClangTidyCheck{name, context}
      , header_extensions_{std::addressof(context->getHeaderFileExtensions())}
  {}

  [[nodiscard]] std::optional<TraversalKind> getCheckTraversalKind() const override
  {
    return TK_IgnoreUnlessSpelledInSource;
  }

  void registerMatchers(MatchFinder* finder) override
  {
    const auto is_special_member = cxxMethodDecl(
      isDefaulted(),
      anyOf(cxxConstructorDecl(), cxxDestructorDecl(), isCopyAssignmentOperator(), isMoveAssignmentOperator()));

    // FunctionDecl includes methods, constructors, destructors, and conversion operators.
    finder->addMatcher(
      functionDecl(
        isDefinition(),
        unless(anyOf(isImplicit(),
                     isDeleted(),
                     isConsteval(),
                     hasAttr(attr::Visibility),
                     cxxMethodDecl(ofClass(cxxRecordDecl(isLambda()))),
                     isExpansionInSystemHeader(),
                     util::isExpansionInTestFile())),
        util::isExpansionInHeaderFile(header_extensions_),
        optionally(is_special_member.bind(SPECIAL_FUNCTION_BIND)))
        .bind(FUNCTION_BIND),
      this);
  }

  void check(const MatchFinder::MatchResult& result) override
  {
    const auto& nodes          = result.Nodes;
    const auto* const function = nodes.getNodeAs<FunctionDecl>(FUNCTION_BIND);
    const auto& SM             = *result.SourceManager;

    if (nodes.getNodeAs<CXXMethodDecl>(SPECIAL_FUNCTION_BIND))
    {
      // Special member functions which are defaulted can't have the usual _CCCL_*_API macros
      // because nvcc forbids attaching explicit __host__ __device__ attributes to
      // these. Instead they need _CCCL_HIDE_FROM_ABI.
      //
      // We don't need to do a search because by virtue of being here (the function has no
      // visibility attributes) we know that it's missing the macro.
      const auto begin = SM.getExpansionLoc(function->getInnerLocStart());

      diag(function->getLocation(), "use _CCCL_HIDE_FROM_ABI to limit ABI visibility")
        << FixItHint::CreateInsertion(begin, "_CCCL_HIDE_FROM_ABI ");
      return;
    }

    handle_regular_functions_(function, SM);
  }

private:
  void handle_regular_functions_(const FunctionDecl* fn, const SourceManager& SM)
  {
    const auto [space, matches] = find_execution_space_(get_annotation_range_(fn, SM), SM);

    if (matches.empty())
    {
      auto loc = fn->getLocation();
      // Unwrap any macros, otherwise given:
      //
      // #define MACRO(name) __device__ name()
      //
      // MACRO(Foo);
      //
      // we emit:
      //
      // error: Use _CCCL_DEVICE_API...
      // | MACRO(Foo);
      // |      ^
      //
      // Which is obviously not useful. We want to point to "__device__ name()" instead. The
      // other path below doesn't need this treatment because it already searches the textual
      // definition pointed to by the function definition and therefore implicitly unwraps
      // macros.
      while (loc.isMacroID() && SM.isMacroArgExpansion(loc))
      {
        loc = SM.getImmediateExpansionRange(loc).getBegin();
      }
      loc = SM.getSpellingLoc(loc);

      // Without an execution-space macro, we cannot choose an API macro for the fix-it.
      diag(loc,
           "function needs an API macro (one of _CCCL_API, _CCCL_DEVICE_API, or _CCCL_HOST_API) to limit ABI "
           "visibility")
        << loc;
      return;
    }

    const auto replacement = get_api_macro_(space);
    auto d                 = diag(matches.front().getBegin(), "use %0 instead to limit ABI visibility")
                          << replacement << matches.front()
          << FixItHint::CreateReplacement(
               CharSourceRange::getTokenRange(matches.front().getBegin(), matches.front().getBegin()), replacement);

    for (auto&& m : llvm::drop_begin(matches))
    {
      d << m << FixItHint::CreateRemoval(m);
    }
  }

  [[nodiscard]] static CharSourceRange get_annotation_range_(const FunctionDecl* fn, const SourceManager& SM)
  {
    // Getting the begin range here is a little fiddly. We are about to textually search the
    // macro annotations of the function, but depending on the compilation arguments many of
    // those macros define as empty. If we just did the naive thing and used
    // `function->getInnerLocStart()`, we would skip over such macro definitions. So instead we
    // need to go all the way back to the *end* of either the template parameter list or
    // constraint list (if any). So basically:
    //
    // #define EMPTY_MACRO /* empty */
    //
    // fn->getDescribedFunctionTemplate()->getTemplateParameters()->getRAngleLoc()
    //                   v        ...->getRequiresClause()->getEndLoc()
    // template <class T>         v
    //   requires my_requirement<T>
    // EMPTY_MACRO inline [[some::other_attribute]] void foo()
    //             ^
    // fn->getInnerLocStart()
    auto begin = fn->getInnerLocStart();

    if (const auto* const templ = fn->getDescribedFunctionTemplate())
    {
      if (const auto* const req = templ->getTemplateParameters()->getRequiresClause())
      {
        begin = req->getEndLoc();
      }
      else
      {
        begin = templ->getTemplateParameters()->getRAngleLoc();
      }
    }

    return CharSourceRange::getCharRange(SM.getExpansionLoc(begin), SM.getExpansionLoc(fn->getLocation()));
  }

  [[nodiscard]] std::pair<execution_space, SmallVector<CharSourceRange>>
  find_execution_space_(CharSourceRange range, const SourceManager& SM) const
  {
    const auto& lang_opts = getLangOpts();
    auto combined_space   = execution_space::none;
    auto matches          = SmallVector<CharSourceRange>{};

    for (auto remaining = range;;)
    {
      const auto match = util::findTokenTextInRange(remaining, SM, lang_opts, [&](const Token& token) {
        const auto spelling = Lexer::getSpelling(token, SM, lang_opts);
        const auto space =
          llvm::StringSwitch<execution_space>{spelling}
            .Cases({"_CCCL_DEVICE", "__device__"}, execution_space::device)
            .Cases({"_CCCL_HOST", "__host__"}, execution_space::host)
            .Cases({"_CCCL_TILE", "__tile__"}, execution_space::tile)
            .Case("_CCCL_HOST_DEVICE", execution_space::host | execution_space::device)
            .Default(execution_space::none);

        combined_space |= space;
        return space != execution_space::none;
      });

      if (match.isInvalid())
      {
        break;
      }

      matches.push_back(match);
      remaining = CharSourceRange::getCharRange(match.getEnd(), range.getEnd());
    }

    return {combined_space, std::move(matches)};
  }

  [[nodiscard]] static StringRef get_api_macro_(execution_space space)
  {
    switch (space)
    {
      case execution_space::none:
        // The caller rejects functions without recognized execution-space annotations.
        llvm::reportFatalUsageError("cccl-hide-from-abi: we should not get exec space none here");
      case execution_space::host:
        return "_CCCL_HOST_API";
      case execution_space::device:
        return "_CCCL_DEVICE_API";
      case execution_space::tile:
        return "_CCCL_TILE_API";
      case execution_space::host | execution_space::device:
        return "_CCCL_HOST_DEVICE_API";
      case execution_space::device | execution_space::tile:
        // We should really have a _CCCL_TILE_DEVICE_API macro for this
        return "_CCCL_TILE _CCCL_DEVICE_API";
      case execution_space::host | execution_space::device | execution_space::tile:
        return "_CCCL_API";
      default:
        const auto err = "cccl-hide-from-abi: unhandled execution space " + std::to_string(llvm::to_underlying(space));

        llvm::reportFatalUsageError(StringRef{err});
    }
  }

  const FileExtensionsSet* header_extensions_{};
};

class HideFromABICheckModule final : public ClangTidyModule
{
public:
  static constexpr StringRef CHECK_NAME{"cccl-hide-from-abi"};

  void addCheckFactories(ClangTidyCheckFactories& check_factories) override
  {
    check_factories.registerCheck<HideFromABICheck>(CHECK_NAME);
  }
};

// NOLINTNEXTLINE(cert-err58-cpp, bugprone-throwing-static-initialization)
ClangTidyModuleRegistry::Add<HideFromABICheckModule> _{
  HideFromABICheckModule::CHECK_NAME, "Checks for missing ABI annotations on function definitions in headers"};
} // namespace
} // namespace clang::tidy::cccl
