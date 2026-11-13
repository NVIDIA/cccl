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
constexpr StringRef CHECK_NAME{"cccl-hide-from-abi"};

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

AST_MATCHER(CXXRecordDecl, isLocalClass) // NOLINT
{
  return Node.isLocalClass() != nullptr;
}

class HideFromABICheck final : public ClangTidyCheck
{
  static constexpr StringRef IGNORE_TEST_FILES_OPTION{"IgnoreTestFiles"};

  static constexpr StringRef FUNCTION_BIND{""};
  static constexpr StringRef DEFAULTED_METHOD_BIND{"d"};

public:
  HideFromABICheck(StringRef name, ClangTidyContext* context)
      : ClangTidyCheck{name, context}
      , header_extensions_{std::addressof(context->getHeaderFileExtensions())}
      , ignore_test_files_{Options.get(IGNORE_TEST_FILES_OPTION, /*Default=*/true)}
  {}

  void storeOptions(ClangTidyOptions::OptionMap& options) override
  {
    Options.store(options, IGNORE_TEST_FILES_OPTION, ignore_test_files_);
  }

  [[nodiscard]] std::optional<TraversalKind> getCheckTraversalKind() const override
  {
    return TK_IgnoreUnlessSpelledInSource;
  }

  void registerMatchers(MatchFinder* finder) override
  {
    // Check tests can opt in because their inputs also reside in test directories.
    const auto file_filter = ignore_test_files_ ? decl(unless(util::isExpansionInTestOrExampleFile())) : decl();

    // FunctionDecl includes methods, constructors, destructors, and conversion operators.
    finder->addMatcher(
      functionDecl(
        // Functions may be declared any number of times, so we need to make sure we check all
        // of them. We could do this by removing the isDefinition() condition below, but I
        // found that iterating over redecls() inside check() to be faster since we visit each
        // function only once.
        isDefinition(),
        unless(anyOf(isImplicit(),
                     isDeleted(),
                     isConsteval(),
                     // Related to the redeclration problem above, there is another issue,
                     // which is that visibility attributes on just one declaration sets the
                     // visibility for all re-decls without those re-decls needing to specify
                     // them again. Since these are inherited, the hasAttr(Visibility) would
                     // mean we skip all such functions.
                     //
                     // On the one hand we could mandate that all redecls repeat the
                     // attributes, to be safe against the scenario where you accidentally
                     // delete the redecl that sets the attribute and now the symbol is
                     // inadvertently public.
                     //
                     // But we are already protected against this; since the symbol is public,
                     // this check would fire and tell you to set a visibility attribute
                     // (helpfully flagging all re-decls as well). Conflicting visibility is
                     // also ok since that's a regular old compiler error.
                     hasAttr(attr::Visibility),
                     // Local class members have no linkage, including members of nested classes.
                     cxxMethodDecl(ofClass(cxxRecordDecl(anyOf(isLambda(), isLocalClass())))),
                     isExpansionInSystemHeader())),
        file_filter,
        util::isExpansionInHeaderFile(header_extensions_),
        optionally(cxxMethodDecl(isDefaulted()).bind(DEFAULTED_METHOD_BIND)))
        .bind(FUNCTION_BIND),
      this);
  }

  void check(const MatchFinder::MatchResult& result) override
  {
    const auto& nodes       = result.Nodes;
    const auto& SM          = *result.SourceManager;
    const auto is_defaulted = nodes.getNodeAs<CXXMethodDecl>(DEFAULTED_METHOD_BIND) != nullptr;
    const auto* const fn    = nodes.getNodeAs<FunctionDecl>(FUNCTION_BIND);

    for (const auto* const decl : fn->redecls())
    {
      if (is_defaulted)
      {
        handle_defaulted_method_(decl, SM);
      }
      else if (decl->hasAttr<CUDAGlobalAttr>())
      {
        handle_kernel_(decl, SM);
      }
      else
      {
        handle_regular_functions_(decl, SM);
      }
    }
  }

private:
  void handle_defaulted_method_(const FunctionDecl* fn, const SourceManager& SM)
  {
    // Special member functions which are defaulted can't have the usual _CCCL_*_API macros
    // because nvcc forbids attaching explicit __host__ __device__ attributes to
    // these. Instead they need _CCCL_HIDE_FROM_ABI.
    //
    // We don't need to do a search because by virtue of being here (the function has no
    // visibility attributes) we know that it's missing the macro.
    const auto begin = SM.getExpansionLoc(fn->getInnerLocStart());

    diag(fn->getLocation(), "use _CCCL_HIDE_FROM_ABI to limit ABI visibility")
      << FixItHint::CreateInsertion(begin, "_CCCL_HIDE_FROM_ABI ");
  }

  void handle_kernel_(const FunctionDecl* fn, const SourceManager& SM)
  {
    const auto& lang_opts = getLangOpts();
    const auto loc        = unwrap_macro_expansions_(SM, fn->getLocation());
    const auto range      = get_annotation_range_(fn, SM);
    auto buf              = SmallString<128>{};
    const auto token      = util::findTokenTextInRange(range, SM, lang_opts, [&](const Token& token) {
      buf.reserve(token.getLength());

      const auto* ptr = buf.data();
      const auto len  = Lexer::getSpelling(token, ptr, SM, lang_opts);

      return StringRef{ptr, len} == "__global__";
    });

    auto d = diag(loc, "use _CCCL_KERNEL_ATTRIBUTES to limit ABI visibility") << loc;

    if (token.isValid())
    {
      d << FixItHint::CreateReplacement(
        CharSourceRange::getTokenRange(token.getBegin(), token.getBegin()), "_CCCL_KERNEL_ATTRIBUTES");
    }
  }

  void handle_regular_functions_(const FunctionDecl* fn, const SourceManager& SM)
  {
    const auto [space, matches] = find_execution_space_(get_annotation_range_(fn, SM), SM);

    if (matches.empty())
    {
      const auto loc = unwrap_macro_expansions_(SM, fn->getLocation());

      // Without an execution-space macro, we cannot choose an API macro for the fix-it.
      diag(loc,
           "function needs an API macro (one of _CCCL_API, _CCCL_DEVICE_API, or _CCCL_HOST_API) to limit ABI "
           "visibility")
        << loc;
      return;
    }

    const auto replacement = get_api_macro_(space);
    const auto loc         = matches.front().getBegin();
    auto d                 = //
      diag(loc, "use %0 instead to limit ABI visibility") //
      << replacement //
      << matches.front() //
      << FixItHint::CreateReplacement(CharSourceRange::getTokenRange(loc, loc), replacement);

    for (auto&& m : llvm::drop_begin(matches))
    {
      d << m << FixItHint::CreateRemoval(m);
    }
  }

  [[nodiscard]] static SourceLocation unwrap_macro_expansions_(const SourceManager& SM, SourceLocation loc)
  {
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
    return SM.getSpellingLoc(loc);
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
    auto remaining        = range;
    auto buf              = SmallString<128>{};

    while (true)
    {
      const auto match = util::findTokenTextInRange(remaining, SM, lang_opts, [&](const Token& token) {
        buf.reserve(token.getLength());

        const auto* ptr = buf.data();
        const auto len  = Lexer::getSpelling(token, ptr, SM, lang_opts);
        const auto space =
          llvm::StringSwitch<execution_space>{{ptr, len}}
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
        llvm::reportFatalUsageError(CHECK_NAME + ": we should not get exec space none here");
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
        llvm::reportFatalUsageError(
          CHECK_NAME + ": unhandled execution space " + std::to_string(llvm::to_underlying(space)));
    }
  }

  const FileExtensionsSet* header_extensions_{};
  bool ignore_test_files_{};
};

class HideFromABICheckModule final : public ClangTidyModule
{
public:
  void addCheckFactories(ClangTidyCheckFactories& check_factories) override
  {
    check_factories.registerCheck<HideFromABICheck>(CHECK_NAME);
  }
};

// NOLINTNEXTLINE(cert-err58-cpp, bugprone-throwing-static-initialization)
ClangTidyModuleRegistry::Add<HideFromABICheckModule> _{
  CHECK_NAME, "Checks for missing ABI annotations on function definitions in headers"};
} // namespace
} // namespace clang::tidy::cccl
