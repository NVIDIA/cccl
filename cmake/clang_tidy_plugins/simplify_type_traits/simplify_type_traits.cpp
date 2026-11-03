//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <optional>
#include <string_view>

#include <clang-tidy/ClangTidy.h>
#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/TypeLoc.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Lex/Lexer.h>
#include <common/type_traits.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Config/llvm-config.h>

namespace clang::tidy::cccl
{
namespace
{
using namespace ast_matchers; // NOLINT(google-build-using-namespace)

#if LLVM_VERSION_MAJOR < 23
// LLVM 22 has no matcher for the number of written template arguments.
AST_MATCHER_P(TemplateSpecializationTypeLoc, templateArgumentLocCountIs, unsigned, count)
{
  static_cast<void>(Builder);
  static_cast<void>(Finder);
  return Node.getNumArgs() == count;
}
#endif

AST_MATCHER(QualType, isSizeT)
{
  static_cast<void>(Builder);

  // NOLINTNEXTLINE(readability-static-accessed-through-instance)
  if (auto&& ctx = Finder->getASTContext(); ctx.hasSameType(Node, ctx.getSizeType()))
  {
    // Follow using declarations without matching implicit nodes or discarding typedef identity.
    const auto* type = Node->getAs<TypedefType>();

    return type && type->getDecl()->getName() == "size_t";
  }

  return false;
}

AST_MATCHER(QualType, isPtrDiffT)
{
  static_cast<void>(Builder);

  // NOLINTNEXTLINE(readability-static-accessed-through-instance)
  if (auto&& ctx = Finder->getASTContext(); ctx.hasSameType(Node, ctx.getPointerDiffType()))
  {
    // Follow using declarations without matching implicit nodes or discarding typedef identity.
    const auto* type = Node->getAs<TypedefType>();

    return type && type->getDecl()->getName() == "ptrdiff_t";
  }

  return false;
}

AST_MATCHER(TemplateArgumentLoc, isSizeOfTypePack)
{
  static_cast<void>(Builder);
  static_cast<void>(Finder);

  if (Node.getArgument().getKind() == TemplateArgument::Expression)
  {
    const auto* size = dyn_cast<SizeOfPackExpr>(Node.getSourceExpression());

    return size && isa<TemplateTypeParmDecl>(size->getPack());
  }

  return false;
}

class SimplifyingReplacement
{
  // StringRef.rsplit() (or rfind()) is strangely not constexpr, so need to do this in
  // string_view.
  [[nodiscard]] static constexpr std::string_view constexpr_get_stem_(std::string_view text)
  {
    constexpr auto sep = std::string_view{"::"};
    const auto pos     = text.rfind(sep);

    if (pos == std::string_view::npos)
    {
      return text;
    }
    return text.substr(pos + sep.size());
  }

public:
  constexpr SimplifyingReplacement(StringRef outer_name, StringRef inner_name, StringRef repl)
      : outer{outer_name}
      , outer_stem{constexpr_get_stem_(outer)}
      , inner{inner_name}
      , inner_stem{constexpr_get_stem_(inner)}
      , replacement{repl}
  {}

  StringRef outer;
  StringRef outer_stem;
  StringRef inner;
  StringRef inner_stem;
  StringRef replacement;
};

template <typename T>
[[nodiscard]] CharSourceRange remove_node(
  const MatchFinder::MatchResult& result,
  const std::optional<T>& prev_node,
  const T& node,
  const std::optional<T>& next_node)
{
  if (next_node.has_value())
  {
    return CharSourceRange::getCharRange(node.getSourceRange().getBegin(), next_node->getSourceRange().getBegin());
  }

  if (prev_node.has_value())
  {
    return CharSourceRange::getTokenRange(
      Lexer::getLocForEndOfToken(
        prev_node->getSourceRange().getEnd(), /*Offset=*/0, *result.SourceManager, result.Context->getLangOpts()),
      node.getSourceRange().getEnd());
  }

  return CharSourceRange::getTokenRange(node.getSourceRange());
}

[[nodiscard]] FixItHint remove_parameter(
  const MatchFinder::MatchResult& result, const TemplateSpecializationTypeLoc& type_loc, std::uint32_t index)
{
  constexpr std::optional<TemplateArgumentLoc> NO_ARG = std::nullopt;

  return FixItHint::CreateRemoval(remove_node(
    result,
    index > 0 ? type_loc.getArgLoc(index - 1) : NO_ARG,
    type_loc.getArgLoc(index),
    index + 1 < type_loc.getNumArgs() ? type_loc.getArgLoc(index + 1) : NO_ARG));
}

class SimplifyTypeTraitsCheck final : public ClangTidyCheck
{
  static constexpr StringRef SIMPLIFYING_BIND{""};

  static constexpr StringRef REPLACE_AND_REMOVE_BIND{"r"};
  static constexpr StringRef DECL_BIND{"d"};

  static constexpr StringRef BUFFER_BIND{"u"};
  static constexpr StringRef MR_BIND{"mr"};

  static constexpr StringRef BOOL_CONSTANT_BIND{"b"};
  static constexpr StringRef BOOL_LITERAL_BIND{"l"};

  static constexpr StringRef MAKE_INDEX_SEQUENCE_BIND{"m"};

public:
  using ClangTidyCheck::ClangTidyCheck;

  [[nodiscard]] std::optional<TraversalKind> getCheckTraversalKind() const override
  {
    return TK_IgnoreUnlessSpelledInSource;
  }

  void registerMatchers(MatchFinder* finder) override
  {
    // Would love to just pass &SimplifyingReplacement::outer here but LLVM 22 does not support
    // this
    const auto outer_names = llvm::map_to_vector<simplifying.size()>(simplifying, [](const auto& r) {
      return r.outer;
    });
    const auto inner_names = llvm::map_to_vector<simplifying.size()>(simplifying, [](const auto& r) {
      return r.inner;
    });

    // LLVM 22 ignores getCheckTraversalKind() when registering TypeLoc matchers. This is fixed
    // in LLVM23, but CI is currently stuck at 22...
    //
    // Explicit traversal prevents rewrites of instantiated types at generic template source
    // locations.
    finder->addMatcher(
      traverse(TK_IgnoreUnlessSpelledInSource,
               templateSpecializationTypeLoc(
                 loc(qualType(hasDeclaration(namedDecl(hasAnyName(outer_names))))),
                 hasTemplateArgumentLoc(
                   /*Index=*/0,
                   hasTypeLoc(
                     templateSpecializationTypeLoc(loc(qualType(hasDeclaration(namedDecl(hasAnyName(inner_names)))))))))
                 .bind(SIMPLIFYING_BIND)),
      this);

    // ==========================================================================================

    finder->addMatcher(
      traverse(TK_IgnoreUnlessSpelledInSource,
               templateSpecializationTypeLoc(
                 loc(qualType(hasDeclaration(namedDecl(hasName("::cuda::std::integral_constant")).bind(DECL_BIND))))
                   .bind(REPLACE_AND_REMOVE_BIND),
                 hasTemplateArgumentLoc(/*Index=*/0, hasTypeLoc(loc(booleanType()))))),
      this);

    finder->addMatcher(
      traverse(TK_IgnoreUnlessSpelledInSource,
               templateSpecializationTypeLoc(
                 loc(qualType(hasDeclaration(namedDecl(hasAnyName("::cuda::std::dextents",
                                                                  "::cuda::std::integer_sequence",
                                                                  "::cuda::std::make_integer_sequence"))
                                               .bind(DECL_BIND))))
                   .bind(REPLACE_AND_REMOVE_BIND),
                 hasTemplateArgumentLoc(/*Index=*/0, hasTypeLoc(loc(isSizeT()))))),
      this);

    finder->addMatcher(
      traverse(TK_IgnoreUnlessSpelledInSource,
               templateSpecializationTypeLoc(
                 loc(qualType(hasDeclaration(namedDecl(hasName("::cuda::dstrides")).bind(DECL_BIND))))
                   .bind(REPLACE_AND_REMOVE_BIND),
                 hasTemplateArgumentLoc(/*Index=*/0, hasTypeLoc(loc(isPtrDiffT()))))),
      this);

    // ==========================================================================================

    finder->addMatcher(
      traverse(
        TK_IgnoreUnlessSpelledInSource,
        templateSpecializationTypeLoc(
          loc(qualType(hasDeclaration(namedDecl(hasName("::cuda::buffer")).bind(DECL_BIND)))).bind(BUFFER_BIND),
          // We must only match exactly cuda::buffer<T, device_accessible>, not
          // cuda::buffer<T, device_accessible, other_prop>
          templateArgumentLocCountIs(2),
          hasTemplateArgumentLoc(
            /*Index=*/1,
            hasTypeLoc(loc(hasCanonicalType(hasDeclaration(
              namedDecl(hasAnyName("::cuda::mr::device_accessible", "::cuda::mr::host_accessible")).bind(MR_BIND)))))))),
      this);

    // ==========================================================================================

    finder->addMatcher(
      traverse(TK_IgnoreUnlessSpelledInSource,
               templateSpecializationTypeLoc(
                 loc(templateSpecializationType(
                   hasDeclaration(namedDecl(hasName("::cuda::std::bool_constant"))),
                   // Keep substitution nodes so instantiated parameters do not match as
                   // written literals. Otherwise we end up matching on things like
                   //
                   // template <bool Value>
                   // inline constexpr auto foo = cuda::std::bool_constant<Value>{};
                   hasTemplateArgument(
                     /*N=*/0, isExpr(traverse(TK_AsIs, ignoringImpCasts(cxxBoolLiteral().bind(BOOL_LITERAL_BIND))))))))
                 .bind(BOOL_CONSTANT_BIND)),
      this);

    finder->addMatcher(
      traverse(TK_IgnoreUnlessSpelledInSource,
               templateSpecializationTypeLoc(
                 loc(qualType(hasDeclaration(namedDecl(hasName("::cuda::std::make_index_sequence")))))
                   .bind(MAKE_INDEX_SEQUENCE_BIND),
                 hasTemplateArgumentLoc(/*Index=*/0, isSizeOfTypePack()))),
      this);
  }

  void check(const MatchFinder::MatchResult& result) override
  {
    auto&& nodes = result.Nodes;

    if (const auto* outer_loc = nodes.getNodeAs<TemplateSpecializationTypeLoc>(SIMPLIFYING_BIND))
    {
      const auto inner_loc =
        outer_loc->getArgLoc(0).getTypeSourceInfo()->getTypeLoc().castAs<TemplateSpecializationTypeLoc>();
      const auto outer_name = outer_loc->getTypePtr()->getTemplateName().getAsTemplateDecl()->getName();
      const auto inner_name = inner_loc.getTypePtr()->getTemplateName().getAsTemplateDecl()->getName();

      const auto it = std::find_if(simplifying.begin(), simplifying.end(), [&](const auto& rule) {
        // Both inner and outer need to match. If only the outer matches then we have a case of
        // the outer being used with one of the other inners, that we cannot simplify. For
        // example remove_const_t<remove_reference_t<...>>;
        return outer_name == rule.outer_stem && inner_name == rule.inner_stem;
      });

      if (it == simplifying.end())
      {
        return;
      }

      // We must do the same is-trait-definition check here as well. For example, we should not
      // emit the following warning:
      //
      // using unwrap_ref_decay_t = unwrap_reference_t<decay_t<_Tp>>;
      //                            ^~~~~~~~~~~~~~~~~~~~~~~~~~     ~
      //                            ::cuda::std::unwrap_ref_decay_t
      {
        const auto alias_name = it->replacement;

        if (util::is_trait_definition(*outer_loc, alias_name, typeAliasDecl, result.Context)
            || util::is_trait_definition(
              *outer_loc,
              alias_name.drop_back(2), // Remove "_t"
              cxxRecordDecl,
              result.Context))
        {
          return;
        }
      }

      diag(outer_loc->getBeginLoc(), "use simplified type alias") //
        << FixItHint::CreateReplacement(
             CharSourceRange::getCharRange(outer_loc->getBeginLoc(), inner_loc.getLAngleLoc()), it->replacement)
        << FixItHint::CreateRemoval(inner_loc.getRAngleLoc());
      return;
    }

    if (const auto* outer_loc = nodes.getNodeAs<TemplateSpecializationTypeLoc>(REPLACE_AND_REMOVE_BIND))
    {
      static const llvm::StringMap<std::pair<StringRef, std::uint32_t>> to_repl = {
        {"integral_constant", {"::cuda::std::bool_constant", 0}},
        {"dextents", {"::cuda::std::dims", 0}},
        {"integer_sequence", {"::cuda::std::index_sequence", 0}},
        {"make_integer_sequence", {"::cuda::std::make_index_sequence", 0}},
        {"dstrides", {"::cuda::steps", 0}},
      };

      const auto name      = nodes.getNodeAs<NamedDecl>(DECL_BIND)->getName();
      auto&& [repl, index] = to_repl.at(name);

      if (util::is_trait_definition(*outer_loc, repl, typeAliasDecl, result.Context))
      {
        return;
      }

      diag(outer_loc->getBeginLoc(), "use simplified type alias") //
        << FixItHint::CreateReplacement(
             CharSourceRange::getCharRange(outer_loc->getBeginLoc(), outer_loc->getLAngleLoc()), repl)
        << remove_parameter(result, *outer_loc, index);
      return;
    }

    if (const auto* outer_loc = nodes.getNodeAs<TemplateSpecializationTypeLoc>(BUFFER_BIND))
    {
      constexpr auto host_buffer   = StringRef{"::cuda::host_buffer"};
      constexpr auto device_buffer = StringRef{"::cuda::device_buffer"};
      const auto repl =
        nodes.getNodeAs<NamedDecl>(MR_BIND)->getName() == "host_accessible" ? host_buffer : device_buffer;

      if (util::is_trait_definition(*outer_loc, repl, typeAliasDecl, result.Context))
      {
        return;
      }

      diag(outer_loc->getBeginLoc(), "use simplified type alias") //
        << FixItHint::CreateReplacement(
             CharSourceRange::getCharRange(outer_loc->getBeginLoc(), outer_loc->getLAngleLoc()), repl)
        << remove_parameter(result, *outer_loc, /*index=*/1);
      return;
    }

    if (const auto* outer_loc = nodes.getNodeAs<TemplateSpecializationTypeLoc>(BOOL_CONSTANT_BIND))
    {
      constexpr auto true_type  = StringRef{"::cuda::std::true_type"};
      constexpr auto false_type = StringRef{"::cuda::std::false_type"};
      const auto repl = nodes.getNodeAs<CXXBoolLiteralExpr>(BOOL_LITERAL_BIND)->getValue() ? true_type : false_type;

      if (util::is_trait_definition(*outer_loc, repl, typeAliasDecl, result.Context))
      {
        return;
      }

      diag(outer_loc->getBeginLoc(), "use simplified type alias")
        << FixItHint::CreateReplacement(outer_loc->getSourceRange(), repl);
      return;
    }

    if (const auto* outer_loc = nodes.getNodeAs<TemplateSpecializationTypeLoc>(MAKE_INDEX_SEQUENCE_BIND))
    {
      constexpr auto alias_name = StringRef{"::cuda::std::index_sequence_for"};

      if (util::is_trait_definition(*outer_loc, alias_name, typeAliasDecl, result.Context))
      {
        return;
      }

      const auto* size = cast<SizeOfPackExpr>(outer_loc->getArgLoc(0).getSourceExpression());
      const auto* pack = cast<TemplateTypeParmDecl>(size->getPack());

      diag(outer_loc->getBeginLoc(), "use simplified type alias") //
        << FixItHint::CreateReplacement(
             CharSourceRange::getCharRange(outer_loc->getBeginLoc(), outer_loc->getLAngleLoc()), alias_name)
        << FixItHint::CreateReplacement(size->getSourceRange(), pack->getNameAsString() + "...");
      return;
    }

    llvm::reportFatalUsageError("cccl-simplify-type-traits: unhandled node kind, this is a bug in the check");
  }

private:
  static constexpr auto simplifying = std::array{
    // remove_cv<remove_reference<T>> -> remove_cvref<T>
    SimplifyingReplacement{"::cuda::std::remove_cv_t", "::cuda::std::remove_reference_t", "::cuda::std::remove_cvref_t"},
    // remove_const<remove_volatile<T>> -> remove_cv<T>
    SimplifyingReplacement{"::cuda::std::remove_const_t", "::cuda::std::remove_volatile_t", "::cuda::std::remove_cv_t"},
    // remove_volatile<remove_const<T>> -> remove_cv<T>
    SimplifyingReplacement{"::cuda::std::remove_volatile_t", "::cuda::std::remove_const_t", "::cuda::std::remove_cv_t"},
    // add_const<add_volatile<T>> -> add_cv<T>
    SimplifyingReplacement{"::cuda::std::add_const_t", "::cuda::std::add_volatile_t", "::cuda::std::add_cv_t"},
    // add_volatile<add_const<T>> -> add_cv<T>
    SimplifyingReplacement{"::cuda::std::add_volatile_t", "::cuda::std::add_const_t", "::cuda::std::add_cv_t"},
    // remove_const<remove_cv<T>> -> remove_cv<T>
    SimplifyingReplacement{"::cuda::std::remove_const_t", "::cuda::std::remove_cv_t", "::cuda::std::remove_cv_t"},
    // remove_volatile<remove_cv<T>> -> remove_cv<T>
    SimplifyingReplacement{"::cuda::std::remove_volatile_t", "::cuda::std::remove_cv_t", "::cuda::std::remove_cv_t"},
    // remove_cv<remove_cvref<T>> -> remove_cvref<T>
    SimplifyingReplacement{"::cuda::std::remove_cv_t", "::cuda::std::remove_cvref_t", "::cuda::std::remove_cvref_t"},
    // remove_reference<remove_cvref<T>> -> remove_cvref<T>
    SimplifyingReplacement{
      "::cuda::std::remove_reference_t", "::cuda::std::remove_cvref_t", "::cuda::std::remove_cvref_t"},
    // decay<remove_reference<T>> -> decay<T>
    SimplifyingReplacement{"::cuda::std::decay_t", "::cuda::std::remove_reference_t", "::cuda::std::decay_t"},
    // decay<decay<T>> -> decay<T>
    SimplifyingReplacement{"::cuda::std::decay_t", "::cuda::std::decay_t", "::cuda::std::decay_t"},
    // remove_cv<decay<T>> -> decay<T>
    SimplifyingReplacement{"::cuda::std::remove_cv_t", "::cuda::std::decay_t", "::cuda::std::decay_t"},
    // remove_reference<decay<T>> -> decay<T>
    SimplifyingReplacement{"::cuda::std::remove_reference_t", "::cuda::std::decay_t", "::cuda::std::decay_t"},
    // remove_cvref<decay<T>> -> decay<T>
    SimplifyingReplacement{"::cuda::std::remove_cvref_t", "::cuda::std::decay_t", "::cuda::std::decay_t"},
    // remove_all_extents<remove_all_extents<T>> -> remove_all_extents<T>
    SimplifyingReplacement{
      "::cuda::std::remove_all_extents_t", "::cuda::std::remove_all_extents_t", "::cuda::std::remove_all_extents_t"},
    // remove_all_extents<remove_extent<T>> -> remove_all_extents<T>
    SimplifyingReplacement{
      "::cuda::std::remove_all_extents_t", "::cuda::std::remove_extent_t", "::cuda::std::remove_all_extents_t"},
    // remove_extent<remove_all_extents<T>> -> remove_all_extents<T>
    SimplifyingReplacement{
      "::cuda::std::remove_extent_t", "::cuda::std::remove_all_extents_t", "::cuda::std::remove_all_extents_t"},
    // unwrap_reference<decay<T>> -> unwrap_ref_decay<T>
    SimplifyingReplacement{"::cuda::std::unwrap_reference_t", "::cuda::std::decay_t", "::cuda::std::unwrap_ref_decay_t"},
    // add_pointer<remove_reference<T>> -> add_pointer<T>
    SimplifyingReplacement{
      "::cuda::std::add_pointer_t", "::cuda::std::remove_reference_t", "::cuda::std::add_pointer_t"},
    // add_lvalue_reference<add_rvalue_reference<T>> -> add_lvalue_reference<T>
    SimplifyingReplacement{"::cuda::std::add_lvalue_reference_t",
                           "::cuda::std::add_rvalue_reference_t",
                           "::cuda::std::add_lvalue_reference_t"},
    // add_rvalue_reference<add_lvalue_reference<T>> -> add_lvalue_reference<T>
    SimplifyingReplacement{"::cuda::std::add_rvalue_reference_t",
                           "::cuda::std::add_lvalue_reference_t",
                           "::cuda::std::add_lvalue_reference_t"},
    // add_const<add_const<T>> -> add_const<T>
    SimplifyingReplacement{"::cuda::std::add_const_t", "::cuda::std::add_const_t", "::cuda::std::add_const_t"},
    // add_volatile<add_volatile<T>> -> add_volatile<T>
    SimplifyingReplacement{"::cuda::std::add_volatile_t", "::cuda::std::add_volatile_t", "::cuda::std::add_volatile_t"},
    // add_cv<add_cv<T>> -> add_cv<T>
    SimplifyingReplacement{"::cuda::std::add_cv_t", "::cuda::std::add_cv_t", "::cuda::std::add_cv_t"},
    // remove_const<remove_const<T>> -> remove_const<T>
    SimplifyingReplacement{"::cuda::std::remove_const_t", "::cuda::std::remove_const_t", "::cuda::std::remove_const_t"},
    // remove_volatile<remove_volatile<T>> -> remove_volatile<T>
    SimplifyingReplacement{
      "::cuda::std::remove_volatile_t", "::cuda::std::remove_volatile_t", "::cuda::std::remove_volatile_t"},
    // remove_cv<remove_cv<T>> -> remove_cv<T>
    SimplifyingReplacement{"::cuda::std::remove_cv_t", "::cuda::std::remove_cv_t", "::cuda::std::remove_cv_t"},
    // remove_reference<remove_reference<T>> -> remove_reference<T>
    SimplifyingReplacement{
      "::cuda::std::remove_reference_t", "::cuda::std::remove_reference_t", "::cuda::std::remove_reference_t"},
    // remove_cvref<remove_cvref<T>> -> remove_cvref<T>
    SimplifyingReplacement{"::cuda::std::remove_cvref_t", "::cuda::std::remove_cvref_t", "::cuda::std::remove_cvref_t"},
    // add_lvalue_reference<add_lvalue_reference<T>> -> add_lvalue_reference<T>
    SimplifyingReplacement{"::cuda::std::add_lvalue_reference_t",
                           "::cuda::std::add_lvalue_reference_t",
                           "::cuda::std::add_lvalue_reference_t"},
    // add_rvalue_reference<add_rvalue_reference<T>> -> add_rvalue_reference<T>
    SimplifyingReplacement{"::cuda::std::add_rvalue_reference_t",
                           "::cuda::std::add_rvalue_reference_t",
                           "::cuda::std::add_rvalue_reference_t"}};
};

class SimplifyTypeTraitsCheckModule final : public ClangTidyModule
{
public:
  static constexpr StringRef CHECK_NAME{"cccl-simplify-type-traits"};

  void addCheckFactories(ClangTidyCheckFactories& check_factories) override
  {
    check_factories.registerCheck<SimplifyTypeTraitsCheck>(CHECK_NAME);
  }
};

// NOLINTNEXTLINE(cert-err58-cpp, bugprone-throwing-static-initialization)
ClangTidyModuleRegistry::Add<SimplifyTypeTraitsCheckModule> _{
  SimplifyTypeTraitsCheckModule::CHECK_NAME, "Simplify composed CCCL type traits"};
} // namespace
} // namespace clang::tidy::cccl
