//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Adapted from:
// https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/clang-tidy/modernize/TypeTraitsCheck.cpp

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

#include <trait_catalog.h>

#include <clang-tidy/ClangTidy.h>
#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/NestedNameSpecifier.h>
#include <clang/AST/ParentMapContext.h>
#include <clang/AST/PrettyPrinter.h>
#include <clang/AST/TypeLoc.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Basic/IdentifierTable.h>
#include <clang/Lex/Lexer.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/raw_ostream.h>

namespace clang::tidy::cccl
{
namespace
{
using namespace ast_matchers; // NOLINT(google-build-using-namespace)

[[nodiscard]] DeclarationName get_name(const DependentScopeDeclRefExpr& expression)
{
  return expression.getDeclName();
}

[[nodiscard]] DeclarationName get_name(const DeclRefExpr& expression)
{
  return expression.getDecl()->getDeclName();
}

AST_POLYMORPHIC_MATCHER(is_value, AST_POLYMORPHIC_SUPPORTED_TYPES(DeclRefExpr, DependentScopeDeclRefExpr))
{
  static_cast<void>(Finder);
  static_cast<void>(Builder);

  const IdentifierInfo* identifier = get_name(Node).getAsIdentifierInfo();

  return identifier && identifier->isStr("value");
}

AST_MATCHER(TypeLoc, is_type)
{
  static_cast<void>(Finder);
  static_cast<void>(Builder);

  if (const auto type_loc = Node.getAs<TypedefTypeLoc>())
  {
    const auto* declaration = type_loc.getDecl();

    return declaration->getDeclName().isIdentifier() && declaration->getName() == "type";
  }

  if (const auto type_loc = Node.getAs<DependentNameTypeLoc>())
  {
    return type_loc.getTypePtr()->getIdentifier()->isStr("type");
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
  constexpr SimplifyingReplacement(StringRef outer_nane, StringRef inner_name, StringRef repl)
      : outer{outer_nane}
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

constexpr auto SMOL_LEN = 256;

[[nodiscard]] SmallString<SMOL_LEN> fully_qualified_name(const NamedDecl& decl)
{
  static const auto policy = [&] {
    auto ret = PrintingPolicy{decl.getASTContext().getLangOpts()};

    ret.SuppressInlineNamespace = llvm::to_underlying(PrintingPolicy::SuppressInlineNamespaceMode::All);
    return ret;
  }();
  auto name   = SmallString<SMOL_LEN>{};
  auto stream = llvm::raw_svector_ostream{name};

  // Need a custom printing policy because we don't want to include inline namespace since the
  // catalog doesn't include them
  decl.printQualifiedName(stream, policy);
  return name;
}

[[nodiscard]] std::optional<SmallString<SMOL_LEN>> get_fully_qualified_name(NestedNameSpecifier specifier)
{
  if (specifier.getKind() != NestedNameSpecifier::Kind::Type)
  {
    // specifier is not a type. For example
    //
    // - cuda::std::is_integral<T>:: names a type and passes.
    // - cuda::std:: names a namespace and fails.
    return std::nullopt;
  }

  // Given
  //
  // specifier = cuda::std::is_integral<T>::
  //
  // we want to extract "cuda::std::is_integral<T>", i.e. the specific specialization
  // requested. This should usually succeed, but if for whatever reason clang cant provide the
  // representation we bail.
  const auto* specialization = specifier.getAsType()->getAs<TemplateSpecializationType>();

  if (!specialization)
  {
    return std::nullopt;
  }

  // Now we need to find the original template declaration so we can match it against the
  // catalog. The reason we cannot just use the specialization's name directly is because we
  // need to remove the template params:
  //
  // Qualified specialization: cuda::std::is_integral<T>
  // Catalog key:              cuda::std::is_integral
  //
  // We must also preserve the correct identity for aliases:
  //
  // template <class T>
  // using local = cuda::std::is_integral<T>;
  //
  // For local<int>, we need the catalog key "local". Resolving its underlying type instead
  // produces "cuda::std::is_integral" which is wrong. We want to instead ask if there is a
  // "local_v" instead.
  const auto* decl = specialization->getTemplateName().getAsTemplateDecl();

  if (!decl)
  {
    return std::nullopt;
  }

  return fully_qualified_name(*decl);
}

[[nodiscard]] bool is_trait_definition(DynTypedNode node, StringRef name, ASTNodeKind target_kind, ASTContext* ctx)
{
  // At this point we have found a cuda::std::foo<T>::type (or ::value, the logic is
  // interchangeable) denoted by `node` that we know how to replace. One final check is to test
  // whether the full statement involving cuda::std::foo<T>::type is in fact:
  //
  // using foo_t = typename cuda::std::foo<T>::type;
  //
  // For which we should not emit a warning. But checking this requires walking *up* the AST,
  // something that it is... not really designed to do and is somewhat expensive.
  //
  // The alternative (and what LLVM itself does) is that we need to NOLINT every single _t and
  // _v declaration.
  constexpr auto max_it = 20;

  for (auto i = 0; i < max_it; ++i)
  {
    const auto parents = ctx->getParents(node);

    if (parents.empty())
    {
      // Somehow have no parents, and we did not pass through a namespace decl or TU decl?
      // Something is fishy.
      return false;
    }

    if (parents.size() > 1)
    {
      // A using decl should not have multiple parents so we can bail early (and emit the
      // warning). The only situation where I have found multiple parents for a false positive
      // is something like:
      //
      // template <class T>
      // using underlying_type_t = decltype(
      //   [](typename underlying_type<T>::type x) { return x; }({})
      // );
      //
      // But this is so obviously bonkers that we should emit the warning anyways because this
      // code should not exist in the first place.
      return false;
    }

    node = parents[0];

    constexpr auto ns_kind = ASTNodeKind::getFromNodeKind<NamespaceDecl>();
    constexpr auto tu_kind = ASTNodeKind::getFromNodeKind<TranslationUnitDecl>();
    const auto kind        = node.getNodeKind();

    if (kind.isSame(ns_kind) || kind.isSame(tu_kind))
    {
      // (kind of) early exit. There's no scenario in which a namespace (or the TU itself) can
      // ever have a parent that names a type so we can stop searching.
      return false;
    }

    if (target_kind.isBaseOf(kind) && (fully_qualified_name(node.getUnchecked<NamedDecl>()) == name))
    {
      // It is in fact some kind of
      //
      // using foo_t = typename cuda::std::foo<T>::type;
      return true;
    }
  }
  // Note if we are here, this is not confirmed to be true, but after 20 rounds of parent
  // searching it probably isn't the definition.
  return false;
}

// Handles cuda::std::foo<int>::value (note the specific type, not T!)
[[nodiscard]] bool decl_needs_fix(const DeclRefExpr& expr, ASTContext* ctx)
{
  if (!expr.hasQualifier())
  {
    return false;
  }

  auto opt_name = get_fully_qualified_name(expr.getQualifier());

  if (!opt_name.has_value())
  {
    return false;
  }

  if (auto& name = *opt_name; VALUE_TRAITS().contains(name))
  {
    name += "_v";
    return !is_trait_definition(DynTypedNode::create(expr), name, ASTNodeKind::getFromNodeKind<VarDecl>(), ctx);
  }
  return false;
}

// Handles cuda::std::foo<int>::value (note the specific type, not T!)
[[nodiscard]] bool decl_needs_fix(const TypedefTypeLoc& type_loc, ASTContext*)
{
  const auto name = get_fully_qualified_name(type_loc.getQualifierLoc().getNestedNameSpecifier());

  return name.has_value() && TYPE_TRAITS().contains(*name);
}

// Handles typename cuda::std::foo<T>::type
[[nodiscard]] bool decl_needs_fix(const DependentNameTypeLoc& loc, ASTContext* ctx)
{
  auto opt_name = get_fully_qualified_name(loc.getQualifierLoc().getNestedNameSpecifier());

  if (!opt_name.has_value())
  {
    return false;
  }

  if (auto& name = *opt_name; TYPE_TRAITS().contains(name))
  {
    name += "_t";
    return !is_trait_definition(DynTypedNode::create(loc), name, ASTNodeKind::getFromNodeKind<TypeAliasDecl>(), ctx);
  }

  return false;
}

// Handles cuda::std::foo<T>::value
[[nodiscard]] bool decl_needs_fix(const DependentScopeDeclRefExpr& reference, ASTContext* ctx)
{
  auto opt_name = get_fully_qualified_name(reference.getQualifier());

  if (!opt_name.has_value())
  {
    return false;
  }

  if (auto& name = *opt_name; VALUE_TRAITS().contains(name))
  {
    name += "_v";
    return !is_trait_definition(DynTypedNode::create(reference), name, ASTNodeKind::getFromNodeKind<VarDecl>(), ctx);
  }
  return false;
}

class ModernizeTypeTraitsCheck final : public ClangTidyCheck
{
  static constexpr StringRef USE_BIND{""};
  static constexpr StringRef SIMPLIFYING_BIND{"s"};

public:
  using ClangTidyCheck::ClangTidyCheck;

  [[nodiscard]] std::optional<TraversalKind> getCheckTraversalKind() const override
  {
    return TK_IgnoreUnlessSpelledInSource;
  }

  void registerMatchers(MatchFinder* finder) override
  {
    // The original clang-tidy modernize-type-traits uses a
    //
    // mapAnyOf(declRefExpr, dependentScopeDeclRefExpr)
    //
    // here, but profiling showed that splitting this into separate matchers improved
    // performance by 2x. I guess mapAnyOf() loses some type information that helps clang more
    // aggressively prune AST traversal.
    finder->addMatcher(declRefExpr(is_value()).bind(USE_BIND), this);
    finder->addMatcher(dependentScopeDeclRefExpr(is_value()).bind(USE_BIND), this);
    finder->addMatcher(typeLoc(is_type()).bind(USE_BIND), this);

    // Would love to just pass &SimplifyingReplacement::outer here but LLVM 22 does not support this
    const auto outer_names = llvm::map_to_vector<simplifying.size()>(simplifying, [](const auto& r) {
      return r.outer;
    });
    const auto inner_names = llvm::map_to_vector<simplifying.size()>(simplifying, [](const auto& r) {
      return r.inner;
    });

    finder->addMatcher(
      templateSpecializationTypeLoc(
        loc(qualType(hasDeclaration(namedDecl(hasAnyName(outer_names))))),
        hasTemplateArgumentLoc(
          /*Index=*/0,
          hasTypeLoc(templateSpecializationTypeLoc(loc(qualType(hasDeclaration(namedDecl(hasAnyName( //
            inner_names)))))))))
        .bind(SIMPLIFYING_BIND),
      this);
  }

  void check(const MatchFinder::MatchResult& result) override
  {
    auto&& nodes = result.Nodes;

    if (const auto* reference = nodes.getNodeAs<DeclRefExpr>(USE_BIND))
    {
      if (decl_needs_fix(*reference, result.Context))
      {
        emit_value_warning_(result, reference->getQualifierLoc(), reference->getEndLoc());
      }

      return;
    }

    if (const auto* type_loc = nodes.getNodeAs<TypedefTypeLoc>(USE_BIND))
    {
      if (decl_needs_fix(*type_loc, result.Context))
      {
        emit_type_warning_(
          result, type_loc->getQualifierLoc(), type_loc->getEndLoc(), type_loc->getElaboratedKeywordLoc());
      }

      return;
    }

    if (const auto* dependent_reference = nodes.getNodeAs<DependentScopeDeclRefExpr>(USE_BIND))
    {
      if (decl_needs_fix(*dependent_reference, result.Context))
      {
        emit_value_warning_(result, dependent_reference->getQualifierLoc(), dependent_reference->getEndLoc());
      }

      return;
    }

    if (const auto* dependent_type_loc = nodes.getNodeAs<DependentNameTypeLoc>(USE_BIND))
    {
      if (decl_needs_fix(*dependent_type_loc, result.Context))
      {
        emit_type_warning_(result,
                           dependent_type_loc->getQualifierLoc(),
                           dependent_type_loc->getEndLoc(),
                           dependent_type_loc->getElaboratedKeywordLoc());
      }

      return;
    }

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

      // We must do the same is-trait-definition check here as well. For example, we should now
      // emit the following warning:
      //
      // using unwrap_ref_decay_t = unwrap_reference_t<decay_t<_Tp>>;
      //                            ^~~~~~~~~~~~~~~~~~~~~~~~~~     ~
      //                            ::cuda::std::unwrap_ref_decay_t
      {
        auto alias_name = it->replacement;

        alias_name.consume_front("::");

        const auto node = DynTypedNode::create(*outer_loc);

        if (is_trait_definition(node, alias_name, ASTNodeKind::getFromNodeKind<TypeAliasDecl>(), result.Context)
            || is_trait_definition(node,
                                   alias_name.drop_back(2), // Remove "_t"
                                   ASTNodeKind::getFromNodeKind<CXXRecordDecl>(),
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

    llvm::reportFatalUsageError("cccl-modernize-type-traits: unhandled node kind, this is a bug in the check");
  }

private:
  void emit_value_warning_(
    const MatchFinder::MatchResult& result, const NestedNameSpecifierLoc& qualifier_loc, SourceLocation end_loc)
  {
    const auto specialization_loc = qualifier_loc.getAsTypeLoc().getAs<TemplateSpecializationTypeLoc>();

    if (!specialization_loc)
    {
      return;
    }

    const auto template_name_end_loc = Lexer::getLocForEndOfToken(
      specialization_loc.getTemplateNameLoc(), /*Offset=*/0, *result.SourceManager, result.Context->getLangOpts());
    // Macro tokens can have no file location after their end.
    auto d = diag(template_name_end_loc.isValid() ? template_name_end_loc : qualifier_loc.getBeginLoc(),
                  "use c++17 style variable templates");

    if (template_name_end_loc.isInvalid() || end_loc.isMacroID() || qualifier_loc.getEndLoc().isMacroID()
        || template_name_end_loc.isMacroID())
    {
      return;
    }

    d << FixItHint::CreateInsertion(template_name_end_loc, "_v")
      << FixItHint::CreateRemoval({qualifier_loc.getEndLoc(), end_loc});
  }

  void emit_type_warning_(const MatchFinder::MatchResult& result,
                          const NestedNameSpecifierLoc& qualifier_loc,
                          SourceLocation end_loc,
                          SourceLocation typename_loc)
  {
    const auto specialization_loc = qualifier_loc.getAsTypeLoc().getAs<TemplateSpecializationTypeLoc>();

    if (!specialization_loc)
    {
      return;
    }

    const auto template_name_end_loc = Lexer::getLocForEndOfToken(
      specialization_loc.getTemplateNameLoc(), /*Offset=*/0, *result.SourceManager, result.Context->getLangOpts());
    // Macro tokens can have no file location after their end.
    auto d = diag(template_name_end_loc.isValid() ? template_name_end_loc : qualifier_loc.getBeginLoc(),
                  "use c++14 style type templates");

    if (template_name_end_loc.isInvalid() || end_loc.isMacroID() || qualifier_loc.getEndLoc().isMacroID()
        || template_name_end_loc.isMacroID() || typename_loc.isMacroID())
    {
      return;
    }

    if (typename_loc.isValid())
    {
      d << FixItHint::CreateRemoval(typename_loc);
    }
    d << FixItHint::CreateInsertion(template_name_end_loc, "_t")
      << FixItHint::CreateRemoval({qualifier_loc.getEndLoc(), end_loc});
  }

  // TODO(jfaibussowit):
  //
  // Consider splitting this into its own separate clang-tidy check. It arguably doesn't have a
  // whole lot to do with using _t and _v forms. The only reason we have it here in this check
  // is because the base modernize-type-traits also has something similar (but only for
  // remove_cv<remove_reference<T>> -> remove_cvref<T>).
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

class ModernizeTypeTraitsCheckModule final : public ClangTidyModule
{
public:
  static constexpr StringRef CHECK_NAME{"cccl-modernize-type-traits"};

  void addCheckFactories(ClangTidyCheckFactories& check_factories) override
  {
    check_factories.registerCheck<ModernizeTypeTraitsCheck>(CHECK_NAME);
  }
};

// NOLINTNEXTLINE(cert-err58-cpp, bugprone-throwing-static-initialization)
ClangTidyModuleRegistry::Add<ModernizeTypeTraitsCheckModule> _{
  ModernizeTypeTraitsCheckModule::CHECK_NAME,
  "Use trait_t<T> and trait_v<T> instead of trait<T>::type and trait<T>::type for CCCL traits"};
} // namespace
} // namespace clang::tidy::cccl
