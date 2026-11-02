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

#include <optional>

#include "trait_catalog.h"
#include <clang-tidy/ClangTidy.h>
#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/NestedNameSpecifier.h>
#include <clang/AST/TypeLoc.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Basic/IdentifierTable.h>
#include <clang/Lex/Lexer.h>
#include <common/type_traits.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>

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

[[nodiscard]] std::optional<SmallString<util::SMOL_LEN>> get_fully_qualified_name(NestedNameSpecifier specifier)
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

  return util::fully_qualified_name(*decl);
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
    return !util::is_trait_definition(expr, "::" + name + "_v", varDecl, ctx);
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
    return !util::is_trait_definition(loc, "::" + name + "_t", typeAliasDecl, ctx);
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
    return !util::is_trait_definition(reference, "::" + name + "_v", varDecl, ctx);
  }
  return false;
}

class ModernizeTypeTraitsCheck final : public ClangTidyCheck
{
  static constexpr StringRef USE_BIND{""};

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
