//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits>
#include <utility>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/PrettyPrinter.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <llvm/Support/raw_ostream.h>

namespace clang::tidy::cccl::util
{
constexpr auto SMOL_LEN = 64;

[[nodiscard]] inline SmallString<SMOL_LEN> fully_qualified_name(const NamedDecl& decl)
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

//! @brief Checks whether a node occurs inside the specified trait definition.
//!
//! Use this check to prevent replacements that make a trait definition refer to itself.
//! `node` must be a `TypeLoc`, `Stmt`, or `Decl`, or a derived type.
//! `ctx` must point to the AST context that owns `node`.
//! `name` must be fully qualified, with a leading `::` and without inline namespaces.
//!
//! @param node The node that requires a replacement.
//! @param name The name of the trait definition to protect.
//! @param declaration The declaration matcher factory, such as `typeAliasDecl` or `varDecl`.
//! @param ctx The AST context.
//!
//! @returns `true` if an ancestor matches the name and declaration category; otherwise `false`.
template <class Node, class DeclFactory>
[[nodiscard]] bool is_trait_definition(const Node& node, StringRef name, const DeclFactory& declaration, ASTContext* ctx)
{
  using namespace clang::ast_matchers;

  const auto ancestor = hasAncestor(declaration(hasName(name)));
  const auto matcher  = [&] {
    if constexpr (std::is_base_of_v<TypeLoc, Node>)
    {
      return typeLoc(ancestor);
    }
    else if constexpr (std::is_base_of_v<Stmt, Node>)
    {
      return stmt(ancestor);
    }
    else if constexpr (std::is_base_of_v<Decl, Node>)
    {
      return decl(ancestor);
    }
    else
    {
      static_assert(!std::is_same_v<Node, Node>, "Unhandled node kind");
    }
  }();

  return !match(std::move(matcher), node, *ctx).empty();
}

template <class Node, class DeclFactory>
[[nodiscard]] bool
is_trait_definition(const Node& node, const Twine& name, const DeclFactory& declaration, ASTContext* ctx)
{
  SmallString<SMOL_LEN> storage;

  return is_trait_definition(node, name.toStringRef(storage), declaration, ctx);
}
} // namespace clang::tidy::cccl::util
