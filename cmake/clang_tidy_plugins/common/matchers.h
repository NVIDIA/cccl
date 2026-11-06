// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <clang-tidy/FileExtensionsSet.h>
#include <clang-tidy/utils/FileExtensionsUtils.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Basic/SourceManager.h>
#include <llvm/Support/Path.h>

namespace clang::tidy::cccl::util
{
AST_MATCHER_P(Decl, isExpansionInHeaderFile, const FileExtensionsSet*, extensions) // NOLINT
{
  const auto begin    = Node.getBeginLoc();
  const auto& SM      = Finder->getASTContext().getSourceManager();
  const auto filename = SM.getFilename(SM.getExpansionLoc(begin));
  // LLVM's header utility rejects extensionless files even when the empty extension is configured.
  return utils::isExpansionLocInHeaderFile(begin, SM, *extensions)
      || (extensions->contains("") && !filename.empty() && llvm::sys::path::extension(filename).empty());
}

using file_matching_type = decltype(ast_matchers::isExpansionInFileMatching(""));

AST_MATCHER_FUNCTION(file_matching_type, isExpansionInTestFile)
{
  return ast_matchers::isExpansionInFileMatching(".*/(test|c2h)/.*");
}
} // namespace clang::tidy::cccl::util
