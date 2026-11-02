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
  const auto& sm      = Finder->getASTContext().getSourceManager();
  const auto filename = sm.getFilename(sm.getExpansionLoc(begin));
  // LLVM's header utility rejects extensionless files even when the empty extension is configured.
  return utils::isExpansionLocInHeaderFile(begin, sm, *extensions)
      || (extensions->contains("") && !filename.empty() && llvm::sys::path::extension(filename).empty());
}
} // namespace clang::tidy::cccl::util
