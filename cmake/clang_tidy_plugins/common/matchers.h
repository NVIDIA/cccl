// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <clang/ASTMatchers/ASTMatchers.h>

namespace clang::tidy::cccl::util
{
using file_matching_type = decltype(ast_matchers::isExpansionInFileMatching(""));

AST_MATCHER_FUNCTION(file_matching_type, isExpansionInTestOrExampleFile)
{
  return ast_matchers::isExpansionInFileMatching(".*/(test|c2h|examples)/.*");
}
} // namespace clang::tidy::cccl::util
