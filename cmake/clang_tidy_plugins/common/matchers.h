// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <clang/ASTMatchers/ASTMatchers.h>

namespace clang::tidy::cccl::util
{
using namespace ast_matchers;

using file_matching_type = decltype(isExpansionInFileMatching(""));

AST_MATCHER_FUNCTION(file_matching_type, isExpansionInTestOrExampleFile)
{
  return isExpansionInFileMatching(".*/(test|c2h|examples)/.*");
}

AST_MATCHER_FUNCTION(DeclarationMatcher, isInNamespaceCudaStd)
{
  // Require a namespace boundary so cuda::std_extra does not match.
  return hasDeclContext(namespaceDecl(matchesName("^::cuda::std($|::)")));
}
} // namespace clang::tidy::cccl::util
