//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Test if dot section guards are movable and if we can them as an optional value
 */

#include <cuda/experimental/stf.cuh>

#include <optional>
#include <vector>

using namespace cuda::experimental::stf;

int main()
{
  context ctx;

  // Ensure the guard is movable
  {
    auto g  = ctx.dot_section("foo");
    auto g2 = mv(g);
  }

  // Ensure the guard can be stored as an optional
  ::std::vector<::std::optional<reserved::dot_section::guard>> nested_sections;

  for (size_t depth = 0; depth < 3; depth++)
  {
    nested_sections.emplace_back(ctx.dot_section("foo"));
  }

  for (size_t depth = 0; depth < 3; depth++)
  {
    nested_sections.pop_back();
  }

  ctx.finalize();
}
