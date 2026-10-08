//===----------------------------------------------------------------------===//
//
// Part of CUDA C++ Core Libraries, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/version>

#include <algorithm>
#include <cstddef>
#include <execution>
#include <vector>

// Ensure that we are indeed using the correct CCCL version
static_assert(CCCL_MAJOR_VERSION == CMAKE_CCCL_VERSION_MAJOR);
static_assert(CCCL_MINOR_VERSION == CMAKE_CCCL_VERSION_MINOR);
static_assert(CCCL_PATCH_VERSION == CMAKE_CCCL_VERSION_PATCH);

int main()
{
  constexpr std::size_t num_items = 1 << 16;

  // Scrambled input with plenty of duplicate keys
  std::vector<int> input(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    input[i] = static_cast<int>((i * 7919) % 1000);
  }

  std::vector<int> expected = input;
  std::sort(expected.begin(), expected.end());

  std::vector<int> values = input;
  std::sort(std::execution::par, values.begin(), values.end());
  if (values != expected)
  {
    return 1;
  }

  constexpr auto greater = [](const int lhs, const int rhs) {
    return lhs > rhs;
  };

  std::sort(expected.begin(), expected.end(), greater);
  values = input;
  std::sort(std::execution::par, values.begin(), values.end(), greater);
  if (values != expected)
  {
    return 1;
  }

  // Sorting an empty range must leave the data untouched
  std::sort(std::execution::par, values.begin(), values.begin());
  std::sort(std::execution::par, values.begin(), values.begin(), greater);
  if (values != expected)
  {
    return 1;
  }

  return 0;
}
