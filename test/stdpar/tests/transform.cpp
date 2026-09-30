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
#include <numeric>
#include <vector>

// Ensure that we are indeed using the correct CCCL version
static_assert(CCCL_MAJOR_VERSION == CMAKE_CCCL_VERSION_MAJOR);
static_assert(CCCL_MINOR_VERSION == CMAKE_CCCL_VERSION_MINOR);
static_assert(CCCL_PATCH_VERSION == CMAKE_CCCL_VERSION_PATCH);

int main()
{
  constexpr std::size_t num_items = 1 << 16;
  std::vector<int> values(num_items);
  const std::vector<int> offsets(num_items, 5);
  std::vector<int> output(num_items, 0);
  std::iota(values.begin(), values.end(), 0);

  constexpr auto double_value = [](const int value) {
    return value * 2;
  };

  constexpr auto add_values = [](const int lhs, const int rhs) {
    return lhs + rhs;
  };

  const auto unary_end =
    std::transform(std::execution::par, values.begin(), values.end(), output.begin(), double_value);

  if (unary_end != output.end())
  {
    return 1;
  }

  for (std::size_t i = 0; i < num_items; ++i)
  {
    if (output[i] != values[i] * 2)
    {
      return 1;
    }
  }

  const auto binary_end =
    std::transform(std::execution::par, values.begin(), values.end(), offsets.begin(), output.begin(), add_values);

  if (binary_end != output.end())
  {
    return 1;
  }

  for (std::size_t i = 0; i < num_items; ++i)
  {
    if (output[i] != values[i] + offsets[i])
    {
      return 1;
    }
  }

  const auto empty_end =
    std::transform(std::execution::par, values.begin(), values.begin(), output.begin(), double_value);

  if (empty_end != output.begin())
  {
    return 1;
  }

  return 0;
}
