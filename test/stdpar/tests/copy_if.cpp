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
  std::vector<int> output(num_items, 0);
  std::iota(values.begin(), values.end(), 0);

  constexpr auto is_odd = [](const int value) {
    return value % 2 != 0;
  };

  const auto output_end = std::copy_if(std::execution::par, values.begin(), values.end(), output.begin(), is_odd);

  constexpr std::size_t expected_size = num_items / 2;
  if (output_end != output.begin() + expected_size)
  {
    return 1;
  }

  for (std::size_t i = 0; i < expected_size; ++i)
  {
    if (output[i] != static_cast<int>(2 * i + 1))
    {
      return 1;
    }
  }

  const auto empty_end = std::copy_if(std::execution::par, values.begin(), values.begin(), output.begin(), is_odd);

  if (empty_end != output.begin())
  {
    return 1;
  }

  return 0;
}
