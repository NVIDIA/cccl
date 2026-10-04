// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
  std::vector<int> values(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    values[i] = static_cast<int>(i % 4);
  }

  const auto matches        = std::count(std::execution::par, values.begin(), values.end(), 2);
  const auto absent_matches = std::count(std::execution::par, values.begin(), values.end(), 4);
  const auto empty_matches  = std::count(std::execution::par, values.begin(), values.begin(), 2);

  if (matches != num_items / 4 || absent_matches != 0 || empty_matches != 0)
  {
    return 1;
  }

  return 0;
}
