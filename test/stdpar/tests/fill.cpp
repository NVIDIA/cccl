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
  std::vector<int> values(num_items, -1);

  std::fill(std::execution::par, values.begin() + 1, values.end() - 1, 42);

  if (values.front() != -1 || values.back() != -1)
  {
    return 1;
  }

  for (auto it = values.begin() + 1; it != values.end() - 1; ++it)
  {
    if (*it != 42)
    {
      return 1;
    }
  }

  std::fill(std::execution::par, values.begin(), values.begin(), 7);
  if (values.front() != -1)
  {
    return 1;
  }

  return 0;
}
