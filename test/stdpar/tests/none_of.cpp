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
  std::vector<int> values(num_items, 0);

  constexpr auto is_one = [](const int value) {
    return value == 1;
  };

  const bool no_matches = std::none_of(std::execution::par, values.begin(), values.end(), is_one);
  if (!no_matches)
  {
    return 1;
  }

  values[num_items / 2]           = 1;
  const bool still_no_matches     = std::none_of(std::execution::par, values.begin(), values.end(), is_one);
  const bool empty_has_no_matches = std::none_of(std::execution::par, values.begin(), values.begin(), is_one);
  if (still_no_matches || !empty_has_no_matches)
  {
    return 1;
  }

  return 0;
}
