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

// Only the key is compared, so the position shows whether equal keys kept their order
struct item
{
  int key;
  int position;
};

bool operator<(const item& lhs, const item& rhs)
{
  return lhs.key < rhs.key;
}

bool operator==(const item& lhs, const item& rhs)
{
  return lhs.key == rhs.key && lhs.position == rhs.position;
}

int main()
{
  constexpr std::size_t num_items = 1 << 16;

  // Scrambled keys with plenty of duplicates, each tagged with its original position
  std::vector<item> input(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    input[i] = item{static_cast<int>((i * 7919) % 1000), static_cast<int>(i)};
  }

  std::vector<item> expected = input;
  std::stable_sort(expected.begin(), expected.end());

  std::vector<item> values = input;
  std::stable_sort(std::execution::par, values.begin(), values.end());
  if (values != expected)
  {
    return 1;
  }

  constexpr auto greater_key = [](const item& lhs, const item& rhs) {
    return lhs.key > rhs.key;
  };

  std::stable_sort(expected.begin(), expected.end(), greater_key);
  values = input;
  std::stable_sort(std::execution::par, values.begin(), values.end(), greater_key);
  if (values != expected)
  {
    return 1;
  }

  // Sorting an empty range must leave the data untouched
  std::stable_sort(std::execution::par, values.begin(), values.begin());
  std::stable_sort(std::execution::par, values.begin(), values.begin(), greater_key);
  if (values != expected)
  {
    return 1;
  }

  return 0;
}
