//===----------------------------------------------------------------------===//
//
// Part of CUDA C++ Core Libraries, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/version>

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
  constexpr int input_value       = 2;
  constexpr int input_weight      = 3;
  constexpr int sentinel_value    = 5;
  constexpr int sentinel_weight   = 7;

  std::vector<int> values(num_items, input_value);
  std::vector<int> weights(num_items, input_weight);
  values[num_items / 2]  = sentinel_value;
  weights[num_items / 2] = sentinel_weight;

  constexpr int dot_product_init = 7;
  const int dot_product =
    std::transform_reduce(std::execution::par, values.begin(), values.end(), weights.begin(), dot_product_init);
  constexpr int regular_item_count = int{num_items} - 1;
  constexpr int expected_dot_product =
    dot_product_init + regular_item_count * input_value * input_weight + sentinel_value * sentinel_weight;

  if (dot_product != expected_dot_product)
  {
    return 1;
  }

  constexpr auto maximum = [](const int lhs, const int rhs) {
    return lhs < rhs ? rhs : lhs;
  };

  constexpr auto squared_difference = [](const int lhs, const int rhs) {
    const int difference = lhs - rhs;
    return difference * difference;
  };

  constexpr int maximum_init           = 0;
  const int maximum_squared_difference = std::transform_reduce(
    std::execution::par, values.begin(), values.end(), weights.begin(), maximum_init, maximum, squared_difference);
  constexpr int sentinel_difference                 = sentinel_value - sentinel_weight;
  constexpr int expected_maximum_squared_difference = sentinel_difference * sentinel_difference;

  if (maximum_squared_difference != expected_maximum_squared_difference)
  {
    return 1;
  }

  constexpr auto add = [](const int lhs, const int rhs) {
    return lhs + rhs;
  };

  constexpr auto square = [](const int value) {
    return value * value;
  };

  constexpr int sum_of_squares_init = 11;
  const int sum_of_squares =
    std::transform_reduce(std::execution::par, values.begin(), values.end(), sum_of_squares_init, add, square);
  constexpr int expected_sum_of_squares =
    sum_of_squares_init + regular_item_count * input_value * input_value + sentinel_value * sentinel_value;

  if (sum_of_squares != expected_sum_of_squares)
  {
    return 1;
  }

  constexpr int empty_init = 42;
  const int empty_result =
    std::transform_reduce(std::execution::par, values.begin(), values.begin(), weights.begin(), empty_init);

  if (empty_result != empty_init)
  {
    return 1;
  }

  return 0;
}
