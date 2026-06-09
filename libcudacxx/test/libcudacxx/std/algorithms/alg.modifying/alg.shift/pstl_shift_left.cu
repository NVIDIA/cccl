//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator>
// void shift_left(const Policy& policy, InputIterator first, InputIterator last, iter_difference_t<InputIterator> n);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_pstl.h"

inline constexpr int size = 1000;

template <class Policy, class T>
void test_shift_left(const Policy& policy, c2h::device_vector<T>& input)
{
  { // Empty does not access anything
    auto res = cuda::std::shift_left(policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr), 42);
    CHECK(res == nullptr);
  }

  const auto expected_none = cuda::transform_iterator{cuda::counting_iterator{0}, cast_to<T>{}};
  thrust::sequence(input.begin(), input.end(), 0);
  { // No shift does nothing
    auto res = cuda::std::shift_left(policy, input.begin(), input.end(), 0);
    CHECK(cuda::std::equal(policy, input.begin(), input.end(), expected_none));
    CHECK(res == input.begin());
  }

  { // Shift larger than size does nothing
    auto res = cuda::std::shift_left(policy, input.begin(), input.end(), size + 1);
    CHECK(cuda::std::equal(policy, input.begin(), input.end(), expected_none));
    CHECK(res == input.begin());
  }

  const int num_shifted_small = 42;
  const auto expected_small   = cuda::transform_iterator{cuda::counting_iterator{num_shifted_small}, cast_to<T>{}};
  thrust::sequence(input.begin(), input.end(), 0);
  { // Small shift
    const auto mid = cuda::std::next(input.begin(), size - num_shifted_small);
    auto res       = cuda::std::shift_left(policy, input.begin(), input.end(), num_shifted_small);
    CHECK(cuda::std::equal(policy, input.begin(), mid, expected_small));
    CHECK(res == mid);
  }

  thrust::sequence(input.begin(), input.end(), 0);
  T* raw_pointer = thrust::raw_pointer_cast(input.data());
  { // Small shift, random_access
    const auto mid = cuda::std::next(input.begin(), size - num_shifted_small);
    auto res       = cuda::std::shift_left(
      policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size}, num_shifted_small);
    CHECK(cuda::std::equal(policy, input.begin(), mid, expected_small));
    CHECK(res == random_access_iterator{raw_pointer + size - num_shifted_small});
  }

  const int num_shifted_large = size / 2 + 10;
  const auto expected_large   = cuda::transform_iterator{cuda::counting_iterator{num_shifted_large}, cast_to<T>{}};
  thrust::sequence(input.begin(), input.end(), 0);
  { // Large shift
    const auto mid = cuda::std::next(input.begin(), size - num_shifted_large);
    auto res       = cuda::std::shift_left(policy, input.begin(), input.end(), num_shifted_large);
    CHECK(cuda::std::equal(policy, input.begin(), mid, expected_large));
    CHECK(res == mid);
  }

  thrust::sequence(input.begin(), input.end(), 0);
  { // Large shift, random_access
    const auto mid = cuda::std::next(input.begin(), size - num_shifted_large);
    auto res       = cuda::std::shift_left(
      policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size}, num_shifted_large);
    CHECK(cuda::std::equal(policy, input.begin(), mid, expected_large));
    CHECK(res == random_access_iterator{raw_pointer + size - num_shifted_large});
  }
}

C2H_TEST("cuda::std::shift_left", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;

    test_shift_left(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);

    test_shift_left(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);

    test_shift_left(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);

    test_shift_left(policy, input);
  }
}
