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
// void rotate(const Policy& policy, InputIterator first, InputIterator middle, InputIterator last);

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

template <class Policy, class T>
void test_rotate(const Policy& policy, c2h::device_vector<T>& input, const int size)
{
  { // Empty does not access anything
    auto res = cuda::std::rotate(policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<T*>(nullptr));
    CHECK(res == nullptr);
  }

  const auto count1 = size < 42 ? 4 : 42;
  const auto count2 = size - count1;

  const auto expected_none = cuda::transform_iterator{cuda::counting_iterator{0}, cast_to<T>{}};
  thrust::sequence(input.begin(), input.end(), 0);
  { // Empty first part
    auto res = cuda::std::rotate(policy, input.begin(), input.begin(), input.end());
    CHECK(cuda::std::equal(policy, input.begin(), input.end(), expected_none));
    CHECK(res == input.begin());
  }

  thrust::sequence(input.begin(), input.end(), 0);
  { // Empty second part
    auto res = cuda::std::rotate(policy, input.begin(), input.end(), input.end());
    CHECK(cuda::std::equal(policy, input.begin(), input.end(), expected_none));
    CHECK(res == input.begin());
  }

  thrust::sequence(input.begin(), input.end(), 0);
  const auto expected_first = cuda::transform_iterator{cuda::counting_iterator{count1}, cast_to<T>{}};
  { // With contiguous iterator
    auto res = cuda::std::rotate(policy, input.begin(), cuda::std::next(input.begin(), count1), input.end());
    CHECK(cuda::std::equal(policy, input.begin(), cuda::std::next(input.begin(), count2), expected_first));
    CHECK(cuda::std::equal(policy, cuda::std::next(input.begin(), count2), input.end(), expected_none));
    CHECK(res == cuda::std::next(input.begin(), count2));
  }

  thrust::sequence(input.begin(), input.end(), 0);
  T* raw_pointer = thrust::raw_pointer_cast(input.data());
  { // With random access iterator
    auto res = cuda::std::rotate(
      policy,
      random_access_iterator{raw_pointer},
      random_access_iterator{raw_pointer + count1},
      random_access_iterator{raw_pointer + size});
    CHECK(cuda::std::equal(policy, input.begin(), cuda::std::next(input.begin(), count2), expected_first));
    CHECK(cuda::std::equal(policy, cuda::std::next(input.begin(), count2), input.end(), expected_none));
    CHECK(res == random_access_iterator{raw_pointer + count2});
  }
}

C2H_TEST("cuda::std::rotate", "[parallel algorithm]", all_types)
{
  const int size = GENERATE(10, 1000, 100000);

  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;

    test_rotate(policy, input, size);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);

    test_rotate(policy, input, size);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);

    test_rotate(policy, input, size);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);

    test_rotate(policy, input, size);
  }
}
