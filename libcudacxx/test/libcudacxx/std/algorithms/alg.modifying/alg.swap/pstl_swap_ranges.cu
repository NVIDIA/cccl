//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator1, class InputIterator2>
// InputIterator2 swap_ranges(const Policy&  policy,
//                            InputIterator1 first1,
//                            InputIterator1 last1,
//                            InputIterator2 first2);

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
void test_swap_ranges(const Policy& policy, c2h::device_vector<T>& input1, c2h::device_vector<T>& input2)
{
  { // Empty should not access anything
    auto res =
      cuda::std::swap_ranges(policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<T*>(nullptr));
    CHECK(res == nullptr);
  }

  const auto expected_first =
    cuda::transform_iterator{cuda::strided_iterator{cuda::counting_iterator{1337}, -1}, cast_to<T>{}};
  const auto expected_second = cuda::transform_iterator{cuda::counting_iterator{42}, cast_to<T>{}};
  { // With contiguous iterator
    auto res = cuda::std::swap_ranges(policy, input1.begin(), input1.end(), input2.begin());
    CHECK(res == input2.end());
    CHECK(cuda::std::equal(policy, input1.begin(), input1.end(), expected_first));
    CHECK(cuda::std::equal(policy, input2.begin(), input2.end(), expected_second));
  }

  T* raw_pointer1 = thrust::raw_pointer_cast(input1.data());
  { // With random access iterator
    auto res = cuda::std::swap_ranges(
      policy, random_access_iterator{raw_pointer1}, random_access_iterator{raw_pointer1 + size}, input2.begin());
    CHECK(res == input2.end());
    CHECK(cuda::std::equal(policy, input1.begin(), input1.end(), expected_second));
    CHECK(cuda::std::equal(policy, input2.begin(), input2.end(), expected_first));
  }

  T* raw_pointer2 = thrust::raw_pointer_cast(input2.data());
  { // With random access iterator
    auto res = cuda::std::swap_ranges(policy, input1.begin(), input1.end(), random_access_iterator{raw_pointer2});
    CHECK(res == random_access_iterator{raw_pointer2 + size});
    CHECK(cuda::std::equal(policy, input1.begin(), input1.end(), expected_first));
    CHECK(cuda::std::equal(policy, input2.begin(), input2.end(), expected_second));
  }

  { // With iterator that specializes iter_move
    auto res = cuda::std::swap_ranges(
      policy,
      cuda::std::reverse_iterator{input1.end()},
      cuda::std::reverse_iterator{input1.begin()},
      cuda::std::reverse_iterator{input2.end()});
    CHECK(res == cuda::std::reverse_iterator{input2.begin()});
    CHECK(cuda::std::equal(policy, input1.begin(), input1.end(), expected_second));
    CHECK(cuda::std::equal(policy, input2.begin(), input2.end(), expected_first));
  }
}

C2H_TEST("cuda::std::swap_ranges", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input2(size, thrust::no_init);
  c2h::device_vector<T> input1(size, thrust::no_init);
  thrust::sequence(input1.begin(), input1.end(), 42);
  thrust::sequence(input2.begin(), input2.end(), 1337, -1);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;

    test_swap_ranges(policy, input1, input2);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);

    test_swap_ranges(policy, input1, input2);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);

    test_swap_ranges(policy, input1, input2);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);

    test_swap_ranges(policy, input1, input2);
  }
}
