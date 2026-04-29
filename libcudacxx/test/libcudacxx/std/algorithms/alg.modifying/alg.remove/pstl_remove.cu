//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class T>
// void remove(const Policy&  policy,
//             InputIterator  first,
//             InputIterator  last,
//             const T&       value);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_pstl.h"

inline constexpr int size = 10000;

template <class Policy, class T>
void test_remove(const Policy& policy, c2h::device_vector<T>& input)
{
  { // empty should not access anything
    const auto res = cuda::std::remove(policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<T>(42));
    CHECK(res == nullptr);
  }

  auto expected     = cuda::transform_iterator{cuda::counting_iterator{0}, cast_to<T>{}};
  auto mid_expected = cuda::transform_iterator{cuda::counting_iterator{43}, cast_to<T>{}};
  auto mid_in       = cuda::std::next(input.begin(), 42);

  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  { // contiguous iterator
    const auto res = cuda::std::remove(policy, input.begin(), input.end(), static_cast<T>(42));
    CHECK(cuda::std::equal(policy, input.begin(), mid_in, expected));
    CHECK(cuda::std::equal(policy, mid_in, res, mid_expected));
    CHECK(cuda::std::distance(input.begin(), res) == size - 1);
  }

  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  T* raw_pointer = thrust::raw_pointer_cast(input.data());
  auto mid_raw   = random_access_iterator{raw_pointer + 42};
  { // random access iterator
    const auto res = cuda::std::remove(
      policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size}, static_cast<T>(42));
    CHECK(cuda::std::equal(policy, input.begin(), mid_in, expected));
    CHECK(cuda::std::equal(policy, mid_raw, res, mid_expected));
    CHECK(cuda::std::distance(random_access_iterator{raw_pointer}, res) == size - 1);
  }

  if constexpr (::cuda::std::__is_cpp17_equality_comparable_v<T, int>)
  {
    thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
    { // contiguous iterator, converting value
      const auto res = cuda::std::remove(policy, input.begin(), input.end(), 42);
      CHECK(cuda::std::equal(policy, input.begin(), mid_in, expected));
      CHECK(cuda::std::equal(policy, mid_in, res, mid_expected));
      CHECK(cuda::std::distance(input.begin(), res) == size - 1);
    }

    thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
    { // random access iterator, converting value
      const auto res =
        cuda::std::remove(policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size}, 42);
      CHECK(cuda::std::equal(policy, input.begin(), mid_in, expected));
      CHECK(cuda::std::equal(policy, mid_raw, res, mid_expected));
      CHECK(cuda::std::distance(random_access_iterator{raw_pointer}, res) == size - 1);
    }
  }
}

C2H_TEST("cuda::std::remove", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_remove(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_remove(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_remove(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_remove(policy, input);
  }
}
