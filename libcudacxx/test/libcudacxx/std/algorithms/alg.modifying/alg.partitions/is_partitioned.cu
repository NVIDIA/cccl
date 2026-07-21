//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, ForwardIterator Iter, UnaryPredicate>
// bool is_partitioned(ExecutionPolicy&& policy, Iter first, Iter last, UnaryPredicate pred);

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_macros.h"

inline constexpr int size = 1000;

template <class T>
struct less_than
{
  T value_;
  TEST_DEVICE_FUNC constexpr bool operator()(const T& value) const noexcept
  {
    return value < value_;
  }
};

template <class Policy>
void test_is_partitioned(const Policy& policy, thrust::device_vector<int>& input)
{
  { // empty should not access anything
    const auto res =
      cuda::std::is_partitioned(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), less_than<int>{42});
    CHECK(res);
  }

  thrust::sequence(input.begin(), input.end(), 0);
  { // contiguous range
    const auto res = cuda::std::is_partitioned(policy, input.begin(), input.end(), less_than<int>{42});
    CHECK(res);
  }

  { // random access range
    const auto res =
      cuda::std::is_partitioned(policy, cuda::counting_iterator{0}, cuda::counting_iterator{size}, less_than<int>{42});
    CHECK(res);
  }

  { // contiguous range, converting predicate
    const auto res = cuda::std::is_partitioned(policy, input.begin(), input.end(), less_than<long>{42ul});
    CHECK(res);
  }

  { // random access range, converting predicate
    const auto res = cuda::std::is_partitioned(
      policy, cuda::counting_iterator{0}, cuda::counting_iterator{size}, less_than<long>{42ul});
    CHECK(res);
  }

  { // unpartitioned contiguous range
    input[10]      = 1337;
    const auto res = cuda::std::is_partitioned(policy, input.begin(), input.end(), less_than<int>{42});
    CHECK(!res);
  }

  { // unpartitioned contiguous range, first element
    input[0]       = 1337;
    const auto res = cuda::std::is_partitioned(policy, input.begin(), input.end(), less_than<int>{42});
    CHECK(!res);
  }

  { // unpartitioned contiguous range, last element
    input[size - 1] = 0;
    const auto res  = cuda::std::is_partitioned(policy, input.begin(), input.end(), less_than<int>{42});
    CHECK(!res);
  }
}

C2H_TEST("cuda::std::is_partitioned(iter, iter, pred)", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_is_partitioned(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_is_partitioned(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_is_partitioned(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_is_partitioned(policy, input);
  }
}
