//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, class ForwardIterator, class T, class BinaryOp, class UnaryOp>
// T transform_reduce(ExecutionPolicy&& exec,
//                    ForwardIterator first,
//                    ForwardIterator last,
//                    T init,
//                    BinaryOp reduce,
//                    UnaryOp transform);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_macros.h"

inline constexpr int size = 100;

template <class T>
struct plus_one
{
  template <class U>
  [[nodiscard]] __device__ constexpr T operator()(const U val) const noexcept
  {
    return static_cast<T>(val + 1);
  }
};

template <class Policy, class Iter>
void test_transform_reduce(const Policy policy, Iter input1)
{
  // N * (N + 1) / 2 for the first N integrals
  // N for plus_one
  // 42 for init
  constexpr int expected = size * (size + 1) / 2 + size + 42;

  { // With matching operators
    auto res = cuda::std::transform_reduce(
      policy, input1, ::cuda::std::next(input1, size), int{42}, cuda::std::plus<>{}, plus_one<int>{});
    static_assert(cuda::std::is_same_v<decltype(res), int>);
    CHECK(res == expected);
  }

  { // With converting operators
    auto res = cuda::std::transform_reduce(
      policy, input1, ::cuda::std::next(input1, size), int{42}, cuda::std::plus<int64_t>{}, plus_one<int64_t>{});
    static_assert(cuda::std::is_same_v<decltype(res), int>);
    CHECK(res == expected);
  }

  { // With matching operators, different init
    auto res = cuda::std::transform_reduce(
      policy, input1, ::cuda::std::next(input1, size), int64_t{42}, cuda::std::plus<>{}, plus_one<int>{});
    static_assert(cuda::std::is_same_v<decltype(res), int64_t>);
    CHECK(res == expected);
  }

  { // With converting operators, different init
    auto res = cuda::std::transform_reduce(
      policy, input1, ::cuda::std::next(input1, size), int64_t{42}, cuda::std::plus<int64_t>{}, plus_one<int64_t>{});
    static_assert(cuda::std::is_same_v<decltype(res), int64_t>);
    CHECK(res == expected);
  }
}

C2H_TEST("cuda::std::transform_reduce(Iter1, Iter1, Iter2, Init)", "[parallel algorithm]")
{
  thrust::device_vector<int> input1(size);
  thrust::sequence(input1.begin(), input1.end(), 1);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_transform_reduce(policy, input1.begin());
    test_transform_reduce(policy, ::cuda::counting_iterator<int>{1});
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_transform_reduce(policy, input1.begin());
    test_transform_reduce(policy, ::cuda::counting_iterator<int>{1});
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_transform_reduce(policy, input1.begin());
    test_transform_reduce(policy, ::cuda::counting_iterator<int>{1});
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_transform_reduce(policy, input1.begin());
    test_transform_reduce(policy, ::cuda::counting_iterator<int>{1});
  }
}
