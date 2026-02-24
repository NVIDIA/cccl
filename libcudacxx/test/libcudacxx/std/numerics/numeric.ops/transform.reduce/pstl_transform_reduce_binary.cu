//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterat2, class T>
// T transform_reduce(ExecutionPolicy&& exec,
//                    ForwardIterator1 first1,
//                    ForwardIterator1 last1,
//                    ForwardIterator2 first2,
//                    T init);
// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterat2, class T,
//          class BinaryOp1, class BinaryOp2>
// T transform_reduce(ExecutionPolicy&& exec,
//                    ForwardIterator1 first1,
//                    ForwardIterator1 last1,
//                    ForwardIterator2 first2,
//                    T init,
//                    BinaryOp1 reduce,
//                    BinaryOp2 transform);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
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

template <class Policy, class Iter>
void test_transform_reduce(const Policy policy, const thrust::device_vector<int>& Input1, Iter input2)
{
  // N * (N + 1) / 2 for the first N integrals
  // 0 for multiplying by 1
  // 42 for init
  constexpr int expected = size * (size + 1) / 2 + 42;

  { // Defaulted operations
    auto res = cuda::std::transform_reduce(policy, Input1.begin(), Input1.end(), input2, int{42});
    static_assert(cuda::std::is_same_v<decltype(res), int>);
    CHECK(res == expected);
  }

  { // Different init
    auto res = cuda::std::transform_reduce(policy, Input1.begin(), Input1.end(), input2, int64_t{42});
    static_assert(cuda::std::is_same_v<decltype(res), int64_t>);
    CHECK(res == expected);
  }

  { // With matching operators
    auto res = cuda::std::transform_reduce(
      policy, Input1.begin(), Input1.end(), input2, int{42}, cuda::std::plus<>{}, cuda::std::multiplies<>{});
    static_assert(cuda::std::is_same_v<decltype(res), int>);
    CHECK(res == expected);
  }

  { // With converting operators
    auto res = cuda::std::transform_reduce(
      policy,
      Input1.begin(),
      Input1.end(),
      input2,
      int{42},
      cuda::std::plus<int64_t>{},
      cuda::std::multiplies<int64_t>{});
    static_assert(cuda::std::is_same_v<decltype(res), int>);
    CHECK(res == expected);
  }

  { // With matching operators, different init
    auto res = cuda::std::transform_reduce(
      policy, Input1.begin(), Input1.end(), input2, int64_t{42}, cuda::std::plus<>{}, cuda::std::multiplies<>{});
    static_assert(cuda::std::is_same_v<decltype(res), int64_t>);
    CHECK(res == expected);
  }

  { // With converting operators, different init
    auto res = cuda::std::transform_reduce(
      policy,
      Input1.begin(),
      Input1.end(),
      input2,
      int64_t{42},
      cuda::std::plus<int64_t>{},
      cuda::std::multiplies<int64_t>{});
    static_assert(cuda::std::is_same_v<decltype(res), int64_t>);
    CHECK(res == expected);
  }
}

C2H_TEST("cuda::std::transform_reduce(Iter1, Iter1, Iter2, Init)", "[parallel algorithm]")
{
  thrust::device_vector<int> input1(size, thrust::no_init);
  thrust::device_vector<int> input2(size, thrust::no_init);
  thrust::sequence(input1.begin(), input1.end(), 1);
  thrust::fill(input2.begin(), input2.end(), 1);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_transform_reduce(policy, input1, input2.begin());
    test_transform_reduce(policy, input1, ::cuda::constant_iterator<int>{1});
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_transform_reduce(policy, input1, input2.begin());
    test_transform_reduce(policy, input1, ::cuda::constant_iterator<int>{1});
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_transform_reduce(policy, input1, input2.begin());
    test_transform_reduce(policy, input1, ::cuda::constant_iterator<int>{1});
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_transform_reduce(policy, input1, input2.begin());
    test_transform_reduce(policy, input1, ::cuda::constant_iterator<int>{1});
  }
}
