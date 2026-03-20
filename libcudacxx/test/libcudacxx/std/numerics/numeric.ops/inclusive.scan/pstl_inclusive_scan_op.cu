//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class Policy, class InputIterator, class OutputIterator, class BinaryOperator>
// OutputIterator inclusive_scan(Policy policy,
//                               InputIterator first,
//                               InputIterator last,
//                               OutputIterator result,
//                               BinaryOperator op)

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

inline constexpr int size = 1000;

struct sum_of_int
{
  __device__ constexpr int operator()(const int val) const noexcept
  { // Eulers sum
    return val * (val + 1) / 2;
  }
};

template <class Policy>
void test_inclusive_scan(
  const Policy& policy, const thrust::device_vector<int>& input, thrust::device_vector<int>& output)
{
  cuda::transform_iterator expected{cuda::counting_iterator{1}, sum_of_int{}};
  { // empty should not access anything
    auto res = cuda::std::inclusive_scan(
      policy,
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      cuda::std::plus<int>{});
    CHECK(res == nullptr);
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // same output type
    auto res = cuda::std::inclusive_scan(policy, input.begin(), input.end(), output.begin(), cuda::std::plus<int>{});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // same output type, conversion with binary op
    auto res = cuda::std::inclusive_scan(policy, input.begin(), input.end(), output.begin(), cuda::std::plus<long>{});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // different output type
    auto res = cuda::std::inclusive_scan(
      policy,
      cuda::counting_iterator{short{1}},
      cuda::counting_iterator{short{size + 1}},
      output.begin(),
      cuda::std::plus<int>{});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  { // different output type, conversion with binary
    auto res = cuda::std::inclusive_scan(
      policy,
      cuda::counting_iterator{short{1}},
      cuda::counting_iterator{short{size + 1}},
      output.begin(),
      cuda::std::plus<long>{});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }
}

C2H_TEST("cuda::std::inclusive_scan(Iter1, Iter1, Iter2, Op)", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size);
  thrust::device_vector<int> output(size);
  thrust::sequence(input.begin(), input.end(), 1);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_inclusive_scan(policy, input, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_inclusive_scan(policy, input, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_inclusive_scan(policy, input, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream).with_memory_resource(device_resource);
    test_inclusive_scan(policy, input, output);
  }
}
