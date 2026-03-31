//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class Policy, class InputIterator, class OutputIterator, class BinaryOp>
// OutputIterator adjacent_difference(Policy policy,
//                               InputIterator first,
//                               InputIterator last,
//                               OutputIterator result,
//                               BinaryOp op)

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

template <class Policy>
void test_adjacent_difference(
  const Policy& policy, const thrust::device_vector<int>& input, thrust::device_vector<int>& output)
{
  { // empty should not access anything
    auto res = cuda::std::adjacent_difference(
      policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), output.begin(), cuda::std::minus<>{});
    CHECK(res == output.begin());
  }

  cuda::constant_iterator<int> expected{1};
  {
    auto res = cuda::std::adjacent_difference(policy, input.begin(), input.end(), output.begin(), cuda::std::minus<>{});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // non contiguous input
    auto res = cuda::std::adjacent_difference(
      policy,
      cuda::counting_iterator{int{1}},
      cuda::counting_iterator{int{size + 1}},
      output.begin(),
      cuda::std::minus<>{});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // different input type
    auto res = cuda::std::adjacent_difference(
      policy,
      cuda::counting_iterator{short{1}},
      cuda::counting_iterator{short{size + 1}},
      output.begin(),
      cuda::std::minus<>{});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }
}

C2H_TEST("cuda::std::adjacent_difference(Iter1, Iter1, Iter2, Init)", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size);
  thrust::device_vector<int> output(size, thrust::no_init);
  thrust::sequence(input.begin(), input.end(), 1);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_adjacent_difference(policy, input, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with(cuda::get_stream, stream);
    test_adjacent_difference(policy, input, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with(cuda::mr::get_memory_resource, device_resource);
    test_adjacent_difference(policy, input, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy                            = cuda::execution::__cub_par_unseq.with(cuda::get_stream, stream)
                          .with(cuda::mr::get_memory_resource, device_resource);
    test_adjacent_difference(policy, input, output);
  }
}
