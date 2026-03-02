//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class OutputIterator, class UnaryOp>
// OutputIter transform(const Policy& policy,
//                      InputIterator first,
//                      InputIterator last,
//                      OutputIterator result,
//                      UnaryOp unary_op);

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 1000;

template <class T = int>
struct minus_five
{
  __device__ constexpr T operator()(const T val) const noexcept
  {
    return static_cast<T>(val - 5);
  }
};

template <class Policy>
void test_transform(const Policy& policy, thrust::device_vector<int>& output)
{
  { // empty should not access anything
    cuda::std::transform(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), output.begin(), minus_five{});
  }

  { // same type
    thrust::fill(output.begin(), output.end(), 0);
    cuda::std::transform(
      policy, cuda::counting_iterator{42}, cuda::counting_iterator{size + 42}, output.begin(), minus_five{});
    CHECK(thrust::equal(output.begin(), output.end(), cuda::counting_iterator{37}));
  }

  { // convertible transform arg
    thrust::fill(output.begin(), output.end(), 0);
    cuda::std::transform(
      policy, cuda::counting_iterator{42}, cuda::counting_iterator{size + 42}, output.begin(), minus_five<short>{});
    CHECK(thrust::equal(output.begin(), output.end(), cuda::counting_iterator{37}));
  }

  { // convertible type
    thrust::fill(output.begin(), output.end(), 0);
    cuda::std::transform(
      policy,
      cuda::counting_iterator<short>{42},
      cuda::counting_iterator<short>{size + 42},
      output.begin(),
      minus_five{});
    CHECK(thrust::equal(output.begin(), output.end(), cuda::counting_iterator{37}));
  }
}

C2H_TEST("cuda::std::transform", "[parallel algorithm]")
{
  thrust::device_vector<int> output(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_transform(policy, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_transform(policy, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_transform(policy, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_transform(policy, output);
  }
}
