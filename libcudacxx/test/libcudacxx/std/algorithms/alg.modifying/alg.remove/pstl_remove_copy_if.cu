//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class OutputIterator, class UnaryPredicate>
// OutputIterator remove_copy_if(const Policy&  policy,
//                               InputIterator  first,
//                               InputIterator  last,
//                               OutputIterator result,
//                               UnaryPredicate pred);

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 10000;

template <class T = int>
struct is_even
{
  __device__ constexpr bool operator()(const T& val) const noexcept
  {
    return (val % 2) == 0;
  }
};

template <class Policy>
void test_remove_copy_if(const Policy& policy, thrust::device_vector<int>& output)
{
  { // empty should not access anything
    const auto res = cuda::std::remove_copy_if(
      policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), static_cast<int*>(nullptr), is_even{});
    CHECK(res == nullptr);
  }

  { // With matching predicate (copy odd elements)
    thrust::fill(output.begin(), output.end(), 0);
    const auto res = cuda::std::remove_copy_if(
      policy, cuda::counting_iterator{0}, cuda::counting_iterator{size}, output.begin(), is_even{});
    CHECK(thrust::equal(output.begin(), res, cuda::strided_iterator{cuda::counting_iterator{1}, 2}));
    CHECK(cuda::std::distance(output.begin(), res) == size / 2);
  }

  { // With conversion for predicate
    thrust::fill(output.begin(), output.end(), 0);
    const auto res = cuda::std::remove_copy_if(
      policy,
      cuda::counting_iterator{0},
      cuda::counting_iterator{size},
      output.begin(),
      is_even<cuda::std::ptrdiff_t>{});
    CHECK(thrust::equal(output.begin(), res, cuda::strided_iterator{cuda::counting_iterator{1}, 2}));
    CHECK(cuda::std::distance(output.begin(), res) == size / 2);
  }
}

C2H_TEST("cuda::std::remove_copy_if", "[parallel algorithm]")
{
  thrust::device_vector<int> output(size / 2, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_remove_copy_if(policy, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_remove_copy_if(policy, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_remove_copy_if(policy, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_remove_copy_if(policy, output);
  }
}
