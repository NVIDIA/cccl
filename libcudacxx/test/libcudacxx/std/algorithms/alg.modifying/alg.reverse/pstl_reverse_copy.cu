//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class OutputIterator>
// OutputIterator reverse_copy(const Policy& policy,
//                             InputIterator first,
//                             InputIterator last,
//                             OutputIterator result);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 1000;

template <class Policy>
void test_replace_copy(const Policy& policy, const thrust::device_vector<int>& input, thrust::device_vector<int>& output)
{
  { // empty should not access anything
    auto res = cuda::std::reverse_copy(
      policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), static_cast<int*>(nullptr));
    CHECK(res == nullptr);
  }

  cuda::std::reverse_iterator expected{::cuda::counting_iterator{int{size}}};
  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // same type, contiguous iterators
    auto res = cuda::std::reverse_copy(policy, input.begin(), input.end(), output.begin());
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // same type, random access iterators
    auto res = cuda::std::reverse_copy(
      policy, ::cuda::counting_iterator{int{0}}, ::cuda::counting_iterator{int{size}}, output.begin());
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // different type, random access iterators
    auto res = cuda::std::reverse_copy(
      policy, ::cuda::counting_iterator{short{0}}, ::cuda::counting_iterator{short{size}}, output.begin());
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // different type, random access iterators
    auto res = cuda::std::reverse_copy(
      policy, ::cuda::counting_iterator{long{0}}, ::cuda::counting_iterator{long{size}}, output.begin());
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }
}

C2H_TEST("cuda::std::reverse_copy", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size, thrust::no_init);
  thrust::device_vector<int> output(size, thrust::no_init);
  thrust::sequence(input.begin(), input.end(), 0);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_replace_copy(policy, input, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_replace_copy(policy, input, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_replace_copy(policy, input, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_replace_copy(policy, input, output);
  }
}
