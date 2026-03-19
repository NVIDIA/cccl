//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator,class T = iter_value_t<InputIterator>>
// void reverse(const Policy& policy,
//                 InputIterator first,
//                 InputIterator last,
//                 const T& old_value,
//                 const T& new_value);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 1000;

template <class Policy>
void test_reverse(const Policy& policy, thrust::device_vector<int>& input)
{
  { // empty should not access anything
    cuda::std::reverse(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr));
  }

  { // same type
    cuda::std::reverse(policy, input.begin(), input.end());

    cuda::strided_iterator expected{cuda::counting_iterator{1000 - 1}, -1};
    CHECK(cuda::std::equal(policy, input.begin(), input.end(), expected));
  }
}

C2H_TEST("cuda::std::reverse", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size, thrust::no_init);
  thrust::sequence(input.begin(), input.end(), 0);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_reverse(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_reverse(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_reverse(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_reverse(policy, input);
  }
}
