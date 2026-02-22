//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, class ForwardIterator, class UnaryPredicate>
// iter_difference<ForwardIterator> count_if(ExecutionPolicy&& exec,
//                                           ForwardIterator first,
//                                           ForwardIterator last,
//                                           UnaryPredicate pred);

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

inline constexpr int size = 1000;

struct equal_to_42
{
  __host__ __device__ constexpr bool operator()(const int& val) const noexcept
  {
    return val == 42;
  }
};

template <class Policy>
void test_count_if(const Policy& policy)
{
  { // empty should not access anything
    const auto res = cuda::std::count_if(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), equal_to_42{});
    CHECK(res == 0);
  }

  { // same type
    const auto res =
      cuda::std::count_if(policy, cuda::counting_iterator{0}, cuda::counting_iterator{size}, equal_to_42{});
    CHECK(res == 1);
  }

  { // convertible type
    const auto res = cuda::std::count_if(
      policy, cuda::counting_iterator<short>{0}, cuda::counting_iterator<short>{size}, equal_to_42{});
    CHECK(res == 1);
  }
}

C2H_TEST("cuda::std::count_if", "[parallel algorithm]")
{
  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_count_if(policy);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_count_if(policy);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_count_if(policy);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_count_if(policy);
  }
}
