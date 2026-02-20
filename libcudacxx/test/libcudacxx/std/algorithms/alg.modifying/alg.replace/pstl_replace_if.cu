//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class UnaryPred, class T = iter_value_t<InputIterator>>
// void replace_if(const Policy& policy,
//                 InputIterator first,
//                 InputIterator last,
//                 UnaryPred pred,
//                 const T& new_value);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>

#include <cuda/cmath>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 1000;

template <class T = int>
struct is_power_of_2
{
  __device__ constexpr bool operator()(const T val) const noexcept
  {
    return cuda::is_power_of_two(val);
  }
};

template <class Policy>
void test_replace_if(const Policy& policy, thrust::device_vector<int>& input)
{
  { // empty should not access anything
    cuda::std::replace_if(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), is_power_of_2{}, 1337);
  }

  { // same type
    thrust::sequence(input.begin(), input.end(), 0);
    cuda::std::replace_if(policy, input.begin(), input.end(), is_power_of_2{}, 1337);
    CHECK(thrust::none_of(input.begin(), input.end(), is_power_of_2{}));
  }

  { // convertible type
    thrust::sequence(input.begin(), input.end(), 0);
    cuda::std::replace_if(policy, input.begin(), input.end(), is_power_of_2<short>{}, 1337);
    CHECK(thrust::none_of(input.begin(), input.end(), is_power_of_2{}));
  }

  { // convertible replacement
    thrust::sequence(input.begin(), input.end(), 0);
    cuda::std::replace_if(policy, input.begin(), input.end(), is_power_of_2{}, static_cast<short>(1337));
    CHECK(thrust::none_of(input.begin(), input.end(), is_power_of_2{}));
  }
}

C2H_TEST("cuda::std::replace_if", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_replace_if(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_replace_if(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_replace_if(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_replace_if(policy, input);
  }
}
