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
#include <cuda/std/execution>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_pstl.h"

inline constexpr int size = 1000;

template <class T = int>
struct is_power_of_2
{
  [[nodiscard]] TEST_DEVICE_FUNC constexpr bool operator()(T val) const noexcept
  {
    if constexpr (cuda::std::is_convertible_v<T, int>)
    {
      return cuda::is_power_of_two(static_cast<int>(val));
    }
    else
    {
      return cuda::is_power_of_two(val.value_);
    }
  }
};

template <class Policy, class T>
void test_replace_if(const Policy& policy, c2h::device_vector<T>& input)
{
  { // empty should not access anything
    cuda::std::replace_if(
      policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr), is_power_of_2<T>{}, static_cast<T>(1337));
  }

  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  { // contiguous
    cuda::std::replace_if(policy, input.begin(), input.end(), is_power_of_2<T>{}, static_cast<T>(1337));
    CHECK(cuda::std::none_of(policy, input.begin(), input.end(), is_power_of_2<T>{}));
  }

  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  T* raw = thrust::raw_pointer_cast(input.data());
  { // random access
    cuda::std::replace_if(
      policy, random_access_iterator{raw}, random_access_iterator{raw + size}, is_power_of_2<T>{}, static_cast<T>(1337));
    CHECK(cuda::std::none_of(policy, input.begin(), input.end(), is_power_of_2<T>{}));
  }

  if constexpr (cuda::std::is_integral_v<T>)
  {
    thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
    { // convertible type
      cuda::std::replace_if(policy, input.begin(), input.end(), is_power_of_2<int>{}, static_cast<T>(1337));
      CHECK(cuda::std::none_of(policy, input.begin(), input.end(), is_power_of_2<T>{}));
    }
  }

  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  { // convertible replacement
    cuda::std::replace_if(policy, input.begin(), input.end(), is_power_of_2<T>{}, static_cast<short>(1337));
    CHECK(cuda::std::none_of(policy, input.begin(), input.end(), is_power_of_2<T>{}));
  }
}

C2H_TEST("cuda::std::replace_if", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_replace_if(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_replace_if(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_replace_if(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_replace_if(policy, input);
  }
}
