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
// void replace(const Policy& policy,
//                 InputIterator first,
//                 InputIterator last,
//                 const T& old_value,
//                 const T& new_value);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_pstl.h"

inline constexpr int size = 1000;

[[nodiscard]] TEST_FUNC constexpr bool operator==(const nontrivial_type& lhs, const int& rhs)
{
  return lhs.value_ == rhs;
}

[[nodiscard]] TEST_FUNC constexpr bool operator==(const int& lhs, const nontrivial_type& rhs)
{
  return lhs == rhs.value_;
}

template <class Policy, class T>
void test_replace(const Policy& policy, c2h::device_vector<T>& input)
{
  { // empty should not access anything
    cuda::std::replace(
      policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<T>(42), static_cast<T>(1));
  }

  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  { // contiguous
    cuda::std::replace(policy, input.begin(), input.end(), static_cast<T>(42), static_cast<T>(1));
    CHECK(cuda::std::none_of(policy, input.begin(), input.end(), cuda::equal_to_value<T>{static_cast<T>(42)}));
  }

  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  T* raw_pointer = thrust::raw_pointer_cast(input.data());
  { // random access
    cuda::std::replace(
      policy,
      random_access_iterator{raw_pointer},
      random_access_iterator{raw_pointer + size},
      static_cast<T>(42),
      static_cast<T>(1));
    CHECK(cuda::std::none_of(policy, input.begin(), input.end(), cuda::equal_to_value<T>{static_cast<T>(42)}));
  }

  if constexpr (::cuda::std::__is_cpp17_equality_comparable_v<T, int>)
  {
    thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
    { // convertible type
      cuda::std::replace(policy, input.begin(), input.end(), 42, 1);
      CHECK(cuda::std::none_of(policy, input.begin(), input.end(), cuda::equal_to_value<T>{static_cast<T>(42)}));
    }
  }
}

C2H_TEST("cuda::std::replace", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_replace(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_replace(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_replace(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_replace(policy, input);
  }
}
