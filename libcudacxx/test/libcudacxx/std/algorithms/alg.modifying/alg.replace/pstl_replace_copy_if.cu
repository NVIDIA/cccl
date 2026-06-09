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
// void replace_copy_if(const Policy& policy,
//                 InputIterator first,
//                 InputIterator last,
//                 UnaryPred pred,
//                 const T& new_value);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>

#include <cuda/cmath>
#include <cuda/iterator>
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

template <class T = int>
struct is_power_of_2
{
  [[nodiscard]] TEST_DEVICE_FUNC constexpr bool operator()(const T val) const noexcept
  {
    return cuda::is_power_of_two(val);
  }
};

template <class Policy, class T>
void test_replace_copy_if(const Policy& policy, c2h::device_vector<T>& output)
{
  T* raw_out = thrust::raw_pointer_cast(output.data());

  { // empty should not access anything
    cuda::std::replace_copy_if(
      policy,
      static_cast<T*>(nullptr),
      static_cast<T*>(nullptr),
      output.begin(),
      is_power_of_2<T>{},
      static_cast<T>(1337));
  }

  { // contiguous
    thrust::sequence(output.begin(), output.end(), static_cast<T>(0));
    cuda::std::replace_copy_if(
      policy,
      cuda::counting_iterator{static_cast<T>(0)},
      cuda::counting_iterator{static_cast<T>(size)},
      output.begin(),
      is_power_of_2<T>{},
      static_cast<T>(1337));
    CHECK(cuda::std::none_of(policy, output.begin(), output.end(), is_power_of_2<T>{}));
  }

  { // random access input
    c2h::device_vector<T> input(size, thrust::no_init);
    thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
    const T* raw_in = thrust::raw_pointer_cast(input.data());
    thrust::sequence(output.begin(), output.end(), static_cast<T>(0));
    cuda::std::replace_copy_if(
      policy,
      random_access_iterator{raw_in},
      random_access_iterator{raw_in + size},
      output.begin(),
      is_power_of_2<T>{},
      static_cast<T>(1337));
    CHECK(cuda::std::none_of(policy, output.begin(), output.end(), is_power_of_2<T>{}));
  }

  { // random access output
    thrust::sequence(output.begin(), output.end(), static_cast<T>(0));
    cuda::std::replace_copy_if(
      policy,
      cuda::counting_iterator{static_cast<T>(0)},
      cuda::counting_iterator{static_cast<T>(size)},
      random_access_iterator{raw_out},
      is_power_of_2<T>{},
      static_cast<T>(1337));
    CHECK(cuda::std::none_of(policy, output.begin(), output.end(), is_power_of_2<T>{}));
  }

  { // convertible type
    thrust::sequence(output.begin(), output.end(), static_cast<T>(0));
    cuda::std::replace_copy_if(
      policy,
      cuda::counting_iterator{static_cast<T>(0)},
      cuda::counting_iterator{static_cast<T>(size)},
      output.begin(),
      is_power_of_2<short>{},
      static_cast<T>(1337));
    CHECK(cuda::std::none_of(policy, output.begin(), output.end(), is_power_of_2<T>{}));
  }

  { // convertible replacement
    thrust::sequence(output.begin(), output.end(), static_cast<T>(0));
    cuda::std::replace_copy_if(
      policy,
      cuda::counting_iterator{static_cast<T>(0)},
      cuda::counting_iterator{static_cast<T>(size)},
      output.begin(),
      is_power_of_2<T>{},
      static_cast<short>(1337));
    CHECK(cuda::std::none_of(policy, output.begin(), output.end(), is_power_of_2<T>{}));
  }

  { // short counting input
    thrust::sequence(output.begin(), output.end(), static_cast<T>(0));
    cuda::std::replace_copy_if(
      policy,
      cuda::counting_iterator<short>{0},
      cuda::counting_iterator<short>{static_cast<short>(size)},
      output.begin(),
      is_power_of_2<T>{},
      static_cast<short>(1337));
    CHECK(cuda::std::none_of(policy, output.begin(), output.end(), is_power_of_2<T>{}));
  }
}

C2H_TEST("cuda::std::replace_copy_if", "[parallel algorithm]", integral_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> output(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_replace_copy_if(policy, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_replace_copy_if(policy, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_replace_copy_if(policy, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_replace_copy_if(policy, output);
  }
}
