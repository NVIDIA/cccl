//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, class ForwardIterator>
// ForwardIterator min_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_pstl.h"

inline constexpr int size = 1000;

template <class Policy, class T>
void test_min_element(const Policy& policy, c2h::device_vector<T>& input)
{
  { // empty should not access anything
    auto res = cuda::std::min_element(policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr));
    static_assert(cuda::std::is_same_v<decltype(res), T*>);
    CHECK(res == nullptr);
  }

  thrust::sequence(input.begin(), input.end(), 1);
  { // first element is smallest, contiguous iterator
    auto res = cuda::std::min_element(policy, input.begin(), input.end());
    CHECK(res == input.begin());
  }

  const T* raw_pointer = thrust::raw_pointer_cast(input.data());
  { // first element is smallest, random access iterator
    auto res =
      cuda::std::min_element(policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size});
    CHECK(res == random_access_iterator{raw_pointer});
  }

  thrust::sequence(input.begin(), input.end(), size, -1);
  { // last element is smallest, contiguous iterator
    auto res = cuda::std::min_element(policy, input.begin(), input.end());
    CHECK(res == --input.end());
  }

  { // last element is smallest, random access iterator
    auto res =
      cuda::std::min_element(policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size});
    CHECK(res == random_access_iterator{raw_pointer + size - 1});
  }

  cuda::std::fill(policy, input.begin(), input.end(), T{42});
  { // all elements equal, contiguous iterator
    auto res = cuda::std::min_element(policy, input.begin(), input.end());
    CHECK(res == input.begin());
  }

  { // all elements equal, random access iterator
    auto res =
      cuda::std::min_element(policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size});
    CHECK(res == random_access_iterator{raw_pointer});
  }
}

C2H_TEST("cuda::std::min_element(Iter, Iter)", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_min_element(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_min_element(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_min_element(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::get_stream, stream).with(cuda::mr::get_memory_resource, device_resource);
    test_min_element(policy, input);
  }
}
