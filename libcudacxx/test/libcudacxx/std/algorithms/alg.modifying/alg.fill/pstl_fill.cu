//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class T>
// void fill(const Policy& policy, InputIterator first, InputIterator last, const T& value);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

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
void test_fill(const Policy& policy, c2h::device_vector<T>& output)
{
  const T val = static_cast<T>(42);

  { // empty should not access anything
    cuda::std::fill(policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr), val);
  }

  { // contiguous input
    cuda::std::fill(policy, output.begin(), output.end(), val);
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), cuda::constant_iterator{val}));
  }

  T* raw_pointer = thrust::raw_pointer_cast(output.data());
  { // sorted random access range{ // random access input
    cuda::std::fill(policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size}, val);
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), cuda::constant_iterator{val}));
  }

  { // convertible type
    cuda::std::fill(policy, output.begin(), output.end(), 42);
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), cuda::constant_iterator{val}));
  }

  { // random access input, convertible type
    cuda::std::fill(policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size}, 42);
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), cuda::constant_iterator{val}));
  }
}

C2H_TEST("cuda::std::fill", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> output(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_fill(policy, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_fill(policy, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_fill(policy, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_fill(policy, output);
  }
}
