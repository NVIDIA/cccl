//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, RandomAccessIterator Iter>
// bool is_heap(ExecutionPolicy&& policy, Iter first, Iter last);

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_pstl.h"

inline constexpr int size = 1000;

template <class Policy, class T>
void test_is_heap(const Policy& policy, c2h::device_vector<T>& input)
{
  { // empty should not access anything
    const auto res = cuda::std::is_heap(policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr));
    CHECK(res);
  }

  { // size-1 should not access anything; using a host pointer would segfault on device
    T host_value{};
    const auto res = cuda::std::is_heap(policy, &host_value, &host_value + 1);
    CHECK(res);
  }

  // Strictly decreasing range is a valid max-heap.
  thrust::sequence(input.begin(), input.end(), size, -1);
  { // valid max-heap, contiguous range
    const auto res = cuda::std::is_heap(policy, input.begin(), input.end());
    CHECK(res);
  }

  T* raw_pointer = thrust::raw_pointer_cast(input.data());
  { // valid max-heap, random access range
    const auto res =
      cuda::std::is_heap(policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size});
    CHECK(res);
  }

  // Break the max-heap property at child index 42 (parent index 20).
  input[42] = static_cast<T>(size + 1);
  { // broken max-heap, contiguous range
    const auto res = cuda::std::is_heap(policy, input.begin(), input.end());
    CHECK(!res);
  }

  { // broken max-heap, random access range
    const auto res =
      cuda::std::is_heap(policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size});
    CHECK(!res);
  }
}

C2H_TEST("cuda::std::is_heap(iter, iter)", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_is_heap(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_is_heap(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_is_heap(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_is_heap(policy, input);
  }
}
