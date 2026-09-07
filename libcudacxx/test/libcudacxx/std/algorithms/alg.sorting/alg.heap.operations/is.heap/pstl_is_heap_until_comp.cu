//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, RandomAccessIterator Iter, BinaryPredicate>
// Iter is_heap_until(ExecutionPolicy&& policy, Iter first, Iter last, BinaryPredicate pred);

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
void test_is_heap_until(const Policy& policy, c2h::device_vector<T>& input)
{
  { // empty should not access anything
    const auto res =
      cuda::std::is_heap_until(policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr), cuda::std::greater<>{});
    CHECK(res == nullptr);
  }

  { // size-1 should not access anything; using a host pointer would segfault on device
    T host_value{};
    const auto res = cuda::std::is_heap_until(policy, &host_value, &host_value + 1, cuda::std::greater<>{});
    CHECK(res == &host_value + 1);
  }

  // Strictly increasing range is a valid min-heap under greater<>
  // (parent always smaller than child).
  thrust::sequence(input.begin(), input.end(), 0);
  { // valid min-heap, contiguous range
    const auto res = cuda::std::is_heap_until(policy, input.begin(), input.end(), cuda::std::greater<>{});
    CHECK(res == input.end());
  }

  T* raw_pointer = thrust::raw_pointer_cast(input.data());
  { // valid min-heap, random access range
    const auto res = cuda::std::is_heap_until(
      policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size}, cuda::std::greater<>{});
    CHECK(res == random_access_iterator{raw_pointer + size});
  }

  // Inject a value at child index 42 smaller than its parent at index 20.
  // Parent value = 20, child value = 0 violates the min-heap property
  // (greater<>(20, 0) == true) -> is_heap_until must return iterator at i=42.
  input[42] = static_cast<T>(0);
  { // broken min-heap, contiguous range
    const auto res = cuda::std::is_heap_until(policy, input.begin(), input.end(), cuda::std::greater<>{});
    CHECK(res == cuda::std::next(input.begin(), 42));
    CHECK(cuda::std::is_heap(policy, input.begin(), res, cuda::std::greater<>{}));
  }

  { // broken min-heap, random access range
    const auto res = cuda::std::is_heap_until(
      policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size}, cuda::std::greater<>{});
    CHECK(res == random_access_iterator{raw_pointer + 42});
    CHECK(cuda::std::is_heap(policy, random_access_iterator{raw_pointer}, res, cuda::std::greater<>{}));
  }
}

C2H_TEST("cuda::std::is_heap_until(iter, iter, comp)", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_is_heap_until(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_is_heap_until(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_is_heap_until(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_is_heap_until(policy, input);
  }
}
