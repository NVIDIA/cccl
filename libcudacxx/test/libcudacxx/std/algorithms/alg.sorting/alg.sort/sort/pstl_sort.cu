//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<RandomAccessIterator Iter>
// void sort(Iter first, Iter last);

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
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
void test_sort(const Policy& policy, c2h::device_vector<T>& input, c2h::device_vector<int>& data)
{
  { // empty should not access anything
    cuda::std::sort(policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr));
  }

  cuda::std::transform(policy, data.begin(), data.end(), input.begin(), cast_to<T>{});
  { // unsorted contiguous range
    cuda::std::sort(policy, input.begin(), input.end());
    CHECK(cuda::std::is_sorted(policy, input.begin(), input.end()));
  }

  { // sorted contiguous range
    cuda::std::sort(policy, input.begin(), input.end());
    CHECK(cuda::std::is_sorted(policy, input.begin(), input.end()));
  }

  cuda::std::transform(policy, data.begin(), data.end(), input.begin(), cast_to<T>{});
  T* raw_pointer = thrust::raw_pointer_cast(input.data());
  { // unsorted random access range
    cuda::std::sort(policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size});
    CHECK(cuda::std::is_sorted(policy, input.begin(), input.end()));
  }

  { // sorted random access range
    cuda::std::sort(policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size});
    CHECK(cuda::std::is_sorted(policy, input.begin(), input.end()));
  }
}

C2H_TEST("cuda::std::sort(iter, iter)", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<int> data(size, thrust::no_init);
  c2h::gen(C2H_SEED(10), data);
  c2h::device_vector<T> input(size);
  cuda::std::transform(cuda::execution::gpu, data.begin(), data.end(), input.begin(), cast_to<T>{});

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_sort(policy, input, data);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_sort(policy, input, data);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_sort(policy, input, data);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_sort(policy, input, data);
  }
}
