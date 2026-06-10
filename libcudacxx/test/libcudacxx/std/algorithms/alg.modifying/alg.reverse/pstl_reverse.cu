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
// void reverse(const Policy& policy,
//                 InputIterator first,
//                 InputIterator last,
//                 const T& old_value,
//                 const T& new_value);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_pstl.h"

inline constexpr int size = 1000;

template <class Policy, class T>
void test_reverse(const Policy& policy, c2h::device_vector<T>& input)
{
  { // empty should not access anything
    cuda::std::reverse(policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr));
  }

  auto expected = cuda::transform_iterator{cuda::strided_iterator{cuda::counting_iterator{size - 1}, -1}, cast_to<T>{}};

  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  { // contiguous
    cuda::std::reverse(policy, input.begin(), input.end());
    CHECK(cuda::std::equal(policy, input.begin(), input.end(), expected));
  }

  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  T* raw_pointer = thrust::raw_pointer_cast(input.data());
  { // random access
    cuda::std::reverse(policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size});
    CHECK(cuda::std::equal(policy, input.begin(), input.end(), expected));
  }
}

C2H_TEST("cuda::std::reverse", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_reverse(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_reverse(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_reverse(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_reverse(policy, input);
  }
}
