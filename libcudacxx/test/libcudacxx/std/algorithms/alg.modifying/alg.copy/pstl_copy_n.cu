//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class OutputIterator, class Size>
// void copy_n(const Policy&  policy,
//             InputIterator  first,
//             Size           count,
//             OutoutIterator result);

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
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

template <class Policy, class T>
void test_copy_n(const Policy& policy,
                 const c2h::device_vector<T>& input,
                 c2h::device_vector<T>& output,
                 const c2h::device_vector<int>& converting)
{
  { // empty should not access anything
    const auto res = cuda::std::copy_n(policy, static_cast<int*>(nullptr), 0, static_cast<int*>(nullptr));
    CHECK(res == nullptr);
  }

  cuda::std::fill(policy, output.begin(), output.end(), static_cast<T>(-1));
  { // With contiguous iterator
    const auto res = cuda::std::copy_n(policy, input.begin(), size, output.begin());
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), input.begin()));
  }

  cuda::std::fill(policy, output.begin(), output.end(), static_cast<T>(-1));
  const T* raw_pointer = thrust::raw_pointer_cast(input.data());
  { // With non-contiguous input iterator
    const auto res = cuda::std::copy_n(policy, random_access_iterator{raw_pointer}, size, output.begin());
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), input.begin()));
  }

  cuda::std::fill(policy, output.begin(), output.end(), static_cast<T>(-1));
  T* raw_out_pointer = thrust::raw_pointer_cast(output.data());
  { // With non-contiguous output iterator
    const auto res = cuda::std::copy_n(policy, input.begin(), size, random_access_iterator{raw_out_pointer});
    CHECK(res == random_access_iterator{raw_out_pointer + size});
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), input.begin()));
  }

  cuda::std::fill(policy, output.begin(), output.end(), static_cast<T>(-1));
  { // All non-contiguous iterators
    const auto res =
      cuda::std::copy_n(policy, random_access_iterator{raw_pointer}, size, random_access_iterator{raw_out_pointer});
    CHECK(res == random_access_iterator{raw_out_pointer + size});
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), input.begin()));
  }

  cuda::std::fill(policy, output.begin(), output.end(), static_cast<T>(-1));
  { // From a converting contiguous sequence
    const auto res = cuda::std::copy_n(policy, converting.begin(), size, output.begin());
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), input.begin()));
  }

  cuda::std::fill(policy, output.begin(), output.end(), static_cast<T>(-1));
  const int* raw_converting_pointer = thrust::raw_pointer_cast(converting.data());
  { // From a converting random access sequence
    const auto res = cuda::std::copy_n(policy, random_access_iterator{raw_converting_pointer}, size, output.begin());
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), input.begin()));
  }
}

C2H_TEST("cuda::std::copy_n", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> output(size, thrust::no_init);
  c2h::device_vector<T> input(size, thrust::no_init);
  c2h::device_vector<int> converting(size, thrust::no_init);
  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  thrust::sequence(converting.begin(), converting.end(), 0);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;

    test_copy_n(policy, input, output, converting);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);

    test_copy_n(policy, input, output, converting);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);

    test_copy_n(policy, input, output, converting);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);

    test_copy_n(policy, input, output, converting);
  }
}
