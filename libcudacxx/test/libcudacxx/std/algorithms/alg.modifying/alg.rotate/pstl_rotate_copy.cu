//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class OutputIterator>
// void rotate_copy_copy(const Policy& policy,
//                  InputIterator first,
//                  InputIterator middle,
//                  InputIterator last,
//                  OutputIterator result);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/cmath>
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
void test_rotate_copy(const Policy& policy, const c2h::device_vector<T>& input, c2h::device_vector<T>& output)
{
  { // Empty does not access anything
    auto res = cuda::std::rotate_copy(
      policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<T*>(nullptr));
    CHECK(res == nullptr);
  }

  const auto count1 = 42;
  const auto count2 = size - count1;

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // Empty first part
    auto res = cuda::std::rotate_copy(policy, input.begin(), input.begin(), input.end(), output.begin());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), input.begin()));
    CHECK(res == output.end());
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // Empty second part
    auto res = cuda::std::rotate_copy(policy, input.begin(), input.end(), input.end(), output.begin());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), input.begin()));
    CHECK(res == output.end());
  }

  const auto expected_first  = cuda::transform_iterator{cuda::counting_iterator{count1}, cast_to<T>{}};
  const auto expected_second = cuda::transform_iterator{cuda::counting_iterator{0}, cast_to<T>{}};
  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // With contiguous iterator
    auto res = cuda::std::rotate_copy(
      policy, input.begin(), cuda::std::next(input.begin(), count1), input.end(), output.begin());
    CHECK(cuda::std::equal(policy, output.begin(), cuda::std::next(output.begin(), count2), expected_first));
    CHECK(cuda::std::equal(policy, cuda::std::next(output.begin(), count2), output.end(), expected_second));
    CHECK(res == output.end());
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  const T* raw_pointer = thrust::raw_pointer_cast(input.data());
  { // With random access iterator
    auto res = cuda::std::rotate_copy(
      policy,
      random_access_iterator{raw_pointer},
      random_access_iterator{raw_pointer + count1},
      random_access_iterator{raw_pointer + size},
      output.begin());
    CHECK(cuda::std::equal(policy, output.begin(), cuda::std::next(output.begin(), count2), expected_first));
    CHECK(cuda::std::equal(policy, cuda::std::next(output.begin(), count2), output.end(), expected_second));
    CHECK(res == output.end());
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  T* raw_pointer_out = thrust::raw_pointer_cast(output.data());
  { // With random access output iterator
    auto res = cuda::std::rotate_copy(
      policy,
      input.begin(),
      cuda::std::next(input.begin(), count1),
      input.end(),
      random_access_iterator{raw_pointer_out});
    CHECK(cuda::std::equal(policy, output.begin(), cuda::std::next(output.begin(), count2), expected_first));
    CHECK(cuda::std::equal(policy, cuda::std::next(output.begin(), count2), output.end(), expected_second));
    CHECK(res == random_access_iterator{raw_pointer_out + size});
  }
}

C2H_TEST("cuda::std::rotate_copy", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);
  c2h::device_vector<T> output(size, thrust::no_init);
  thrust::sequence(input.begin(), input.end(), 0);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;

    test_rotate_copy(policy, input, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);

    test_rotate_copy(policy, input, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);

    test_rotate_copy(policy, input, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);

    test_rotate_copy(policy, input, output);
  }
}
