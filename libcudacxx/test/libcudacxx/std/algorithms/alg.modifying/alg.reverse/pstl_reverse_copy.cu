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
// OutputIterator reverse_copy(const Policy& policy,
//                             InputIterator first,
//                             InputIterator last,
//                             OutputIterator result);

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
void test_reverse_copy(const Policy& policy,
                       const c2h::device_vector<T>& input,
                       c2h::device_vector<T>& output,
                       const c2h::device_vector<int>& converting)
{
  { // empty should not access anything
    const auto res =
      cuda::std::reverse_copy(policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<T*>(nullptr));
    CHECK(res == nullptr);
  }

  auto expected = cuda::transform_iterator{cuda::strided_iterator{cuda::counting_iterator{size - 1}, -1}, cast_to<T>{}};

  cuda::std::fill(policy, output.begin(), output.end(), static_cast<T>(-1));
  { // contiguous input and output
    const auto res = cuda::std::reverse_copy(policy, input.begin(), input.end(), output.begin());
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), static_cast<T>(-1));
  const T* raw_pointer_in = thrust::raw_pointer_cast(input.data());
  { // random access input
    const auto res = cuda::std::reverse_copy(
      policy, random_access_iterator{raw_pointer_in}, random_access_iterator{raw_pointer_in + size}, output.begin());
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), static_cast<T>(-1));
  T* raw_pointer_out = thrust::raw_pointer_cast(output.data());
  { // contiguous input, random access output
    const auto res =
      cuda::std::reverse_copy(policy, input.begin(), input.end(), random_access_iterator{raw_pointer_out});
    CHECK(res == random_access_iterator{raw_pointer_out + size});
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), static_cast<T>(-1));
  { // all random access
    const auto res = cuda::std::reverse_copy(
      policy,
      random_access_iterator{raw_pointer_in},
      random_access_iterator{raw_pointer_in + size},
      random_access_iterator{raw_pointer_out});
    CHECK(res == random_access_iterator{raw_pointer_out + size});
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), static_cast<T>(-1));
  { // converting contiguous input
    const auto res = cuda::std::reverse_copy(policy, converting.begin(), converting.end(), output.begin());
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), static_cast<T>(-1));
  const int* raw_pointer_converting = thrust::raw_pointer_cast(converting.data());
  { // random access input
    const auto res = cuda::std::reverse_copy(
      policy,
      random_access_iterator{raw_pointer_converting},
      random_access_iterator{raw_pointer_converting + size},
      output.begin());
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }
}

C2H_TEST("cuda::std::reverse_copy", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);
  c2h::device_vector<T> output(size, thrust::no_init);
  c2h::device_vector<int> converting(size, thrust::no_init);
  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  thrust::sequence(converting.begin(), converting.end(), 0);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_reverse_copy(policy, input, output, converting);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_reverse_copy(policy, input, output, converting);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_reverse_copy(policy, input, output, converting);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_reverse_copy(policy, input, output, converting);
  }
}
