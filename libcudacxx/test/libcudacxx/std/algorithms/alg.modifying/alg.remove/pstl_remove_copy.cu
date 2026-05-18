//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class OutputIterator, class T>
// OutputIterator remove_copy(const Policy&  policy,
//                            InputIterator  first,
//                            InputIterator  last,
//                            OutputIterator result,
//                            const T&       value);

#include <thrust/device_vector.h>
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

inline constexpr int size = 10000;

[[nodiscard]] TEST_FUNC constexpr bool operator==(const nontrivial_type& lhs, const int& rhs)
{
  return lhs.value_ == rhs;
}

[[nodiscard]] TEST_FUNC constexpr bool operator==(const int& lhs, const nontrivial_type& rhs)
{
  return lhs == rhs.value_;
}

template <class Policy, class T>
void test_remove_copy(const Policy& policy,
                      const c2h::device_vector<T>& input,
                      c2h::device_vector<T>& output,
                      const c2h::device_vector<int>& converting)
{
  { // empty should not access anything
    const auto res = cuda::std::remove_copy(
      policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<T>(42));
    CHECK(res == nullptr);
  }

  auto mid_in  = cuda::std::next(input.begin(), 43);
  auto mid_out = cuda::std::next(output.begin(), 42);

  cuda::std::fill(policy, output.begin(), output.end(), 0);
  { // contiguous iterator
    const auto res = cuda::std::remove_copy(policy, input.begin(), input.end(), output.begin(), static_cast<T>(42));
    CHECK(cuda::std::equal(policy, output.begin(), mid_out, input.begin()));
    CHECK(cuda::std::equal(policy, mid_out, res, mid_in));
    CHECK(cuda::std::distance(output.begin(), res) == size - 1);
  }

  cuda::std::fill(policy, output.begin(), output.end(), 0);
  const T* raw_pointer = thrust::raw_pointer_cast(input.data());
  { // random access iterator
    const auto res = cuda::std::remove_copy(
      policy,
      random_access_iterator{raw_pointer},
      random_access_iterator{raw_pointer + size},
      output.begin(),
      static_cast<T>(42));
    CHECK(cuda::std::equal(policy, output.begin(), mid_out, input.begin()));
    CHECK(cuda::std::equal(policy, mid_out, res, mid_in));
    CHECK(cuda::std::distance(output.begin(), res) == size - 1);
  }

  if constexpr (::cuda::std::__is_cpp17_equality_comparable_v<T, int>)
  {
    cuda::std::fill(policy, output.begin(), output.end(), 0);
    { // contiguous iterator, converting value
      const auto res = cuda::std::remove_copy(policy, input.begin(), input.end(), output.begin(), 42);
      CHECK(cuda::std::equal(policy, output.begin(), mid_out, input.begin()));
      CHECK(cuda::std::equal(policy, mid_out, res, mid_in));
      CHECK(cuda::std::distance(output.begin(), res) == size - 1);
    }

    cuda::std::fill(policy, output.begin(), output.end(), 0);
    { // random access iterator, converting value
      const auto res = cuda::std::remove_copy(
        policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size}, output.begin(), 42);
      CHECK(cuda::std::equal(policy, output.begin(), mid_out, input.begin()));
      CHECK(cuda::std::equal(policy, mid_out, res, mid_in));
      CHECK(cuda::std::distance(output.begin(), res) == size - 1);
    }
  }

  cuda::std::fill(policy, output.begin(), output.end(), 0);
  { // contiguous iterator, converting input sequence
    const auto res = cuda::std::remove_copy(policy, converting.begin(), converting.end(), output.begin(), 42);
    CHECK(cuda::std::equal(policy, output.begin(), mid_out, input.begin()));
    CHECK(cuda::std::equal(policy, mid_out, res, mid_in));
    CHECK(cuda::std::distance(output.begin(), res) == size - 1);
  }

  cuda::std::fill(policy, output.begin(), output.end(), 0);
  const int* raw_converting_pointer = thrust::raw_pointer_cast(converting.data());
  { // random access iterator, converting input sequence
    const auto res = cuda::std::remove_copy(
      policy,
      random_access_iterator{raw_converting_pointer},
      random_access_iterator{raw_converting_pointer + size},
      output.begin(),
      42);
    CHECK(cuda::std::equal(policy, output.begin(), mid_out, input.begin()));
    CHECK(cuda::std::equal(policy, mid_out, res, mid_in));
    CHECK(cuda::std::distance(output.begin(), res) == size - 1);
  }
}

C2H_TEST("cuda::std::remove_copy", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);
  c2h::device_vector<T> output(size - 1, thrust::no_init);
  c2h::device_vector<int> converting(size, thrust::no_init);
  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  thrust::sequence(converting.begin(), converting.end(), 0);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_remove_copy(policy, input, output, converting);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_remove_copy(policy, input, output, converting);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_remove_copy(policy, input, output, converting);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_remove_copy(policy, input, output, converting);
  }
}
