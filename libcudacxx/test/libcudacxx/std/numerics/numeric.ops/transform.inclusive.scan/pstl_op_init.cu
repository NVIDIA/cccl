//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class Policy, class InputIterator, class OutputIterator, class BinaryOperator>
// OutputIterator transform_inclusive_scan(Policy policy,
//                               InputIterator first,
//                               InputIterator last,
//                               OutputIterator result,
//                               BinaryOperator op)

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_macros.h"

inline constexpr int size = 1000;

template <class T>
struct plus_two
{
  TEST_DEVICE_FUNC constexpr T operator()(const T val) const noexcept
  {
    return val + 2;
  }
};

struct sum_of_int
{
  TEST_DEVICE_FUNC constexpr int operator()(const int val) const noexcept
  { // Eulers sum with initial value, account for plus_two and start at one
    plus_two<int> op{};
    return 42 + op(val) * (op(val) + 1) / 2 - op(1);
  }
};

template <class Policy>
void test_inclusive_scan(
  const Policy& policy, const thrust::device_vector<int>& input, thrust::device_vector<int>& output)
{
  cuda::transform_iterator expected{cuda::counting_iterator{1}, sum_of_int{}};
  { // empty should not access anything
    auto res = cuda::std::transform_inclusive_scan(
      policy,
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      cuda::std::plus<int>{},
      plus_two<int>{},
      42);
    CHECK(res == nullptr);
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // same output type
    auto res = cuda::std::transform_inclusive_scan(
      policy, input.begin(), input.end(), output.begin(), cuda::std::plus<int>{}, plus_two<int>{}, int{42});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // same output type, conversion with binary op
    auto res = cuda::std::transform_inclusive_scan(
      policy, input.begin(), input.end(), output.begin(), cuda::std::plus<long>{}, plus_two<int>{}, int{42});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // same output type, conversion with unary op
    auto res = cuda::std::transform_inclusive_scan(
      policy, input.begin(), input.end(), output.begin(), cuda::std::plus<int>{}, plus_two<long>{}, int{42});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // same output type, different initial value
    auto res = cuda::std::transform_inclusive_scan(
      policy, input.begin(), input.end(), output.begin(), cuda::std::plus<int>{}, plus_two<int>{}, short{42});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // same output type, different initial value, conversion with binary op
    auto res = cuda::std::transform_inclusive_scan(
      policy, input.begin(), input.end(), output.begin(), cuda::std::plus<long>{}, plus_two<int>{}, short{42});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // same output type, different initial value, conversion with unary op
    auto res = cuda::std::transform_inclusive_scan(
      policy, input.begin(), input.end(), output.begin(), cuda::std::plus<int>{}, plus_two<long>{}, short{42});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // different output type
    auto res = cuda::std::transform_inclusive_scan(
      policy,
      cuda::counting_iterator{short{1}},
      cuda::counting_iterator{short{size + 1}},
      output.begin(),
      cuda::std::plus<int>{},
      plus_two<int>{},
      int{42});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  { // different output type, different initial value
    auto res = cuda::std::transform_inclusive_scan(
      policy,
      cuda::counting_iterator{short{1}},
      cuda::counting_iterator{short{size + 1}},
      output.begin(),
      cuda::std::plus<int>{},
      plus_two<int>{},
      long{42});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  { // different output type, conversion with binary
    auto res = cuda::std::transform_inclusive_scan(
      policy,
      cuda::counting_iterator{short{1}},
      cuda::counting_iterator{short{size + 1}},
      output.begin(),
      cuda::std::plus<long>{},
      plus_two<int>{},
      int{42});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  { // different output type, conversion with unary
    auto res = cuda::std::transform_inclusive_scan(
      policy,
      cuda::counting_iterator{short{1}},
      cuda::counting_iterator{short{size + 1}},
      output.begin(),
      cuda::std::plus<int>{},
      plus_two<long>{},
      int{42});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  { // different output type, different initial value, conversion with binary
    auto res = cuda::std::transform_inclusive_scan(
      policy,
      cuda::counting_iterator{short{1}},
      cuda::counting_iterator{short{size + 1}},
      output.begin(),
      cuda::std::plus<long>{},
      plus_two<int>{},
      int8_t{42});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }

  { // different output type, different initial value, conversion with unary
    auto res = cuda::std::transform_inclusive_scan(
      policy,
      cuda::counting_iterator{short{1}},
      cuda::counting_iterator{short{size + 1}},
      output.begin(),
      cuda::std::plus<int>{},
      plus_two<long>{},
      int8_t{42});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), expected));
  }
}

C2H_TEST("cuda::std::transform_inclusive_scan(Iter1, Iter1, Iter2, Op, Init)", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size);
  thrust::device_vector<int> output(size);
  thrust::sequence(input.begin(), input.end(), 1);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_inclusive_scan(policy, input, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_inclusive_scan(policy, input, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_inclusive_scan(policy, input, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::get_stream, stream).with(cuda::mr::get_memory_resource, device_resource);
    test_inclusive_scan(policy, input, output);
  }
}
