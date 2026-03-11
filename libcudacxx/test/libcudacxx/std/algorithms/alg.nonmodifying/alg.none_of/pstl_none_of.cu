//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, class ForwardIterator, class UnaryPredicate>
// void none_of(ExecutionPolicy&& exec ForwardIterator first, ForwardIterator last, UnaryPredicate pred);

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr short size = 1000;

template <class T = int>
struct less_than_val
{
  T val_;

  constexpr less_than_val(const T val) noexcept
      : val_(val)
  {}

  template <class U>
  __device__ constexpr bool operator()(const U& val) const noexcept
  {
    return static_cast<T>(val) < val_;
  }
};

template <class Policy>
void test_none_of(const Policy& policy)
{
  { // empty should not access anything
    const auto res =
      cuda::std::none_of(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), less_than_val{0});
    CHECK(res);
  }

  { // same type
    const auto res = cuda::std::none_of(
      policy, cuda::counting_iterator{short{0}}, cuda::counting_iterator{size}, less_than_val<short>{0});
    CHECK(res);
  }

  { // same type false
    const auto res = cuda::std::none_of(
      policy, cuda::counting_iterator{short{0}}, cuda::counting_iterator{size}, less_than_val<short>{1});
    CHECK(!res);
  }

  { // convertible pred
    const auto res = cuda::std::none_of(
      policy, cuda::counting_iterator{short{0}}, cuda::counting_iterator{size}, less_than_val<int>{0});
    CHECK(res);
  }

  { // convertible pred
    const auto res = cuda::std::none_of(
      policy, cuda::counting_iterator{short{0}}, cuda::counting_iterator{size}, less_than_val<int>{1});
    CHECK(!res);
  }
}

C2H_TEST("cuda::std::none_of", "[parallel algorithm]")
{
  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_none_of(policy);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_none_of(policy);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_none_of(policy);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream).with_memory_resource(device_resource);
    test_none_of(policy);
  }
}
