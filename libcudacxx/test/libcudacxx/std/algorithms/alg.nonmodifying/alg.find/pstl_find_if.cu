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
// void find_if(ExecutionPolicy&& exec ForwardIterator first, ForwardIterator last, UnaryPredicate pred);

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr size_t size = 1000;

template <class T = int>
struct equal_to_val
{
  T val_;

  constexpr equal_to_val(const T val) noexcept
      : val_(val)
  {}

  template <class U>
  __device__ constexpr bool operator()(const U& val) const noexcept
  {
    return static_cast<T>(val) == val_;
  }
};

template <class Policy>
void test_find_if(const Policy& policy)
{
  const size_t expected = 42;

  { // empty should not access anything
    const auto res =
      cuda::std::find_if(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), equal_to_val{expected});
    CHECK(res == nullptr);
  }

  { // same type
    const auto res = cuda::std::find_if(
      policy, cuda::counting_iterator{size_t{0}}, cuda::counting_iterator{size}, equal_to_val<size_t>{expected});
    CHECK(res == cuda::counting_iterator{expected});
  }

  { // convertible type
    const auto res = cuda::std::find_if(
      policy,
      cuda::counting_iterator{size_t{0}},
      cuda::counting_iterator{size},
      equal_to_val<int>{static_cast<int>(expected)});
    CHECK(res == cuda::counting_iterator{expected});
  }
}

C2H_TEST("cuda::std::find_if", "[parallel algorithm]")
{
  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_find_if(policy);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_find_if(policy);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_find_if(policy);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream).with_memory_resource(device_resource);
    test_find_if(policy);
  }
}
