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
// void all_of(ExecutionPolicy&& exec ForwardIterator first, ForwardIterator last, UnaryPredicate pred);

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr size_t size = 1000;

struct less_than_val
{
  size_t val_;

  constexpr less_than_val(const size_t val) noexcept
      : val_(val)
  {}

  template <class T>
  __device__ constexpr bool operator()(const T& val) const noexcept
  {
    return val < val_;
  }
};

C2H_TEST("cuda::std::all_of", "[parallel algorithm]")
{
  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    { // true case
      const auto res = cuda::std::all_of(
        policy, cuda::counting_iterator{size_t{0}}, cuda::counting_iterator{size}, less_than_val{size + 1});
      CHECK(res);
    }

    { // false case
      const auto res = cuda::std::all_of(
        policy, cuda::counting_iterator{size_t{0}}, cuda::counting_iterator{size}, less_than_val{size - 1});
      CHECK(!res);
    }
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    { // true case
      const auto res = cuda::std::all_of(
        policy, cuda::counting_iterator{size_t{0}}, cuda::counting_iterator{size}, less_than_val{size + 1});
      CHECK(res);
    }

    { // false case
      const auto res = cuda::std::all_of(
        policy, cuda::counting_iterator{size_t{0}}, cuda::counting_iterator{size}, less_than_val{size - 1});
      CHECK(!res);
    }
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    { // true case
      const auto res = cuda::std::all_of(
        policy, cuda::counting_iterator{size_t{0}}, cuda::counting_iterator{size}, less_than_val{size + 1});
      CHECK(res);
    }

    { // false case
      const auto res = cuda::std::all_of(
        policy, cuda::counting_iterator{size_t{0}}, cuda::counting_iterator{size}, less_than_val{size - 1});
      CHECK(!res);
    }
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream).with_memory_resource(device_resource);
    { // true case
      const auto res = cuda::std::all_of(
        policy, cuda::counting_iterator{size_t{0}}, cuda::counting_iterator{size}, less_than_val{size + 1});
      CHECK(res);
    }

    { // false case
      const auto res = cuda::std::all_of(
        policy, cuda::counting_iterator{size_t{0}}, cuda::counting_iterator{size}, less_than_val{size - 1});
      CHECK(!res);
    }
  }
}
