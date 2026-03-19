//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, ForwardIterator Iter, BinaryPredicate>
// bool is_sorted(ExecutionPolicy&& policy, Iter first, Iter last, BinaryPredicate pred);

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 1000;

struct turn_42_into_1337
{
  __device__ constexpr int operator()(const int val) const noexcept
  {
    return val == 42 ? 1337 : val;
  }
};

template <class Policy>
void test_is_sorted(const Policy& policy, thrust::device_vector<int>& input)
{
  { // empty should not access anything
    const auto res =
      cuda::std::is_sorted(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), cuda::std::greater<>{});
    CHECK(res);
  }

  thrust::sequence(input.begin(), input.end(), size, -1);
  { // sorted contiguous range
    const auto res = cuda::std::is_sorted(policy, input.begin(), input.end(), cuda::std::greater<>{});
    CHECK(res);
  }

  { // sorted random access range
    const auto res = cuda::std::is_sorted(
      policy,
      cuda::strided_iterator{cuda::counting_iterator{size}, -1},
      cuda::strided_iterator{cuda::counting_iterator{0}, -1},
      cuda::std::greater<>{});
    CHECK(res);
  }

  { // unsorted contiguous range
    input[42]      = 1337;
    const auto res = cuda::std::is_sorted(policy, input.begin(), input.end(), cuda::std::greater<>{});
    CHECK(!res);
  }

  { // unsorted random access range
    const auto res = cuda::std::is_sorted(
      policy,
      cuda::transform_iterator{cuda::strided_iterator{cuda::counting_iterator{size}, -1}, turn_42_into_1337{}},
      cuda::transform_iterator{cuda::strided_iterator{cuda::counting_iterator{0}, -1}, turn_42_into_1337{}},
      cuda::std::greater<>{});
    CHECK(!res);
  }
}

C2H_TEST("cuda::std::is_sorted(iter, iter, pred)", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_is_sorted(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_is_sorted(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_is_sorted(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_is_sorted(policy, input);
  }
}
