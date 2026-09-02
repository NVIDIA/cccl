//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class UnaryPredicate>
// void stable_partition(const Policy&  policy,
//                       InputIterator  first,
//                       InputIterator  last,
//                       UnaryPredicate pred);

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

#include "test_macros.h"

inline constexpr int size = 1000;

template <class T>
struct is_even
{
  [[nodiscard]] TEST_DEVICE_FUNC constexpr bool operator()(T value) const noexcept
  {
    return value % 2 == 0;
  }
};

template <class Policy>
void test_partition(const Policy& policy, thrust::device_vector<int>& input)
{
  { // Empty does not access anything
    auto res =
      cuda::std::stable_partition(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), is_even<int>{});
    CHECK(res == nullptr);
  }

  const auto mid = size / 2;
  thrust::sequence(input.begin(), input.end(), 0);
  { // With matching predicate
    auto res = cuda::std::stable_partition(policy, input.begin(), input.end(), is_even<int>{});
    CHECK(res == cuda::std::next(input.begin(), mid));
    CHECK(cuda::std::equal(policy, input.begin(), res, cuda::strided_iterator{cuda::counting_iterator{0}, 2}));
    CHECK(cuda::std::equal(policy, res, input.end(), cuda::strided_iterator{cuda::counting_iterator{1}, 2}));
  }

  thrust::sequence(input.begin(), input.end(), 0);
  { // With converting predicate
    auto res = cuda::std::stable_partition(policy, input.begin(), input.end(), is_even<long>{});
    CHECK(res == cuda::std::next(input.begin(), mid));
    CHECK(cuda::std::equal(policy, input.begin(), res, cuda::strided_iterator{cuda::counting_iterator{0}, 2}));
    CHECK(cuda::std::equal(policy, res, input.end(), cuda::strided_iterator{cuda::counting_iterator{1}, 2}));
  }
}

C2H_TEST("cuda::std::stable_partition", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;

    test_partition(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);

    test_partition(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);

    test_partition(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);

    test_partition(policy, input);
  }
}
