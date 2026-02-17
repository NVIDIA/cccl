//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class T>
// void remove(const Policy&  policy,
//             InputIterator  first,
//             InputIterator  last,
//             const T&       value);

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 10000;

struct is_42
{
  __device__ constexpr bool operator()(const int& value) const noexcept
  {
    return value == 42;
  }
};

template <class Policy>
void test_remove(const Policy& policy, thrust::device_vector<int>& input)
{
  { // empty should not access anything
    const auto res = cuda::std::remove(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), 42);
    CHECK(res == nullptr);
  }

  { // With matching predicate
    thrust::sequence(input.begin(), input.end(), 0);
    const auto res = cuda::std::remove(policy, input.begin(), input.end(), 42);
    CHECK(cuda::std::distance(input.begin(), res) == size - 1);

    const auto mid = cuda::std::next(input.begin(), 42);
    CHECK(thrust::equal(input.begin(), mid, cuda::counting_iterator{0}));
    CHECK(thrust::equal(mid, res, cuda::counting_iterator{43}));
  }

  { // With convertion for predicate
    thrust::sequence(input.begin(), input.end(), 0);
    const auto res = cuda::std::remove(policy, input.begin(), input.end(), short{42});
    CHECK(cuda::std::distance(input.begin(), res) == size - 1);

    const auto mid = cuda::std::next(input.begin(), 42);
    CHECK(thrust::equal(input.begin(), mid, cuda::counting_iterator{0}));
    CHECK(thrust::equal(mid, res, cuda::counting_iterator{43}));
  }
}

C2H_TEST("cuda::std::remove", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_remove(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_remove(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_remove(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_remove(policy, input);
  }
}
