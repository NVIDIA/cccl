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
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 10000;

template <class Policy>
void test_remove_copy(const Policy& policy, thrust::device_vector<int>& output)
{
  { // empty should not access anything
    const auto res = cuda::std::remove_copy(
      policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), static_cast<int*>(nullptr), 42);
    CHECK(res == nullptr);
  }

  { // With matching value
    thrust::fill(output.begin(), output.end(), 0);
    const auto res =
      cuda::std::remove_copy(policy, cuda::counting_iterator{0}, cuda::counting_iterator{size}, output.begin(), 42);
    CHECK(cuda::std::distance(output.begin(), res) == size - 1);

    const auto mid = cuda::std::next(output.begin(), 42);
    CHECK(thrust::equal(output.begin(), mid, cuda::counting_iterator{0}));
    CHECK(thrust::equal(mid, res, cuda::counting_iterator{43}));
  }

  { // With conversion for value type
    thrust::fill(output.begin(), output.end(), 0);
    const auto res = cuda::std::remove_copy(
      policy, cuda::counting_iterator{0}, cuda::counting_iterator{size}, output.begin(), short{42});
    CHECK(cuda::std::distance(output.begin(), res) == size - 1);

    const auto mid = cuda::std::next(output.begin(), 42);
    CHECK(thrust::equal(output.begin(), mid, cuda::counting_iterator{0}));
    CHECK(thrust::equal(mid, res, cuda::counting_iterator{43}));
  }
}

C2H_TEST("cuda::std::remove_copy", "[parallel algorithm]")
{
  thrust::device_vector<int> output(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_remove_copy(policy, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_remove_copy(policy, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_remove_copy(policy, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_remove_copy(policy, output);
  }
}
