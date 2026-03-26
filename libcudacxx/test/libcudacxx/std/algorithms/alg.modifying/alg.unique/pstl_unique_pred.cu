//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class ExecutionPolicy, class InputIterator, class BinaryPredicate>
// InputIterator unique(ExecutionPolicy&& policy,
//                      InputIterator first,
//                      InputIterator last,
//                      BinaryPredicate pred);

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 1000;

template <class Policy>
void test_unique(const Policy& policy, thrust::device_vector<int>& input)
{
  { // empty should not access anything
    const auto res =
      cuda::std::unique(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), cuda::std::equal_to<>{});
    CHECK(res == nullptr);
  }

  thrust::sequence(input.begin(), input.end(), 0);
  { // no duplicates
    const auto res = cuda::std::unique(policy, input.begin(), input.end(), cuda::std::equal_to<>{});
    CHECK(res == input.end());
    CHECK(cuda::std::equal(policy, input.begin(), input.end(), cuda::counting_iterator{0}));
  }

  thrust::sequence(input.begin(), input.end(), 0);
  { // one match,
    input[43]      = input[42];
    const auto res = cuda::std::unique(policy, input.begin(), input.end(), cuda::std::equal_to<>{});
    CHECK(res == cuda::std::prev(input.end()));

    auto mid_in = input.begin() + 43;
    CHECK(cuda::std::equal(policy, input.begin(), mid_in, cuda::counting_iterator{0}));
    CHECK(cuda::std::equal(policy, mid_in, res, cuda::counting_iterator{44}));
  }

  cuda::std::fill(input.begin(), input.end(), 42);
  { // all equal, not_equal to ensure we are actually using the predicate
    const auto res = cuda::std::unique(policy, input.begin(), input.end(), cuda::std::not_equal_to<>{});
    CHECK(res == input.end());
    CHECK(cuda::std::equal(policy, input.begin(), input.end(), cuda::constant_iterator{42}));
  }

  { // all equal, continuous iterator
    const auto res = cuda::std::unique(policy, input.begin(), input.end(), cuda::std::equal_to<>{});
    CHECK(res == cuda::std::next(input.begin()));
    CHECK(input[0] == 42);
  }
}

C2H_TEST("cuda::std::unique", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_unique(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with(cuda::get_stream, stream);
    test_unique(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with(cuda::mr::get_memory_resource, device_resource);
    test_unique(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with(cuda::mr::get_memory_resource, device_resource)
                          .with(cuda::get_stream, stream);
    test_unique(policy, input);
  }
}
