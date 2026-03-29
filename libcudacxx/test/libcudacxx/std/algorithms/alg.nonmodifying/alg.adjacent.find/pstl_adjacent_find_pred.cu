//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class Policy, class InputIterator, class BinaryPredicate>
// InputIterator adjacent_find(Policy policy,
//                             InputIterator first,
//                             InputIterator last,
//                             BinaryPredicate pred)

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_iterators.h"
#include "test_macros.h"

inline constexpr int size = 1000;

template <class Policy>
void test_adjacent_find(const Policy& policy, const thrust::device_vector<int>& input)
{
  { // empty should not access anything
    auto res =
      cuda::std::adjacent_find(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), cuda::std::greater<>{});
    CHECK(res == static_cast<int*>(nullptr));
  }

  {
    auto res = cuda::std::adjacent_find(policy, input.begin(), input.end(), cuda::std::greater<>{});
    CHECK(*res == *(input.begin() + 42));
  }

  { // non contiguous input
    auto* inptr = thrust::raw_pointer_cast(input.data());
    auto res    = cuda::std::adjacent_find(
      policy, random_access_iterator{inptr}, random_access_iterator{inptr + size}, cuda::std::greater<>{});
    CHECK(res == random_access_iterator{inptr + 42});
  }
}

C2H_TEST("cuda::std::adjacent_find(Iter, Iter, comp)", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size);
  thrust::sequence(input.begin(), input.end(), 1);
  input[42] = 1337;

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_adjacent_find(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with(cuda::get_stream, stream);
    test_adjacent_find(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with(cuda::mr::get_memory_resource, device_resource);
    test_adjacent_find(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy                            = cuda::execution::__cub_par_unseq.with(cuda::get_stream, stream)
                          .with(cuda::mr::get_memory_resource, device_resource);
    test_adjacent_find(policy, input);
  }
}
