//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class Size, class T>
// InputIterator fill_n(const Policy& policy, InputIterator first, Size count, const T& value);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 1000;

template <class Policy>
void test_fill_n(const Policy& policy, thrust::device_vector<int>& output)
{
  { // empty should not access anything
    const int val = 42;
    cuda::std::fill_n(policy, static_cast<int*>(nullptr), 0, val);
  }

  { // same type
    const int val = 42;
    cuda::std::fill_n(policy, output.begin(), size, val);
    CHECK(thrust::equal(output.begin(), output.end(), cuda::constant_iterator{val}));
  }

  { // convertible type
    const short val = 1337;
    cuda::std::fill_n(policy, output.begin(), size, val);
    CHECK(thrust::equal(output.begin(), output.end(), cuda::constant_iterator{val}));
  }
}

C2H_TEST("cuda::std::fill_n", "[parallel algorithm]")
{
  thrust::device_vector<int> output(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_fill_n(policy, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_fill_n(policy, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_fill_n(policy, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_fill_n(policy, output);
  }
}
