//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class T = iter_value_t<T>>
// void iota(const Policy&  policy,
//           InputIterator  first,
//           InputIterator  last,
//           const T& init = T{});

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 1000;

template <class Policy>
void test_iota(const Policy& policy, thrust::device_vector<int>& output)
{
  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // With default value
    cuda::iota(policy, output.begin(), output.end());
    CHECK(cuda::std::equal(output.begin(), output.end(), cuda::counting_iterator{0}));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // With init
    cuda::iota(policy, output.begin(), output.end(), 42);
    CHECK(cuda::std::equal(output.begin(), output.end(), cuda::counting_iterator{42}));
  }

  cuda::std::fill(policy, output.begin(), output.end(), -1);
  { // With init and step
    cuda::iota(policy, output.begin(), output.end(), 42, 2);
    CHECK(cuda::std::equal(output.begin(), output.end(), cuda::strided_iterator{cuda::counting_iterator{42}, 2}));
  }
}

C2H_TEST("cuda::iota", "[parallel algorithm]")
{
  thrust::device_vector<int> output(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;

    test_iota(policy, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);

    test_iota(policy, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);

    test_iota(policy, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);

    test_iota(policy, output);
  }
}
