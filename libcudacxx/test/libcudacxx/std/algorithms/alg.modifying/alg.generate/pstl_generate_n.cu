//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class Generator>
// void generate_n(const Policy& policy,
//               InputIterator first,
//               InputIterator last,
//               Generator gen);

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 1000;

template <class T = int>
struct gen_val
{
  int val_;

  constexpr gen_val(const int val) noexcept
      : val_(val)
  {}

  __device__ constexpr T operator()() const noexcept
  {
    return static_cast<T>(val_);
  }
};

template <class Policy>
void test_generate_n(const Policy& policy, thrust::device_vector<int>& output)
{
  { // empty should not access anything
    cuda::std::generate_n(policy, static_cast<int*>(nullptr), 0, gen_val{42});
  }

  { // same type
    thrust::fill(output.begin(), output.end(), 0);
    cuda::std::generate_n(policy, output.begin(), size, gen_val{42});
    CHECK(thrust::equal(output.begin(), output.end(), cuda::constant_iterator{42}));
  }

  { // convertible type
    thrust::fill(output.begin(), output.end(), 0);
    cuda::std::generate_n(policy, output.begin(), size, gen_val<short>{42});
    CHECK(thrust::equal(output.begin(), output.end(), cuda::constant_iterator{42}));
  }
}

C2H_TEST("cuda::std::generate_n", "[parallel algorithm]")
{
  thrust::device_vector<int> output(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_generate_n(policy, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_generate_n(policy, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_generate_n(policy, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_generate_n(policy, output);
  }
}
