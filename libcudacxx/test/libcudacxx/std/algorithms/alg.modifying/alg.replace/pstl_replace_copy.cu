//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class UnaryPred, class T = iter_value_t<InputIterator>>
// void replace_copy(const Policy& policy,
//                 InputIterator first,
//                 InputIterator last,
//                 UnaryPred pred,
//                 const T& new_value);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 1000;

struct not_42
{
  __device__ constexpr bool operator()(const int val) const noexcept
  {
    return val == 42;
  }
};

C2H_TEST("cuda::std::replace_copy", "[parallel algorithm]")
{
  thrust::device_vector<int> output(size, thrust::no_init);

  SECTION("with default stream")
  {
    thrust::sequence(output.begin(), output.end(), 0);

    const auto policy = cuda::execution::__cub_par_unseq;
    cuda::std::replace_copy(policy, cuda::counting_iterator{0}, cuda::counting_iterator{size}, output.begin(), 42, 1337);
    CHECK(thrust::none_of(output.begin(), output.end(), not_42{}));
  }

  SECTION("with provided stream")
  {
    thrust::sequence(output.begin(), output.end(), 0);

    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    cuda::std::replace_copy(policy, cuda::counting_iterator{0}, cuda::counting_iterator{size}, output.begin(), 42, 1337);
    CHECK(thrust::none_of(output.begin(), output.end(), not_42{}));
  }

  SECTION("with provided memory_resource")
  {
    thrust::sequence(output.begin(), output.end(), 0);

    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    cuda::std::replace_copy(policy, cuda::counting_iterator{0}, cuda::counting_iterator{size}, output.begin(), 42, 1337);
    CHECK(thrust::none_of(output.begin(), output.end(), not_42{}));
  }

  SECTION("with provided stream and memory_resource")
  {
    thrust::sequence(output.begin(), output.end(), 0);

    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    cuda::std::replace_copy(policy, cuda::counting_iterator{0}, cuda::counting_iterator{size}, output.begin(), 42, 1337);
    CHECK(thrust::none_of(output.begin(), output.end(), not_42{}));
  }
}
