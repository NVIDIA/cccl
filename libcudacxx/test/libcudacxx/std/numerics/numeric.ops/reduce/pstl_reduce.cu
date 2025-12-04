//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, class ForwardIterator>
//   typename iterator_traits<ForwardIterator>::value_type
//     reduce(ExecutionPolicy&& exec,
//            ForwardIterator first, ForwardIterator last);
// template<class ExecutionPolicy, class ForwardIterator, class T, class BinaryOperation>
//   T reduce(ExecutionPolicy&& exec,
//            ForwardIterator first, ForwardIterator last, T init,
//            BinaryOperation binary_op);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/iterator>
#include <cuda/std/__pstl/reduce.h>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_macros.h"

inline constexpr int size = 100;

C2H_TEST("cuda::std::reduce(Iter, Iter)", "[parallel algorithm]")
{
  SECTION("with default stream")
  {
    thrust::device_vector<int> data(size);
    thrust::sequence(data.begin(), data.end(), 1);

    const auto policy  = cuda::execution::__cub_par_unseq;
    decltype(auto) res = cuda::std::reduce(policy, data.begin(), data.end());
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), int>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)

    constexpr int expected = size * (size + 1) / 2;
    CHECK(res == expected);
  }

  SECTION("with provided stream")
  {
    thrust::device_vector<int> data(size);
    thrust::sequence(data.begin(), data.end(), 1);

    ::cuda::stream stream{::cuda::device_ref{0}};
    const auto policy  = cuda::execution::__cub_par_unseq.set_stream(stream);
    decltype(auto) res = cuda::std::reduce(policy, data.begin(), data.end());
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), int>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)

    constexpr int expected = size * (size + 1) / 2;
    CHECK(res == expected);
  }
}

C2H_TEST("cuda::std::reduce(Iter, Iter, Tp)", "[parallel algorithm]")
{
  SECTION("with default stream")
  {
    thrust::device_vector<int> data(size);
    thrust::sequence(data.begin(), data.end(), 1);

    const auto policy  = cuda::execution::__cub_par_unseq;
    decltype(auto) res = cuda::std::reduce(policy, data.begin(), data.end(), 42);
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), int>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)

    constexpr int expected = size * (size + 1) / 2 + 42;
    CHECK(res == expected);
  }

  SECTION("with provided stream")
  {
    thrust::device_vector<int> data(size);
    thrust::sequence(data.begin(), data.end(), 1);

    ::cuda::stream stream{::cuda::device_ref{0}};
    const auto policy  = cuda::execution::__cub_par_unseq.set_stream(stream);
    decltype(auto) res = cuda::std::reduce(policy, data.begin(), data.end(), 42);
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), int>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)

    constexpr int expected = size * (size + 1) / 2 + 42;
    CHECK(res == expected);
  }
}

struct plus_two
{
  __host__ __device__ constexpr int operator()(const int lhs, const int rhs) const noexcept
  {
    return lhs + rhs + 2;
  };
};

C2H_TEST("cuda::std::reduce(Iter, Iter, Tp, Fn)", "[parallel algorithm]")
{
  SECTION("with default stream")
  {
    thrust::device_vector<int> data(size);
    thrust::sequence(data.begin(), data.end(), 1);

    const auto policy  = cuda::execution::__cub_par_unseq;
    decltype(auto) res = cuda::std::reduce(policy, data.begin(), data.end(), 42, plus_two{});
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), int>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)

    constexpr int expected = size * (size + 1) / 2 + 42 + size * 2;
    CHECK(res == expected);
  }

  SECTION("with provided stream")
  {
    thrust::device_vector<int> data(size);
    thrust::sequence(data.begin(), data.end(), 1);

    ::cuda::stream stream{::cuda::device_ref{0}};
    const auto policy  = cuda::execution::__cub_par_unseq.set_stream(stream);
    decltype(auto) res = cuda::std::reduce(policy, data.begin(), data.end(), 42, plus_two{});
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), int>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)

    constexpr int expected = size * (size + 1) / 2 + 42 + size * 2;
    CHECK(res == expected);
  }
}
