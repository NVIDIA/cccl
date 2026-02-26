//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_macros.h"

inline constexpr int size = 100;

template <class Policy>
void test_reduce(const Policy& policy, const thrust::device_vector<int>& data)
{
  { // empty should not access anything
    decltype(auto) res = cuda::std::reduce(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr));
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), int>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    CHECK(res == 0);
  }

  { // same type
    decltype(auto) res = cuda::std::reduce(policy, data.begin(), data.end());
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), int>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)

    constexpr int expected = size * (size + 1) / 2;
    CHECK(res == expected);
  }
}

C2H_TEST("cuda::std::reduce(Iter, Iter)", "[parallel algorithm]")
{
  thrust::device_vector<int> data(size);
  thrust::sequence(data.begin(), data.end(), 1);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_reduce(policy, data);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_reduce(policy, data);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_reduce(policy, data);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream).with_memory_resource(device_resource);
    test_reduce(policy, data);
  }
}

template <class Policy>
void test_reduce_init(const Policy& policy, const thrust::device_vector<int>& data)
{
  { // empty should not access anything
    decltype(auto) res = cuda::std::reduce(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), 42);
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), int>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    CHECK(res == 42);
  }

  { // same type
    decltype(auto) res = cuda::std::reduce(policy, data.begin(), data.end(), 42);
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), int>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)

    constexpr int expected = size * (size + 1) / 2 + 42;
    CHECK(res == expected);
  }

  { // convertible type
    decltype(auto) res = cuda::std::reduce(policy, data.begin(), data.end(), static_cast<unsigned long>(42));
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), unsigned long>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)

    constexpr int expected = size * (size + 1) / 2 + 42;
    CHECK(res == expected);
  }
}

C2H_TEST("cuda::std::reduce(Iter, Iter, Tp)", "[parallel algorithm]")
{
  thrust::device_vector<int> data(size);
  thrust::sequence(data.begin(), data.end(), 1);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_reduce_init(policy, data);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_reduce_init(policy, data);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_reduce_init(policy, data);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream).with_memory_resource(device_resource);
    test_reduce_init(policy, data);
  }
}

struct plus_two
{
  __host__ __device__ constexpr int operator()(const int lhs, const int rhs) const noexcept
  {
    return lhs + rhs + 2;
  };
};

template <class Policy>
void test_reduce_init_fn(const Policy& policy, const thrust::device_vector<int>& data)
{
  { // empty should not access anything
    decltype(auto) res =
      cuda::std::reduce(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), 42, plus_two{});
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), int>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    CHECK(res == 42);
  }

  { // same type
    decltype(auto) res = cuda::std::reduce(policy, data.begin(), data.end(), 42, plus_two{});
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), int>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)

    constexpr int expected = size * (size + 1) / 2 + 42 + size * 2;
    CHECK(res == expected);
  }

  { // convertible type
    decltype(auto) res =
      cuda::std::reduce(policy, data.begin(), data.end(), static_cast<unsigned long>(42), plus_two{});
#if !TEST_CUDA_COMPILER(NVCC, <, 12, 5)
    static_assert(cuda::std::is_same_v<decltype(res), unsigned long>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 5)

    constexpr int expected = size * (size + 1) / 2 + 42 + size * 2;
    CHECK(res == expected);
  }
}

C2H_TEST("cuda::std::reduce(Iter, Iter, Tp, Fn)", "[parallel algorithm]")
{
  thrust::device_vector<int> data(size);
  thrust::sequence(data.begin(), data.end(), 1);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_reduce_init_fn(policy, data);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_reduce_init_fn(policy, data);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_reduce_init_fn(policy, data);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream).with_memory_resource(device_resource);
    test_reduce_init_fn(policy, data);
  }
}
