//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, class ForwardIterator, class SizeType, class Function>
//   void for_each_n(ExecutionPolicy&& exec, ForwardIterator first, SizeType count, Function f);

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

struct mark_present_for_each
{
  bool* ptr_;

  template <typename T>
  __host__ __device__ void operator()(T val) const noexcept
  {
    ptr_[val] = true;
  }
};

template <class Policy>
void test_for_each_n(const Policy& policy, thrust::device_vector<bool>& res)
{
  mark_present_for_each fn{thrust::raw_pointer_cast(res.data())};

  { // empty should not access anything
    const auto result = cuda::std::for_each_n(policy, static_cast<int*>(nullptr), 0, fn);
    CHECK(result == nullptr);
  }

  { // same type
    thrust::fill(res.begin(), res.end(), false);
    const auto result = cuda::std::for_each_n(policy, cuda::counting_iterator{0}, size, fn);
    CHECK(thrust::all_of(res.begin(), res.end(), cuda::std::identity{}));
    CHECK(result == cuda::counting_iterator{size});
  }

  { // convertible type
    thrust::fill(res.begin(), res.end(), false);
    const auto result = cuda::std::for_each_n(policy, cuda::counting_iterator<short>{0}, size, fn);
    CHECK(thrust::all_of(res.begin(), res.end(), cuda::std::identity{}));
    CHECK(result == cuda::counting_iterator<short>{size});
  }
}

C2H_TEST("cuda::std::for_each_n", "[parallel algorithm]")
{
  thrust::device_vector<bool> res(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_for_each_n(policy, res);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_for_each_n(policy, res);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_for_each_n(policy, res);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_for_each_n(policy, res);
  }
}
