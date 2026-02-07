//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, class ForwardIterator, class Function>
//   void for_each(ExecutionPolicy&& exec,
//                 ForwardIterator first, ForwardIterator last,
//                 Function f);

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

  __host__ __device__ void operator()(int val) const noexcept
  {
    ptr_[val] = true;
  }
};

template <class Policy>
void test_for_each(const Policy& policy, thrust::device_vector<bool>& res)
{
  mark_present_for_each fn{thrust::raw_pointer_cast(res.data())};

  { // empty should not access anything
    cuda::std::for_each(policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), fn);
  }

  { // same type
    thrust::fill(res.begin(), res.end(), false);
    cuda::std::for_each(policy, cuda::counting_iterator{0}, cuda::counting_iterator{size}, fn);
    CHECK(thrust::all_of(res.begin(), res.end(), cuda::std::identity{}));
  }

  { // convertible type
    thrust::fill(res.begin(), res.end(), false);
    cuda::std::for_each(policy, cuda::counting_iterator<short>{0}, cuda::counting_iterator<short>{size}, fn);
    CHECK(thrust::all_of(res.begin(), res.end(), cuda::std::identity{}));
  }
}

C2H_TEST("cuda::std::for_each", "[parallel algorithm]")
{
  thrust::device_vector<bool> res(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_for_each(policy, res);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_for_each(policy, res);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_for_each(policy, res);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_for_each(policy, res);
  }
}
