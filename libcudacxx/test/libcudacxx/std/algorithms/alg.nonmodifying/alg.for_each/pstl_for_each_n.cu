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
#include <cuda/std/__pstl/for_each_n.h>
#include <cuda/std/execution>
#include <cuda/std/functional>

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

C2H_TEST("cuda::std::for_each_n", "[parallel algorithm]")
{
  thrust::device_vector<bool> res(size, false);
  mark_present_for_each fn{thrust::raw_pointer_cast(res.data())};

  const auto policy = cuda::execution::__cub_par_unseq;
  cuda::std::for_each_n(policy, cuda::counting_iterator{0}, size, fn);
  CHECK(thrust::all_of(res.begin(), res.end(), cuda::std::identity{}));
}
