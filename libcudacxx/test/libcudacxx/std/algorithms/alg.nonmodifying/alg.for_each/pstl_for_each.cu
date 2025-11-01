//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class ExecutionPolicy, class ForwardIterator, class Function>
//   void for_each(ExecutionPolicy&& exec,
//                 ForwardIterator first, ForwardIterator last,
//                 Function f);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <cuda/iterator>
#include <cuda/std/__pstl/for_each.h>
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

C2H_TEST("cuda::std::for_each", "[parallel algorithm]")
{
  SECTION("with default stream")
  {
    thrust::device_vector<bool> res(size, false);
    mark_present_for_each fn{thrust::raw_pointer_cast(res.data())};

    cuda::std::for_each(cuda::std::execution::par_unseq, cuda::counting_iterator{0}, cuda::counting_iterator{size}, fn);
    CHECK(thrust::all_of(res.begin(), res.end(), cuda::std::identity{}));
  }

  SECTION("with unique stream")
  {
    ::cuda::stream stream{::cuda::device_ref{0}};
    thrust::device_vector<bool> res(size, false);
    mark_present_for_each fn{thrust::raw_pointer_cast(res.data())};

    cuda::std::for_each(
      cuda::std::execution::par_unseq.set_stream(stream), cuda::counting_iterator{0}, cuda::counting_iterator{size}, fn);
    CHECK(thrust::all_of(res.begin(), res.end(), cuda::std::identity{}));
  }
}
