//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// template<class R>
//   explicit join_view(R&&) -> join_view<views::all_t<R>>;

// Tests that the deduction guide is explicit.

#include <cuda/std/ranges>

#include "test_iterators.h"

template <class T>
struct Range
{
  __host__ __device__ friend T* begin(Range&)
  {
    return nullptr;
  }
  __host__ __device__ friend T* begin(Range const&)
  {
    return nullptr;
  }
  __host__ __device__ friend sentinel_wrapper<T*> end(Range&)
  {
    return sentinel_wrapper<T*>(nullptr);
  }
  __host__ __device__ friend sentinel_wrapper<T*> end(Range const&)
  {
    return sentinel_wrapper<T*>(nullptr);
  }
};

__host__ __device__ void testExplicitCTAD()
{
  Range<Range<int>> r;
  cuda::std::ranges::join_view v = r; // expected-error {{no viable constructor or deduction guide for deduction of
                                      // template arguments of 'join_view'}}
}

int main(int, char**)
{
  return 0;
}
