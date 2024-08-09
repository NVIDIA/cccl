//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<class R, class F>
//   transform_view(R&&, F) -> transform_view<views::all_t<R>, F>;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

struct PlusOne
{
  __host__ __device__ int operator()(int x) const;
};

struct View : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct Range
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct BorrowedRange
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BorrowedRange> = true;

// gcc falls over it feets trying to evaluate this otherwise
using result_rvlaue_range = cuda::std::ranges::transform_view<cuda::std::ranges::owning_view<Range>, PlusOne>;
using result_rvlaue_borrowed_range =
  cuda::std::ranges::transform_view<cuda::std::ranges::owning_view<BorrowedRange>, PlusOne>;

__host__ __device__ void testCTAD()
{
  View v;
  Range r;
  BorrowedRange br;
  PlusOne f;

  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::transform_view(v, f)),
                                   cuda::std::ranges::transform_view<View, PlusOne>>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::transform_view(cuda::std::move(v), f)),
                                   cuda::std::ranges::transform_view<View, PlusOne>>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::transform_view(r, f)),
                                   cuda::std::ranges::transform_view<cuda::std::ranges::ref_view<Range>, PlusOne>>);
  static_assert(
    cuda::std::same_as<decltype(cuda::std::ranges::transform_view(cuda::std::move(r), f)), result_rvlaue_range>);
  static_assert(
    cuda::std::same_as<decltype(cuda::std::ranges::transform_view(br, f)),
                       cuda::std::ranges::transform_view<cuda::std::ranges::ref_view<BorrowedRange>, PlusOne>>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::transform_view(cuda::std::move(br), f)),
                                   result_rvlaue_borrowed_range>);

  unused(v);
  unused(r);
  unused(br);
  unused(f);
}

int main(int, char**)
{
  return 0;
}
