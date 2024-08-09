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

// template<class R>
//   take_view(R&&, range_difference_t<R>) -> take_view<views::all_t<R>>;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

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

using result_take_view                 = cuda::std::ranges::take_view<View>;
using result_take_view_ref             = cuda::std::ranges::take_view<cuda::std::ranges::ref_view<Range>>;
using result_take_view_ref_borrowed    = cuda::std::ranges::take_view<cuda::std::ranges::ref_view<BorrowedRange>>;
using result_take_view_owning          = cuda::std::ranges::take_view<cuda::std::ranges::owning_view<Range>>;
using result_take_view_owning_borrowed = cuda::std::ranges::take_view<cuda::std::ranges::owning_view<BorrowedRange>>;

__host__ __device__ void testCTAD()
{
  View v;
  Range r;
  BorrowedRange br;

  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::take_view(v, 0)), result_take_view>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::take_view(cuda::std::move(v), 0)), result_take_view>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::take_view(r, 0)), result_take_view_ref>);
  static_assert(
    cuda::std::same_as<decltype(cuda::std::ranges::take_view(cuda::std::move(r), 0)), result_take_view_owning>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::take_view(br, 0)), result_take_view_ref_borrowed>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::take_view(cuda::std::move(br), 0)),
                                   result_take_view_owning_borrowed>);

  unused(v);
  unused(r);
  unused(br);
}

int main(int, char**)
{
  return 0;
}
