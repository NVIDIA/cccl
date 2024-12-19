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
//   explicit join_view(R&&) -> join_view<views::all_t<R>>;

#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

struct Child
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct View : cuda::std::ranges::view_base
{
  __host__ __device__ Child* begin() const;
  __host__ __device__ Child* end() const;
};

struct Range
{
  __host__ __device__ Child* begin() const;
  __host__ __device__ Child* end() const;
};

struct BorrowedRange
{
  __host__ __device__ Child* begin() const;
  __host__ __device__ Child* end() const;
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BorrowedRange> = true;

struct NestedChildren : cuda::std::ranges::view_base
{
  __host__ __device__ View* begin() const;
  __host__ __device__ View* end() const;
};

// GCC really does not like local type defs...
using result_join_view                 = cuda::std::ranges::join_view<View>;
using result_join_view_ref             = cuda::std::ranges::join_view<cuda::std::ranges::ref_view<Range>>;
using result_join_view_ref_borrowed    = cuda::std::ranges::join_view<cuda::std::ranges::ref_view<BorrowedRange>>;
using result_join_view_owning          = cuda::std::ranges::join_view<cuda::std::ranges::owning_view<Range>>;
using result_join_view_owning_borrowed = cuda::std::ranges::join_view<cuda::std::ranges::owning_view<BorrowedRange>>;

__host__ __device__ void testCTAD()
{
  View v;
  Range r;
  BorrowedRange br;

  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::join_view(v)), result_join_view>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::join_view(cuda::std::move(v))), result_join_view>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::join_view(r)), result_join_view_ref>);
  static_assert(
    cuda::std::same_as<decltype(cuda::std::ranges::join_view(cuda::std::move(r))), result_join_view_owning>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::join_view(br)), result_join_view_ref_borrowed>);
  static_assert(
    cuda::std::same_as<decltype(cuda::std::ranges::join_view(cuda::std::move(br))), result_join_view_owning_borrowed>);
  unused(v);
  unused(r);
  unused(br);

  NestedChildren n;
  cuda::std::ranges::join_view jv(n);

  // CTAD generated from the copy constructor instead of joining the join_view
  auto view = cuda::std::ranges::join_view(jv);
  static_assert(cuda::std::same_as<decltype(view), decltype(jv)>);
  unused(view);

  // CTAD generated from the move constructor instead of joining the join_view
  auto view2 = cuda::std::ranges::join_view(cuda::std::move(jv));
  static_assert(cuda::std::same_as<decltype(view2), decltype(jv)>);
  unused(view2);
}

int main(int, char**)
{
  return 0;
}
