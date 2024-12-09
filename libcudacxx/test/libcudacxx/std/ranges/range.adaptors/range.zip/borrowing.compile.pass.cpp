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

// template<class... Views>
// inline constexpr bool enable_borrowed_range<zip_view<Views...>> =
//      (enable_borrowed_range<Views> && ...);

#include <cuda/std/ranges>
#include <cuda/std/tuple>

struct Borrowed : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const
  {
    return nullptr;
  }
  __host__ __device__ int* end() const
  {
    return nullptr;
  }
};

template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<Borrowed> = true;

static_assert(cuda::std::ranges::borrowed_range<Borrowed>);

struct NonBorrowed : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const
  {
    return nullptr;
  }
  __host__ __device__ int* end() const
  {
    return nullptr;
  }
};
static_assert(!cuda::std::ranges::borrowed_range<NonBorrowed>);

// test borrowed_range
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::zip_view<Borrowed>>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::zip_view<Borrowed, Borrowed>>);
static_assert(!cuda::std::ranges::borrowed_range<cuda::std::ranges::zip_view<Borrowed, NonBorrowed>>);
static_assert(!cuda::std::ranges::borrowed_range<cuda::std::ranges::zip_view<NonBorrowed>>);
static_assert(!cuda::std::ranges::borrowed_range<cuda::std::ranges::zip_view<NonBorrowed, NonBorrowed>>);

int main(int, char**)
{
  return 0;
}
