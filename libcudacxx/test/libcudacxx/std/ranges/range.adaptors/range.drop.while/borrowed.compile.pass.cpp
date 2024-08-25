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
//
// template<class T, class Pred>
//   inline constexpr bool enable_borrowed_range<drop_while_view<T, Pred>> =
//     enable_borrowed_range<T>;

#include <cuda/std/ranges>

struct NonBorrowed : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin();
  __host__ __device__ int* end();
};

struct Borrowed : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin();
  __host__ __device__ int* end();
};

struct Pred
{
  __host__ __device__ bool operator()(int) const;
};

template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<Borrowed> = true;

static_assert(!cuda::std::ranges::borrowed_range<cuda::std::ranges::drop_while_view<NonBorrowed, Pred>>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::drop_while_view<Borrowed, Pred>>);

int main(int, char**)
{
  return 0;
}
