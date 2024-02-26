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
//
// template<class T, size_t N>
//   inline constexpr bool enable_borrowed_range<elements_view<T, N>> =
//     enable_borrowed_range<T>;

#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "test_macros.h"

struct NonBorrowed : cuda::std::ranges::view_base
{
  __host__ __device__ cuda::std::tuple<int>* begin()
  {
    return nullptr;
  }
  __host__ __device__ cuda::std::tuple<int>* end()
  {
    return nullptr;
  }
};

struct Borrowed : cuda::std::ranges::view_base
{
  __host__ __device__ cuda::std::tuple<int>* begin()
  {
    return nullptr;
  }
  __host__ __device__ cuda::std::tuple<int>* end()
  {
    return nullptr;
  }
};

template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<Borrowed> = true;

static_assert(!cuda::std::ranges::borrowed_range<cuda::std::ranges::elements_view<NonBorrowed, 0>>);

#if !defined(TEST_COMPILER_NVRTC) // NVRTC somehow fails this?
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::elements_view<Borrowed, 0>>);
#endif // !TEST_COMPILER_NVRTC

int main(int, char**)
{
  return 0;
}
