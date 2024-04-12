//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr void swap(unexpected& other) noexcept(is_nothrow_swappable_v<E>);
//
// Mandates: is_swappable_v<E> is true.
//
// Effects: Equivalent to: using cuda::std::swap; swap(unex, other.unex);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/utility>

#include "test_macros.h"

// test noexcept
struct NoexceptSwap
{
  __host__ __device__ friend void swap(NoexceptSwap&, NoexceptSwap&) noexcept;
};

struct MayThrowSwap
{
  __host__ __device__ friend void swap(MayThrowSwap&, MayThrowSwap&);
};

template <class T, class = void>
constexpr bool MemberSwapNoexcept = false;

template <class T>
constexpr bool
  MemberSwapNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<T&>().swap(cuda::std::declval<T&>()))>> =
    noexcept(cuda::std::declval<T&>().swap(cuda::std::declval<T&>()));

static_assert(MemberSwapNoexcept<cuda::std::unexpected<NoexceptSwap>>, "");
#ifndef TEST_COMPILER_ICC
static_assert(!MemberSwapNoexcept<cuda::std::unexpected<MayThrowSwap>>, "");
#endif // TEST_COMPILER_ICC

struct ADLSwap
{
  __host__ __device__ constexpr ADLSwap(int ii)
      : i(ii)
  {}
  ADLSwap& operator=(const ADLSwap&) = delete;
  int i;
  __host__ __device__ constexpr friend void swap(ADLSwap& x, ADLSwap& y)
  {
    cuda::std::swap(x.i, y.i);
  }
};

__host__ __device__ constexpr bool test()
{
  // using cuda::std::swap;
  {
    cuda::std::unexpected<int> unex1(5);
    cuda::std::unexpected<int> unex2(6);
    unex1.swap(unex2);
    assert(unex1.error() == 6);
    assert(unex2.error() == 5);
  }

  // adl swap
  {
    cuda::std::unexpected<ADLSwap> unex1(5);
    cuda::std::unexpected<ADLSwap> unex2(6);
    unex1.swap(unex2);
    assert(unex1.error().i == 6);
    assert(unex2.error().i == 5);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
