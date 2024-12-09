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

// constexpr auto size() requires sized_range<V>
// constexpr auto size() const requires sized_range<const V>

#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

#if TEST_STD_VER >= 2020
template <class T>
concept SizeInvocable = requires(T t) { t.size(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool SizeInvocable = false;

template <class T>
inline constexpr bool SizeInvocable<T, cuda::std::void_t<decltype(cuda::std::declval<T>().size())>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::transform_view transformView(MoveOnlyView{}, PlusOne{});
    assert(transformView.size() == 8);
  }

  {
    const cuda::std::ranges::transform_view transformView(MoveOnlyView{globalBuff, 4}, PlusOne{});
    assert(transformView.size() == 4);
  }

  static_assert(!SizeInvocable<cuda::std::ranges::transform_view<ForwardView, PlusOne>>);

  static_assert(SizeInvocable<cuda::std::ranges::transform_view<SizedSentinelNotConstView, PlusOne>>);
  static_assert(!SizeInvocable<const cuda::std::ranges::transform_view<SizedSentinelNotConstView, PlusOne>>);

  return true;
}

int main(int, char**)
{
  test();
#ifndef TEST_COMPILER_CUDACC_BELOW_11_3
  static_assert(test(), "");
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3

  return 0;
}
