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

// class transform_view::<sentinel>;

#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

#if TEST_STD_VER >= 2020
template <class T>
concept EndIsIter = requires(T t) { ++t.end(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool EndIsIter = false;

template <class T>
inline constexpr bool EndIsIter<T, cuda::std::void_t<decltype(++cuda::std::declval<T>().end())>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  cuda::std::ranges::transform_view<SizedSentinelView, PlusOne> transformView1{};
  // Going to const and back.
  auto sent1 = transformView1.end();
  cuda::std::ranges::sentinel_t<const cuda::std::ranges::transform_view<SizedSentinelView, PlusOne>> sent2{sent1};
  cuda::std::ranges::sentinel_t<const cuda::std::ranges::transform_view<SizedSentinelView, PlusOne>> sent3{sent2};
  unused(sent3);

  static_assert(!EndIsIter<decltype(sent1)>);
  static_assert(!EndIsIter<decltype(sent2)>);
  assert(sent1.base() == globalBuff + 8);

  cuda::std::ranges::transform_view transformView2(SizedSentinelView{4}, PlusOne());
  auto sent4 = transformView2.end();
  auto iter  = transformView1.begin();
  {
    assert(iter != sent1);
    assert(iter != sent2);
    assert(iter != sent4);
  }

  {
    assert(iter + 8 == sent1);
    assert(iter + 8 == sent2);
    assert(iter + 4 == sent4);
  }

  {
    assert(sent1 - iter == 8);
    assert(sent4 - iter == 4);
    assert(iter - sent1 == -8);
    assert(iter - sent4 == -4);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF) && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  static_assert(test(), "");
#endif // _LIBCUDACXX_ADDRESSOF && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)

  return 0;
}
