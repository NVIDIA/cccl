//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// gcc-10 segfaults with any use of constant_wrapper, gcc-11 fails to evaluate:
//   typename decltype(__cw_fixed_value(_Xp))::type
// UNSUPPORTED: gcc-10 || gcc-11

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
// UNSUPPORTED: nvrtc

// REQUIRES: !c++17

// constant_wrapper

// template<class T, size_t Extent>
//   cw-fixed-value(T (&)[Extent]) -> cw-fixed-value<T[Extent]>;                   // exposition only

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

constexpr int arr[] = {1, 2, 3};
using T1            = cuda::std::__constant_wrapper<arr>;
static_assert(cuda::std::is_same_v<T1::value_type, const int[3]>);

using T2 = cuda::std::__constant_wrapper<"hello world">;
static_assert(cuda::std::is_same_v<T2::value_type, const char[12]>);

struct S
{
  int value;
};

constexpr S s_arr[] = {{1}, {2}, {3}};
using T3            = cuda::std::__constant_wrapper<s_arr>;
static_assert(cuda::std::is_same_v<T3::value_type, const S[3]>);

int main(int, char**)
{
  return 0;
}
