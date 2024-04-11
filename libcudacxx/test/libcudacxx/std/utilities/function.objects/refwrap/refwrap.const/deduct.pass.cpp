//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: gcc-6, gcc-7

// <functional>

// template <class T>
//   reference_wrapper(T&) -> reference_wrapper<T>;

// #include <cuda/std/functional>
#include <cuda/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  int i = 0;
  cuda::std::reference_wrapper ri(i);
  static_assert(cuda::std::is_same_v<decltype(ri), cuda::std::reference_wrapper<int>>);
  cuda::std::reference_wrapper ri2(ri);
  static_assert(cuda::std::is_same_v<decltype(ri2), cuda::std::reference_wrapper<int>>);
  unused(ri2);

  const int j = 0;
  cuda::std::reference_wrapper rj(j);
  static_assert(cuda::std::is_same_v<decltype(rj), cuda::std::reference_wrapper<const int>>);
  cuda::std::reference_wrapper rj2(rj);
  static_assert(cuda::std::is_same_v<decltype(rj2), cuda::std::reference_wrapper<const int>>);
  unused(rj2);

  return 0;
}
