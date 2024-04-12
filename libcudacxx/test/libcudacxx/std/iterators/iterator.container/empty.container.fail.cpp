// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>
// template <class C> constexpr auto empty(const C& c) -> decltype(c.empty());

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// UNSUPPORTED: clang-3.3, clang-3.4, clang-3.5, clang-3.6, clang-3.7, clang-3.8
// nvrtc will not generate warnings/failures on nodiscard attribute
// UNSUPPORTED: nvrtc

#include <cuda/std/array>
#include <cuda/std/iterator>

#include "test_macros.h"

int main(int, char**)
{
  cuda::std::array<int, 4> c;
  cuda::std::empty(c); // expected-error {{ignoring return value of function declared with 'nodiscard' attribute}}

  return 0;
}
