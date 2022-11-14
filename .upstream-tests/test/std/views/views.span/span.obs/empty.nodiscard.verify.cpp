//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: nvrtc

// <span>

// [[nodiscard]] constexpr bool empty() const noexcept;

#include <cuda/std/span>

void test() {
  cuda::std::span<int> s1;
  s1.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  int arr[] = {1, 2, 3};
  cuda::std::span<int, 3> s2{arr};
  s2.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

int main(int, char**)
{
  return 0;
}
