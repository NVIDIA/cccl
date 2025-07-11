//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/std/mdspan>

#include <test_macros.h>

int main(int, char**)
{
  using T = int;
  // conversion from smaller aligned accessor
  cuda::std::aligned_accessor<T, sizeof(T)> aligned_x1{};
  cuda::std::aligned_accessor<T, sizeof(T) * 2> aligned_x2{aligned_x1};
  unused(aligned_x2);

  // alignment to small
  cuda::std::aligned_accessor<T, sizeof(T) / 2> aligned_half{};
  unused(aligned_half);

  // alignment non-power of 2
  cuda::std::aligned_accessor<T, 6> aligned6{};
  unused(aligned_half);

  // non-convertible
  cuda::std::aligned_accessor<T, sizeof(T)> aligned_int{};
  cuda::std::aligned_accessor<float, sizeof(T)> aligned_float{aligned_int};
  unused(aligned_float);
  return 0;
}
