//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// template <class Rep2>
//   explicit duration(const Rep2& r);

// treat_as_floating_point<Rep2>::value shall be false

#include <cuda/std/chrono>

int main(int, char**)
{
  cuda::std::chrono::duration<int> d(1.);

  return 0;
}
