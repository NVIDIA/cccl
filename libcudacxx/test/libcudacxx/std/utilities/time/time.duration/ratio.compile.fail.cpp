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

// Period shall be a specialization of ratio, diagnostic required.

#include <cuda/std/chrono>

template <int N, int D = 1>
class Ratio
{
public:
  static const int num = N;
  static const int den = D;
};

int main(int, char**)
{
  using D = cuda::std::chrono::duration<int, Ratio<1>>;
  D d;

  return 0;
}
