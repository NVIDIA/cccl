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

// Period::num shall be positive, diagnostic required.

#include <cuda/std/chrono>

int main(int, char**)
{
  using D = cuda::std::chrono::duration<int, cuda::std::ratio<5, -1>>;
  D d;

  return 0;
}
