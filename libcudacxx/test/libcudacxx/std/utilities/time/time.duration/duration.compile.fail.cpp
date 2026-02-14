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

// If a program instantiates duration with a duration type for the template
// argument Rep a diagnostic is required.

#include <cuda/std/chrono>

int main(int, char**)
{
  using D = cuda::std::chrono::duration<cuda::std::chrono::milliseconds>;
  D d;

  return 0;
}
