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
  cuda::std::default_accessor<int> acc{};
  cuda::std::aligned_accessor<int, sizeof(int)> acc_aligned = acc;
  return 0;
}
