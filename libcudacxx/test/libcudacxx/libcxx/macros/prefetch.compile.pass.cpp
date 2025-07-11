//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/detail/__config>

#include <test_macros.h>

int main(int, char**)
{
  int memory[8];
  _CCCL_BUILTIN_PREFETCH(memory);
  _CCCL_BUILTIN_PREFETCH(memory, /*read-only=*/0);
  _CCCL_BUILTIN_PREFETCH(memory, /*read-only=*/0, /*medium cache utilization=*/1);
  unused(memory);
  return 0;
}
