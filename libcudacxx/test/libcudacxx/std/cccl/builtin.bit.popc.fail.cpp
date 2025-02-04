//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14

#include <cuda/std/__cccl/builtin.h>

int main(int, char**)
{
#if !defined(_CCCL_BUILTIN_POPCOUNT)
  auto x = _CCCL_BUILTIN_POPCOUNT(0b10101010);
  unused(x);
#else
  static_assert(false);
#endif
  return 0;
}
