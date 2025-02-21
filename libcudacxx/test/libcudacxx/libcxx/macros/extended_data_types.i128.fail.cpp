//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/std/__cccl/extended_data_types.h>

#include "test_macros.h"

int main(int, char**)
{
#if !_CCCL_HAS_INT128()
  __int128 x    = __int128(123456789123) + __int128(123456789123);
  __uint128_t y = __uint128_t(123456789123) + __uint128_t(123456789123);
  unused(x);
  unused(y);
#else
  static_assert(false);
#endif
  return 0;
}
