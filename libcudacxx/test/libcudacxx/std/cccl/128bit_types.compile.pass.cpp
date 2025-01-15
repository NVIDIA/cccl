//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cccl/128bit_types.h>

#include "test_macros.h"

int main(int, char**)
{
#if _CCCL_HAS_INT128()
  auto x = __int128(123456789123) + __int128(123456789123);
  auto y = __uint128_t(123456789123) + __uint128_t(123456789123);
  unused(x);
  unused(y);
#endif
#if _CCCL_HAS_FLOAT128()
  auto z = __float128(3.14) + __float128(3.14);
  unused(z);
#endif
  return 0;
}
