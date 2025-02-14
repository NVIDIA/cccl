//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cccl/builtin.h>

int main(int, char**)
{
#if defined(_CCCL_BUILTIN_POPCOUNT)
  static_assert(_CCCL_BUILTIN_POPCOUNT(0b10101010) == 4);
#endif
#if defined(_CCCL_BUILTIN_POPCOUNTLL)
  static_assert(_CCCL_BUILTIN_POPCOUNTLL(0b10101010) == 4);
#endif
#if defined(_CCCL_BUILTIN_CLZ)
  static_assert(_CCCL_BUILTIN_CLZ(0b10101010) == 24);
#endif
#if defined(_CCCL_BUILTIN_CLZLL)
  static_assert(_CCCL_BUILTIN_CLZLL(0b10101010) == 56);
#endif
#if defined(_CCCL_BUILTIN_CTZ)
  static_assert(_CCCL_BUILTIN_CTZ(0b10101010) == 1);
#endif
#if defined(_CCCL_BUILTIN_CTZLL)
  static_assert(_CCCL_BUILTIN_CTZLL(0b10101010) == 1);
#endif
  return 0;
}
