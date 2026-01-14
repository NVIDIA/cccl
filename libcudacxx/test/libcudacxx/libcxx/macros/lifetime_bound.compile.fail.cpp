//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// FORCE_ALL_WARNINGS.

#include <cuda/std/__cccl/attributes.h>

#include "test_macros.h"

#if _CCCL_HAS_CPP_ATTRIBUTE(clang::lifetimebound) || _CCCL_COMPILER(CLANG)
#elif _CCCL_HAS_CPP_ATTRIBUTE(msvc::lifetimebound) || _CCCL_COMPILER(MSVC, >=, 19, 37)
#else
#  error "lifetimebound attribute not supported"
#endif

struct S
{
  char data[32];

  __host__ __device__ const char* get() const _CCCL_LIFETIMEBOUND
  {
    return data;
  }
};

__host__ __device__ bool test()
{
  auto sv = S{"abc"}.get();
  return true;
}

int main(int, char**)
{
  assert(test());
  return 0;
}
