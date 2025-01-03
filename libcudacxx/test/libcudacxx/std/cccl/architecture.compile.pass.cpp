//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cccl/architecture.h>
#include <cuda/std/__cccl/compiler.h>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#elif _CCCL_COMPILER(GCC) || _CCCL_COMPILER(CLANG)
#  include <cpuid.h>
#endif

#if _CCCL_ARCH(ARM64)
#  include <arm_acle.h>
#endif

int main(int, char**)
{
#if _CCCL_ARCH(64BIT)
  static_assert(sizeof(void*) == 8, "");
#endif
#if _CCCL_ARCH(32BIT)
  static_assert(sizeof(void*) == 4, "");
#endif
#if _CCCL_ARCH(X86)
#  if _CCCL_COMPILER(MSVC)
  static_cast<void>(__cpuid);
#  elif _CCCL_COMPILER(GCC) || _CCCL_COMPILER(CLANG)
  static_cast<void>(__get_cpuid);
#  endif
#endif
#if _CCCL_ARCH(ARM64)
  static_cast<void>(__clz);
#endif
  return 0;
}
