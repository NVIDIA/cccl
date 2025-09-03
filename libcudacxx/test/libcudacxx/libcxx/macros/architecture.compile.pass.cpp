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

#if !_CCCL_COMPILER(NVRTC)
#  if _CCCL_ARCH(X86_64)
#    if _CCCL_COMPILER(MSVC)
#      include <intrin.h>
#    elif _CCCL_COMPILER(GCC) || _CCCL_COMPILER(CLANG)
#      include <cpuid.h>
#    endif // _CCCL_COMPILER(GCC) || _CCCL_COMPILER(CLANG)
#  endif // _CCCL_ARCH(X86_64)

#  if !_CCCL_COMPILER(NVHPC) // nvbug5395777
#    if _CCCL_ARCH(ARM64) && defined(__ARM_ACLE)
#      include <arm_acle.h>
#    endif // _CCCL_ARCH(ARM64) && defined(__ARM_ACLE)
#  endif // !_CCCL_COMPILER(NVHPC)
#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  static_assert(sizeof(void*) == 8, "");
  return 0;
}
