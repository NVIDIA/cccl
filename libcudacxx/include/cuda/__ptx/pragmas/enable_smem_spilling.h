//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___PTX_ENABLE_SMEM_SPILLING_H
#define _CUDA___PTX_ENABLE_SMEM_SPILLING_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_PTX

#if __cccl_ptx_isa >= 900

extern "C" _CCCL_DEVICE void __cuda_ptx_pragma_enable_smem_spilling_is_not_supported_before_SM_75__();
_CCCL_DEVICE static _CCCL_FORCEINLINE void enable_smem_spilling() noexcept
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 750
  asm volatile(".pragma \"enable_smem_spilling\";");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  ::cuda::ptx::__cuda_ptx_pragma_enable_smem_spilling_is_not_supported_before_SM_75__();
#  endif
}

#endif // __cccl_ptx_isa >= 900

_CCCL_END_NAMESPACE_CUDA_PTX

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___PTX_ENABLE_SMEM_SPILLING_H
