// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX_MBARRIER_INIT_H_
#define _CUDA_PTX_MBARRIER_INIT_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/ptx_dot_variants.h>
#include <cuda/__ptx/ptx_helper_functions.h>
#include <cuda/std/cstdint>

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

// 9.7.12.15.9. Parallel Synchronization and Communication Instructions: mbarrier.init
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init
/*
// mbarrier.init.shared.b64 [addr], count; // PTX ISA 70, SM_80
template <typename=void>
__device__ static inline void mbarrier_init(
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 700
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_init_is_not_supported_before_SM_80__();
template <typename = void>
_CCCL_DEVICE static inline void mbarrier_init(_CUDA_VSTD::uint64_t* __addr, const _CUDA_VSTD::uint32_t& __count)
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_80,
    (asm("mbarrier.init.shared.b64 [%0], %1;"
         :
         : "r"(__as_ptr_smem(__addr)), "r"(__count)
         : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_mbarrier_init_is_not_supported_before_SM_80__();));
}
#endif // __cccl_ptx_isa >= 700

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_MBARRIER_INIT_H_
