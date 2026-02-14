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

#ifndef _CUDA___MEMCPY_ASYNC_CP_ASYNC_SHARED_GLOBAL_H_
#define _CUDA___MEMCPY_ASYNC_CP_ASYNC_SHARED_GLOBAL_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()

#  include <cuda/__ptx/ptx_dot_variants.h>
#  include <cuda/__ptx/ptx_helper_functions.h>
#  include <cuda/std/cstdint>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_shared_global_is_not_supported_before_SM_80__();

#  if _CCCL_CUDA_COMPILER(NVCC, <, 12, 1) // WAR for compiler state space issues
template <size_t _Copy_size>
inline _CCCL_DEVICE void __cp_async_shared_global(char* __dest, const char* __src)
{
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async

  // If `if constexpr` is not available, this function gets instantiated even
  // if is not called. Do not static_assert in that case.
  static_assert(_Copy_size == 4 || _Copy_size == 8 || _Copy_size == 16,
                "cp.async.shared.global requires a copy size of 4, 8, or 16.");

  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_80,
    (asm volatile(R"XYZ(
      {
        .reg .b64 tmp;
        .reg .b32 dst;

        cvta.to.shared.u64 tmp, %0;
        cvt.u32.u64 dst, tmp;
        cvta.to.global.u64 tmp, %1;
        cp.async.ca.shared.global [dst], [tmp], %2, %2;
      }
      )XYZ" : : "l"(__dest),
                  "l"(__src),
                  "n"(_Copy_size) : "memory");),
    (::cuda::__cuda_ptx_cp_async_shared_global_is_not_supported_before_SM_80__();));
}
template <>
inline _CCCL_DEVICE void __cp_async_shared_global<16>(char* __dest, const char* __src)
{
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
  // When copying 16 bytes, it is possible to skip L1 cache (.cg).
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
                    (asm volatile(R"XYZ(
      {
        .reg .u64 tmp;
        .reg .u32 dst;

        cvta.to.shared.u64 tmp, %0;
        cvt.u32.u64 dst, tmp;
        cvta.to.global.u64 tmp, %1;
        cp.async.cg.shared.global [dst], [tmp], 16, 16;
      }
      )XYZ" : : "l"(__dest),
                                  "l"(__src) : "memory");),
                    (::cuda::__cuda_ptx_cp_async_shared_global_is_not_supported_before_SM_80__();));
}
#  else // ^^^^ NVCC 12.0 / !NVCC 12.0 vvvvv WAR for compiler state space issues
template <size_t _Copy_size>
inline _CCCL_DEVICE void __cp_async_shared_global(char* __dest, const char* __src)
{
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async

  // If `if constexpr` is not available, this function gets instantiated even
  // if is not called. Do not static_assert in that case.
  static_assert(_Copy_size == 4 || _Copy_size == 8 || _Copy_size == 16,
                "cp.async.shared.global requires a copy size of 4, 8, or 16.");

  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_80,
    (asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %2;" : : "r"(
                    static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(__dest))),
                  "l"(static_cast<::cuda::std::uint64_t>(::__cvta_generic_to_global(__src))),
                  "n"(_Copy_size) : "memory");),
    (::cuda::__cuda_ptx_cp_async_shared_global_is_not_supported_before_SM_80__();));
}
template <>
inline _CCCL_DEVICE void __cp_async_shared_global<16>(char* __dest, const char* __src)
{
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
  // When copying 16 bytes, it is possible to skip L1 cache (.cg).
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_80,
    (asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %2;" : : "r"(
                    static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(__dest))),
                  "l"(static_cast<::cuda::std::uint64_t>(::__cvta_generic_to_global(__src))),
                  "n"(16) : "memory");),
    (::cuda::__cuda_ptx_cp_async_shared_global_is_not_supported_before_SM_80__();));
}
#  endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 1)

template <size_t _Alignment, typename _Group>
inline _CCCL_DEVICE void
__cp_async_shared_global_mechanism(_Group __g, char* __dest, const char* __src, ::cuda::std::size_t __size)
{
  // If `if constexpr` is not available, this function gets instantiated even
  // if is not called. Do not static_assert in that case.
  static_assert(4 <= _Alignment, "cp.async requires at least 4-byte alignment");

  // Maximal copy size is 16.
  constexpr int __copy_size = (_Alignment > 16) ? 16 : _Alignment;
  // We use an int offset here, because we are copying to shared memory,
  // which is easily addressable using int.
  const int __group_size = __g.size();
  const int __group_rank = __g.thread_rank();
  const int __stride     = __group_size * __copy_size;
  for (int __offset = __group_rank * __copy_size; __offset < static_cast<int>(__size); __offset += __stride)
  {
    ::cuda::__cp_async_shared_global<__copy_size>(__dest + __offset, __src + __offset);
  }
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION()

#endif // _CUDA___MEMCPY_ASYNC_CP_ASYNC_SHARED_GLOBAL_H_
