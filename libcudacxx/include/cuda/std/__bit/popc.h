//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX__BIT_POPC_H
#define _LIBCUDACXX__BIT_POPC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// It is not possible to understand if we are in constant evaluation context in GCC < 9. For this reason, we provide an
// optimized version of runtime popc that is used in device code.
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __constexpr_popc_32bit(uint32_t __x) noexcept
{
#if defined(_CCCL_BUILTIN_POPC)
  return _CCCL_BUILTIN_POPC(__x);
#else
  __x = __x - ((__x >> 1) & 0x55555555);
  __x = (__x & 0x33333333) + ((__x >> 2) & 0x33333333);
  return (((__x + (__x >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif // _CCCL_BUILTIN_POPC
}

template <typename _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __constexpr_popc(_Tp __x) noexcept
{
  static_assert(is_same_v<_Tp, uint32_t> || is_same_v<_Tp, uint64_t>);
  if constexpr (is_same_v<_Tp, uint32_t>)
  {
    return __constexpr_popc_32bit(__x);
  }
  else
  {
#if defined(_CCCL_BUILTIN_POPCLL)
    return _CCCL_BUILTIN_POPCLL(__x);
#else
    return __constexpr_popc_32bit(static_cast<uint32_t>(__x))
         + __constexpr_popc_32bit(static_cast<uint32_t>(__x >> 32));
#endif // defined(_CCCL_BUILTIN_POPCLL)
  }
}

#if !_CCCL_COMPILER(NVRTC)

template <typename _Tp>
_CCCL_HIDE_FROM_ABI int __host_runtime_popc(_Tp __x) noexcept
{
#  if _CCCL_COMPILER(MSVC)
#    if !defined(_M_ARM64)
  auto __ret = sizeof(_Tp) == sizeof(uint32_t) //
               ? __popcnt(static_cast<uint32_t>(__x))
               : __popcnt64(static_cast<uint64_t>(__x));
#    else
  auto __ret = sizeof(_Tp) == sizeof(uint32_t) //
               ? _CountOneBits(static_cast<uint32_t>(__x))
               : _CountOneBits64(static_cast<uint64_t>(__x));
#    endif // !defined(_M_ARM64)
  return static_cast<int>(__ret);
#  else
  return __constexpr_popc(__x);
#  endif
}

#endif // !_CCCL_COMPILER(NVRTC)

template <typename _Tp>
_LIBCUDACXX_HIDE_FROM_ABI int __runtime_popc(_Tp __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (return sizeof(_Tp) == sizeof(uint32_t) ? __popc(static_cast<uint32_t>(__x)) //
                                                            : __popcll(static_cast<uint64_t>(__x));),
                    (return _CUDA_VSTD::__host_runtime_popc(__x);))
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int __cccl_popc(_Tp __x) noexcept
{
  static_assert(is_same_v<_Tp, uint32_t> || is_same_v<_Tp, uint64_t>);
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  return is_constant_evaluated() ? _CUDA_VSTD::__constexpr_popc(__x) : _CUDA_VSTD::__runtime_popc(__x);
#else
  return _CUDA_VSTD::__constexpr_popc(__x);
#endif
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX__BIT_POPC_H
