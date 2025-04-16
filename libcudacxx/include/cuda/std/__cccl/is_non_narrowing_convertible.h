//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_IS_NON_NARROWING_CONVERTIBLE_H
#define __CCCL_IS_NON_NARROWING_CONVERTIBLE_H

#include <cuda/std/__cccl/compiler.h>

//! There is compiler bug that results in incorrect results for the below `__is_non_narrowing_convertible` check.
//! This breaks some common functionality, so this *must* be included outside of a system header. See nvbug4867473.
#if defined(_CCCL_FORCE_SYSTEM_HEADER_GCC) || defined(_CCCL_FORCE_SYSTEM_HEADER_CLANG) \
  || defined(_CCCL_FORCE_SYSTEM_HEADER_MSVC)
#  error \
    "This header must be included only within the <cuda/std/__cccl/system_header>. This most likely means a mix and match of different versions of CCCL."
#endif // system header detected

namespace __cccl_internal
{

#if _CCCL_HAS_CUDA_COMPILER() && (defined(__CUDACC__) || defined(_NVHPC_CUDA) || _CCCL_COMPILER(NVRTC))
template <class _Tp>
__host__ __device__ _Tp&& __cccl_declval(int);
template <class _Tp>
__host__ __device__ _Tp __cccl_declval(long);
template <class _Tp>
__host__ __device__ decltype(__cccl_internal::__cccl_declval<_Tp>(0)) __cccl_declval() noexcept;

// This requires a type to be implicitly convertible (also non-arithmetic)
template <class _Tp>
__host__ __device__ void __cccl_accepts_implicit_conversion(_Tp) noexcept;
#else // ^^^ CUDA compilation ^^^ / vvv no CUDA compilation
template <class _Tp>
_Tp&& __cccl_declval(int);
template <class _Tp>
_Tp __cccl_declval(long);
template <class _Tp>
decltype(__cccl_internal::__cccl_declval<_Tp>(0)) __cccl_declval() noexcept;

// This requires a type to be implicitly convertible (also non-arithmetic)
template <class _Tp>
void __cccl_accepts_implicit_conversion(_Tp) noexcept;
#endif // no CUDA compilation

template <class...>
using __cccl_void_t = void;

template <class _Dest, class _Source, class = void>
inline constexpr bool __cccl_is_non_narrowing_v = false;

template <class _Dest, class _Source>
inline constexpr bool
  __cccl_is_non_narrowing_v<_Dest, _Source, __cccl_void_t<decltype(_Dest{__cccl_internal::__cccl_declval<_Source>()})>> =
    true;

template <class _Dest, class _Source, class = void>
inline constexpr bool __cccl_accepts_conversion_v = false;

template <class _Dest, class _Source>
inline constexpr bool __cccl_accepts_conversion_v<
  _Dest,
  _Source,
  __cccl_void_t<decltype(__cccl_internal::__cccl_accepts_implicit_conversion<_Dest>(
    __cccl_internal::__cccl_declval<_Source>()))>> = true;

template <class _Dest, class _Source>
inline constexpr bool __is_non_narrowing_convertible_v =
  __cccl_is_non_narrowing_v<_Dest, _Source> && __cccl_accepts_conversion_v<_Dest, _Source>;

} // namespace __cccl_internal

#endif // __CCCL_IS_NON_NARROWING_CONVERTIBLE_H
