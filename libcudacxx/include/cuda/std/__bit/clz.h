//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX__BIT_CLZ_H
#define _LIBCUDACXX__BIT_CLZ_H

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
// optimized version of runtime clz that is used in device code.
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __constexpr_clz_32bit(uint32_t __x) noexcept
{
  constexpr auto __digits = numeric_limits<uint32_t>::digits;
  if (__x == 0)
  {
    return __digits;
  }
#if defined(_CCCL_BUILTIN_CLZ)
  return _CCCL_BUILTIN_CLZ(__x);
#else
  unsigned __res = 0;
  for (unsigned __i = __digits / 2; __i >= 1; __i /= 2)
  {
    const auto __mark = (~uint32_t{0} >> (__digits - __i)) << __i;
    if (__x & __mark)
    {
      __x >>= __i;
      __res |= __i;
    }
  }
  return __digits - 1 - __res;
#endif // defined(_CCCL_BUILTIN_CLZ)
}

template <typename _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __constexpr_clz(_Tp __x) noexcept
{
  static_assert(is_same_v<_Tp, uint32_t> || is_same_v<_Tp, uint64_t>);
  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return __constexpr_clz_32bit(__x);
  }
  else
  {
#if defined(_CCCL_BUILTIN_CLZLL)
    return __x == 0 ? numeric_limits<_Tp>::digits : _CCCL_BUILTIN_CLZLL(__x);
#else
    auto __higher = __constexpr_clz_32bit(static_cast<uint32_t>(__x >> 32));
    return __higher != numeric_limits<uint32_t>::digits
           ? __higher
           : numeric_limits<uint32_t>::digits + __constexpr_clz_32bit(static_cast<uint32_t>(__x));
#endif // defined(_CCCL_BUILTIN_CLZLL)
  }
}

#if !_CCCL_COMPILER(NVRTC)

template <typename _Tp>
_CCCL_HIDE_FROM_ABI int __host_runtime_clz(_Tp __x) noexcept
{
#  if _CCCL_COMPILER(MSVC)
  constexpr auto __digits = numeric_limits<_Tp>::digits;
  unsigned long __where;
  auto __res = sizeof(_Tp) == sizeof(uint32_t)
               ? _BitScanReverse(&__where, static_cast<uint32_t>(__x))
               : _BitScanReverse64(&__where, static_cast<uint64_t>(__x));
  return __res ? __digits - 1 - static_cast<int>(__where) : __digits;
#  else
  return __constexpr_clz(__x);
#  endif // _CCCL_COMPILER(MSVC)
}

#endif // !_CCCL_COMPILER(NVRTC)

template <typename _Tp>
_LIBCUDACXX_HIDE_FROM_ABI int __runtime_clz(_Tp __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (return sizeof(_Tp) == sizeof(uint32_t) ? __clz(static_cast<uint32_t>(__x)) //
                                                            : __clzll(static_cast<uint64_t>(__x));),
                    (return _CUDA_VSTD::__host_runtime_clz(__x);))
}

// __cccl_clz returns numeric_limits<_Tp>::digits if __x == 0 on both host and device
template <typename _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int __cccl_clz(_Tp __x) noexcept
{
  static_assert(is_same_v<_Tp, uint32_t> || is_same_v<_Tp, uint64_t>);
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  return is_constant_evaluated() ? _CUDA_VSTD::__constexpr_clz(__x) : _CUDA_VSTD::__runtime_clz(__x);
#else
  return _CUDA_VSTD::__constexpr_clz(__x);
#endif
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX__BIT_CLZ_H
