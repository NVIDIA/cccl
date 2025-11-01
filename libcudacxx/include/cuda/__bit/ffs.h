//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BIT_FFS_H
#define _CUDA___BIT_FFS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <nv/target>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

using ::cuda::std::uint32_t;
using ::cuda::std::uint64_t;

#if _CCCL_HAS_BUILTIN(__builtin_ffs) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FFS(...)   __builtin_ffs(__VA_ARGS__)
#  define _CCCL_BUILTIN_FFSLL(...) __builtin_ffsll(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__builtin_ffs) || _CCCL_COMPILER(GCC)

template <typename _Tp>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr int __ffs_impl_constexpr(_Tp __v) noexcept
{
  static_assert(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>, "_Tp must be unsigned");
  if (__v == 0)
  {
    return 0;
  }
  int __pos = 1;
  while ((__v & 1) == 0)
  {
    __v >>= 1;
    ++__pos;
  }
  return __pos;
}

#if !_CCCL_COMPILER(NVRTC)

template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API int __ffs_impl_host(_Tp __v) noexcept
{
#  if defined(_CCCL_BUILTIN_FFS)
  if constexpr (sizeof(_Tp) <= sizeof(int))
  {
    return _CCCL_BUILTIN_FFS(static_cast<int>(__v));
  }
  else
  {
    return _CCCL_BUILTIN_FFSLL(static_cast<long long>(__v));
  }
#  elif _CCCL_COMPILER(MSVC)
  unsigned long __where{};
  unsigned char __res{};
  if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
  {
    __res = ::_BitScanForward(&__where, static_cast<uint32_t>(__v));
  }
  else
  {
    __res = ::_BitScanForward64(&__where, static_cast<uint64_t>(__v));
  }
  return __res ? (static_cast<int>(__where) + 1) : 0;
#  else
  return __ffs_impl_constexpr(__v);
#  endif // _CCCL_COMPILER(MSVC)
}

#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE_API int __ffs_impl_device(_Tp __v) noexcept
{
  if constexpr (sizeof(_Tp) <= sizeof(int))
  {
    return ::__ffs(static_cast<int>(__v));
  }
  else
  {
    return ::__ffsll(static_cast<long long>(__v));
  }
}

#endif // _CCCL_CUDA_COMPILATION()

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr int ffs(_Tp __v) noexcept
{
#if _CCCL_HAS_INT128()
  if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    const auto __lo = static_cast<uint64_t>(__v);
    const auto __hi = static_cast<uint64_t>(static_cast<__uint128_t>(__v) >> 64);

    if (const auto __result = ffs(__lo))
    {
      return __result;
    }
    if (const auto __result = ffs(__hi))
    {
      return __result + 64;
    }
    return 0;
  }
  else
#endif // _CCCL_HAS_INT128()
  {
    int __result{};
    if (!::cuda::std::__cccl_default_is_constant_evaluated())
    {
      if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
      {
        const uint32_t __vu = static_cast<uint32_t>(__v);
        NV_IF_ELSE_TARGET(NV_IS_HOST, (__result = __ffs_impl_host(__vu);), (__result = __ffs_impl_device(__vu);));
      }
      else
      {
        const uint64_t __vu = static_cast<uint64_t>(__v);
        NV_IF_ELSE_TARGET(NV_IS_HOST, (__result = __ffs_impl_host(__vu);), (__result = __ffs_impl_device(__vu);));
      }
    }
    else
    {
      __result = __ffs_impl_constexpr(__v);
    }
    _CCCL_ASSUME(__result >= 0 && __result <= ::cuda::std::numeric_limits<_Tp>::digits);
    return __result;
  }
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BIT_FFS_H
