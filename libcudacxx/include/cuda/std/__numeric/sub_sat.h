//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___NUMERIC_SUB_SAT_H
#define _CUDA_STD___NUMERIC_SUB_SAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__numeric/sub_overflow.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/cstdint>

#include <nv/target>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_BUILTIN(__builtin_elementwise_sub_sat)
#  define _CCCL_BUILTIN_ELEMENTWISE_SUB_SAT(...) __builtin_elementwise_sub_sat(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__builtin_elementwise_sub_sat)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __sub_sat_impl_generic(_Tp __x, _Tp __y) noexcept
{
  if constexpr (is_signed_v<_Tp>)
  {
    const auto [__result, __overflow] = ::cuda::sub_overflow(__x, __y);
    if (__overflow)
    {
      return (__y > _Tp{0}) ? numeric_limits<_Tp>::min() : numeric_limits<_Tp>::max();
    }
    return __result;
  }
  else
  {
    const auto __result = static_cast<_Tp>(__x - __y);
    if (__result > __x)
    {
      return numeric_limits<_Tp>::min();
    }
    return __result;
  }
}

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp __sub_sat_impl_host(_Tp __x, _Tp __y) noexcept
{
#  if defined(_CCCL_BUILTIN_ELEMENTWISE_SUB_SAT)
  return _CCCL_BUILTIN_ELEMENTWISE_SUB_SAT(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_ELEMENTWISE_SUB_SAT ^^^ / vvv !_CCCL_BUILTIN_ELEMENTWISE_SUB_SAT vvv
  if constexpr (is_signed_v<_Tp>)
  {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(int8_t))
    {
      return ::_sat_sub_i8(__x, __y);
    }
    else if constexpr (sizeof(_Tp) == sizeof(int16_t))
    {
      return ::_sat_sub_i16(__x, __y);
    }
    else if constexpr (sizeof(_Tp) == sizeof(int32_t))
    {
      return ::_sat_sub_i32(__x, __y);
    }
    else if constexpr (sizeof(_Tp) == sizeof(int64_t))
    {
      return ::_sat_sub_i64(__x, __y);
    }
    else
#    endif // _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
    {
      return ::cuda::std::__sub_sat_impl_generic(__x, __y);
    }
  }
  else
  {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(uint8_t))
    {
      return ::_sat_sub_u8(__x, __y);
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
    {
      return ::_sat_sub_u16(__x, __y);
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
    {
      return ::_sat_sub_u32(__x, __y);
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
    {
      return ::_sat_sub_u64(__x, __y);
    }
    else
#    endif // _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
    {
      return ::cuda::std::__sub_sat_impl_generic(__x, __y);
    }
  }
#  endif // ^^^ !_CCCL_BUILTIN_ELEMENTWISE_SUB_SAT ^^^
}
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()
template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __sub_sat_impl_device(_Tp __x, _Tp __y) noexcept
{
  if constexpr (is_signed_v<_Tp>)
  {
    if constexpr (sizeof(_Tp) < sizeof(int32_t))
    {
      constexpr auto __min = int32_t{numeric_limits<_Tp>::min()};
      constexpr auto __max = int32_t{numeric_limits<_Tp>::max()};
      return static_cast<_Tp>(::cuda::std::clamp(int32_t{__x} - int32_t{__y}, __min, __max));
    }
    // Disabled due to nvbug 5033045
    // else if constexpr (sizeof(_Tp) == sizeof(int32_t))
    // {
    //   int32_t __result;
    //   asm volatile("sub.sat.s32 %0, %1, %2;" : "=r"(__result) : "r"(__x), "r"(__y));
    //   return __result;
    // }
    return ::cuda::std::__sub_sat_impl_generic(__x, __y);
  }
  else
  {
    return static_cast<_Tp>(::cuda::std::max<_Tp>(__x, __y) - __y);
  }
}
#endif // _CCCL_CUDA_COMPILATION()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp sub_sat(_Tp __x, _Tp __y) noexcept
{
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (return ::cuda::std::__sub_sat_impl_host(__x, __y);),
                      (return ::cuda::std::__sub_sat_impl_device(__x, __y);))
  }
  return ::cuda::std::__sub_sat_impl_generic(__x, __y);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NUMERIC_SUB_SAT_H
