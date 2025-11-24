//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_MUL_HI_H
#define _CUDA___CMATH_MUL_HI_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cstdint>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

/***********************************************************************************************************************
 * Extract higher bits after multiplication
 **********************************************************************************************************************/

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __mul_hi_fallback(_Tp __lhs, _Tp __rhs) noexcept
{
  static_assert(::cuda::std::is_unsigned_v<_Tp>, "__mul_hi_fallback: T is required to be a unsigned integer type");
  constexpr int __half_bits = ::cuda::std::__num_bits_v<_Tp> / 2;
  using __half_bits_t       = ::cuda::std::__make_nbit_uint_t<__half_bits>;
  const auto __lhs_low      = static_cast<__half_bits_t>(__lhs); // 32-bit
  const auto __lhs_high     = static_cast<__half_bits_t>(__lhs >> __half_bits); // 32-bit
  const auto __rhs_low      = static_cast<__half_bits_t>(__rhs); // 32-bit
  const auto __rhs_high     = static_cast<__half_bits_t>(__rhs >> __half_bits); // 32-bit
  const auto __po_half      = (static_cast<_Tp>(__lhs_low) * __rhs_low) >> __half_bits;
  const auto __p1           = static_cast<_Tp>(__lhs_low) * __rhs_high; // 64-bit
  const auto __p2           = static_cast<_Tp>(__lhs_high) * __rhs_low; // 64-bit
  const auto __p3           = static_cast<_Tp>(__lhs_high) * __rhs_high; // 64-bit
  const auto __p1_half      = static_cast<__half_bits_t>(__p1); // 32-bit
  const auto __p2_half      = static_cast<__half_bits_t>(__p2); // 32-bit
  const auto __carry        = (__po_half + __p1_half + __p2_half) >> __half_bits; // 64-bit
  return __p3 + (__p1 >> __half_bits) + (__p2 >> __half_bits) + __carry;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
[[nodiscard]]
_CCCL_API constexpr _Tp mul_hi(_Tp __lhs, _Tp __rhs) noexcept
{
  using ::cuda::std::int64_t;
  using ::cuda::std::is_signed_v;
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    if constexpr (sizeof(_Tp) == sizeof(int))
    {
      if constexpr (is_signed_v<_Tp>)
      {
        [[maybe_unused]] const auto __lhs1 = static_cast<int>(__lhs);
        [[maybe_unused]] const auto __rhs1 = static_cast<int>(__rhs);
        NV_IF_TARGET(NV_IS_DEVICE, (return ::__mulhi(__lhs1, __rhs1);));
      }
      else // is_unsigned_v<_Tp>
      {
        [[maybe_unused]] const auto __lhs1 = static_cast<unsigned>(__lhs);
        [[maybe_unused]] const auto __rhs1 = static_cast<unsigned>(__rhs);
        NV_IF_TARGET(NV_IS_DEVICE, (return ::__umulhi(__lhs1, __rhs1);));
      }
    }
    else if constexpr (sizeof(_Tp) == sizeof(int64_t))
    {
      if constexpr (is_signed_v<_Tp>)
      {
        [[maybe_unused]] const auto __lhs1 = static_cast<long long>(__lhs);
        [[maybe_unused]] const auto __rhs1 = static_cast<long long>(__rhs);
        NV_IF_TARGET(NV_IS_DEVICE, (return ::__mul64hi(__lhs1, __rhs1);));
#if _CCCL_COMPILER(MSVC)
        NV_IF_TARGET(NV_IS_HOST, (return ::__mulh(__lhs1, __rhs1);));
#endif // _CCCL_COMPILER(MSVC)
      }
      else // is_unsigned_v<_Tp>
      {
        [[maybe_unused]] const auto __lhs1 = static_cast<unsigned long long>(__lhs);
        [[maybe_unused]] const auto __rhs1 = static_cast<unsigned long long>(__rhs);
        NV_IF_TARGET(NV_IS_DEVICE, (return ::__umul64hi(__lhs1, __rhs1);));
#if _CCCL_COMPILER(MSVC)
        NV_IF_TARGET(NV_IS_HOST, (return ::__umulh(__lhs1, __rhs1);));
#endif // _CCCL_COMPILER(MSVC)
      }
    }
  }
  if constexpr (sizeof(_Tp) < sizeof(int64_t) || (sizeof(_Tp) == sizeof(int64_t) && _CCCL_HAS_INT128()))
  {
    constexpr auto __bits = ::cuda::std::__num_bits_v<_Tp>;
    using __larger_t      = ::cuda::std::__make_nbit_int_t<__bits * 2, is_signed_v<_Tp>>;
    const auto __ret      = (static_cast<__larger_t>(__lhs) * __rhs) >> __bits;
    return static_cast<_Tp>(__ret);
  }
  else // sizeof(_Tp) >= sizeof(int64_t) && !_CCCL_HAS_INT128()
  {
    if constexpr (is_signed_v<_Tp>)
    {
      using _Up         = ::cuda::std::make_unsigned_t<_Tp>;
      const auto __lhs1 = static_cast<_Up>(__lhs);
      const auto __rhs1 = static_cast<_Up>(__rhs);
      auto __hi         = ::cuda::__mul_hi_fallback(__lhs1, __rhs1);
      if (__lhs < 0)
      {
        __hi -= __rhs1;
      }
      if (__rhs < 0)
      {
        __hi -= __lhs1;
      }
      return static_cast<_Tp>(__hi);
    }
    else
    {
      return ::cuda::__mul_hi_fallback(__lhs, __rhs);
    }
  }
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_MULTIPLY_HIGH_HALF_H
