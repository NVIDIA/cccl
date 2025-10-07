//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_MULTIPLY_HIGH_HALF_H
#define _CUDA___CMATH_MULTIPLY_HIGH_HALF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_constant_evaluated.h>
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
[[nodiscard]] _CCCL_API constexpr _Tp __multiply_half_high_fallback(_Tp __lhs, _Tp __rhs)
{
  constexpr int __shift = ::cuda::std::__num_bits_v<_Tp> / 2;
  auto __x_high         = __lhs >> __shift;
  auto __x_low          = __lhs;
  auto __y_high         = __rhs >> __shift;
  auto __y_low          = __rhs;
  auto __p0             = __x_low * __y_low;
  auto __p1             = __x_low * __y_high;
  auto __p2             = __x_high * __y_low;
  auto __p3             = __x_high * __y_high;
  auto __mid            = __p1 + __p2;
  auto __carry          = static_cast<_Tp>(__mid < __p1);
  auto __po_half        = __p0 >> __shift;
  __mid                 = __mid + __po_half;
  __carry += (__mid < __po_half);
  return __p3 + (__mid >> __shift) + (__carry << __shift);
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
[[nodiscard]]
_CCCL_API constexpr _Tp multiply_half_high(_Tp __lhs, _Tp __rhs)
{
  if constexpr (::cuda::std::is_signed_v<_Tp>)
  {
    _CCCL_ASSERT(__lhs >= 0, "__lhs must be non-negative");
    _CCCL_ASSERT(__rhs >= 0, "__rhs must be non-negative");
  }
  using ::cuda::std::uint32_t;
  using ::cuda::std::uint64_t;
  using _Up         = ::cuda::std::make_unsigned_t<_Tp>;
  const auto __lhs1 = static_cast<_Up>(__lhs);
  const auto __rhs1 = static_cast<_Up>(__rhs);
  if (!::cuda::std::__cccl_default_is_constant_evaluated())
  {
    if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return ::__umulhi(__lhs1, __rhs1);));
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return ::__umul64hi(__lhs1, __rhs1);));
#if _CCCL_COMPILER(MSVC)
      NV_IF_TARGET(NV_IS_HOST, (return ::__umulh(__lhs1, __rhs1);));
#endif // _CCCL_COMPILER(MSVC)
    }
  }
  if constexpr (sizeof(_Tp) < sizeof(uint64_t) || (sizeof(_Tp) == sizeof(uint64_t) && _CCCL_HAS_INT128()))
  {
    using __larger_t      = ::cuda::std::__make_nbit_uint_t<::cuda::std::__num_bits_v<_Tp> * 2>;
    constexpr auto __bits = ::cuda::std::__num_bits_v<_Tp>;
    auto __ret            = (static_cast<__larger_t>(__lhs1) * __rhs1) >> __bits;
    return static_cast<_Tp>(__ret);
  }
  else // sizeof(_Tp) >= sizeof(uint64_t) && !_CCCL_HAS_INT128()
  {
    return ::cuda::__multiply_half_high_fallback(__lhs1, __rhs1);
  }
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_MULTIPLY_HIGH_HALF_H
