//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_ABS_DIFF_H
#define _CUDA___CMATH_ABS_DIFF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Computes absolute difference.
//! @param[in] __lhs The left-hand side input.
//! @param[in] __rhs The right-hand side input.
//! @return An unsigned value containing the absolute difference of each pair of input elements.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr ::cuda::std::make_unsigned_t<_Tp> abs_diff(_Tp __lhs, _Tp __rhs) noexcept
{
  using _Up _CCCL_NODEBUG = ::cuda::std::make_unsigned_t<_Tp>;

  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_TARGET(NV_IS_DEVICE, ({
                   if constexpr (::cuda::std::is_signed_v<_Tp> && sizeof(_Tp) <= sizeof(::cuda::std::int32_t))
                   {
                     return static_cast<_Up>(::__sad(__lhs, __rhs, 0));
                   }
                   else if constexpr (::cuda::std::is_unsigned_v<_Tp> && sizeof(_Tp) <= sizeof(::cuda::std::uint32_t))
                   {
                     return static_cast<_Up>(::__usad(__lhs, __rhs, 0));
                   }
                 }))
  }

  const auto __minuend    = static_cast<_Up>((__lhs > __rhs) ? __lhs : __rhs);
  const auto __subtrahend = static_cast<_Up>((__lhs > __rhs) ? __rhs : __lhs);
  return static_cast<_Up>(__minuend - __subtrahend);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_ABS_DIFF_H
