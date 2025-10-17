//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CMATH_COPYSIGN_H
#define _CUDA_STD___CMATH_COPYSIGN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr auto copysign(_Tp __x, [[maybe_unused]] _Tp __y) noexcept
{
  if constexpr (is_integral_v<_Tp>)
  {
    if constexpr (!numeric_limits<_Tp>::is_signed)
    {
      return static_cast<double>(__x);
    }
    else
    {
      const auto __x_dbl = static_cast<double>(__x);
      if (__y < 0)
      {
        return (__x < 0) ? __x_dbl : -__x_dbl;
      }
      else
      {
        return (__x < 0) ? -__x_dbl : __x_dbl;
      }
    }
  }
  else // ^^^ integral ^^^ / vvv floating_point vvv
  {
    if constexpr (!numeric_limits<_Tp>::is_signed)
    {
      return __x;
    }
    else
    {
      const auto __val = (::cuda::std::__fp_get_storage(__x) & __fp_exp_mant_mask_of_v<_Tp>)
                       | (::cuda::std::__fp_get_storage(__y) & __fp_sign_mask_of_v<_Tp>);
      return ::cuda::std::__fp_from_storage<_Tp>(static_cast<__fp_storage_of_t<_Tp>>(__val));
    }
  }
}

[[nodiscard]] _CCCL_API constexpr float copysignf(float __x, float __y) noexcept
{
  return ::cuda::std::copysign(__x, __y);
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API constexpr long double copysignl(long double __x, long double __y) noexcept
{
  return ::cuda::std::copysign(__x, __y);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_COPYSIGN_H
