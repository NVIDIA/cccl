//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___NUMERIC_MUL_SAT_H
#define _CUDA_STD___NUMERIC_MUL_SAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/mul_hi.h>
#include <cuda/__cmath/neg.h>
#include <cuda/__cmath/uabs.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// todo: use cuda::mul_overflow when available

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp mul_sat(_Tp __x, _Tp __y) noexcept
{
  [[maybe_unused]] constexpr auto __min = numeric_limits<_Tp>::min();
  constexpr auto __max                  = numeric_limits<_Tp>::max();

  if constexpr (is_signed_v<_Tp>)
  {
    // Upcast to a greater integer type if possible, fallback to slow implementation otherwise.
    if constexpr (sizeof(_Tp) < sizeof(__cccl_intmax_t))
    {
      using _Up = conditional_t<(sizeof(_Tp) < sizeof(int32_t)), int32_t, __make_nbit_int_t<2 * __num_bits_v<_Tp>>>;
      return static_cast<_Tp>(::cuda::std::clamp(_Up{__x} * _Up{__y}, _Up{__min}, _Up{__max}));
    }
    else
    {
      // We need to use unsigned types to avoid undefined behavior.
      const auto __negative_result = (__x >= 0) != (__y >= 0);
      const auto __ux              = ::cuda::uabs(__x);
      const auto __uy              = ::cuda::uabs(__y);
      const auto __uresult_lo      = __ux * __uy;
      const auto __uresult_hi      = ::cuda::mul_hi(__ux, __uy);
      const auto __uresult_max     = (__negative_result) ? ::cuda::uabs(__min) : ::cuda::uabs(__max);

      // Check for overflow.
      if (__uresult_hi != 0 || __uresult_lo > __uresult_max)
      {
        return (__negative_result) ? __min : __max;
      }
      return static_cast<_Tp>((__negative_result) ? ::cuda::neg(__uresult_lo) : __uresult_lo);
    }
  }
  else // ^^^ signed types ^^^ / vvv unsigned types vvv
  {
    if constexpr (sizeof(_Tp) < sizeof(uint32_t))
    {
      return static_cast<_Tp>(::cuda::std::min(uint32_t{__x} * uint32_t{__y}, uint32_t{__max}));
    }
    else
    {
      return (::cuda::mul_hi(__x, __y) == _Tp{0}) ? (__x * __y) : __max;
    }
  } // ^^^ unsigned types ^^^
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NUMERIC_MUL_SAT_H
