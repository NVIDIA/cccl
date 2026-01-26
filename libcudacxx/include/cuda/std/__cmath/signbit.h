//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CMATH_SIGNBIT_H
#define _CUDA_STD___CMATH_SIGNBIT_H

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

// This function may or may not be implemented as a function macro so we need to handle that case
#ifdef signbit
// No fallback implementation as we implement it natively anyhow
#  pragma push_macro("signbit")
#  undef signbit
#  define _CCCL_POP_MACRO_signbit
#endif // signbit

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr bool signbit([[maybe_unused]] _Tp __x) noexcept
{
  if constexpr (!numeric_limits<_Tp>::is_signed)
  {
    return false;
  }
  else if constexpr (is_integral_v<_Tp>)
  {
    return __x < 0;
  }
  else
  {
    return ::cuda::std::__fp_get_storage(__x) & __fp_sign_mask_of_v<_Tp>;
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#ifdef _CCCL_POP_MACRO_signbit
#  pragma pop_macro("signbit")
#  undef _CCCL_POP_MACRO_signbit
#endif

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_SIGNBIT_H
