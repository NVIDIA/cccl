//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_NEG_H
#define _CUDA___CMATH_NEG_H

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
#include <cuda/std/__type_traits/make_unsigned.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief Returns the negation of the given number
//! @param __x The input number
//! @pre \p __v must be an integer type
//! @return The negation of \p __x
//! @warning The result may overflow the result type, prefer to use cuda::neg_overflow() if this is a concern
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_cv_integer, _Tp))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __neg(_Tp __v) noexcept
{
  using _Up = _CUDA_VSTD::make_unsigned_t<_Tp>;
  return static_cast<_Tp>(~static_cast<_Up>(__v) + _Up(1));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___CMATH_NEG_H
