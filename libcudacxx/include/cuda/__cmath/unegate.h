//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_UNEGATE_H
#define _CUDA___CMATH_UNEGATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/make_signed.h>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief Returns the *signed* negative value of the given *unsigned* number.
//! @param __v The input number
//! @pre \p __v must be an unsigned integer type
//! @pre \p __v must be less than or equal to the maximum representable value of the signed type + 1
//! @return The signed negative value of \p __v
_CCCL_TEMPLATE(class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_cv_unsigned_integer, _Up))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _CUDA_VSTD::make_signed_t<_Up> __unegate(_Up __v) noexcept
{
  using _Tp = _CUDA_VSTD::make_signed_t<_Up>;
  _CCCL_ASSERT(__v <= _Up(_CUDA_VSTD::numeric_limits<_Tp>::max()) + 1,
               "__unegate called with a value greater than the maximum representable value");
  return static_cast<_Tp>(~__v + 1);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___CMATH_UNEGATE_H
