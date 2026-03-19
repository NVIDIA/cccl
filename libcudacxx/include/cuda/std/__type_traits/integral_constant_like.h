//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_INTEGRAL_CONSTANT_LIKE_H
#define _CUDA_STD___TYPE_TRAITS_INTEGRAL_CONSTANT_LIKE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
_CCCL_CONCEPT __integral_constant_like = _CCCL_REQUIRES_EXPR((_Tp)) //
  ( //
    requires(is_integral_v<remove_cvref_t<decltype(_Tp::value)>>), //
    requires(!is_same_v<bool, remove_cvref_t<decltype(_Tp::value)>>), //
    requires(convertible_to<_Tp, decltype(_Tp::value)>), //
    requires(equality_comparable_with<_Tp, decltype(_Tp::value)>), //
    requires(bool_constant<(_Tp() == _Tp::value)>::value), //
    requires(bool_constant<(static_cast<decltype(_Tp::value)>(_Tp()) == _Tp::value)>::value) //
  );

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_INTEGRAL_CONSTANT_LIKE_H
