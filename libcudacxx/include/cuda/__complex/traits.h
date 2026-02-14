//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___COMPLEX_TRAITS_H
#define _CUDA___COMPLEX_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/complex.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_like.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// __is_complex_compatible_tuple_like

template <class _Tp>
_CCCL_CONCEPT __is_complex_compatible_tuple_like = _CCCL_REQUIRES_EXPR(
  (_Tp))(requires(::cuda::std::tuple_size<_Tp>::value == 2),
         requires(::cuda::std::is_same_v<::cuda::std::remove_cvref_t<::cuda::std::tuple_element_t<0, _Tp>>,
                                         ::cuda::std::remove_cvref_t<::cuda::std::tuple_element_t<1, _Tp>>>));

// __complex_tuple_like_value_type

template <class _Tp>
using __complex_tuple_like_value_type_t = ::cuda::std::remove_cvref_t<::cuda::std::tuple_element_t<0, _Tp>>;

// __is_cccl_complex_v

template <class _Tp>
inline constexpr bool __is_cccl_complex_v = __is_cuda_complex_v<_Tp> || ::cuda::std::__is_cuda_std_complex_v<_Tp>;

// __is_any_complex_v

template <class _Tp>
inline constexpr bool __is_any_complex_v = ::cuda::std::__is_std_complex_v<_Tp> || __is_cccl_complex_v<_Tp>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___COMPLEX_TRAITS_H
