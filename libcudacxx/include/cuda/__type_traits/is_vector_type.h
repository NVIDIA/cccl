//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__TYPE_TRAITS_IS_VECTOR_TYPE_H
#define _CUDA__TYPE_TRAITS_IS_VECTOR_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()
#  include <cuda/__type_traits/scalar_type.h>
#  include <cuda/__type_traits/vector_size.h>
#  include <cuda/std/__floating_point/traits.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__type_traits/void_t.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// is_vector_type

template <class _Tp>
inline constexpr bool is_vector_type_v = (vector_size_v<_Tp> != 0);

template <class _Tp>
using is_vector_type = ::cuda::std::bool_constant<is_vector_type_v<_Tp>>;

// is_extended_fp_vector_type

template <class _Tp, class = void>
inline constexpr bool is_extended_fp_vector_type_v = false;
template <class _Tp>
inline constexpr bool is_extended_fp_vector_type_v<_Tp, ::cuda::std::void_t<typename scalar_type<_Tp>::type>> =
  ::cuda::std::__is_ext_nv_fp_v<scalar_type_t<_Tp>>;

template <class _Tp>
using is_extended_fp_vector_type = ::cuda::std::bool_constant<is_extended_fp_vector_type_v<_Tp>>;

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_HAS_CTK()
#endif // _CUDA__TYPE_TRAITS_IS_VECTOR_TYPE_H
