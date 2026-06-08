//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA__TYPE_TRAITS_HAS_NO_FLOATING_POINT_H
#define __CUDA__TYPE_TRAITS_HAS_NO_FLOATING_POINT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/complex.h>
#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/__type_traits/is_vector_type.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__type_traits/aggregate_members_all_of.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_aggregate.h>
#include <cuda/std/__type_traits/remove_cv.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Tp, typename = void>
inline constexpr bool __has_no_floating_point_aggregate_v = true;

template <typename _Tp>
inline constexpr bool __has_no_floating_point_cv_v =
  !::cuda::is_floating_point_v<_Tp>
#if _CCCL_HAS_CTK()
  && !is_extended_fp_vector_type_v<_Tp>
#endif // _CCCL_HAS_CTK()
  && __has_no_floating_point_aggregate_v<_Tp>;

template <typename _Tp>
inline constexpr bool __has_no_floating_point_v = __has_no_floating_point_cv_v<::cuda::std::remove_cv_t<_Tp>>;

//----------------------------------------------------------------------------------------------------------------------
// specializations for array, pair, tuple, and complex types

template <typename _Tp>
inline constexpr bool __has_no_floating_point_cv_v<_Tp[]> = __has_no_floating_point_v<_Tp>;

template <typename _Tp, ::cuda::std::size_t _Size>
inline constexpr bool __has_no_floating_point_cv_v<_Tp[_Size]> = __has_no_floating_point_v<_Tp>;

template <typename _Tp, ::cuda::std::size_t _Size>
inline constexpr bool __has_no_floating_point_cv_v<::cuda::std::array<_Tp, _Size>> = __has_no_floating_point_v<_Tp>;

template <typename _T1, typename _T2>
inline constexpr bool __has_no_floating_point_cv_v<::cuda::std::pair<_T1, _T2>> =
  __has_no_floating_point_v<_T1> && __has_no_floating_point_v<_T2>;

template <typename... _Ts>
inline constexpr bool __has_no_floating_point_cv_v<::cuda::std::tuple<_Ts...>> =
  (__has_no_floating_point_v<_Ts> && ...);

template <typename _Tp>
inline constexpr bool __has_no_floating_point_cv_v<::cuda::std::complex<_Tp>> = __has_no_floating_point_v<_Tp>;

template <typename _Tp>
inline constexpr bool __has_no_floating_point_cv_v<complex<_Tp>> = __has_no_floating_point_v<_Tp>;

//----------------------------------------------------------------------------------------------------------------------
// if all the previous conditions fail, check if the type is an aggregate and none of its members are floating-point

template <typename _Tp>
using __has_no_floating_point_callable = ::cuda::std::bool_constant<__has_no_floating_point_v<_Tp>>;

template <typename _Tp>
inline constexpr bool
  __has_no_floating_point_aggregate_v<_Tp, ::cuda::std::enable_if_t<::cuda::std::is_aggregate_v<_Tp>>> =
    ::cuda::std::__aggregate_all_of_v<__has_no_floating_point_callable, _Tp>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA__TYPE_TRAITS_HAS_NO_FLOATING_POINT_H
