//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA__TYPE_TRAITS_IS_BITWISE_COMPARABLE_H
#define __CUDA__TYPE_TRAITS_IS_BITWISE_COMPARABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/complex.h>
#include <cuda/__type_traits/is_vector_type.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__tuple_dir/tuple.h> // required for sizeof
#include <cuda/std/__type_traits/aggregate_members_all_of.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/has_unique_object_representation.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_aggregate.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__utility/pair.h> // required for sizeof

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Tp, typename = void>
inline constexpr bool __is_aggregate_bitwise_comparable_v = true;

template <typename _Tp>
inline constexpr bool __is_bitwise_comparable_v =
  ::cuda::std::has_unique_object_representations_v<_Tp> // no padding, no float/double
#if _CCCL_HAS_CTK()
  && !::cuda::std::__is_extended_floating_point_v<_Tp> // scalar extended floating-point types
  && !is_extended_fp_vector_type_v<_Tp> // vector extended floating-point types
#endif // _CCCL_HAS_CTK()
  && __is_aggregate_bitwise_comparable_v<_Tp>;

template <typename _Tp>
inline constexpr bool __is_bitwise_comparable_v<_Tp[]> = __is_bitwise_comparable_v<_Tp>;

template <typename _Tp, ::cuda::std::size_t _Size>
inline constexpr bool __is_bitwise_comparable_v<_Tp[_Size]> = __is_bitwise_comparable_v<_Tp>;

template <typename _Tp, ::cuda::std::size_t _Size>
inline constexpr bool __is_bitwise_comparable_v<::cuda::std::array<_Tp, _Size>> = __is_bitwise_comparable_v<_Tp>;

template <typename _T1, typename _T2>
inline constexpr bool __is_bitwise_comparable_v<::cuda::std::pair<_T1, _T2>> =
  (sizeof(::cuda::std::pair<_T1, _T2>) == sizeof(_T1) + sizeof(_T2))
  && __is_bitwise_comparable_v<_T1> && __is_bitwise_comparable_v<_T2>;

template <typename... _Ts>
inline constexpr bool __is_bitwise_comparable_v<::cuda::std::tuple<_Ts...>> =
  (sizeof...(_Ts) == 0 || sizeof(::cuda::std::tuple<_Ts...>) == (sizeof(_Ts) + ... + 0))
  && (__is_bitwise_comparable_v<_Ts> && ...);

template <typename _Tp>
inline constexpr bool __is_bitwise_comparable_v<::cuda::std::complex<_Tp>> = false;

template <typename _Tp>
inline constexpr bool __is_bitwise_comparable_v<complex<_Tp>> = false;

// if all the previous conditions fail, check if the type is an aggregate and all its members are bitwise comparable
template <typename _Tp>
using __is_bitwise_comparable_callable = ::cuda::std::bool_constant<__is_bitwise_comparable_v<_Tp>>;

// __aggregate_all_of_v returns true if the type is not an aggregate (or empty)
template <typename _Tp>
inline constexpr bool
  __is_aggregate_bitwise_comparable_v<_Tp, ::cuda::std::enable_if_t<::cuda::std::is_aggregate_v<_Tp>>> =
    ::cuda::std::__aggregate_all_of_v<__is_bitwise_comparable_callable, _Tp>;

//----------------------------------------------------------------------------------------------------------------------
// public variable template and alias

// Users are allowed to specialize this variable template for their own types
template <typename _Tp>
inline constexpr bool is_bitwise_comparable_v = __is_bitwise_comparable_v<_Tp>;

template <typename _Tp>
inline constexpr bool is_bitwise_comparable_v<const _Tp> = is_bitwise_comparable_v<_Tp>;

template <typename _Tp>
inline constexpr bool is_bitwise_comparable_v<volatile _Tp> = is_bitwise_comparable_v<_Tp>;

template <typename _Tp>
inline constexpr bool is_bitwise_comparable_v<const volatile _Tp> = is_bitwise_comparable_v<_Tp>;

// defined as alias so users cannot specialize it (they should specialize the variable template instead)
template <typename _Tp>
using is_bitwise_comparable = ::cuda::std::bool_constant<is_bitwise_comparable_v<_Tp>>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA__TYPE_TRAITS_IS_BITWISE_COMPARABLE_H
