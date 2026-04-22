//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_CONCEPTS_H
#define _CUDA_STD___SIMD_CONCEPTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__floating_point/conversion_rank_order.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.expos], explicitly-convertible-to concept

template <typename _From, typename _To>
_CCCL_CONCEPT __explicitly_convertible_to = _CCCL_REQUIRES_EXPR((_From, _To))((static_cast<_To>(declval<_From>())));

// [simd.expos], constexpr-wrapper-like concept

template <typename _Tp>
_CCCL_CONCEPT __constexpr_wrapper_like = _CCCL_REQUIRES_EXPR((_Tp))(
  requires(convertible_to<_Tp, decltype(_Tp::value)>),
  requires(equality_comparable_with<_Tp, decltype(_Tp::value)>),
  requires(bool_constant<(_Tp() == _Tp::value)>::value),
  requires(bool_constant<(static_cast<decltype(_Tp::value)>(_Tp()) == _Tp::value)>::value));

// Covers all integral types including character types (char16_t, char32_t, wchar_t, char8_t),
// which are excluded by __cccl_is_integer_v
template <typename _From, typename _To>
inline constexpr bool __is_integral__value_preserving_v =
  is_integral_v<_From> && is_integral_v<_To> && numeric_limits<_From>::digits <= numeric_limits<_To>::digits
  && (!is_signed_v<_From> || is_signed_v<_To>);

// [conv.rank], integer conversion rank for [simd.ctor] p7

template <typename _Tp>
inline constexpr int __integer_conversion_rank = 0;

template <>
inline constexpr int __integer_conversion_rank<signed char> = 1;
template <>
inline constexpr int __integer_conversion_rank<unsigned char> = 1;
template <>
inline constexpr int __integer_conversion_rank<char> = 1;
template <>
inline constexpr int __integer_conversion_rank<short> = 2;
template <>
inline constexpr int __integer_conversion_rank<unsigned short> = 2;
template <>
inline constexpr int __integer_conversion_rank<int> = 3;
template <>
inline constexpr int __integer_conversion_rank<unsigned int> = 3;
template <>
inline constexpr int __integer_conversion_rank<long> = 4;
template <>
inline constexpr int __integer_conversion_rank<unsigned long> = 4;
template <>
inline constexpr int __integer_conversion_rank<long long> = 5;
template <>
inline constexpr int __integer_conversion_rank<unsigned long long> = 5;
#if _CCCL_HAS_INT128()
template <>
inline constexpr int __integer_conversion_rank<__int128_t> = 6;
template <>
inline constexpr int __integer_conversion_rank<__uint128_t> = 6;
#endif // _CCCL_HAS_INT128()

// The conversion from an arithmetic type U to a vectorizable type T is value-preserving if all possible
// values of U can be represented with type T. For floating-point pairs we defer to
// __fp_is_implicit_conversion_v, which correctly handles unordered pairs such as __half / __nv_bfloat16.
template <typename _From, typename _To>
inline constexpr bool __is_value_preserving_v =
  __is_integral__value_preserving_v<_From, _To>
  || (::cuda::is_floating_point_v<_From> && ::cuda::is_floating_point_v<_To>
      && __fp_is_implicit_conversion_v<_From, _To>)
  || (is_integral_v<_From> && ::cuda::is_floating_point_v<_To>
      && numeric_limits<_From>::digits <= numeric_limits<_To>::digits);

template <typename _From, typename _ValueType, typename = void>
inline constexpr bool __is_constexpr_wrapper_value_preserving_v = false;

// The standard requires checking whether the specific compile-time value From::value is representable by _ValueType,
// not whether the entire source type is value-preserving.
template <typename _From, typename _ValueType>
inline constexpr bool __is_constexpr_wrapper_value_preserving_v<_From, _ValueType, void_t<decltype(_From::value)>> =
  is_arithmetic_v<remove_cvref_t<decltype(_From::value)>>
  && (static_cast<remove_cvref_t<decltype(_From::value)>>(static_cast<_ValueType>(_From::value)) == _From::value);

// [simd.ctor] implicit value constructor
// - From is not an arithmetic type and does not satisfy constexpr-wrapper-like,
// - From is an arithmetic type and the conversion from From to value_type is value-preserving
// - From satisfies constexpr-wrapper-like, remove_cvref_t<decltype(From​::​value)> is an arithmetic type, and
//   From​::​value is representable by value_type.
template <typename _Up, typename _ValueType, typename _From = remove_cvref_t<_Up>>
_CCCL_CONCEPT __is_value_ctor_implicit =
  convertible_to<_Up, _ValueType>
  && ((!is_arithmetic_v<_From> && !__constexpr_wrapper_like<_From>)
      || (is_arithmetic_v<_From> && __is_value_preserving_v<_From, _ValueType>)
      || (__constexpr_wrapper_like<_From> && __is_constexpr_wrapper_value_preserving_v<_From, _ValueType>) );

// [simd.ctor] p7: explicit(see below) for basic_vec(const basic_vec<U, UAbi>&)
// explicit evaluates to true if either:
//   - conversion from U to value_type is not value-preserving, or
//   - both U and value_type are integral and integer_conversion_rank(U) > rank(value_type), or
//   - both U and value_type are floating-point and fp_conversion_rank(U) > rank(value_type)
template <typename _Up, typename _ValueType>
inline constexpr bool __is_vec_ctor_explicit =
  !__is_value_preserving_v<_Up, _ValueType>
  || (is_integral_v<_Up> && is_integral_v<_ValueType>
      && __integer_conversion_rank<_Up> > __integer_conversion_rank<_ValueType>)
  || (::cuda::is_floating_point_v<_Up> && ::cuda::is_floating_point_v<_ValueType>
      && __fp_conv_rank_order_v<_Up, _ValueType> == __fp_conv_rank_order::__greater);

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_CONCEPTS_H
