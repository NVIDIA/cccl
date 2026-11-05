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

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/uabs.h>
#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/__utility/in_range.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__floating_point/conversion_rank_order.h>
#include <cuda/std/__fwd/simd.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__simd/exposition.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_signed.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.expos], explicitly-convertible-to concept

template <typename _From, typename _To>
_CCCL_CONCEPT __explicitly_convertible_to = _CCCL_REQUIRES_EXPR((_From, _To))((static_cast<_To>(declval<_From>())));

// [simd.expos], simd-vec-type concept

template <typename _Vp>
_CCCL_CONCEPT __simd_vec_type = _CCCL_REQUIRES_EXPR(
  (_Vp))(requires(same_as<_Vp, basic_vec<typename _Vp::value_type, typename _Vp::abi_type>>),
         requires(is_default_constructible_v<_Vp>));

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
template <>
inline constexpr int __integer_conversion_rank<wchar_t> = __integer_conversion_rank<make_signed_t<wchar_t>>;
template <>
inline constexpr int __integer_conversion_rank<char16_t> = __integer_conversion_rank<make_signed_t<char16_t>>;
template <>
inline constexpr int __integer_conversion_rank<char32_t> = __integer_conversion_rank<make_signed_t<char32_t>>;
#if _CCCL_HAS_CHAR8_T()
template <>
inline constexpr int __integer_conversion_rank<char8_t> = __integer_conversion_rank<unsigned char>;
#endif // _CCCL_HAS_CHAR8_T()
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

template <typename _From, typename _To>
inline constexpr bool __is_value_preserving_v<_From, ::cuda::std::complex<_To>> = __is_value_preserving_v<_From, _To>;

template <typename _From, typename _To>
inline constexpr bool __is_value_preserving_v<_From, ::cuda::complex<_To>> = __is_value_preserving_v<_From, _To>;

#if _CCCL_HAS_HOST_STD_LIB()
template <typename _From, typename _To>
inline constexpr bool __is_value_preserving_v<_From, ::std::complex<_To>> = __is_value_preserving_v<_From, _To>;
#endif // _CCCL_HAS_HOST_STD_LIB()

template <typename _Int, typename _Fp>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL bool __is_fp_in_integer_range(_Fp __value) noexcept
{
  // The target integer range covers all floating-point values
  if constexpr (numeric_limits<_Int>::digits >= numeric_limits<_Fp>::max_exponent)
  {
    constexpr auto __low = is_signed_v<_Int> ? numeric_limits<_Fp>::lowest() : _Fp{0};
    return ::cuda::in_range(__value, __low, numeric_limits<_Fp>::max());
  }
  else // The target integer range is smaller than floating-point range
  {
    // find the smallest floating-point value that is not representable by the integer type
    constexpr auto __upper_bound = ::cuda::ceil_div(numeric_limits<_Int>::max(), 2) * _Fp{2};
    constexpr auto __low         = is_signed_v<_Int> ? -__upper_bound : _Fp{0};
    return __value >= __low && __value < __upper_bound;
  }
}

template <typename _From, typename _To>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL bool __is_constexpr_wrapper_value_preserving() noexcept
{
  using _FromValue = remove_cvref_t<decltype(_From::value)>;
  if constexpr (!__constexpr_wrapper_like<_From> || !is_arithmetic_v<_FromValue>)
  {
    return false;
  }
  else if constexpr (__is_complex_vectorizable_v<_To>)
  {
    using __value_type _CCCL_NODEBUG = typename _To::value_type;
    return ::cuda::std::simd::__is_constexpr_wrapper_value_preserving<_From, __value_type>();
  }
  else if constexpr (is_integral_v<_FromValue> && is_integral_v<_To>)
  {
    // Promote bool and character types, which cmp_equal does not accept directly.
    return ::cuda::std::cmp_equal(+static_cast<_To>(_From::value), +_From::value);
  }
  else if constexpr (is_integral_v<_To>)
  {
    auto __value = _From::value; // non-const avoids diagnostics on CUDA 12 and NVHPC
    return ::cuda::std::simd::__is_fp_in_integer_range<_To>(__value) // e.g. 1e100 is not representable by int
        && static_cast<_FromValue>(static_cast<_To>(__value)) == __value; // e.g. 3.0 <-> 3
  }
  else if constexpr (is_integral_v<_FromValue> && is_floating_point_v<_To>)
  {
    // Promote bool and character types before computing the unsigned magnitude
    const auto __value         = ::cuda::uabs(+_From::value);
    const auto __width         = ::cuda::std::bit_width(__value);
    const auto __mantissa_bits = __width - ::cuda::std::countr_zero(__value);
    return __value == 0
        || (__width <= numeric_limits<_To>::max_exponent && __mantissa_bits <= numeric_limits<_To>::digits);
  }
  else
  {
    return static_cast<_FromValue>(static_cast<_To>(_From::value)) == _From::value;
  }
}

template <typename _From, typename _To, typename = void>
inline constexpr bool __is_constexpr_wrapper_value_preserving_v = false;

// The standard requires checking whether the specific compile-time value From::value is representable by _To,
// not whether the entire source type is value-preserving.
template <typename _From, typename _To>
inline constexpr bool __is_constexpr_wrapper_value_preserving_v<
  _From,
  _To,
  void_t<decltype(_From::value), decltype(bool_constant<__is_constexpr_wrapper_value_preserving<_From, _To>()>{})>> =
  __is_constexpr_wrapper_value_preserving<_From, _To>();

// [simd.ctor] implicit value constructor
// - From is not an arithmetic type and does not satisfy constexpr-wrapper-like,
// - From is an arithmetic type and the conversion from From to value_type is value-preserving
// - From satisfies constexpr-wrapper-like, remove_cvref_t<decltype(From::value)> is an arithmetic type, and
//   From::value is representable by value_type.
template <typename _Up, typename _ValueType, typename _From = remove_cvref_t<_Up>>
_CCCL_CONCEPT __is_value_ctor_implicit =
  (convertible_to<_Up, _ValueType> && !is_arithmetic_v<_From> && !__constexpr_wrapper_like<_From>)
  || (is_arithmetic_v<_From> && __is_value_preserving_v<_From, _ValueType>)
  || (__constexpr_wrapper_like<_From> && __is_constexpr_wrapper_value_preserving_v<_From, _ValueType>);

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
