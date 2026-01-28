//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_FUNCTIONAL_OPERATOR_PROPERTIES_H
#define _CUDA_FUNCTIONAL_OPERATOR_PROPERTIES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__functional/maximum.h>
#include <cuda/__functional/minimum.h>
#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__floating_point/arithmetic.h>
#include <cuda/std/__floating_point/constants.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/void_t.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

/***********************************************************************************************************************
 * Associativity
 **********************************************************************************************************************/

template <class _Op>
[[nodiscard]] _CCCL_API constexpr bool __is_associative_static_assert()
{
  static_assert(::cuda::std::__always_false_v<_Op>,
                "operator_properties is not specialized for this operator and type combination");
  return false;
}

template <class _Op, class _Tp, class Enable = void>
inline constexpr bool is_associative_v = __is_associative_static_assert<_Op>();

// strictly speaking, plus (+) and multiply (*) are not associative because of overflow UB
template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = false;

template <class _Tp>
inline constexpr bool is_associative_v<::cuda::std::plus<>, _Tp> = is_associative_v<::cuda::std::plus<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::multiplies<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::multiplies<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool is_associative_v<::cuda::std::multiplies<>, _Tp> =
  is_associative_v<::cuda::std::multiplies<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::bit_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool is_associative_v<::cuda::std::bit_and<>, _Tp> =
  is_associative_v<::cuda::std::bit_and<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::bit_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool is_associative_v<::cuda::std::bit_or<>, _Tp> =
  is_associative_v<::cuda::std::bit_or<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::bit_xor<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool is_associative_v<::cuda::std::bit_xor<>, _Tp> =
  is_associative_v<::cuda::std::bit_xor<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::logical_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>> =
    true;

template <class _Tp>
inline constexpr bool is_associative_v<::cuda::std::logical_and<>, _Tp> =
  is_associative_v<::cuda::std::logical_and<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::logical_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>> =
    true;

template <class _Tp>
inline constexpr bool is_associative_v<::cuda::std::logical_or<>, _Tp> =
  is_associative_v<::cuda::std::logical_or<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = true;

template <class _Tp>
inline constexpr bool is_associative_v<::cuda::minimum<>, _Tp> = is_associative_v<::cuda::minimum<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = true;

template <class _Tp>
inline constexpr bool is_associative_v<::cuda::maximum<>, _Tp> = is_associative_v<::cuda::maximum<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::minus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::minus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = false;

template <class _Tp>
inline constexpr bool is_associative_v<::cuda::std::minus<>, _Tp> =
  is_associative_v<::cuda::std::minus<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::divides<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::divides<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = false;

template <class _Tp>
inline constexpr bool is_associative_v<::cuda::std::divides<>, _Tp> =
  is_associative_v<::cuda::std::divides<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_associative_v<::cuda::std::modulus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool is_associative_v<::cuda::std::modulus<>, _Tp> =
  is_associative_v<::cuda::std::modulus<_Tp>, _Tp, void>;

/***********************************************************************************************************************
 * Commutativity
 **********************************************************************************************************************/

template <class _Op>
[[nodiscard]] _CCCL_API constexpr bool __is_commutative_static_assert()
{
  static_assert(::cuda::std::__always_false_v<_Op>,
                "operator_properties is not specialized for this operator and type combination");
  return false;
}

template <class _Op, class _Tp, class Enable = void>
inline constexpr bool is_commutative_v = ::cuda::__is_commutative_static_assert<_Op>();

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = true;

template <class _Tp>
inline constexpr bool is_commutative_v<::cuda::std::plus<>, _Tp> = is_commutative_v<::cuda::std::plus<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::multiplies<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::multiplies<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool is_commutative_v<::cuda::std::multiplies<>, _Tp> =
  is_commutative_v<::cuda::std::multiplies<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::bit_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool is_commutative_v<::cuda::std::bit_and<>, _Tp> =
  is_commutative_v<::cuda::std::bit_and<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::bit_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool is_commutative_v<::cuda::std::bit_or<>, _Tp> =
  is_commutative_v<::cuda::std::bit_or<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::bit_xor<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool is_commutative_v<::cuda::std::bit_xor<>, _Tp> =
  is_commutative_v<::cuda::std::bit_xor<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::logical_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>> =
    true;

template <class _Tp>
inline constexpr bool is_commutative_v<::cuda::std::logical_and<>, _Tp> =
  is_commutative_v<::cuda::std::logical_and<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::logical_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>> =
    true;

template <class _Tp>
inline constexpr bool is_commutative_v<::cuda::std::logical_or<>, _Tp> =
  is_commutative_v<::cuda::std::logical_or<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = true;

template <class _Tp>
inline constexpr bool is_commutative_v<::cuda::minimum<>, _Tp> = is_commutative_v<::cuda::minimum<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = true;

template <class _Tp>
inline constexpr bool is_commutative_v<::cuda::maximum<>, _Tp> = is_commutative_v<::cuda::maximum<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::minus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::minus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = false;

template <class _Tp>
inline constexpr bool is_commutative_v<::cuda::std::minus<>, _Tp> =
  is_commutative_v<::cuda::std::minus<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::divides<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::divides<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = false;

template <class _Tp>
inline constexpr bool is_commutative_v<::cuda::std::divides<>, _Tp> =
  is_commutative_v<::cuda::std::divides<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  is_commutative_v<::cuda::std::modulus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool is_commutative_v<::cuda::std::modulus<>, _Tp> =
  is_commutative_v<::cuda::std::modulus<_Tp>, _Tp, void>;

/***********************************************************************************************************************
 * Identity Element
 **********************************************************************************************************************/

struct __no_identity_element
{};

template <class _Op, class _Tp, class Enable = void>
inline constexpr auto identity_element_v = __no_identity_element{};

// cuda::std::plus

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    _Tp{};

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<_Tp>>> =
    -_Tp{}; // -0.0 to preserve negative zero: -0.0 + (-0.0) = -0.0

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::std::plus<>,
                     _Tp,
                     ::cuda::std::enable_if_t<!::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    identity_element_v<::cuda::std::plus<_Tp>, _Tp>;

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto
  identity_element_v<::cuda::std::plus<_Tp>,
                     _Tp,
                     ::cuda::std::enable_if_t<::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    ::cuda::std::__fp_neg(_Tp{}); // -0.0 to preserve negative zero: -0.0 + (-0.0) = -0.0

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto
  identity_element_v<::cuda::std::plus<>,
                     _Tp,
                     ::cuda::std::enable_if_t<::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    identity_element_v<::cuda::std::plus<_Tp>, _Tp>;

// cuda::std::multiplies

template <class _Tp>
inline constexpr auto identity_element_v<::cuda::std::multiplies<_Tp>,
                                         _Tp,
                                         ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> = _Tp{1};

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::std::multiplies<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<_Tp>>> =
    _Tp{1};

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::std::multiplies<>,
                     _Tp,
                     ::cuda::std::enable_if_t<!::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    identity_element_v<::cuda::std::multiplies<_Tp>, _Tp>;

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto
  identity_element_v<::cuda::std::multiplies<_Tp>,
                     _Tp,
                     ::cuda::std::enable_if_t<::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    ::cuda::std::__fp_one<_Tp>();

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto
  identity_element_v<::cuda::std::multiplies<>,
                     _Tp,
                     ::cuda::std::enable_if_t<::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    identity_element_v<::cuda::std::multiplies<_Tp>, _Tp>;

// cuda::std::bit_and

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::std::bit_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    static_cast<_Tp>(~_Tp{});

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto identity_element_v<::cuda::std::bit_and<>, _Tp> =
  identity_element_v<::cuda::std::bit_and<_Tp>, _Tp>;

// cuda::std::bit_or

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::std::bit_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    _Tp{};

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto identity_element_v<::cuda::std::bit_or<>, _Tp> =
  identity_element_v<::cuda::std::bit_or<_Tp>, _Tp>;

// cuda::std::bit_xor

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::std::bit_xor<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    _Tp{};

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto identity_element_v<::cuda::std::bit_xor<>, _Tp> =
  identity_element_v<::cuda::std::bit_xor<_Tp>, _Tp>;

// cuda::std::logical_and

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::std::logical_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>> =
    true;

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto identity_element_v<::cuda::std::logical_and<>, _Tp> =
  identity_element_v<::cuda::std::logical_and<_Tp>, _Tp>;

// cuda::std::logical_or

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::std::logical_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>> =
    false;

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto identity_element_v<::cuda::std::logical_or<>, _Tp> =
  identity_element_v<::cuda::std::logical_or<_Tp>, _Tp>;

// cuda::minimum

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    ::cuda::std::numeric_limits<_Tp>::max();

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<_Tp>>> =
    ::cuda::std::numeric_limits<_Tp>::infinity();

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::minimum<>,
                     _Tp,
                     ::cuda::std::enable_if_t<!::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    identity_element_v<::cuda::minimum<_Tp>, _Tp>;

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto
  identity_element_v<::cuda::minimum<_Tp>,
                     _Tp,
                     ::cuda::std::enable_if_t<::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    ::cuda::std::numeric_limits<_Tp>::infinity();

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto
  identity_element_v<::cuda::minimum<>,
                     _Tp,
                     ::cuda::std::enable_if_t<::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    identity_element_v<::cuda::minimum<_Tp>, _Tp>;

// cuda::maximum

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    ::cuda::std::numeric_limits<_Tp>::lowest();

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<_Tp>>> =
    -cuda::std::numeric_limits<_Tp>::infinity();

template <class _Tp>
inline constexpr auto
  identity_element_v<::cuda::maximum<>,
                     _Tp,
                     ::cuda::std::enable_if_t<!::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    identity_element_v<::cuda::maximum<_Tp>, _Tp>;

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto
  identity_element_v<::cuda::maximum<_Tp>,
                     _Tp,
                     ::cuda::std::enable_if_t<::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    ::cuda::std::__fp_neg(::cuda::std::__fp_inf<_Tp>());

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto
  identity_element_v<::cuda::maximum<>,
                     _Tp,
                     ::cuda::std::enable_if_t<::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    identity_element_v<::cuda::maximum<_Tp>, _Tp>;

template <class _Op, class _Tp, class = void>
inline constexpr bool has_identity_element_v = false;

template <class _Op, class _Tp>
inline constexpr bool has_identity_element_v<
  _Op,
  _Tp,
  ::cuda::std::enable_if_t<!::cuda::std::is_same_v<decltype(identity_element_v<_Op, _Tp>), const __no_identity_element>>> =
  true;

/***********************************************************************************************************************
 * Absorbing Element
 **********************************************************************************************************************/

struct __no_absorbing_element
{};

template <class _Op, class _Tp, class Enable = void>
inline constexpr auto absorbing_element_v = __no_absorbing_element{};

// cuda::std::multiplies (no absorbing element for floating-point due to NaN, infinity, and -1.0 * +0.0 = -0.0  (!=
// +0.0))

template <class _Tp>
inline constexpr auto absorbing_element_v<::cuda::std::multiplies<_Tp>,
                                          _Tp,
                                          ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> = _Tp{};

template <class _Tp>
inline constexpr auto absorbing_element_v<::cuda::std::multiplies<>, _Tp> =
  absorbing_element_v<::cuda::std::multiplies<_Tp>, _Tp>;

// cuda::std::bit_and

template <class _Tp>
inline constexpr auto
  absorbing_element_v<::cuda::std::bit_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    _Tp{};

template <class _Tp>
inline constexpr auto absorbing_element_v<::cuda::std::bit_and<>, _Tp> =
  absorbing_element_v<::cuda::std::bit_and<_Tp>, _Tp>;

// cuda::std::bit_or

template <class _Tp>
inline constexpr auto
  absorbing_element_v<::cuda::std::bit_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    static_cast<_Tp>(~_Tp{});

template <class _Tp>
inline constexpr auto absorbing_element_v<::cuda::std::bit_or<>, _Tp> =
  absorbing_element_v<::cuda::std::bit_or<_Tp>, _Tp>;

// cuda::std::logical_and

template <class _Tp>
inline constexpr auto
  absorbing_element_v<::cuda::std::logical_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>> =
    false;

template <class _Tp>
inline constexpr auto absorbing_element_v<::cuda::std::logical_and<>, _Tp> =
  absorbing_element_v<::cuda::std::logical_and<_Tp>, _Tp>;

// cuda::std::logical_or

template <class _Tp>
inline constexpr auto
  absorbing_element_v<::cuda::std::logical_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>> =
    true;

template <class _Tp>
inline constexpr auto absorbing_element_v<::cuda::std::logical_or<>, _Tp> =
  absorbing_element_v<::cuda::std::logical_or<_Tp>, _Tp>;

// cuda::minimum

template <class _Tp>
inline constexpr auto
  absorbing_element_v<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    ::cuda::std::numeric_limits<_Tp>::lowest();

template <class _Tp>
inline constexpr auto
  absorbing_element_v<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<_Tp>>> =
    ::cuda::std::__fp_neg(::cuda::std::numeric_limits<_Tp>::infinity());

template <class _Tp>
inline constexpr auto
  absorbing_element_v<::cuda::minimum<>,
                      _Tp,
                      ::cuda::std::enable_if_t<!::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    absorbing_element_v<::cuda::minimum<_Tp>, _Tp>;

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto
  absorbing_element_v<::cuda::minimum<_Tp>,
                      _Tp,
                      ::cuda::std::enable_if_t<::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    ::cuda::std::__fp_neg(::cuda::std::__fp_inf<_Tp>());

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto
  absorbing_element_v<::cuda::minimum<>,
                      _Tp,
                      ::cuda::std::enable_if_t<::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    absorbing_element_v<::cuda::minimum<_Tp>, _Tp>;

// cuda::maximum

template <class _Tp>
inline constexpr auto
  absorbing_element_v<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    ::cuda::std::numeric_limits<_Tp>::max();

template <class _Tp>
inline constexpr auto
  absorbing_element_v<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<_Tp>>> =
    ::cuda::std::numeric_limits<_Tp>::infinity();

template <class _Tp>
inline constexpr auto
  absorbing_element_v<::cuda::maximum<>,
                      _Tp,
                      ::cuda::std::enable_if_t<!::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    absorbing_element_v<::cuda::maximum<_Tp>, _Tp>;

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto
  absorbing_element_v<::cuda::maximum<_Tp>,
                      _Tp,
                      ::cuda::std::enable_if_t<::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    ::cuda::std::__fp_inf<_Tp>();

template <class _Tp>
_CCCL_GLOBAL_CONSTANT auto
  absorbing_element_v<::cuda::maximum<>,
                      _Tp,
                      ::cuda::std::enable_if_t<::cuda::std::__is_cv_extended_floating_point_v<_Tp>>> =
    absorbing_element_v<::cuda::maximum<_Tp>, _Tp>;

template <class _Op, class _Tp, class = void>
inline constexpr bool has_absorbing_element_v = false;

template <class _Op, class _Tp>
inline constexpr bool has_absorbing_element_v<
  _Op,
  _Tp,
  ::cuda::std::enable_if_t<
    !::cuda::std::is_same_v<decltype(absorbing_element_v<_Op, _Tp>), const __no_absorbing_element>>> = true;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_FUNCTIONAL_OPERATOR_PROPERTIES_H
