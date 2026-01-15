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

#include <cuda/std/__cccl/execution_space.h>

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
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/arithmetic.h>
#include <cuda/std/__floating_point/constants.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

/***********************************************************************************************************************
 * Associativity
 **********************************************************************************************************************/

template <class _Op, class _Tp, class Enable = void>
struct __is_associative
{
  static_assert(::cuda::std::__always_false_v<_Op>,
                "operator_properties is not specialized for this operator and type combination");
};

// strictly speaking, plus (+) and multiply (*) are not associative because of overflow UB
template <class _Tp>
struct __is_associative<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
    : ::cuda::std::false_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::plus<>, _Tp> : __is_associative<::cuda::std::plus<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_associative<::cuda::std::multiplies<_Tp>,
                        _Tp,
                        ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>> : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::multiplies<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
    : ::cuda::std::false_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::multiplies<>, _Tp> : __is_associative<::cuda::std::multiplies<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_associative<::cuda::std::bit_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::bit_and<>, _Tp> : __is_associative<::cuda::std::bit_and<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_associative<::cuda::std::bit_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::bit_or<>, _Tp> : __is_associative<::cuda::std::bit_or<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_associative<::cuda::std::bit_xor<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::bit_xor<>, _Tp> : __is_associative<::cuda::std::bit_xor<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_associative<::cuda::std::logical_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::logical_and<>, _Tp> : __is_associative<::cuda::std::logical_and<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_associative<::cuda::std::logical_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::logical_or<>, _Tp> : __is_associative<::cuda::std::logical_or<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_associative<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_associative<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_associative<::cuda::minimum<>, _Tp> : __is_associative<::cuda::minimum<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_associative<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_associative<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_associative<::cuda::maximum<>, _Tp> : __is_associative<::cuda::maximum<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_associative<::cuda::std::minus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::false_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::minus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
    : ::cuda::std::false_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::minus<>, _Tp> : __is_associative<::cuda::std::minus<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_associative<::cuda::std::divides<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::false_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::divides<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
    : ::cuda::std::false_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::divides<>, _Tp> : __is_associative<::cuda::std::divides<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_associative<::cuda::std::modulus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::false_type
{};

template <class _Tp>
struct __is_associative<::cuda::std::modulus<>, _Tp> : __is_associative<::cuda::std::modulus<_Tp>, _Tp>
{};

template <class _Op, class _Tp>
struct is_associative : __is_associative<_Op, _Tp>
{};

template <class _Op, class _Tp>
inline constexpr bool is_associative_v = is_associative<_Op, _Tp>::value;

/***********************************************************************************************************************
 * Commutativity
 **********************************************************************************************************************/

template <class _Op, class _Tp, class Enable = void>
struct __is_commutative
{
  static_assert(::cuda::std::__always_false_v<_Op>,
                "operator_properties is not specialized for this operator and type combination");
};

template <class _Tp>
struct __is_commutative<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::plus<>, _Tp> : __is_commutative<::cuda::std::plus<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_commutative<::cuda::std::multiplies<_Tp>,
                        _Tp,
                        ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>> : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::multiplies<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::multiplies<>, _Tp> : __is_commutative<::cuda::std::multiplies<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_commutative<::cuda::std::bit_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::bit_and<>, _Tp> : __is_commutative<::cuda::std::bit_and<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_commutative<::cuda::std::bit_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::bit_or<>, _Tp> : __is_commutative<::cuda::std::bit_or<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_commutative<::cuda::std::bit_xor<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::bit_xor<>, _Tp> : __is_commutative<::cuda::std::bit_xor<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_commutative<::cuda::std::logical_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::logical_and<>, _Tp> : __is_commutative<::cuda::std::logical_and<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_commutative<::cuda::std::logical_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::logical_or<>, _Tp> : __is_commutative<::cuda::std::logical_or<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_commutative<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_commutative<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_commutative<::cuda::minimum<>, _Tp> : __is_commutative<::cuda::minimum<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_commutative<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_commutative<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
    : ::cuda::std::true_type
{};

template <class _Tp>
struct __is_commutative<::cuda::maximum<>, _Tp> : __is_commutative<::cuda::maximum<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_commutative<::cuda::std::minus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::false_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::minus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
    : ::cuda::std::false_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::minus<>, _Tp> : __is_commutative<::cuda::std::minus<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_commutative<::cuda::std::divides<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::false_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::divides<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
    : ::cuda::std::false_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::divides<>, _Tp> : __is_commutative<::cuda::std::divides<_Tp>, _Tp>
{};

template <class _Tp>
struct __is_commutative<::cuda::std::modulus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
    : ::cuda::std::false_type
{};

template <class _Tp>
struct __is_commutative<::cuda::std::modulus<>, _Tp> : __is_commutative<::cuda::std::modulus<_Tp>, _Tp>
{};

template <class _Op, class _Tp>
struct is_commutative : __is_commutative<_Op, _Tp>
{};

template <class _Op, class _Tp>
inline constexpr bool is_commutative_v = is_commutative<_Op, _Tp>::value;

/***********************************************************************************************************************
 * Element Existence
 **********************************************************************************************************************/

template <template <class...> class _Trait, class... _Tp>
_CCCL_CONCEPT element_exists = _CCCL_REQUIRES_EXPR((_Tp...))((_Trait<_Tp...>::value));

/***********************************************************************************************************************
 * Identity Element
 **********************************************************************************************************************/

template <class _Op, class _Tp, class Enable = void>
struct __identity_element
{};

template <class _Tp>
struct __identity_element<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
{
  static constexpr auto value = _Tp{};
};

template <class _Tp>
struct __identity_element<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
{
  static constexpr auto value = ::cuda::std::__fp_neg(_Tp{}); // -0.0 to preserve negative zero: -0.0 + (-0.0) = -0.0
};

template <class _Tp>
struct __identity_element<::cuda::std::plus<>, _Tp> : __identity_element<::cuda::std::plus<_Tp>, _Tp>
{};

template <class _Tp>
struct __identity_element<::cuda::std::multiplies<_Tp>,
                          _Tp,
                          ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
{
  static constexpr auto value = _Tp{1};
};

template <class _Tp>
struct __identity_element<::cuda::std::multiplies<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
{
  static constexpr auto value = ::cuda::std::__fp_one<_Tp>();
};

template <class _Tp>
struct __identity_element<::cuda::std::multiplies<>, _Tp> : __identity_element<::cuda::std::multiplies<_Tp>, _Tp>
{};

template <class _Tp>
struct __identity_element<::cuda::std::bit_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
{
  static constexpr auto value = static_cast<_Tp>(~_Tp{});
};

template <class _Tp>
struct __identity_element<::cuda::std::bit_and<>, _Tp> : __identity_element<::cuda::std::bit_and<_Tp>, _Tp>
{};

template <class _Tp>
struct __identity_element<::cuda::std::bit_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
{
  static constexpr auto value = _Tp{};
};

template <class _Tp>
struct __identity_element<::cuda::std::bit_or<>, _Tp> : __identity_element<::cuda::std::bit_or<_Tp>, _Tp>
{};

template <class _Tp>
struct __identity_element<::cuda::std::bit_xor<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
{
  static constexpr auto value = _Tp{};
};

template <class _Tp>
struct __identity_element<::cuda::std::bit_xor<>, _Tp> : __identity_element<::cuda::std::bit_xor<_Tp>, _Tp>
{};

template <class _Tp>
struct __identity_element<::cuda::std::logical_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>>
{
  static constexpr auto value = true;
};

template <class _Tp>
struct __identity_element<::cuda::std::logical_and<>, _Tp> : __identity_element<::cuda::std::logical_and<_Tp>, _Tp>
{};

template <class _Tp>
struct __identity_element<::cuda::std::logical_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>>
{
  static constexpr auto value = false;
};

template <class _Tp>
struct __identity_element<::cuda::std::logical_or<>, _Tp> : __identity_element<::cuda::std::logical_or<_Tp>, _Tp>
{};

template <class _Tp>
struct __identity_element<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
{
  static constexpr auto value = ::cuda::std::numeric_limits<_Tp>::max();
};

template <class _Tp>
struct __identity_element<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
{
  static constexpr auto value = ::cuda::std::numeric_limits<_Tp>::infinity();
};

template <class _Tp>
struct __identity_element<::cuda::minimum<>, _Tp> : __identity_element<::cuda::minimum<_Tp>, _Tp>
{};

template <class _Tp>
struct __identity_element<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
{
  static constexpr auto value = ::cuda::std::numeric_limits<_Tp>::lowest();
};

template <class _Tp>
struct __identity_element<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
{
  static constexpr auto value = ::cuda::std::__fp_neg(::cuda::std::__fp_inf<_Tp>());
};

template <class _Tp>
struct __identity_element<::cuda::maximum<>, _Tp> : __identity_element<::cuda::maximum<_Tp>, _Tp>
{};

template <class _Op, class _Tp>
struct identity_element : __identity_element<_Op, _Tp>
{};

template <class _Op, class _Tp>
_CCCL_GLOBAL_VARIABLE constexpr auto identity_element_v = identity_element<_Op, _Tp>::value;

template <class _Op, class _Tp>
inline constexpr bool has_identity_element_v = element_exists<identity_element, _Op, _Tp>;

/***********************************************************************************************************************
 * Absorbing Element
 **********************************************************************************************************************/

template <class _Op, class _Tp, class Enable = void>
struct __absorbing_element
{};

template <class _Tp>
struct __absorbing_element<::cuda::std::multiplies<_Tp>,
                           _Tp,
                           ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
{
  static constexpr auto value = _Tp{};
};

// no absorbing element for floating-point due to NaN, infinity, and -1.0 * +0.0 = -0.0  (!= +0.0)

template <class _Tp>
struct __absorbing_element<::cuda::std::multiplies<>, _Tp> : __absorbing_element<::cuda::std::multiplies<_Tp>, _Tp>
{};

template <class _Tp>
struct __absorbing_element<::cuda::std::bit_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
{
  static constexpr auto value = _Tp{};
};

template <class _Tp>
struct __absorbing_element<::cuda::std::bit_and<>, _Tp> : __absorbing_element<::cuda::std::bit_and<_Tp>, _Tp>
{};

template <class _Tp>
struct __absorbing_element<::cuda::std::bit_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
{
  static constexpr auto value = static_cast<_Tp>(~_Tp{});
};

template <class _Tp>
struct __absorbing_element<::cuda::std::bit_or<>, _Tp> : __absorbing_element<::cuda::std::bit_or<_Tp>, _Tp>
{};

template <class _Tp>
struct __absorbing_element<::cuda::std::logical_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>>
{
  static constexpr auto value = false;
};

template <class _Tp>
struct __absorbing_element<::cuda::std::logical_and<>, _Tp> : __absorbing_element<::cuda::std::logical_and<_Tp>, _Tp>
{};

template <class _Tp>
struct __absorbing_element<::cuda::std::logical_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>>
{
  static constexpr auto value = true;
};

template <class _Tp>
struct __absorbing_element<::cuda::std::logical_or<>, _Tp> : __absorbing_element<::cuda::std::logical_or<_Tp>, _Tp>
{};

template <class _Tp>
struct __absorbing_element<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
{
  static constexpr auto value = ::cuda::std::numeric_limits<_Tp>::lowest();
};

template <class _Tp>
struct __absorbing_element<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
{
  static constexpr auto value = ::cuda::std::__fp_neg(::cuda::std::numeric_limits<_Tp>::infinity());
};

template <class _Tp>
struct __absorbing_element<::cuda::minimum<>, _Tp> : __absorbing_element<::cuda::minimum<_Tp>, _Tp>
{};

template <class _Tp>
struct __absorbing_element<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_integer_v<_Tp>>>
{
  static constexpr auto value = ::cuda::std::numeric_limits<_Tp>::max();
};

template <class _Tp>
struct __absorbing_element<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>>
{
  static constexpr auto value = ::cuda::std::numeric_limits<_Tp>::infinity();
};

template <class _Tp>
struct __absorbing_element<::cuda::maximum<>, _Tp> : __absorbing_element<::cuda::maximum<_Tp>, _Tp>
{};

template <class _Op, class _Tp>
struct absorbing_element : __absorbing_element<_Op, _Tp>
{};

template <class _Op, class _Tp>
_CCCL_GLOBAL_VARIABLE constexpr auto absorbing_element_v = absorbing_element<_Op, _Tp>::value;

template <class _Op, class _Tp>
inline constexpr bool has_absorbing_element_v = element_exists<absorbing_element, _Op, _Tp>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_FUNCTIONAL_OPERATOR_PROPERTIES_H
