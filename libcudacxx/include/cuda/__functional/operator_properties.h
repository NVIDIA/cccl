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
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

/// @brief Trait class describing algebraic properties of binary operators.
///
/// @tparam Op The binary operator type (e.g., cuda::std::plus<T>)
/// @tparam T The value type the operator operates on
///
/// Properties:
/// - `is_commutative` - whether `op(a, b) == op(b, a)`
/// - `is_associative` - whether `op(op(a, b), c) == op(a, op(b, c))` (exact/bit-accurate)
/// - `has_identity_element` - whether an identity element exists
/// - `identity_element()` - the value `e` where `op(a, e) == op(e, a) == a`
/// - `has_absorbing_element` - whether an absorbing element exists
/// - `absorbing_element()` - the value `z` where `op(a, z) == op(z, a) == z`
///
/// Users can specialize this template for custom operators and types.
template <class Op, class T = void>
struct operator_properties
{
  static_assert(::cuda::std::__always_false_v<Op>,
                "operator_properties is not specialized for this operator and type combination");
};

/***********************************************************************************************************************
 * cuda::std::plus specializations
 **********************************************************************************************************************/

template <class T>
struct operator_properties<::cuda::std::plus<T>, T>
{
  static constexpr bool is_commutative        = true;
  static constexpr bool is_associative        = ::cuda::std::is_integral_v<T>;
  static constexpr bool has_identity_element  = true;
  static constexpr bool has_absorbing_element = false;

  [[nodiscard]] _CCCL_API static constexpr T identity_element() noexcept
  {
    return T{};
  }

  [[nodiscard]] _CCCL_API static constexpr T absorbing_element() noexcept
  {
    static_assert(has_absorbing_element, "cuda::std::plus does not have an absorbing element");
    return T{};
  }
};

template <class T>
struct operator_properties<::cuda::std::plus<>, T> : operator_properties<::cuda::std::plus<T>, T>
{};

/***********************************************************************************************************************
 * cuda::std::multiplies specializations
 **********************************************************************************************************************/

template <class T>
struct operator_properties<::cuda::std::multiplies<T>, T>
{
  static constexpr bool is_commutative        = true;
  static constexpr bool is_associative        = ::cuda::std::is_integral_v<T>;
  static constexpr bool has_identity_element  = true;
  static constexpr bool has_absorbing_element = true;

  [[nodiscard]] _CCCL_API static constexpr T identity_element() noexcept
  {
    return T{1};
  }

  [[nodiscard]] _CCCL_API static constexpr T absorbing_element() noexcept
  {
    return T{};
  }
};

template <class T>
struct operator_properties<::cuda::std::multiplies<>, T> : operator_properties<::cuda::std::multiplies<T>, T>
{};

/***********************************************************************************************************************
 * cuda::std::bit_and specializations
 **********************************************************************************************************************/

template <class T>
struct operator_properties<::cuda::std::bit_and<T>, T>
{
  static_assert(::cuda::std::__cccl_is_unsigned_integer_v<T>, "cuda::std::bit_and requires an unsigned integer type");

  static constexpr bool is_commutative        = true;
  static constexpr bool is_associative        = true;
  static constexpr bool has_identity_element  = true;
  static constexpr bool has_absorbing_element = true;

  [[nodiscard]] _CCCL_API static constexpr T identity_element() noexcept
  {
    return static_cast<T>(~T{});
  }

  [[nodiscard]] _CCCL_API static constexpr T absorbing_element() noexcept
  {
    return T{};
  }
};

template <class T>
struct operator_properties<::cuda::std::bit_and<>, T> : operator_properties<::cuda::std::bit_and<T>, T>
{};

/***********************************************************************************************************************
 * cuda::std::bit_or specializations
 **********************************************************************************************************************/

template <class T>
struct operator_properties<::cuda::std::bit_or<T>, T>
{
  static_assert(::cuda::std::__cccl_is_unsigned_integer_v<T>, "cuda::std::bit_or requires an unsigned integer type");

  static constexpr bool is_commutative        = true;
  static constexpr bool is_associative        = true;
  static constexpr bool has_identity_element  = true;
  static constexpr bool has_absorbing_element = true;

  [[nodiscard]] _CCCL_API static constexpr T identity_element() noexcept
  {
    return T{};
  }

  [[nodiscard]] _CCCL_API static constexpr T absorbing_element() noexcept
  {
    return static_cast<T>(~T{});
  }
};

template <class T>
struct operator_properties<::cuda::std::bit_or<>, T> : operator_properties<::cuda::std::bit_or<T>, T>
{};

/***********************************************************************************************************************
 * cuda::std::bit_xor specializations
 **********************************************************************************************************************/

template <class T>
struct operator_properties<::cuda::std::bit_xor<T>, T>
{
  static_assert(::cuda::std::__cccl_is_unsigned_integer_v<T>, "cuda::std::bit_xor requires an unsigned integer type");

  static constexpr bool is_commutative        = true;
  static constexpr bool is_associative        = true;
  static constexpr bool has_identity_element  = true;
  static constexpr bool has_absorbing_element = false;

  [[nodiscard]] _CCCL_API static constexpr T identity_element() noexcept
  {
    return T{};
  }

  [[nodiscard]] _CCCL_API static constexpr T absorbing_element() noexcept
  {
    static_assert(has_absorbing_element, "cuda::std::bit_xor does not have an absorbing element");
    return T{};
  }
};

template <class T>
struct operator_properties<::cuda::std::bit_xor<>, T> : operator_properties<::cuda::std::bit_xor<T>, T>
{};

/***********************************************************************************************************************
 * cuda::std::logical_and specializations
 **********************************************************************************************************************/

template <class T>
struct operator_properties<::cuda::std::logical_and<T>, T>
{
  static_assert(::cuda::std::is_same_v<T, bool>, "cuda::std::logical_and requires a boolean type");

  static constexpr bool is_commutative        = true;
  static constexpr bool is_associative        = true;
  static constexpr bool has_identity_element  = true;
  static constexpr bool has_absorbing_element = true;

  [[nodiscard]] _CCCL_API static constexpr T identity_element() noexcept
  {
    return static_cast<T>(true);
  }

  [[nodiscard]] _CCCL_API static constexpr T absorbing_element() noexcept
  {
    return static_cast<T>(false);
  }
};

template <class T>
struct operator_properties<::cuda::std::logical_and<>, T> : operator_properties<::cuda::std::logical_and<T>, T>
{};

/***********************************************************************************************************************
 * cuda::std::logical_or specializations
 **********************************************************************************************************************/

template <class T>
struct operator_properties<::cuda::std::logical_or<T>, T>
{
  static_assert(::cuda::std::is_same_v<T, bool>, "cuda::std::logical_and requires a boolean type");

  static constexpr bool is_commutative        = true;
  static constexpr bool is_associative        = true;
  static constexpr bool has_identity_element  = true;
  static constexpr bool has_absorbing_element = true;

  [[nodiscard]] _CCCL_API static constexpr T identity_element() noexcept
  {
    return static_cast<T>(false);
  }

  [[nodiscard]] _CCCL_API static constexpr T absorbing_element() noexcept
  {
    return static_cast<T>(true);
  }
};

template <class T>
struct operator_properties<::cuda::std::logical_or<>, T> : operator_properties<::cuda::std::logical_or<T>, T>
{};

/***********************************************************************************************************************
 * cuda::minimum specializations
 **********************************************************************************************************************/

template <class T>
struct operator_properties<::cuda::minimum<T>, T>
{
  static constexpr bool is_commutative        = true;
  static constexpr bool is_associative        = true;
  static constexpr bool has_identity_element  = true;
  static constexpr bool has_absorbing_element = true;

  [[nodiscard]] _CCCL_API static constexpr T identity_element() noexcept
  {
    return ::cuda::std::numeric_limits<T>::max();
  }

  [[nodiscard]] _CCCL_API static constexpr T absorbing_element() noexcept
  {
    return ::cuda::std::numeric_limits<T>::lowest();
  }
};

template <class T>
struct operator_properties<::cuda::minimum<>, T> : operator_properties<::cuda::minimum<T>, T>
{};

/***********************************************************************************************************************
 * cuda::maximum specializations
 **********************************************************************************************************************/

template <class T>
struct operator_properties<::cuda::maximum<T>, T>
{
  static constexpr bool is_commutative        = true;
  static constexpr bool is_associative        = true;
  static constexpr bool has_identity_element  = true;
  static constexpr bool has_absorbing_element = true;

  [[nodiscard]] _CCCL_API static constexpr T identity_element() noexcept
  {
    return ::cuda::std::numeric_limits<T>::lowest();
  }

  [[nodiscard]] _CCCL_API static constexpr T absorbing_element() noexcept
  {
    return ::cuda::std::numeric_limits<T>::max();
  }
};

template <class T>
struct operator_properties<::cuda::maximum<>, T> : operator_properties<::cuda::maximum<T>, T>
{};

/***********************************************************************************************************************
 * cuda::std::minus specializations (not commutative, not associative, no identity/absorbing)
 **********************************************************************************************************************/

template <class T>
struct operator_properties<::cuda::std::minus<T>, T>
{
  static constexpr bool is_commutative        = false;
  static constexpr bool is_associative        = false;
  static constexpr bool has_identity_element  = false;
  static constexpr bool has_absorbing_element = false;

  [[nodiscard]] _CCCL_API static constexpr T identity_element() noexcept
  {
    static_assert(has_identity_element, "cuda::std::minus does not have an identity element");
    return T{};
  }

  [[nodiscard]] _CCCL_API static constexpr T absorbing_element() noexcept
  {
    static_assert(has_absorbing_element, "cuda::std::minus does not have an absorbing element");
    return T{};
  }
};

template <class T>
struct operator_properties<::cuda::std::minus<>, T> : operator_properties<::cuda::std::minus<T>, T>
{};

/***********************************************************************************************************************
 * cuda::std::divides specializations (not commutative, not associative, no identity/absorbing)
 **********************************************************************************************************************/

template <class T>
struct operator_properties<::cuda::std::divides<T>, T>
{
  static constexpr bool is_commutative        = false;
  static constexpr bool is_associative        = false;
  static constexpr bool has_identity_element  = false;
  static constexpr bool has_absorbing_element = false;

  [[nodiscard]] _CCCL_API static constexpr T identity_element() noexcept
  {
    static_assert(has_identity_element, "cuda::std::divides does not have an identity element");
    return T{};
  }

  [[nodiscard]] _CCCL_API static constexpr T absorbing_element() noexcept
  {
    static_assert(has_absorbing_element, "cuda::std::divides does not have an absorbing element");
    return T{};
  }
};

template <class T>
struct operator_properties<::cuda::std::divides<>, T> : operator_properties<::cuda::std::divides<T>, T>
{};

/***********************************************************************************************************************
 * cuda::std::modulus specializations (not commutative, not associative, no identity/absorbing)
 **********************************************************************************************************************/

template <class T>
struct operator_properties<::cuda::std::modulus<T>, T>
{
  static constexpr bool is_commutative        = false;
  static constexpr bool is_associative        = false;
  static constexpr bool has_identity_element  = false;
  static constexpr bool has_absorbing_element = false;

  [[nodiscard]] _CCCL_API static constexpr T identity_element() noexcept
  {
    static_assert(has_identity_element, "cuda::std::modulus does not have an identity element");
    return T{};
  }

  [[nodiscard]] _CCCL_API static constexpr T absorbing_element() noexcept
  {
    static_assert(has_absorbing_element, "cuda::std::modulus does not have an absorbing element");
    return T{};
  }
};

template <class T>
struct operator_properties<::cuda::std::modulus<>, T> : operator_properties<::cuda::std::modulus<T>, T>
{};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_FUNCTIONAL_OPERATOR_PROPERTIES_H
