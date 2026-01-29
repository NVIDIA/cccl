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
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>

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
inline constexpr bool __is_associative_v = __is_associative_static_assert<_Op>();

// strictly speaking, plus (+) and multiply (*) are not associative because of overflow UB
template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = false;

template <class _Tp>
inline constexpr bool __is_associative_v<::cuda::std::plus<>, _Tp> =
  __is_associative_v<::cuda::std::plus<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool __is_associative_v<::cuda::std::multiplies<_Tp>,
                                         _Tp,
                                         ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> = true;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::std::multiplies<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool __is_associative_v<::cuda::std::multiplies<>, _Tp> =
  __is_associative_v<::cuda::std::multiplies<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::std::bit_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool __is_associative_v<::cuda::std::bit_and<>, _Tp> =
  __is_associative_v<::cuda::std::bit_and<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::std::bit_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool __is_associative_v<::cuda::std::bit_or<>, _Tp> =
  __is_associative_v<::cuda::std::bit_or<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::std::bit_xor<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool __is_associative_v<::cuda::std::bit_xor<>, _Tp> =
  __is_associative_v<::cuda::std::bit_xor<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::std::logical_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>> =
    true;

template <class _Tp>
inline constexpr bool __is_associative_v<::cuda::std::logical_and<>, _Tp> =
  __is_associative_v<::cuda::std::logical_and<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::std::logical_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>> =
    true;

template <class _Tp>
inline constexpr bool __is_associative_v<::cuda::std::logical_or<>, _Tp> =
  __is_associative_v<::cuda::std::logical_or<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = true;

template <class _Tp>
inline constexpr bool __is_associative_v<::cuda::minimum<>, _Tp> = __is_associative_v<::cuda::minimum<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = true;

template <class _Tp>
inline constexpr bool __is_associative_v<::cuda::maximum<>, _Tp> = __is_associative_v<::cuda::maximum<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::std::minus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::std::minus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = false;

template <class _Tp>
inline constexpr bool __is_associative_v<::cuda::std::minus<>, _Tp> =
  __is_associative_v<::cuda::std::minus<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::std::divides<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::std::divides<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool __is_associative_v<::cuda::std::divides<>, _Tp> =
  __is_associative_v<::cuda::std::divides<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_associative_v<::cuda::std::modulus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool __is_associative_v<::cuda::std::modulus<>, _Tp> =
  __is_associative_v<::cuda::std::modulus<_Tp>, _Tp, void>;

template <class _Op, class _Tp>
inline constexpr bool is_associative_v = __is_associative_v<_Op, ::cuda::std::remove_cv_t<_Tp>>;

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
inline constexpr bool __is_commutative_v = ::cuda::__is_commutative_static_assert<_Op>();

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::std::plus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = true;

template <class _Tp>
inline constexpr bool __is_commutative_v<::cuda::std::plus<>, _Tp> =
  __is_commutative_v<::cuda::std::plus<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool __is_commutative_v<::cuda::std::multiplies<_Tp>,
                                         _Tp,
                                         ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> = true;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::std::multiplies<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool __is_commutative_v<::cuda::std::multiplies<>, _Tp> =
  __is_commutative_v<::cuda::std::multiplies<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::std::bit_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool __is_commutative_v<::cuda::std::bit_and<>, _Tp> =
  __is_commutative_v<::cuda::std::bit_and<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::std::bit_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool __is_commutative_v<::cuda::std::bit_or<>, _Tp> =
  __is_commutative_v<::cuda::std::bit_or<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::std::bit_xor<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool __is_commutative_v<::cuda::std::bit_xor<>, _Tp> =
  __is_commutative_v<::cuda::std::bit_xor<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::std::logical_and<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>> =
    true;

template <class _Tp>
inline constexpr bool __is_commutative_v<::cuda::std::logical_and<>, _Tp> =
  __is_commutative_v<::cuda::std::logical_and<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::std::logical_or<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Tp, bool>>> =
    true;

template <class _Tp>
inline constexpr bool __is_commutative_v<::cuda::std::logical_or<>, _Tp> =
  __is_commutative_v<::cuda::std::logical_or<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::minimum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = true;

template <class _Tp>
inline constexpr bool __is_commutative_v<::cuda::minimum<>, _Tp> = __is_commutative_v<::cuda::minimum<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    true;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::maximum<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = true;

template <class _Tp>
inline constexpr bool __is_commutative_v<::cuda::maximum<>, _Tp> = __is_commutative_v<::cuda::maximum<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::std::minus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::std::minus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> = false;

template <class _Tp>
inline constexpr bool __is_commutative_v<::cuda::std::minus<>, _Tp> =
  __is_commutative_v<::cuda::std::minus<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::std::divides<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::std::divides<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::is_floating_point_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool __is_commutative_v<::cuda::std::divides<>, _Tp> =
  __is_commutative_v<::cuda::std::divides<_Tp>, _Tp, void>;

template <class _Tp>
inline constexpr bool
  __is_commutative_v<::cuda::std::modulus<_Tp>, _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_integer_v<_Tp>>> =
    false;

template <class _Tp>
inline constexpr bool __is_commutative_v<::cuda::std::modulus<>, _Tp> =
  __is_commutative_v<::cuda::std::modulus<_Tp>, _Tp, void>;

template <class _Op, class _Tp>
inline constexpr bool is_commutative_v = __is_commutative_v<_Op, ::cuda::std::remove_cv_t<_Tp>>;

/***********************************************************************************************************************
 * Internal helpers
 **********************************************************************************************************************/

template <typename>
inline constexpr bool __is_cuda_std_plus_v = false;

template <typename... _Tp>
inline constexpr bool __is_cuda_std_plus_v<::cuda::std::plus<_Tp...>> = true;

template <typename>
inline constexpr bool __is_cuda_std_multiplies_v = false;

template <typename... _Tp>
inline constexpr bool __is_cuda_std_multiplies_v<::cuda::std::multiplies<_Tp...>> = true;

template <typename>
inline constexpr bool __is_cuda_std_bit_and_v = false;

template <typename... _Tp>
inline constexpr bool __is_cuda_std_bit_and_v<::cuda::std::bit_and<_Tp...>> = true;

template <typename>
inline constexpr bool __is_cuda_std_bit_or_v = false;

template <typename... _Tp>
inline constexpr bool __is_cuda_std_bit_or_v<::cuda::std::bit_or<_Tp...>> = true;

template <typename>
inline constexpr bool __is_cuda_std_bit_xor_v = false;

template <typename... _Tp>
inline constexpr bool __is_cuda_std_bit_xor_v<::cuda::std::bit_xor<_Tp...>> = true;

template <typename>
inline constexpr bool __is_cuda_std_logical_and_v = false;

template <typename... _Tp>
inline constexpr bool __is_cuda_std_logical_and_v<::cuda::std::logical_and<_Tp...>> = true;

template <typename>
inline constexpr bool __is_cuda_std_logical_or_v = false;

template <typename... _Tp>
inline constexpr bool __is_cuda_std_logical_or_v<::cuda::std::logical_or<_Tp...>> = true;

template <typename>
inline constexpr bool __is_cuda_minimum_v = false;

template <typename... _Tp>
inline constexpr bool __is_cuda_minimum_v<::cuda::minimum<_Tp...>> = true;

template <typename>
inline constexpr bool __is_cuda_maximum_v = false;

template <typename... _Tp>
inline constexpr bool __is_cuda_maximum_v<::cuda::maximum<_Tp...>> = true;

/***********************************************************************************************************************
 * Identity Element
 **********************************************************************************************************************/

struct __no_identity_element
{};

template <class _Op, class _Tp>
[[nodiscard]] _CCCL_API constexpr auto identity_element() noexcept
{
  using _Up = ::cuda::std::remove_cv_t<_Tp>;
  if constexpr (__is_cuda_std_plus_v<_Op>)
  {
    if constexpr (::cuda::std::__cccl_is_integer_v<_Up>)
    {
      return _Up{};
    }
    else if constexpr (::cuda::is_floating_point_v<_Up>)
    {
      return ::cuda::std::__fp_neg(_Up{}); // -0.0 to preserve negative zero: -0.0 + (-0.0) = -0.0
    }
    else
    {
      return __no_identity_element{};
    }
  }
  else if constexpr (__is_cuda_std_multiplies_v<_Op>)
  {
    if constexpr (::cuda::std::__cccl_is_integer_v<_Up> || ::cuda::std::is_floating_point_v<_Up>)
    {
      return _Up{1};
    }
    else if constexpr (::cuda::std::__is_extended_floating_point_v<_Up>)
    {
      return ::cuda::std::__fp_one<_Up>();
    }
    else
    {
      return __no_identity_element{};
    }
  }
  else if constexpr (__is_cuda_std_bit_and_v<_Op>)
  {
    if constexpr (::cuda::std::__cccl_is_integer_v<_Up>)
    {
      return static_cast<_Up>(~_Up{});
    }
    else
    {
      return __no_identity_element{};
    }
  }
  else if constexpr (__is_cuda_std_bit_or_v<_Op>)
  {
    if constexpr (::cuda::std::__cccl_is_integer_v<_Up>)
    {
      return _Up{};
    }
    else
    {
      return __no_identity_element{};
    }
  }
  else if constexpr (__is_cuda_std_bit_xor_v<_Op>)
  {
    if constexpr (::cuda::std::__cccl_is_integer_v<_Up>)
    {
      return _Up{};
    }
    else
    {
      return __no_identity_element{};
    }
  }
  else if constexpr (__is_cuda_std_logical_and_v<_Op>)
  {
    if constexpr (::cuda::std::is_same_v<_Up, bool>)
    {
      return true;
    }
    else
    {
      return __no_identity_element{};
    }
  }
  else if constexpr (__is_cuda_std_logical_or_v<_Op>)
  {
    if constexpr (::cuda::std::is_same_v<_Up, bool>)
    {
      return false;
    }
    else
    {
      return __no_identity_element{};
    }
  }
  else if constexpr (__is_cuda_minimum_v<_Op>)
  {
    if constexpr (::cuda::std::__cccl_is_integer_v<_Up>)
    {
      return ::cuda::std::numeric_limits<_Up>::max();
    }
    else if constexpr (::cuda::is_floating_point_v<_Up>)
    {
      return ::cuda::std::numeric_limits<_Up>::infinity();
    }
    else
    {
      return __no_identity_element{};
    }
  }
  else if constexpr (__is_cuda_maximum_v<_Op>)
  {
    if constexpr (::cuda::std::__cccl_is_integer_v<_Up>)
    {
      return ::cuda::std::numeric_limits<_Up>::lowest();
    }
    else if constexpr (::cuda::is_floating_point_v<_Up>)
    {
      return ::cuda::std::__fp_neg(::cuda::std::__fp_inf<_Up>());
    }
    else
    {
      return __no_identity_element{};
    }
  }
  else
  {
    return __no_identity_element{};
  }
}

template <class _Op, class _Tp, class = void>
inline constexpr bool has_identity_element_v = false;

template <class _Op, class _Tp>
inline constexpr bool has_identity_element_v<
  _Op,
  _Tp,
  ::cuda::std::enable_if_t<!::cuda::std::is_same_v<decltype(identity_element<_Op, _Tp>()), __no_identity_element>>> =
  true;

/***********************************************************************************************************************
 * Absorbing Element
 **********************************************************************************************************************/

struct __no_absorbing_element
{};

template <class _Op, class _Tp>
[[nodiscard]] _CCCL_API constexpr auto absorbing_element() noexcept
{
  using _Up = ::cuda::std::remove_cv_t<_Tp>;
  if constexpr (__is_cuda_std_multiplies_v<_Op>)
  {
    // no absorbing element for floating-point due to NaN, infinity, and -1.0 * +0.0 = -0.0 (!= +0.0)
    if constexpr (::cuda::std::__cccl_is_integer_v<_Up>)
    {
      return _Up{};
    }
    else
    {
      return __no_absorbing_element{};
    }
  }
  else if constexpr (__is_cuda_std_bit_and_v<_Op>)
  {
    if constexpr (::cuda::std::__cccl_is_integer_v<_Up>)
    {
      return _Up{};
    }
    else
    {
      return __no_absorbing_element{};
    }
  }
  else if constexpr (__is_cuda_std_bit_or_v<_Op>)
  {
    if constexpr (::cuda::std::__cccl_is_integer_v<_Up>)
    {
      return static_cast<_Up>(~_Up{});
    }
    else
    {
      return __no_absorbing_element{};
    }
  }
  else if constexpr (__is_cuda_std_logical_and_v<_Op>)
  {
    if constexpr (::cuda::std::is_same_v<_Up, bool>)
    {
      return false;
    }
    else
    {
      return __no_absorbing_element{};
    }
  }
  else if constexpr (__is_cuda_std_logical_or_v<_Op>)
  {
    if constexpr (::cuda::std::is_same_v<_Up, bool>)
    {
      return true;
    }
    else
    {
      return __no_absorbing_element{};
    }
  }
  else if constexpr (__is_cuda_minimum_v<_Op>)
  {
    if constexpr (::cuda::std::__cccl_is_integer_v<_Up>)
    {
      return ::cuda::std::numeric_limits<_Up>::lowest();
    }
    else if constexpr (::cuda::is_floating_point_v<_Up>)
    {
      return ::cuda::std::__fp_neg(::cuda::std::__fp_inf<_Up>());
    }
    else
    {
      return __no_absorbing_element{};
    }
  }
  else if constexpr (__is_cuda_maximum_v<_Op>)
  {
    if constexpr (::cuda::std::__cccl_is_integer_v<_Up>)
    {
      return ::cuda::std::numeric_limits<_Up>::max();
    }
    else if constexpr (::cuda::is_floating_point_v<_Up>)
    {
      return ::cuda::std::__fp_inf<_Up>();
    }
    else
    {
      return __no_absorbing_element{};
    }
  }
  else
  {
    return __no_absorbing_element{};
  }
}

template <class _Op, class _Tp, class = void>
inline constexpr bool has_absorbing_element_v = false;

template <class _Op, class _Tp>
inline constexpr bool has_absorbing_element_v<
  _Op,
  _Tp,
  ::cuda::std::enable_if_t<!::cuda::std::is_same_v<decltype(absorbing_element<_Op, _Tp>()), __no_absorbing_element>>> =
  true;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_FUNCTIONAL_OPERATOR_PROPERTIES_H
