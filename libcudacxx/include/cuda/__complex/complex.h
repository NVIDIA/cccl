//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___COMPLEX_COMPLEX_H
#define _CUDA___COMPLEX_COMPLEX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__complex/get_real_imag.h>
#include <cuda/__complex/traits.h>
#include <cuda/__fwd/complex.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__floating_point/conversion_rank_order.h>
#include <cuda/std/__floating_point/traits.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp, class _Up>
_CCCL_CONCEPT __complex_can_implicitly_convert_float = _CCCL_REQUIRES_EXPR((_Tp, _Up))(
  requires(__is_any_complex_v<_Up>),
  requires(::cuda::std::__is_fp_v<_Tp>),
  requires(::cuda::std::__is_fp_v<typename _Up::value_type>),
  requires(::cuda::std::__fp_is_implicit_conversion_v<typename _Up::value_type, _Tp>));

template <class _Tp, class _Up>
_CCCL_CONCEPT __complex_can_implicitly_convert_int = _CCCL_REQUIRES_EXPR((_Tp, _Up)) //
  (requires(__is_any_complex_v<_Up>),
   requires((!::cuda::std::__is_fp_v<_Tp> || !::cuda::std::__is_fp_v<typename _Up::value_type>) ));

template <class _Tp, class _Up>
_CCCL_CONCEPT __complex_can_explicitly_convert = _CCCL_REQUIRES_EXPR((_Tp, _Up))(
  requires(__is_any_complex_v<_Up>),
  requires(::cuda::std::__is_fp_v<_Tp>),
  requires(::cuda::std::__is_fp_v<typename _Up::value_type>),
  requires(::cuda::std::__fp_is_explicit_conversion_v<typename _Up::value_type, _Tp>));

//! @brief Class representing a complex number.
//!
//! @tparam _Tp The type of the real and imaginary parts. Must be a NumericType.
//!
//! @note Compared to std::complex, this type is trivially default constructible and is aligned to 2 * sizeof(_Tp).
template <class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_ALIGNAS(2 * sizeof(_Tp)) complex
{
  static_assert(!::cuda::std::__is_ext_nv_fp_v<_Tp>, "Extended NVIDIA floating point types are not supported yet.");

  _Tp __re_; //!< The real part of the complex number.
  _Tp __im_; //!< The imaginary part of the complex number.

public:
  using value_type = _Tp; //!< The type of the values.

  //! @brief Trivial default constructor.
  _CCCL_HIDE_FROM_ABI constexpr complex() noexcept = default;

  //! @brief Constructs a complex number with the given real and imaginary parts.
  //!
  //! @param __re The real part of the complex number.
  //! @param __im The imaginary part of the complex number, defaults to 0.
  _CCCL_API constexpr complex(const _Tp& __re, const _Tp& __im = _Tp{}) noexcept
      : __re_{__re}
      , __im_{__im}
  {}

  //! @brief Defaulted copy constructor.
  _CCCL_HIDE_FROM_ABI constexpr complex(const complex&) noexcept = default;

  //! @brief Constructs a complex number from another complex number of a different type.
  //!
  //! @tparam _Up The type of the other complex number.
  //! @param __other The other complex number.
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__complex_can_implicitly_convert_float<_Tp, _Up> || __complex_can_implicitly_convert_int<_Tp, _Up>)
  _CCCL_API constexpr complex(const _Up& __other) noexcept
      : __re_{static_cast<_Tp>(::cuda::__get_real(__other))}
      , __im_{static_cast<_Tp>(::cuda::__get_imag(__other))}
  {}

  //! @brief Constructs a complex number from another complex number of a different type.
  //!
  //! @tparam _Up The type of the other complex number.
  //! @param __other The other complex number.
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__complex_can_explicitly_convert<_Tp, _Up>)
  _CCCL_API explicit constexpr complex(const _Up& __other) noexcept
      : __re_{static_cast<_Tp>(::cuda::__get_real(__other))}
      , __im_{static_cast<_Tp>(::cuda::__get_imag(__other))}
  {}

#if _CCCL_STD_VER >= 2020
  //! @brief Constructs a complex number from a tuple-like object.
  //!
  //! @tparam _Up The type of the tuple-like object.
  //! @param __other The tuple-like object.
  //!
  //! @note The tuple-like object must have two elements of the same type (ignoring cvref-qualifiers).
  //!       The tuple element type must be the same as the complex type's value type (ignoring cvref-qualifiers)
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!__is_any_complex_v<_Up>) _CCCL_AND //
                   __is_complex_compatible_tuple_like<_Up> _CCCL_AND //
                 ::cuda::std::is_same_v<__complex_tuple_like_value_type_t<_Up>, _Tp>)
  _CCCL_API explicit constexpr complex(const _Up& __other) noexcept(
    noexcept(get<0>(__other)) && noexcept(get<1>(__other)))
      : __re_{static_cast<_Tp>(get<0>(__other))}
      , __im_{static_cast<_Tp>(get<1>(__other))}
  {}
#endif // _CCCL_STD_VER >= 2020

  //! @brief Defaulted copy assignment operator.
  _CCCL_HIDE_FROM_ABI constexpr complex& operator=(const complex&) noexcept = default;

  //! @brief Assigns a value to the complex number.
  //!
  //! @param __v The value to assign to the complex number.
  //! @return A reference to this complex number.
  _CCCL_API constexpr complex& operator=(const _Tp& __v) noexcept
  {
    __re_ = __v;
    __im_ = _Tp{};
    return *this;
  }

  //! @brief Assigns another complex number to this complex number.
  //!
  //! @tparam _Up The type of the other complex number.
  //! @param __v The other complex number.
  //! @return A reference to this complex number.
  template <class _Up>
  _CCCL_API constexpr complex& operator=(const complex<_Up>& __v) noexcept
  {
    __re_ = static_cast<_Tp>(__v.real());
    __im_ = static_cast<_Tp>(__v.imag());
    return *this;
  }

  //! @brief Returns the real part of the complex number.
  //!
  //! @return The real part of the complex number.
  [[nodiscard]] _CCCL_API constexpr _Tp real() const noexcept
  {
    return __re_;
  }

  //! @brief Returns the real part of the complex number.
  //!
  //! @return The real part of the complex number.
  [[nodiscard]] _CCCL_API constexpr _Tp real() const volatile noexcept
  {
    return __re_;
  }

  //! @brief Returns the imaginary part of the complex number.
  //!
  //! @return The imaginary part of the complex number.
  [[nodiscard]] _CCCL_API constexpr _Tp imag() const noexcept
  {
    return __im_;
  }

  //! @brief Returns the imaginary part of the complex number.
  //!
  //! @return The imaginary part of the complex number.
  [[nodiscard]] _CCCL_API constexpr _Tp imag() const volatile noexcept
  {
    return __im_;
  }

  //! @brief Sets the real part of the complex number.
  //!
  //! @param __re The new real part of the complex number.
  _CCCL_API constexpr void real(_Tp __re) noexcept
  {
    __re_ = __re;
  }

  //! @brief Sets the real part of the complex number.
  //!
  //! @param __re The new real part of the complex number.
  _CCCL_API constexpr void real(_Tp __re) volatile noexcept
  {
    __re_ = __re;
  }

  //! @brief Sets the imaginary part of the complex number.
  //!
  //! @param __im The new imaginary part of the complex number.
  _CCCL_API constexpr void imag(_Tp __im) noexcept
  {
    __im_ = __im;
  }

  //! @brief Sets the imaginary part of the complex number.
  //!
  //! @param __im The new imaginary part of the complex number.
  _CCCL_API constexpr void imag(_Tp __im) volatile noexcept
  {
    __im_ = __im;
  }
};

//! @brief Deduction guide for construction from complex types.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_any_complex_v<_Tp>)
_CCCL_HOST_DEVICE complex(const _Tp&) -> complex<typename _Tp::value_type>;

#if _CCCL_STD_VER >= 2020
//! @brief Deduction guide for construction from a tuple-like object.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!__is_any_complex_v<_Tp>) _CCCL_AND __is_complex_compatible_tuple_like<_Tp>)
_CCCL_HOST_DEVICE complex(const _Tp&) -> complex<__complex_tuple_like_value_type_t<_Tp>>;
#endif // _CCCL_STD_VER >= 2020

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___COMPLEX_COMPLEX_H
