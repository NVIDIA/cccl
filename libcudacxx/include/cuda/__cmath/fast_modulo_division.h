//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_FAST_MODULO_DIVISION_H
#define _LIBCUDACXX___CMATH_FAST_MODULO_DIVISION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/ilog.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

/***********************************************************************************************************************
 * Extract higher bits after multiplication
 **********************************************************************************************************************/

template <typename _Tp, typename _Up>
[[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::common_type_t<_Tp, _Up> __multiply_extract_higher_bits(_Tp __x, _Up __y)
{
  using namespace _CUDA_VSTD;
  static_assert(__cccl_is_integer_v<_Tp> && sizeof(_Tp) <= 8, "unsupported type");
  static_assert(__cccl_is_integer_v<_Up> && sizeof(_Up) <= 8, "unsupported type");
  if constexpr (is_signed_v<_Tp>)
  {
    _CCCL_ASSERT(__x >= 0, "__x must be non-negative");
    _CCCL_ASSUME(__x >= 0);
  }
  if constexpr (is_signed_v<_Up>)
  {
    _CCCL_ASSERT(__y >= 0, "__y must be non-negative");
    _CCCL_ASSUME(__y >= 0);
  }
  constexpr auto __mul_bits = ::cuda::next_power_of_two(__num_bits_v<_Tp> + __num_bits_v<_Up>);
  using __larger_t          = __make_nbit_uint_t<__mul_bits>;
  using __ret_t             = common_type_t<_Tp, _Up>;
  auto __ret                = (static_cast<__larger_t>(__x) * __y) >> (__mul_bits / 2);
  return static_cast<__ret_t>(__ret);
}

/***********************************************************************************************************************
 * Fast Modulo/Division based on Precomputation
 **********************************************************************************************************************/

template <typename _Tp, bool _IsDivisorNotOne = false>
class fast_mod_div
{
  static_assert(_CUDA_VSTD::__cccl_is_integer_v<_Tp> && sizeof(_Tp) <= 8, "unsupported type");
  using __unsigned_t = _CUDA_VSTD::make_unsigned_t<_Tp>;

public:
  fast_mod_div() = delete;

  _CCCL_API explicit fast_mod_div(_Tp __divisor1) noexcept
      : __divisor{__divisor1}
  {
    using namespace _CUDA_VSTD;
    using __larger_t = __make_nbit_uint_t<__num_bits_v<_Tp> * 2>;
    _CCCL_ASSERT(!_IsDivisorNotOne || __divisor1 != 1, "divisor must not be one");
    if constexpr (is_signed_v<_Tp>)
    {
      __shift      = ::cuda::ceil_ilog2(__divisor1) - 1; // is_pow2(x) ? log2(x) - 1 : log2(x)
      auto __k     = __num_bits_v<_Tp> + __shift;
      __multiplier = ::cuda::ceil_div(__larger_t{1} << __k, __divisor);
    }
    else
    {
      __shift = ::cuda::ilog2(__divisor1);
      if (::cuda::is_power_of_two(__divisor))
      {
        __multiplier = 0;
        return;
      }
      auto __k              = __num_bits_v<_Tp> + __shift;
      __multiplier          = ((__larger_t{1} << __k) + (__larger_t{1} << __shift)) / __divisor;
      auto __multiplier_low = (__larger_t{1} << __k) / __divisor;
      __add                 = (__multiplier_low == __multiplier);
    }
  }

  template <typename _Up>
  [[nodiscard]] _CCCL_API friend _CUDA_VSTD::common_type_t<_Tp, _Up>
  operator/(_Up __dividend, fast_mod_div<_Tp> __divisor1) noexcept
  {
    using namespace _CUDA_VSTD;
    static_assert(__cccl_is_integer_v<_Up> && sizeof(_Up) <= 8, "unsupported type");
    static_assert(sizeof(_Up) <= sizeof(_Tp), "dividend type must be not larger than the divisor type");
    static_assert(!(is_unsigned_v<_Up> && is_signed_v<_Tp>), "unsupported types");
    if constexpr (is_signed_v<_Up>)
    {
      _CCCL_ASSERT(__dividend >= 0, "dividend must be non-negative");
    }
    using __common_t    = common_type_t<_Tp, _Up>;
    const auto __div    = __divisor1.__divisor; // cannot use structure binding because of clang-14
    const auto __mul    = __divisor1.__multiplier;
    const auto __add_   = __divisor1.__add; // cannot use __add because of shadowing warning with clang-cuda
    const auto __shift_ = __divisor1.__shift;
    if (!_IsDivisorNotOne && __div == 1)
    {
      return __dividend;
    }
    if constexpr (is_unsigned_v<_Tp>)
    {
      if (__mul == 0) // divisor is a power of two
      {
        return static_cast<__common_t>(__dividend >> __shift_);
      }
      if (__dividend != numeric_limits<_Up>::max()) // avoid overflow
      {
        __dividend += __add_;
      }
    }
    auto __higher_bits = ::cuda::__multiply_extract_higher_bits(__dividend, __mul);
    auto __quotient    = __higher_bits >> __shift_;
    _CCCL_ASSERT(__quotient == (static_cast<__unsigned_t>(__dividend) - __add_) / __div, "wrong __quotient");
    return static_cast<__common_t>(__quotient);
  }

  template <typename _Up>
  [[nodiscard]] _CCCL_API friend _CUDA_VSTD::common_type_t<_Tp, _Up>
  operator%(_Up __dividend, fast_mod_div<_Tp> __divisor1) noexcept
  {
    return __dividend - (__dividend / __divisor1) * __divisor1.__divisor;
  }

  [[nodiscard]] _CCCL_API operator _Tp() const noexcept
  {
    return static_cast<_Tp>(__divisor);
  }

private:
  _Tp __divisor             = 1;
  __unsigned_t __multiplier = 0;
  int __add                 = 0;
  int __shift               = 0;
};

/***********************************************************************************************************************
 * Non-member functions
 **********************************************************************************************************************/

template <typename _Tp, typename _Up>
[[nodiscard]] _CCCL_API _CUDA_VSTD::pair<_Tp, _Up> div(_Tp __dividend, fast_mod_div<_Up> __divisor) noexcept
{
  auto __quotient  = __dividend / __divisor;
  auto __remainder = __dividend - __quotient * __divisor;
  return {__quotient, __remainder};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_FAST_MODULO_DIVISION_H
