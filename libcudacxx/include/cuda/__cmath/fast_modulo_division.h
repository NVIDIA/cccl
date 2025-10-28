//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_FAST_MODULO_DIVISION_H
#define _CUDA___CMATH_FAST_MODULO_DIVISION_H

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
#include <cuda/__cmath/mul_hi.h>
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

_CCCL_BEGIN_NAMESPACE_CUDA

/***********************************************************************************************************************
 * Fast Modulo/Division based on Precomputation
 **********************************************************************************************************************/

// The implementation is based on the following references depending on the data type:
// - Hacker's Delight, Second Edition, Chapter 10
// - Labor of Division (Episode III): Faster Unsigned Division by Constants (libdivide)
//   https://ridiculousfish.com/blog/posts/labor-of-division-episode-iii.html
// - Classic Round-Up Variant of Fast Unsigned Division by Constants
//   https://arxiv.org/pdf/2412.03680
template <typename _Tp, bool _DivisorIsNeverOne = false>
class fast_mod_div
{
#if !_CCCL_HAS_INT128()
  using __max_supported_t = uint32_t;
#else
  using __max_supported_t = uint64_t;
#endif

  static_assert(::cuda::std::__cccl_is_integer_v<_Tp> && sizeof(_Tp) <= sizeof(__max_supported_t),
                "fast_mod_div: T is required to be an integer type");

  using __unsigned_t = ::cuda::std::make_unsigned_t<_Tp>;

public:
  fast_mod_div() = delete;

  _CCCL_API explicit fast_mod_div(_Tp __divisor1) noexcept
      : __divisor{__divisor1}
  {
    using ::cuda::std::__num_bits_v;
    using __larger_t = ::cuda::std::__make_nbit_uint_t<__num_bits_v<_Tp> * 2>;
    _CCCL_ASSERT(__divisor > 0, "divisor must be positive");
    _CCCL_ASSERT(!_DivisorIsNeverOne || __divisor1 != 1, "cuda::fast_mod_div: divisor must not be one");
    if constexpr (::cuda::std::is_signed_v<_Tp>)
    {
      __shift            = ::cuda::ceil_ilog2(__divisor) - 1; // is_pow2(x) ? log2(x) - 1 : log2(x)
      auto __k           = __num_bits_v<_Tp> + __shift; // k: [N, 2*N-2]
      auto __multiplier1 = ::cuda::ceil_div(__larger_t{1} << __k, __divisor); // ceil(2^k / divisor)
      __multiplier       = static_cast<__unsigned_t>(__multiplier1);
    }
    else
    {
      __shift = ::cuda::ilog2(__divisor);
      if (::cuda::is_power_of_two(__divisor))
      {
        __multiplier = 0;
        return;
      }
      const auto __k        = __num_bits_v<_Tp> + __shift;
      __multiplier          = ((__larger_t{1} << __k) + (__larger_t{1} << __shift)) / __divisor;
      auto __multiplier_low = (__larger_t{1} << __k) / __divisor;
      __add                 = (__multiplier_low == __multiplier);
    }
  }

  template <typename _Lhs>
  [[nodiscard]] _CCCL_API friend ::cuda::std::common_type_t<_Tp, _Lhs>
  operator/(_Lhs __dividend, fast_mod_div<_Tp> __divisor1) noexcept
  {
    using ::cuda::std::is_same_v;
    using ::cuda::std::is_signed_v;
    using ::cuda::std::is_unsigned_v;
    static_assert(::cuda::std::__cccl_is_integer_v<_Lhs> && sizeof(_Tp) <= sizeof(__max_supported_t),
                  "cuda::fast_mod_div: T is required to be an integer type");
    static_assert(sizeof(_Lhs) < sizeof(_Tp) || is_same_v<_Lhs, _Tp> || (is_signed_v<_Lhs> && is_unsigned_v<_Tp>),
                  "cuda::fast_mod_div: if dividend and divisor have the same size, dividend must be signed and divisor "
                  "must be unsigned");
    if constexpr (is_signed_v<_Lhs>)
    {
      _CCCL_ASSERT(__dividend >= 0, "dividend must be non-negative");
    }
    using __common_t    = ::cuda::std::common_type_t<_Tp, _Lhs>;
    using __ucommon_t   = ::cuda::std::make_unsigned_t<__common_t>;
    using _Up           = ::cuda::std::make_unsigned_t<_Lhs>;
    const auto __div    = __divisor1.__divisor; // cannot use structure binding because of clang-14
    const auto __mul    = __divisor1.__multiplier;
    const auto __shift_ = __divisor1.__shift;
    auto __udividend    = static_cast<_Up>(__dividend);
    if constexpr (is_unsigned_v<_Tp>)
    {
      if (__mul == 0) // divisor is a power of two
      {
        return static_cast<__common_t>(__udividend >> __shift_);
      }
      // if dividend is a signed type, overflow is not possible
      if (is_signed_v<_Lhs> || __udividend != ::cuda::std::numeric_limits<_Up>::max()) // avoid overflow
      {
        __udividend += static_cast<_Up>(__divisor1.__add);
      }
    }
    else if (!_DivisorIsNeverOne && __div == 1)
    {
      return static_cast<__common_t>(__dividend);
    }
    auto __higher_bits = ::cuda::mul_hi(static_cast<__ucommon_t>(__udividend), static_cast<__ucommon_t>(__mul));
    auto __quotient    = static_cast<__common_t>(__higher_bits >> __shift_);
    _CCCL_ASSERT(__quotient == static_cast<__common_t>(__dividend / __div), "wrong __quotient");
    return __quotient;
  }

  template <typename _Lhs>
  [[nodiscard]] _CCCL_API friend ::cuda::std::common_type_t<_Tp, _Lhs>
  operator%(_Lhs __dividend, fast_mod_div<_Tp> __divisor1) noexcept
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
  unsigned __add            = 0;
  int __shift               = 0;
};

/***********************************************************************************************************************
 * Non-member functions
 **********************************************************************************************************************/

template <typename _Tp, typename _Lhs>
[[nodiscard]] _CCCL_API ::cuda::std::pair<_Tp, _Lhs> div(_Tp __dividend, fast_mod_div<_Lhs> __divisor) noexcept
{
  auto __quotient  = __dividend / __divisor;
  auto __remainder = __dividend - __quotient * __divisor;
  return {__quotient, __remainder};
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_FAST_MODULO_DIVISION_H
