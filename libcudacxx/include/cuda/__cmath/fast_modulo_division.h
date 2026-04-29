//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/__cmath/ilog.h>
#include <cuda/__cmath/mul_hi.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/pair.h>

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

//! @brief Fast modulo and division by precomputation
//! @tparam _Tp The integer type of the divisor
//! @tparam _DivisorIsNeverOne If \c true, the divisor is guaranteed to never be one, enabling optimizations
template <typename _Tp, bool _DivisorIsNeverOne = false>
class fast_mod_div
{
  static_assert(::cuda::std::__cccl_is_integer_v<_Tp>, "cuda::fast_mod_div: T is required to be an integer type");

  using __unsigned_t = ::cuda::std::make_unsigned_t<_Tp>;

public:
  fast_mod_div() = delete;

  //! @brief Constructs a fast_mod_div object from a divisor value
  //! @param[in] __divisor1 The divisor value, must be positive
  //! @pre \p __divisor1 must be positive
  _CCCL_API explicit fast_mod_div(_Tp __divisor1) noexcept
      : __divisor{__divisor1}
  {
    constexpr int __num_bits = ::cuda::std::__num_bits_v<_Tp>;
    _CCCL_ASSERT(__divisor > 0, "divisor must be positive");
    _CCCL_ASSERT(!_DivisorIsNeverOne || __divisor1 != 1, "cuda::fast_mod_div: divisor must not be one");
    const auto __u_divisor = static_cast<__unsigned_t>(__divisor);
    if constexpr (::cuda::std::is_signed_v<_Tp>)
    {
      __shift        = ::cuda::ceil_ilog2(__divisor) - 1; // is_pow2(x) ? log2(x) : ceil(log2(x))
      const auto __k = __num_bits + __shift; // k: [N, 2*N-2]
      // __multiplier: ceil(2^k / divisor)
      //   computed as 2^k / divisor + (remainder != 0)
      const auto __pow2_div = __divmod_pow2(__k, __u_divisor);
      __multiplier          = __pow2_div.first + (__pow2_div.second != 0);
    }
    else
    {
      __shift = ::cuda::ilog2(__divisor); // floor(log2(divisor))
      if (::cuda::is_power_of_two(__divisor))
      {
        __multiplier = 0;
        return;
      }
      const auto __k        = __num_bits + __shift;
      const auto __pow2_div = __divmod_pow2(__k, __u_divisor);
      // __multiplier: (2^k + 2^shift) / divisor
      //  computed as         2^k / divisor + (2^k % divisor + 2^shift) / divisor
      //  we know             0 < (divisor - 2^shift) < 2^shift
      //  so __multiplier is  2^k / divisor + (2^k % divisor) >= (divisor - 2^shift)
      //  where (divisor - 2^shift) is the threshold
      const auto __threshold = __u_divisor - (__unsigned_t{1} << __shift);
      __multiplier           = __pow2_div.first + (__pow2_div.second >= __threshold);
      __add                  = (__pow2_div.second < __threshold);
    }
  }

  //! @brief Divides the dividend by the precomputed divisor
  //! @param[in] __dividend The dividend value, must be non-negative
  //! @param[in] __divisor1 The precomputed divisor
  //! @return The quotient of the division
  template <typename _Lhs>
  [[nodiscard]] _CCCL_API friend ::cuda::std::common_type_t<_Tp, _Lhs>
  operator/(_Lhs __dividend, fast_mod_div __divisor1) noexcept
  {
    using ::cuda::std::is_same_v;
    using ::cuda::std::is_signed_v;
    using ::cuda::std::is_unsigned_v;
    static_assert(::cuda::std::__cccl_is_integer_v<_Lhs>, "cuda::fast_mod_div: T is required to be an integer type");
    static_assert(
      ::cuda::std::cmp_less_equal(::cuda::std::numeric_limits<_Lhs>::max(), ::cuda::std::numeric_limits<_Tp>::max()),
      "cuda::fast_mod_div: dividend type must be less than or equal to divisor type");
    if constexpr (is_signed_v<_Lhs>)
    {
      _CCCL_ASSERT(__dividend >= 0, "dividend must be non-negative");
    }
    using __common_t       = ::cuda::std::common_type_t<_Tp, _Lhs>;
    using __ucommon_t      = ::cuda::std::make_unsigned_t<__common_t>;
    using __unsigned_lhs_t = ::cuda::std::make_unsigned_t<_Lhs>;
    const auto __div       = __divisor1.__divisor; // cannot use structure binding because of clang-14
    const auto __mul       = __divisor1.__multiplier;
    const auto __shift_    = __divisor1.__shift;
    auto __udividend       = static_cast<__unsigned_lhs_t>(__dividend);
    if constexpr (is_unsigned_v<_Tp>)
    {
      if (__mul == 0) // divisor is a power of two
      {
        return static_cast<__common_t>(static_cast<__ucommon_t>(__udividend) >> __shift_);
      }
      // if dividend is a signed type, overflow is not possible
      if (is_signed_v<_Lhs> || __udividend != ::cuda::std::numeric_limits<__unsigned_lhs_t>::max()) // avoid overflow
      {
        __udividend += static_cast<__unsigned_lhs_t>(__divisor1.__add);
      }
    }
    else if (!_DivisorIsNeverOne && __div == 1)
    {
      return static_cast<__common_t>(__dividend);
    }
    const auto __higher_bits = ::cuda::mul_hi(static_cast<__ucommon_t>(__udividend), static_cast<__ucommon_t>(__mul));
    const auto __quotient    = static_cast<__common_t>(__higher_bits >> __shift_);
    _CCCL_ASSERT(__quotient == static_cast<__common_t>(__dividend / __div), "wrong __quotient");
    return __quotient;
  }

  //! @brief Computes the remainder of dividing the dividend by the precomputed divisor
  //! @param[in] __dividend The dividend value
  //! @param[in] __divisor1 The precomputed divisor
  //! @return The remainder of the division
  template <typename _Lhs>
  [[nodiscard]] _CCCL_API friend ::cuda::std::common_type_t<_Tp, _Lhs>
  operator%(_Lhs __dividend, fast_mod_div __divisor1) noexcept
  {
    return __dividend - (__dividend / __divisor1) * __divisor1.__divisor;
  }

  //! @brief Converts to the underlying divisor value
  //! @return The divisor value
  [[nodiscard]] _CCCL_API operator _Tp() const noexcept
  {
    return static_cast<_Tp>(__divisor);
  }

private:
  //! @brief Computes {2^power / divisor, 2^power % divisor}
  //!
  //! @param[in] __power The exponent, in the range [0, 2*N) where N is the bit-width of \c __unsigned_t
  //! @param[in] __divisor The divisor, must be positive
  //! @return A pair of (quotient, remainder)
  [[nodiscard]] _CCCL_API static ::cuda::std::pair<__unsigned_t, __unsigned_t>
  __divmod_pow2(int __power, __unsigned_t __divisor) noexcept
  {
    constexpr int __num_bits = ::cuda::std::__num_bits_v<__unsigned_t>;
    _CCCL_ASSERT(__power >= 0, "power must be non-negative");
    _CCCL_ASSERT(__power < 2 * __num_bits, "power must be less than 2 * N");
    _CCCL_ASSERT(__divisor > 0, "divisor must be positive");
    __unsigned_t __quotient  = 0;
    __unsigned_t __remainder = 0;
    // Algorithm: restoring binary division
    // Reference: https://marz.utk.edu/my-courses/cosc130/lectures/binary-arithmetic/
    // The dividend 2^power may exceed the range of __unsigned_t. The algorithm processes the dividend bit-by-bit from
    // MSB to LSB without materializing it.
    // Since 2^power has exactly one set bit, the loop contributes to 1 only when the current bit index equals __power,
    // and 0 otherwise.
    // At each iteration, the remainder is shifted left, the next dividend bit is appended.
    // If the result is >= __divisor, the divisor is subtracted and 1-bit is recorded in the quotient.
    for (int __bit = __power; __bit >= 0; --__bit)
    {
      // __carry_flag only matters for unsigned types with large divisors (MSB set: >= 2^(N-1))
      const bool __carry_flag = (__remainder >> (__num_bits - 1)) != 0; // __remainder / 2^N != 0
      __remainder <<= 1;
      __remainder |= (__bit == __power); // append the remainder bit
      const bool __quotient_bit = __carry_flag || (__remainder >= __divisor);
      __quotient <<= 1; // shift
      __quotient |= unsigned{__quotient_bit}; // append the quotient bit
      if (__quotient_bit)
      {
        __remainder -= __divisor;
      }
    }
    return {__quotient, __remainder};
  }

  _Tp __divisor             = 1;
  __unsigned_t __multiplier = 0;
  unsigned __add            = 0;
  int __shift               = 0;
};

/***********************************************************************************************************************
 * Non-member functions
 **********************************************************************************************************************/

//! @brief Computes both quotient and remainder of dividing the dividend by the precomputed divisor
//! @param[in] __dividend The dividend value
//! @param[in] __divisor The precomputed divisor
//! @return A pair of (quotient, remainder)
template <typename _Tp, typename _Lhs, bool _DivisorIsNeverOne>
[[nodiscard]] _CCCL_API ::cuda::std::pair<_Tp, _Lhs>
div(_Tp __dividend, fast_mod_div<_Lhs, _DivisorIsNeverOne> __divisor) noexcept
{
  const auto __quotient  = __dividend / __divisor;
  const auto __remainder = __dividend - __quotient * __divisor;
  return {__quotient, __remainder};
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_FAST_MODULO_DIVISION_H
