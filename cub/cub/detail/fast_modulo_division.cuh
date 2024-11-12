/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#include <cuda/std/type_traits>

#include <climits>
#include <type_traits>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h> // cuda::std::ceil_div
#include <cuda/std/__bit/has_single_bit.h> // std::has_single_bit
#include <cuda/std/__bit/integral.h> // std::bit_width
#include <cuda/std/climits> // CHAR_BIT
#include <cuda/std/cstdint> // uint64_t

#include "cub/util_type.cuh" // CUB_IS_INT128_ENABLED
#include "cuda/std/__cccl/assert.h"
#include "cuda/std/__type_traits/is_integral.h"
#include "nv/detail/__target_macros"

#if defined(__NVCC_DIAG_PRAGMA_SUPPORT__) && defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
#  pragma nv_diag_suppress 186
#endif

CUB_NAMESPACE_BEGIN

namespace detail
{

/***********************************************************************************************************************
 * larger_unsigned_type
 **********************************************************************************************************************/

template <typename T, typename = void>
struct larger_unsigned_type
{
  static_assert(sizeof(T) >= 8, "64-bit integer are only supported from CUDA >= 11.5");
  using type = void;
};

template <typename T>
struct larger_unsigned_type<T, typename ::cuda::std::enable_if<(sizeof(T) < 4)>::type>
{
  using type = ::cuda::std::uint32_t;
};

template <typename T>
struct larger_unsigned_type<T, typename ::cuda::std::enable_if<(sizeof(T) == 4)>::type>
{
  using type = ::cuda::std::uint64_t;
};

#if CUB_IS_INT128_ENABLED

template <typename T>
struct larger_unsigned_type<T, typename ::cuda::std::enable_if<(sizeof(T) == 8)>::type>
{
  using type = __uint128_t;
};

#endif // CUB_IS_INT128_ENABLED

template <typename T>
using larger_unsigned_type_t = typename larger_unsigned_type<T>::type;

template <typename T>
using implicit_prom_t = decltype(+T{});

template <typename T>
using unsigned_implicit_prom_t = typename ::cuda::std::make_unsigned<implicit_prom_t<T>>::type;

/***********************************************************************************************************************
 * Extract higher bits after multiplication
 **********************************************************************************************************************/

template <typename T, typename R>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE unsigned_implicit_prom_t<T>
multiply_extract_higher_bits(T value, R multiplier)
{
  static_assert(::cuda::std::is_integral<T>::value && !::cuda::std::is_same<T, bool>::value, "unsupported type");
  static_assert(::cuda::std::is_integral<R>::value && !::cuda::std::is_same<R, bool>::value, "unsupported type");
  _CCCL_ASSERT(value >= 0, "value must be non-negative");
  _CCCL_ASSERT(multiplier >= 0, "multiplier must be non-negative");
  using unsigned_t           = unsigned_implicit_prom_t<T>;
  using larger_t             = larger_unsigned_type_t<T>;
  constexpr unsigned BitSize = sizeof(T) * CHAR_BIT;
  // clang-format off
  NV_IF_TARGET(
    NV_IS_HOST,
      (return static_cast<unsigned_t>((static_cast<larger_t>(value) * multiplier) >> BitSize);),
    //NV_IS_DEVICE
      (return (sizeof(T) == 8)
        ? static_cast<unsigned_t>(__umul64hi(value, multiplier))
        : static_cast<unsigned_t>((static_cast<larger_t>(value) * multiplier) >> BitSize);));
  // clang-format on
}

/***********************************************************************************************************************
 * Fast Modulo/Division based on Precomputation
 **********************************************************************************************************************/

template <typename T>
struct fast_div_mod
{
  static_assert(::cuda::std::is_integral<T>::value && !::cuda::std::is_same<T, bool>::value, "unsupported type");

  using unsigned_t = unsigned_implicit_prom_t<T>;
  using prom_t     = implicit_prom_t<T>;
  using larger_t   = larger_unsigned_type_t<T>;

  struct result
  {
    prom_t quotient;
    prom_t remainder;
  };

  fast_div_mod() = delete;

  _CCCL_NODISCARD _CCCL_HOST_DEVICE explicit fast_div_mod(T divisor) noexcept
      : _divisor{static_cast<prom_t>(divisor)}
  {
    _CCCL_ASSERT(divisor > 0, "divisor must be positive");
    auto udivisor = static_cast<unsigned_t>(divisor);
    if (::cuda::std::has_single_bit(udivisor)) // power of two
    {
      _shift_right = ::cuda::std::bit_width(udivisor) - 1;
      return;
    }
    else if (sizeof(T) == 8 && divisor == 3)
    {
      return;
    }
    constexpr int BitSize   = sizeof(T) * CHAR_BIT;
    constexpr int BitOffset = BitSize / 16;
    int num_bits            = ::cuda::std::bit_width(udivisor);
    // without explicit power-of-two check, num_bits needs to add !::cuda::std::has_single_bit(udivisor)
    _multiplier  = static_cast<unsigned_t>(::cuda::ceil_div(larger_t{1} << (num_bits + BitSize - BitOffset), //
                                                           static_cast<larger_t>(divisor)));
    _shift_right = num_bits - BitOffset;
  }

  fast_div_mod(const fast_div_mod&) noexcept = default;

  fast_div_mod(fast_div_mod&&) noexcept = default;

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE result operator()(T dividend) const noexcept
  {
    _CCCL_ASSERT(dividend >= 0, "divisor must be non-negative");
    // the following branches are needed to avoid negative shift
    if (_divisor == 1)
    {
      return result{dividend, 0};
    }
    if (sizeof(T) == 8 && _divisor == 3)
    {
      return result{static_cast<prom_t>(dividend / unsigned_t{3}), static_cast<prom_t>(dividend % unsigned_t{3})};
    }
    auto quotient =
      ((_multiplier == 0) ? dividend : multiply_extract_higher_bits(dividend, _multiplier)) >> _shift_right;
    auto remainder = dividend - (quotient * _divisor);
    _CCCL_ASSERT(quotient == dividend / _divisor, "wrong quotient");
    _CCCL_ASSERT(remainder >= 0 && remainder < _divisor, "remainder out of range");
    return result{static_cast<prom_t>(quotient), static_cast<prom_t>(remainder)};
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE friend prom_t operator/(T dividend, fast_div_mod div) noexcept
  {
    return div(dividend).quotient;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE friend prom_t operator%(T dividend, fast_div_mod div) noexcept
  {
    return div(dividend).remainder;
  }

private:
  prom_t _divisor        = 1;
  unsigned_t _multiplier = 0;
  unsigned _shift_right  = 0;
};

} // namespace detail

CUB_NAMESPACE_END
