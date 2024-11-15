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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/cmath> // cuda::std::ceil_div
#include <cuda/std/bit> // std::has_single_bit
#include <cuda/std/climits> // CHAR_BIT
#include <cuda/std/cstdint> // uint64_t
#include <cuda/std/limits> // numeric_limits
#include <cuda/std/type_traits> // std::is_integral

#include "cub/util_type.cuh" // CUB_IS_INT128_ENABLED

#if defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
_CCCL_NV_DIAG_SUPPRESS(186) // pointless comparison of unsigned integer with zero
#endif // CCCL_ENABLE_DEVICE_ASSERTIONS

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
struct larger_unsigned_type<T, ::cuda::std::__enable_if_t<(sizeof(T) < 4)>>
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

template <typename T>
using supported_integral =
     ::cuda::std::bool_constant<::cuda::std::is_integral<T>::value && !::cuda::std::is_same<T, bool>::value
                                 && (sizeof(T) <= 8)>;

/***********************************************************************************************************************
 * Extract higher bits after multiplication
 **********************************************************************************************************************/

template <typename DivisorType, typename T, typename R>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE unsigned_implicit_prom_t<DivisorType>
multiply_extract_higher_bits(T value, R multiplier)
{
  static_assert(supported_integral<T>::value, "unsupported type");
  static_assert(supported_integral<R>::value, "unsupported type");
  _CCCL_ASSERT(value >= 0, "value must be non-negative");
  _CCCL_ASSERT(multiplier >= 0, "multiplier must be non-negative");
  static constexpr int NumBits = sizeof(DivisorType) * CHAR_BIT;
  using unsigned_t             = unsigned_implicit_prom_t<DivisorType>;
  using larger_t               = larger_unsigned_type_t<DivisorType>;
  // clang-format off
  NV_IF_TARGET(
    NV_IS_HOST,
      (return static_cast<unsigned_t>((static_cast<larger_t>(value) * multiplier) >> NumBits);),
    //NV_IS_DEVICE
      (return (sizeof(T) == 8)
        ? static_cast<unsigned_t>(__umul64hi(value, multiplier))
        : static_cast<unsigned_t>((static_cast<larger_t>(value) * multiplier) >> NumBits);));
  // clang-format on
}

/***********************************************************************************************************************
 * Fast Modulo/Division based on Precomputation
 **********************************************************************************************************************/

template <typename T>
class fast_div_mod
{
  static_assert(supported_integral<T>::value, "unsupported type");

  using prom_t     = implicit_prom_t<T>;
  using unsigned_t = unsigned_implicit_prom_t<T>;

public:
  template <typename R>
  struct result
  {
    using Common = decltype(R{} / T{});
    Common quotient;
    Common remainder;
  };

  fast_div_mod() = delete;

  _CCCL_NODISCARD _CCCL_HOST_DEVICE explicit fast_div_mod(T divisor) noexcept
      : _divisor{static_cast<prom_t>(divisor)}
  {
    using larger_t = larger_unsigned_type_t<T>;
    _CCCL_ASSERT(divisor > 0, "divisor must be positive");
    auto udivisor = static_cast<unsigned_t>(divisor);
    // the following branches are needed to avoid negative shift
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
    int num_bits            = ::cuda::std::bit_width(udivisor) + 1;
    // without explicit power-of-two check, num_bits needs to replace +1 with !::cuda::std::has_single_bit(udivisor)
    _multiplier  = static_cast<unsigned_t>(::cuda::ceil_div(larger_t{1} << (num_bits + BitSize - BitOffset), //
                                                           static_cast<larger_t>(divisor)));
    _shift_right = num_bits - BitOffset;
  }

  fast_div_mod(const fast_div_mod&) noexcept = default;

  fast_div_mod(fast_div_mod&&) noexcept = default;

  template <typename R>
  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE result<T, R> operator()(R dividend) const noexcept
  {
    static_assert(supported_integral<R>::value, "unsupported type");
    using Common   = decltype(R{} / T{});
    using UCommon  = ::cuda::std::make_unsigned_t<Common>;
    using result_t = result<T, R>;
    _CCCL_ASSERT(dividend >= 0, "divisor must be non-negative");
    auto udividend = static_cast<UCommon>(dividend);
    if (_divisor == 1)
    {
      return result_t{static_cast<Common>(dividend), Common{}};
    }
    if (sizeof(T) == 8 && _divisor == 3)
    {
      return result_t{static_cast<Common>(udividend / 3), static_cast<Common>(udividend % 3)};
    }
    auto higher_bits = (_multiplier == 0) ? udividend : multiply_extract_higher_bits<T>(dividend, _multiplier);
    auto quotient    = higher_bits >> _shift_right;
    auto remainder   = udividend - (quotient * _divisor);
    _CCCL_ASSERT(quotient == dividend / _divisor, "wrong quotient");
    _CCCL_ASSERT(remainder < _divisor, "remainder out of range");
    return result_t{static_cast<Common>(quotient), static_cast<Common>(remainder)};
  }

  template <typename R>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE friend decltype(R{} / T{}) operator/(R dividend, fast_div_mod div) noexcept
  {
    return div(dividend).quotient;
  }

  template <typename R>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE friend decltype(R{} / T{}) operator%(R dividend, fast_div_mod div) noexcept
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

#if defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
_CCCL_NV_DIAG_DEFAULT(186)
#endif // CCCL_ENABLE_DEVICE_ASSERTIONS
