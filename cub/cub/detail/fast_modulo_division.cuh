// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/type_traits.cuh> // implicit_prom_t
#include <cub/util_type.cuh> // _CCCL_HAS_INT128()

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/climits> // CHAR_BIT
#include <cuda/std/cstdint> // uint64_t
#include <cuda/std/limits>

#if defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
_CCCL_BEGIN_NV_DIAG_SUPPRESS(186) // pointless comparison of unsigned integer with zero
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
  using type = void;
};

template <typename T>
struct larger_unsigned_type<T, ::cuda::std::enable_if_t<(sizeof(T) < 4)>>
{
  using type = ::cuda::std::uint32_t;
};

template <typename T>
struct larger_unsigned_type<T, ::cuda::std::enable_if_t<(sizeof(T) == 4)>>
{
  using type = ::cuda::std::uint64_t;
};

#if _CCCL_HAS_INT128()

template <typename T>
struct larger_unsigned_type<T, ::cuda::std::enable_if_t<(sizeof(T) == 8)>>
{
  using type = __uint128_t;
};

#endif // _CCCL_HAS_INT128()

template <typename T>
using larger_unsigned_type_t = typename larger_unsigned_type<T>::type;

template <typename T>
using unsigned_implicit_prom_t = ::cuda::std::make_unsigned_t<implicit_prom_t<T>>;

template <typename T>
using supported_integral =
  ::cuda::std::bool_constant<::cuda::std::is_integral_v<T> && !::cuda::std::is_same_v<T, bool> && (sizeof(T) <= 8)>;

/***********************************************************************************************************************
 * Extract higher bits after multiplication
 **********************************************************************************************************************/

template <typename DivisorType, typename T, typename R>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE unsigned_implicit_prom_t<DivisorType>
multiply_extract_higher_bits(T value, R multiplier)
{
  static_assert(supported_integral<T>::value, "unsupported type");
  static_assert(supported_integral<R>::value, "unsupported type");
  if constexpr (::cuda::std::is_signed_v<T>)
  {
    _CCCL_ASSERT(value >= 0, "value must be non-negative");
  }
  if constexpr (::cuda::std::is_signed_v<R>)
  {
    _CCCL_ASSERT(multiplier >= 0, "multiplier must be non-negative");
  }
  static constexpr int NumBits = sizeof(DivisorType) * CHAR_BIT;
  using unsigned_t             = unsigned_implicit_prom_t<DivisorType>;
  using larger_t               = larger_unsigned_type_t<DivisorType>;
  // clang-format off
  NV_IF_ELSE_TARGET(
    NV_IS_HOST,
      (return static_cast<unsigned_t>((static_cast<larger_t>(value) * multiplier) >> NumBits);),
      ({return (sizeof(T) == 8)
        ? static_cast<unsigned_t>(__umul64hi(value, multiplier))
        : static_cast<unsigned_t>((static_cast<larger_t>(value) * multiplier) >> NumBits);}));
  // clang-format on
}

/***********************************************************************************************************************
 * Fast division by a precomputed unsigned constant
 *
 * A divisor of zero selects the identity operation. This provides a safe default state and lets callers use zero to
 * represent an inactive division without introducing undefined behavior.
 **********************************************************************************************************************/

template <typename UInt>
class fast_divide_by_constant
{
  static_assert(::cuda::std::is_unsigned_v<UInt>, "fast_divide_by_constant requires an unsigned integer type");
  static_assert(sizeof(UInt) == 4 || sizeof(UInt) == 8, "fast_divide_by_constant supports 32- or 64-bit integers");

  static constexpr int bits = static_cast<int>(sizeof(UInt) * CHAR_BIT);

  enum class mode : unsigned char
  {
    identity,
    shift,
    multiply_shift,
    hardware
  };

  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static int ceil_log2(UInt divisor) noexcept
  {
    return divisor <= UInt{1} ? 0 : bits - ::cuda::std::countl_zero(divisor - UInt{1});
  }

  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static UInt multiply_high(UInt lhs, UInt rhs) noexcept
  {
    if constexpr (sizeof(UInt) == 4)
    {
      return static_cast<UInt>(
        (static_cast<::cuda::std::uint64_t>(lhs) * static_cast<::cuda::std::uint64_t>(rhs)) >> bits);
    }
    else
    {
#if _CCCL_HAS_INT128()
      NV_IF_ELSE_TARGET(
        NV_IS_DEVICE,
        (return static_cast<UInt>(
                  __umul64hi(static_cast<unsigned long long>(lhs), static_cast<unsigned long long>(rhs)));),
        (return static_cast<UInt>((static_cast<__uint128_t>(lhs) * static_cast<__uint128_t>(rhs)) >> bits);));
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
      NV_IF_ELSE_TARGET(
        NV_IS_DEVICE,
        (return static_cast<UInt>(
                  __umul64hi(static_cast<unsigned long long>(lhs), static_cast<unsigned long long>(rhs)));),
        ({
          const ::cuda::std::uint64_t lhs_low   = static_cast<::cuda::std::uint32_t>(lhs);
          const ::cuda::std::uint64_t lhs_high  = lhs >> 32;
          const ::cuda::std::uint64_t rhs_low   = static_cast<::cuda::std::uint32_t>(rhs);
          const ::cuda::std::uint64_t rhs_high  = rhs >> 32;
          const ::cuda::std::uint64_t low_low   = lhs_low * rhs_low;
          const ::cuda::std::uint64_t low_high  = lhs_low * rhs_high;
          const ::cuda::std::uint64_t high_low  = lhs_high * rhs_low;
          const ::cuda::std::uint64_t high_high = lhs_high * rhs_high;
          const ::cuda::std::uint64_t middle    = (low_low >> 32) + static_cast<::cuda::std::uint32_t>(low_high)
                                                + static_cast<::cuda::std::uint32_t>(high_low);
          return static_cast<UInt>(high_high + (low_high >> 32) + (high_low >> 32) + (middle >> 32));
        }));
#endif // !_CCCL_HAS_INT128()
    }
  }

public:
  _CCCL_HOST_DEVICE constexpr fast_divide_by_constant() noexcept {}

  _CCCL_HOST_DEVICE explicit fast_divide_by_constant(UInt divisor) noexcept
  {
    init(divisor);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void init(UInt divisor) noexcept
  {
    if (divisor <= UInt{1})
    {
      magic_ = UInt{0};
      shift_ = 0;
      mode_  = mode::identity;
      return;
    }
    if ((divisor & (divisor - UInt{1})) == UInt{0})
    {
      magic_ = UInt{0};
      shift_ = static_cast<unsigned char>(ceil_log2(divisor));
      mode_  = mode::shift;
      return;
    }

    const int log2_divisor = ceil_log2(divisor);
    if (log2_divisor == bits)
    {
      magic_ = divisor;
      shift_ = 0;
      mode_  = mode::hardware;
      return;
    }
    if constexpr (sizeof(UInt) == 8)
    {
#if _CCCL_HAS_INT128()
      const __uint128_t numerator   = static_cast<__uint128_t>(1) << (bits + log2_divisor);
      const __uint128_t denominator = static_cast<__uint128_t>(divisor);
      magic_                        = static_cast<UInt>((numerator + denominator - 1) / denominator);
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
      UInt quotient  = 0;
      UInt remainder = 0;
      for (int bit = bits + log2_divisor; bit >= 0; --bit)
      {
        UInt next_remainder     = (remainder << 1) | (bit == bits + log2_divisor ? UInt{1} : UInt{0});
        const bool carry        = (remainder >> (bits - 1)) != 0;
        const UInt quotient_bit = (carry || next_remainder >= divisor) ? UInt{1} : UInt{0};
        if (quotient_bit != 0)
        {
          next_remainder -= divisor;
        }
        remainder = next_remainder;
        quotient  = (quotient << 1) | quotient_bit;
      }
      magic_ = quotient + (remainder != 0 ? UInt{1} : UInt{0});
#endif // !_CCCL_HAS_INT128()
    }
    else
    {
      const ::cuda::std::uint64_t numerator   = ::cuda::std::uint64_t{1} << (bits + log2_divisor);
      const ::cuda::std::uint64_t denominator = static_cast<::cuda::std::uint64_t>(divisor);
      magic_                                  = static_cast<UInt>((numerator + denominator - 1) / denominator);
    }
    shift_ = static_cast<unsigned char>(log2_divisor);
    mode_  = mode::multiply_shift;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UInt divide(UInt numerator) const noexcept
  {
    if (mode_ == mode::identity)
    {
      return numerator;
    }
    if (mode_ == mode::shift)
    {
      return numerator >> shift_;
    }
    if (mode_ == mode::hardware)
    {
      return numerator / magic_;
    }

    const UInt high = multiply_high(magic_, numerator);
    return (((numerator - high) >> 1) + high) >> (shift_ - 1);
  }

private:
  UInt magic_          = UInt{0};
  unsigned char shift_ = 0;
  mode mode_           = mode::identity;
};

/***********************************************************************************************************************
 * Fast Modulo/Division based on Precomputation
 **********************************************************************************************************************/

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4127) /* conditional expression is constant */

template <typename T1>
class fast_div_mod
{
  static_assert(supported_integral<T1>::value, "unsupported type");

  // uint16_t is a special case that would requires complex logic. Workaround: convert to int
  using T          = ::cuda::std::conditional_t<::cuda::std::is_same_v<T1, ::cuda::std::uint16_t>, int, T1>;
  using unsigned_t = unsigned_implicit_prom_t<T>;

public:
  template <typename R>
  struct result
  {
    using common_t = decltype(R{} / T{});
    common_t quotient;
    common_t remainder;
  };

  fast_div_mod() = delete;

  _CCCL_HOST_DEVICE explicit fast_div_mod(T divisor) noexcept
      : _divisor{static_cast<unsigned_t>(divisor)}
  {
    using larger_t = larger_unsigned_type_t<T>;
    _CCCL_ASSERT(divisor > 0, "divisor must be positive");
    auto udivisor = static_cast<unsigned_t>(divisor);
    // the following branches are needed to avoid negative shift
    if (::cuda::is_power_of_two(udivisor))
    {
      _shift_right = ::cuda::std::bit_width(udivisor) - 1;
      return;
    }
    else if (sizeof(T) == 8 && divisor == 3)
    {
      return;
    }
    constexpr int BitSize   = sizeof(T) * CHAR_BIT; // 32
    constexpr int BitOffset = BitSize / 16; // 2
    const int num_bits      = ::cuda::std::bit_width(udivisor) + 1;
    _CCCL_ASSERT(static_cast<size_t>(num_bits + BitSize - BitOffset) < sizeof(larger_t) * CHAR_BIT, "overflow error");
    // without explicit power-of-two check, num_bits needs to replace +1 with !::cuda::is_power_of_two(udivisor)
    _multiplier  = static_cast<unsigned_t>(::cuda::ceil_div(larger_t{1} << (num_bits + BitSize - BitOffset), //
                                                            static_cast<larger_t>(divisor)));
    _shift_right = num_bits - BitOffset;
    _CCCL_ASSERT(_multiplier != 0, "overflow error");
  }

  fast_div_mod(const fast_div_mod&) noexcept = default;

  fast_div_mod(fast_div_mod&&) noexcept = default;

  template <typename R>
  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE result<R> operator()(R dividend) const noexcept
  {
    static_assert(supported_integral<R>::value, "unsupported type");
    using common_t  = decltype(R{} / T{});
    using ucommon_t = ::cuda::std::make_unsigned_t<common_t>;
    using result_t  = result<R>;
    _CCCL_ASSERT(dividend >= 0, "divisor must be non-negative");
    auto udividend = static_cast<ucommon_t>(dividend);
    if (_divisor == 1)
    {
      return result_t{static_cast<common_t>(dividend), common_t{}};
    }
    else if (_divisor > unsigned_t{::cuda::std::numeric_limits<T>::max() / 2})
    {
      auto quotient = udividend >= static_cast<ucommon_t>(_divisor);
      return result_t{static_cast<common_t>(quotient), static_cast<common_t>(udividend - (quotient * _divisor))};
    }
    else if (sizeof(T) == 8 && _divisor == 3)
    {
      return result_t{static_cast<common_t>(udividend / 3), static_cast<common_t>(udividend % 3)};
    }
    auto higher_bits = (_multiplier == 0) ? udividend : multiply_extract_higher_bits<T>(dividend, _multiplier);
    auto quotient    = higher_bits >> _shift_right;
    auto remainder   = udividend - (quotient * _divisor);
    _CCCL_ASSERT(quotient == udividend / _divisor, "wrong quotient");
    _CCCL_ASSERT(remainder < (ucommon_t) _divisor, "remainder out of range");
    return result_t{static_cast<common_t>(quotient), static_cast<common_t>(remainder)};
  }

  template <typename R>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE friend implicit_prom_t<T> operator/(R dividend, fast_div_mod div) noexcept
  {
    return div(dividend).quotient;
  }

  template <typename R>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE friend implicit_prom_t<T> operator%(R dividend, fast_div_mod div) noexcept
  {
    return div(dividend).remainder;
  }

private:
  unsigned_t _divisor    = 1;
  unsigned_t _multiplier = 0;
  unsigned _shift_right  = 0;
};
_CCCL_DIAG_POP
} // namespace detail

CUB_NAMESPACE_END

#if defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
_CCCL_END_NV_DIAG_SUPPRESS()
#endif // CCCL_ENABLE_DEVICE_ASSERTIONS
