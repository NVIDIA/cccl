//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANDOM_PCG_ENGINE_H
#define _CUDA_STD___RANDOM_PCG_ENGINE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/mul_hi.h>
#include <cuda/std/__bit/rotate.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__random/is_seed_sequence.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// Keep this here even when we have __int128 support, so that we can test it against native __int128
class __pcg_uint128_fallback
{
private:
  ::cuda::std::uint64_t __hi_;
  ::cuda::std::uint64_t __lo_;

public:
  _CCCL_API constexpr __pcg_uint128_fallback() noexcept
      : __hi_{0}
      , __lo_{0}
  {}

  _CCCL_API constexpr __pcg_uint128_fallback(::cuda::std::uint64_t __val) noexcept
      : __hi_{0}
      , __lo_{__val}
  {}

  _CCCL_API constexpr __pcg_uint128_fallback(::cuda::std::uint64_t __hi, ::cuda::std::uint64_t __lo) noexcept
      : __hi_{__hi}
      , __lo_{__lo}
  {}

  [[nodiscard]] _CCCL_API constexpr explicit operator ::cuda::std::uint64_t() const noexcept
  {
    return __lo_;
  }

  [[nodiscard]] _CCCL_API constexpr explicit operator ::cuda::std::uint8_t() const noexcept
  {
    return static_cast<::cuda::std::uint8_t>(__lo_);
  }

  [[nodiscard]] _CCCL_API constexpr __pcg_uint128_fallback operator|(::cuda::std::uint64_t __rhs) const noexcept
  {
    return __pcg_uint128_fallback(__hi_, __lo_ | __rhs);
  }

  [[nodiscard]] _CCCL_API constexpr __pcg_uint128_fallback operator^(__pcg_uint128_fallback __rhs) const noexcept
  {
    return __pcg_uint128_fallback(__hi_ ^ __rhs.__hi_, __lo_ ^ __rhs.__lo_);
  }

  [[nodiscard]] _CCCL_API constexpr int operator&(int __rhs) const noexcept
  {
    return __lo_ & static_cast<::cuda::std::uint64_t>(__rhs);
  }

  [[nodiscard]] _CCCL_API constexpr __pcg_uint128_fallback operator<<(int __shift) const noexcept
  {
    _CCCL_ASSERT(__shift >= 0 && __shift < 128, "shift value out of range");
    if (__shift == 0)
    {
      return *this;
    }
    if (__shift >= 128)
    {
      return __pcg_uint128_fallback(0, 0);
    }
    if (__shift >= 64)
    {
      return __pcg_uint128_fallback(__lo_ << (__shift - 64), 0);
    }
    return __pcg_uint128_fallback((__hi_ << __shift) | (__lo_ >> (64 - __shift)), __lo_ << __shift);
  }

  [[nodiscard]] _CCCL_API constexpr __pcg_uint128_fallback operator>>(int __shift) const noexcept
  {
    _CCCL_ASSERT(__shift >= 0 && __shift < 128, "shift value out of range");
    if (__shift == 0)
    {
      return *this;
    }
    if (__shift >= 128)
    {
      return __pcg_uint128_fallback(0, 0);
    }
    if (__shift >= 64)
    {
      return __pcg_uint128_fallback(0, __hi_ >> (__shift - 64));
    }
    return __pcg_uint128_fallback(__hi_ >> __shift, (__lo_ >> __shift) | (__hi_ << (64 - __shift)));
  }

  [[nodiscard]] _CCCL_API constexpr __pcg_uint128_fallback operator+(__pcg_uint128_fallback __rhs) const noexcept
  {
    // TODO: optimize with PTX add.cc
    ::cuda::std::uint64_t __new_lo = __lo_ + __rhs.__lo_;
    ::cuda::std::uint64_t __carry  = (__new_lo < __lo_) ? 1 : 0;
    return __pcg_uint128_fallback(__hi_ + __rhs.__hi_ + __carry, __new_lo);
  }

  [[nodiscard]] _CCCL_API constexpr __pcg_uint128_fallback operator*(__pcg_uint128_fallback __rhs) const noexcept
  {
    __pcg_uint128_fallback __c(::cuda::mul_hi(__lo_, __rhs.__lo_), __lo_ * __rhs.__lo_);
    __c.__hi_ += __hi_ * __rhs.__lo_ + __lo_ * __rhs.__hi_;
    return __c;
  }

  _CCCL_API constexpr __pcg_uint128_fallback& operator*=(__pcg_uint128_fallback __rhs) noexcept
  {
    return *this = *this * __rhs;
  }

  [[nodiscard]] _CCCL_API constexpr bool operator>(int __x) const noexcept
  {
    return __hi_ != 0 || __lo_ > static_cast<::cuda::std::uint64_t>(__x);
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(__pcg_uint128_fallback __lhs, __pcg_uint128_fallback __rhs) noexcept
  {
    return __lhs.__hi_ == __rhs.__hi_ && __lhs.__lo_ == __rhs.__lo_;
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(__pcg_uint128_fallback __lhs, __pcg_uint128_fallback __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#endif // _CCCL_STD_VER <= 2017
};

//! @brief A 64-bit permuted congruential generator (PCG) random number engine.
//!
//! This is a high-quality, fast random number generator based on the PCG family
//! of algorithms. It uses a 128-bit internal state and produces 64-bit output
//! values using a permutation function applied to a linear congruential generator.
//!
//! Most users should use the predefined `pcg64` type alias instead of this class directly.
//!
//! @tparam _AHi The high 64 bits of the multiplier constant for the LCG.
//! @tparam _ALo The low 64 bits of the multiplier constant for the LCG.
//! @tparam _CHi The high 64 bits of the increment constant for the LCG.
//! @tparam _CLo The low 64 bits of the increment constant for the LCG.
//!
//! @see https://www.pcg-random.org/ for details on the PCG family of generators.
template <::cuda::std::uint64_t _AHi, ::cuda::std::uint64_t _ALo, ::cuda::std::uint64_t _CHi, ::cuda::std::uint64_t _CLo>
class pcg64_engine
{
public:
  using result_type = ::cuda::std::uint64_t;

private:
#if _CCCL_HAS_INT128()
  using __pcg64_uint128_t = __uint128_t;
#else
  using __pcg64_uint128_t = __pcg_uint128_fallback;
#endif
  using __bitcount_t = ::cuda::std::uint8_t;

  static constexpr __pcg64_uint128_t __multiplier = (static_cast<__pcg64_uint128_t>(_AHi) << 64) | _ALo;
  static constexpr __pcg64_uint128_t __increment  = (static_cast<__pcg64_uint128_t>(_CHi) << 64) | _CLo;

  [[nodiscard]] _CCCL_API static constexpr result_type __output_transform(__pcg64_uint128_t __internal) noexcept
  {
    const int __rot = static_cast<__bitcount_t>(__internal >> 122);
    __internal      = __internal ^ (__internal >> 64);
    return ::cuda::std::rotr(result_type(__internal), __rot);
  }

  [[nodiscard]] _CCCL_API constexpr ::cuda::std::pair<__pcg64_uint128_t, __pcg64_uint128_t>
  __power_mod(__pcg64_uint128_t __delta) noexcept
  {
    __pcg64_uint128_t __acc_mult = 1;
    __pcg64_uint128_t __acc_plus = 0;
    __pcg64_uint128_t __cur_mult = __multiplier;
    __pcg64_uint128_t __cur_plus = __increment;
    while (__delta > 0)
    {
      if (__delta & 1)
      {
        __acc_mult *= __cur_mult;
        __acc_plus = __acc_plus * __cur_mult + __cur_plus;
      }
      __cur_plus = (__cur_mult + 1) * __cur_plus;
      __cur_mult *= __cur_mult;
      __delta = __delta >> 1;
    }
    return ::cuda::std::pair{__acc_mult, __acc_plus};
  }
  __pcg64_uint128_t __x_{};

public:
  static constexpr result_type default_seed = 0xcafef00dd15ea5e5ULL;

  //! @brief Returns the smallest value the engine can produce.
  //! @return Always 0 for pcg64_engine.
  [[nodiscard]] _CCCL_API static constexpr result_type min() noexcept
  {
    return 0;
  }
  //! @brief Returns the largest value the engine can produce.
  //! @return The maximum representable `result_type`.
  [[nodiscard]] _CCCL_API static constexpr result_type max() noexcept
  {
    return ::cuda::std::numeric_limits<result_type>::max();
  }

  // constructors and seeding functions
  //! @brief Default-constructs the engine using `default_seed`.
  _CCCL_API constexpr pcg64_engine() noexcept
      : pcg64_engine(default_seed)
  {}
  //! @brief Constructs the engine and seeds it with `__seed`.
  //! @param __seed The seed value used to initialize the engine state.
  _CCCL_API constexpr explicit pcg64_engine(result_type __seed) noexcept
  {
    seed(__seed);
  }

  //! @brief Constructs the engine and seeds it from a SeedSequence-like object.
  //! @tparam _Sseq A SeedSequence-like type satisfying the project's seed concept.
  //! @param __seq The seed sequence used to initialize the internal state.
  _CCCL_TEMPLATE(class _Sseq)
  _CCCL_REQUIRES(::cuda::std::__is_seed_sequence<_Sseq, pcg64_engine>)
  _CCCL_API constexpr explicit pcg64_engine(_Sseq& __seq)
  {
    seed(__seq);
  }
  //! @brief Seed the engine with an integer seed.
  //! @param __seed The seed value; defaults to `default_seed`.
  _CCCL_API constexpr void seed(result_type __seed = default_seed) noexcept
  {
    __x_ = (__pcg64_uint128_t(__seed) + __increment) * __multiplier + __increment;
  }

  //! @brief Seed the engine from a SeedSequence-like object.
  //! @tparam _Sseq A SeedSequence-like type providing entropy words.
  //! @param __seq A SeedSequence-like object providing 128 bits of entropy.
  _CCCL_TEMPLATE(class _Sseq)
  _CCCL_REQUIRES(::cuda::std::__is_seed_sequence<_Sseq, pcg64_engine>)
  _CCCL_API constexpr void seed(_Sseq& __seq)
  {
    ::cuda::std::array<::cuda::std::uint32_t, 4> data = {};
    __seq.generate(data.begin(), data.end());
    __pcg64_uint128_t seed_val = data[0];
    seed_val                   = (seed_val << 32) | data[1];
    seed_val                   = (seed_val << 32) | data[2];
    seed_val                   = (seed_val << 32) | data[3];
    __x_                       = (seed_val + __increment) * __multiplier + __increment;
  }

  //! @brief Generate the next pseudo-random value.
  //!
  //! Advances the internal LCG state and applies the PCG output
  //! permutation to produce a 64-bit result.
  //! @return A 64-bit pseudo-random value.
  _CCCL_API constexpr result_type operator()() noexcept
  {
    __x_ = __x_ * __multiplier + __increment;
    return __output_transform(__x_);
  }

  //! @brief Advance the engine state by `__z` steps, discarding outputs.
  //! @param __z Number of values to discard.
  _CCCL_API constexpr void discard(unsigned long long __z) noexcept
  {
    const auto [__mult, __plus] = __power_mod(__z);
    __x_                        = __x_ * __mult + __plus;
  }

  //! @brief Equality comparison for two engines.
  //! @return True if both engines have identical internal state.
  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const pcg64_engine& __x, const pcg64_engine& __y) noexcept
  {
    return __x.__x_ == __y.__x_;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison for two engines.
  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const pcg64_engine& __x, const pcg64_engine& __y) noexcept
  {
    return !(__x == __y);
  }
#endif // _CCCL_STD_VER <= 2017

#if !_CCCL_COMPILER(NVRTC)

  template <typename _CharT, typename _Traits>
  _CCCL_API friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const pcg64_engine& __e)
  {
    using ostream_type = ::std::basic_ostream<_CharT, _Traits>;
    using ios_base     = typename ostream_type::ios_base;

    // save old flags & fill character
    const typename ios_base::fmtflags __flags = __os.flags();
    const _CharT __fill                       = __os.fill();

    __os.flags(ios_base::dec | ios_base::fixed | ios_base::left);
    __os.fill(__os.widen(' '));
    // Write 64 bits at a time
    ::cuda::std::uint64_t __low = static_cast<::cuda::std::uint64_t>(__e.__x_);
    ::cuda::std::uint64_t __hi  = static_cast<::cuda::std::uint64_t>(__e.__x_ >> 64);
    __os << __low;
    __os << __os.widen(' ');
    __os << __hi;
    __os << __os.widen(' ');
    // restore flags & fill character
    __os.flags(__flags);
    __os.fill(__fill);

    return __os;
  }

  template <typename _CharT, typename _Traits>
  _CCCL_API friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, pcg64_engine& __e)
  {
    using istream_type = ::std::basic_istream<_CharT, _Traits>;
    using ios_base     = typename istream_type::ios_base;

    // save old flags
    const typename ios_base::fmtflags __flags = __is.flags();

    __is.flags(ios_base::dec | ios_base::skipws);

    ::cuda::std::uint64_t __low, __hi;
    __is >> __low;
    __is >> __hi;
    // Read engine state from stream: low 64 bits then high 64 bits.
    __e.__x_ = (static_cast<__pcg64_uint128_t>(__hi) << 64) | __low;
    // restore flags
    __is.flags(__flags);

    return __is;
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

//! @class pcg64
//! @brief A 128-bit state PCG engine producing 64-bit output values.
//!
//! This class implements the PCG XSL RR 128/64 generator described in:
//! O'neill, Melissa E. "PCG: A family of simple fast space-efficient statistically good algorithms for random number
//! generation." ACM Transactions on Mathematical Software 204 (2014): 1-46. The engine keeps a 128-bit internal state
//! and returns 64-bit pseudo-random values. PCG64 is a fast general purpose PRNG that passes common statistical tests,
//! has a long period (2^128), and can discard values in O(log n) time.
//!
//! PCG64 produces the 10000th value 11135645891219275043 when seeded with the default seed.
//!
//! Usage example:
//! @code
//!   #include <cuda/random>
//!
//!   cuda::pcg64 eng;                 // default seed
//!   uint64_t v = eng();                     // draw value
//!   eng.seed(42);                           // reseed
//!   eng.discard(10);                        // skip 10 outputs
//! @endcode
//!
using pcg64 =
  pcg64_engine<2549297995355413924ull, 4865540595714422341ull, 6364136223846793005ull, 1442695040888963407ull>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_PCG_ENGINE_H
