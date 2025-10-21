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

#include <cuda/std/__bit/rotate.h>
#include <cuda/std/__random/is_seed_sequence.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/utility>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#if _CCCL_HAS_INT128()

/// @class pcg64_engine
/// @brief A 128-bit state PCG engine producing 64-bit output values.
///
/// This class implements the PCG XSL RR 128/64 generator described in:
/// O’neill, Melissa E. "PCG: A family of simple fast space-efficient statistically good algorithms for random number
/// generation." ACM Transactions on Mathematical Software 204 (2014): 1-46. The engine keeps a 128-bit internal state
/// and returns 64-bit pseudo-random values. PCG64 is a fast general purpose PRNG that passes common statistical tests,
/// has a long period (2^128), and can discard values in O(log n) time.
///
/// PCG64 produces the 10000th value 11135645891219275043 when seeded with the default seed.
///
/// Usage example:
/// @code
///   #include <cuda/std/random>
///
///   cuda::pcg64_engine eng;                 // default seed
///   uint64_t v = eng();                     // draw value
///   eng.seed(42);                           // reseed
///   eng.discard(10);                        // skip 10 outputs
/// @endcode
///
class pcg64_engine
{
public:
  using result_type                         = ::cuda::std::uint64_t;
  static constexpr result_type default_seed = 0xcafef00dd15ea5e5ULL;

  /// @brief Returns the smallest value the engine can produce.
  /// @return Always 0 for pcg64_engine.
  [[nodiscard]] _CCCL_API static constexpr result_type min() noexcept
  {
    return 0;
  }
  /// @brief Returns the largest value the engine can produce.
  /// @return The maximum representable `result_type`.
  [[nodiscard]] _CCCL_API static constexpr result_type max() noexcept
  {
    return ::cuda::std::numeric_limits<result_type>::max();
  }

  // constructors and seeding functions
  /// @brief Default-constructs the engine using `default_seed`.
  constexpr _CCCL_API pcg64_engine() noexcept
      : pcg64_engine(default_seed)
  {}
  /// @brief Constructs the engine and seeds it with `__seed`.
  /// @param __seed The seed value used to initialize the engine state.
  constexpr _CCCL_API explicit pcg64_engine(result_type __seed) noexcept
  {
    seed(__seed);
  }

  _CCCL_TEMPLATE(class _Sseq)
  /// @brief Constructs the engine and seeds it from a SeedSequence-like object.
  /// @tparam _Sseq A SeedSequence-like type satisfying the project's seed concept.
  /// @param __seq The seed sequence used to initialize the internal state.
  _CCCL_REQUIRES(::cuda::std::__is_seed_sequence<_Sseq, pcg64_engine>)
  constexpr _CCCL_API explicit pcg64_engine(_Sseq& __seq)
  {
    seed(__seq);
  }
  /// @brief Seed the engine with an integer seed.
  /// @param __seed The seed value; defaults to `default_seed`.
  constexpr _CCCL_API void seed(result_type __seed = default_seed) noexcept
  {
    __x_ = (__seed + __increment) * __multiplier + __increment;
  }

  /// @brief Seed the engine from a SeedSequence-like object.
  /// @tparam _Sseq A SeedSequence-like type providing entropy words.
  /// @param __seq A SeedSequence-like object providing 128 bits of entropy.
  _CCCL_TEMPLATE(class _Sseq)
  _CCCL_REQUIRES(::cuda::std::__is_seed_sequence<_Sseq, pcg64_engine>)
  constexpr _CCCL_API void seed(_Sseq& __seq)
  {
    ::cuda::std::array<uint32_t, 4> data = {};
    __seq.generate(data.begin(), data.end());
    __uint128_t seed_val = data[0];
    seed_val             = (seed_val << 32) | data[1];
    seed_val             = (seed_val << 32) | data[2];
    seed_val             = (seed_val << 32) | data[3];
    __x_                 = (seed_val + __increment) * __multiplier + __increment;
  }

  /// @brief Generate the next pseudo-random value.
  ///
  /// Advances the internal LCG state and applies the PCG output
  /// permutation to produce a 64-bit result.
  /// @return A 64-bit pseudo-random value.
  constexpr _CCCL_API result_type operator()() noexcept
  {
    __x_ = __x_ * __multiplier + __increment;
    return __output_transform(__x_);
  }

  /// @brief Advance the engine state by `__z` steps, discarding outputs.
  /// @param __z Number of values to discard.
  constexpr _CCCL_API void discard(unsigned long long __z) noexcept
  {
    auto [__mult, __plus] = __power_mod(__z);
    __x_                  = __x_ * __mult + __plus;
  }

  /// @brief Equality comparison for two engines.
  /// @return True if both engines have identical internal state.
  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const pcg64_engine& __x, const pcg64_engine& __y) noexcept
  {
    return __x.__x_ == __y.__x_;
  }
  /// @brief Inequality comparison for two engines.
  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const pcg64_engine& __x, const pcg64_engine& __y) noexcept
  {
    return !(__x == __y);
  }
#  if !_CCCL_COMPILER(NVRTC)
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
    __e.__x_ = (static_cast<__uint128_t>(__hi) << 64) | __low;
    // restore flags
    __is.flags(__flags);

    return __is;
  }
#  endif // !_CCCL_COMPILER(NVRTC)

private:
  using __bitcount_t = ::cuda::std::uint8_t;

  static constexpr __uint128_t __multiplier = ((__uint128_t) 2549297995355413924ULL << 64) | 4865540595714422341ULL;
  static constexpr __uint128_t __increment  = ((__uint128_t) 6364136223846793005ULL << 64) | 1442695040888963407ULL;
  [[nodiscard]] _CCCL_API constexpr result_type __output_transform(__uint128_t __internal) noexcept
  {
    __bitcount_t __rot = __bitcount_t(__internal >> 122);
    __internal ^= __internal >> 64;
    return ::cuda::std::rotr(result_type(__internal), __rot);
  }

  [[nodiscard]] _CCCL_API constexpr ::cuda::std::pair<__uint128_t, __uint128_t> __power_mod(__uint128_t __delta) noexcept
  {
    __uint128_t __acc_mult = 1;
    __uint128_t __acc_plus = 0;
    __uint128_t __cur_mult = __multiplier;
    __uint128_t __cur_plus = __increment;
    while (__delta > 0)
    {
      if (__delta & 1)
      {
        __acc_mult *= __cur_mult;
        __acc_plus = __acc_plus * __cur_mult + __cur_plus;
      }
      __cur_plus = (__cur_mult + 1) * __cur_plus;
      __cur_mult *= __cur_mult;
      __delta >>= 1;
    }
    return ::cuda::std::pair{__acc_mult, __acc_plus};
  }
  __uint128_t __x_{};
};

#endif // _CCCL_HAS_INT128()
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_LINEAR_CONGRUENTIAL_ENGINE_H
