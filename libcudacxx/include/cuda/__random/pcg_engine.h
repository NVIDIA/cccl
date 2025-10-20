//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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

#include <cuda/std/__random/is_seed_sequence.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/utility>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// PCG XSL RR 128/64
class pcg64_engine
{
public:
  using result_type                         = ::cuda::std::uint64_t;
  static constexpr result_type default_seed = 0xcafef00dd15ea5e5ULL;

  [[nodiscard]] _CCCL_API static constexpr result_type min() noexcept
  {
    return 0;
  }
  [[nodiscard]] _CCCL_API static constexpr result_type max() noexcept
  {
    return ::cuda::std::numeric_limits<result_type>::max();
  }

  // constructors and seeding functions
  constexpr _CCCL_API pcg64_engine() noexcept
      : pcg64_engine(default_seed)
  {}
  constexpr _CCCL_API explicit pcg64_engine(result_type __seed) noexcept
  {
    seed(__seed);
  }

  _CCCL_TEMPLATE(class _Sseq)
  _CCCL_REQUIRES(::cuda::std::__is_seed_sequence<_Sseq, pcg64_engine>)
  constexpr _CCCL_API explicit pcg64_engine(_Sseq& __seq)
  {
    seed(__seq);
  }
  constexpr _CCCL_API void seed(result_type __seed = default_seed) noexcept
  {
    __x_ = (__seed + increment) * multiplier + increment;
  }

  _CCCL_TEMPLATE(class _Sseq)
  _CCCL_REQUIRES(::cuda::std::__is_seed_sequence<_Sseq, pcg64_engine>)
  constexpr _CCCL_API void seed(_Sseq& __seq)
  {
    ::cuda::std::array<uint32_t, 4> data = {};
    __seq.generate(data.begin(), data.end());
    itype seed_val = data[0];
    seed_val       = (seed_val << 32) | data[1];
    seed_val       = (seed_val << 32) | data[2];
    seed_val       = (seed_val << 32) | data[3];
    __x_           = (seed_val + increment) * multiplier + increment;
  }

  // generating functions
  constexpr _CCCL_API result_type operator()() noexcept
  {
    __x_ = __x_ * multiplier + increment;
    return OutputTransform(__x_);
  }

  constexpr _CCCL_API void discard(unsigned long long __z) noexcept
  {
    for (; __z; --__z)
    {
      (void) operator()();
    }
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const pcg64_engine& __x, const pcg64_engine& __y) noexcept
  {
    return __x.__x_ == __y.__x_;
  }
  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const pcg64_engine& __x, const pcg64_engine& __y) noexcept
  {
    return !(__x == __y);
  }
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
    ::cuda::std::uint64_t low = static_cast<::cuda::std::uint64_t>(__e.__x_);
    ::cuda::std::uint64_t hi  = static_cast<::cuda::std::uint64_t>(__e.__x_ >> 64);
    __os << low;
    __os << __os.widen(' ');
    __os << hi;
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

    ::cuda::std::uint64_t low, hi;
    __is >> low;
    __is >> hi;
    __e.__x_ = (static_cast<itype>(hi) << 64) | low;

    // restore flags
    __is.flags(__flags);

    return __is;
  }
#endif // !_CCCL_COMPILER(NVRTC)

private:
  using xtype      = uint64_t;
  using itype      = unsigned __int128;
  using bitcount_t = ::cuda::std::uint8_t;

  static constexpr itype multiplier = ((itype) 2549297995355413924ULL << 64) | 4865540595714422341ULL;
  static constexpr itype increment  = ((itype) 6364136223846793005ULL << 64) | 1442695040888963407ULL;

  [[nodiscard]] _CCCL_API constexpr xtype rotr(xtype value, bitcount_t rot) noexcept
  {
    constexpr bitcount_t bits = sizeof(xtype) * 8;
    constexpr bitcount_t mask = bits - 1;
    return (value >> rot) | (value << ((-rot) & mask));
  }
  [[nodiscard]] _CCCL_API constexpr xtype OutputTransform(itype internal) noexcept
  {
    bitcount_t rot = bitcount_t(internal >> 122);
    internal ^= internal >> 64;
    return rotr(xtype(internal), rot);
  }

  [[nodiscard]] _CCCL_API constexpr auto power_mod(itype delta) noexcept
  {
    constexpr itype ZERO = 0u;
    constexpr itype ONE  = 1u;
    itype acc_mult       = 1;
    itype acc_plus       = 0;
    itype cur_mult       = multiplier;
    itype cur_plus       = increment;
    while (delta > ZERO)
    {
      if (delta & ONE)
      {
        acc_mult *= cur_mult;
        acc_plus = acc_plus * cur_mult + cur_plus;
      }
      cur_plus = (cur_mult + ONE) * cur_plus;
      cur_mult *= cur_mult;
      delta >>= 1;
    }
    return ::cuda::std::make_pair(acc_mult, acc_plus);
  }
  itype __x_{};
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_LINEAR_CONGRUENTIAL_ENGINE_H
