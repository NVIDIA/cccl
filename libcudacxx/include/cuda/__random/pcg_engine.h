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

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// PCG XSL RR 128/64
template <class _UIntType>
class pcg64_engine
{
public:
  // types
  using result_type = _UIntType;

public:
  [[nodiscard]] _CCCL_API static constexpr result_type min() noexcept
  {
    return 0;
  }
  [[nodiscard]] _CCCL_API static constexpr result_type max() noexcept
  {
    return ~result_type(0);
  }
  // engine characteristics

  // constructors and seeding functions
  _CCCL_API pcg64_engine() noexcept
      : pcg64_engine(default_seed)
  {}
  _CCCL_API explicit pcg64_engine(result_type __s) noexcept
  {
    seed(__s);
  }

  template <class _Sseq, enable_if_t<__is_seed_sequence<_Sseq, pcg64_engine>, int> = 0>
  _CCCL_API explicit pcg64_engine(_Sseq& __q) noexcept
  {
    seed(__q);
  }
  _CCCL_API void seed(result_type __s = default_seed)
  {
    __x_ = (__s + increment) * multiplier + increment;
  }
  template <class _Sseq, enable_if_t<__is_seed_sequence<_Sseq, pcg64_engine>, int> = 0>
  _CCCL_API void seed(_Sseq& __q) noexcept
  {
    // TODO
  }

  // generating functions
  [[nodiscard]] _CCCL_API result_type operator()() noexcept
  {
    __x_ = __x_ * multiplier + increment;
    return OutputTransform(__x_);
  }

  _CCCL_API void discard(uint64_t __z) noexcept
  {
    for (; __z; --__z)
    {
      (void) operator()();
    }
  }

  [[nodiscard]] _CCCL_API friend bool operator==(const pcg64_engine& __x, const pcg64_engine& __y) noexcept
  {
    return __x.__x_ == __y.__x_;
  }
  [[nodiscard]] _CCCL_API friend bool operator!=(const pcg64_engine& __x, const pcg64_engine& __y) noexcept
  {
    return !(__x == __y);
  }

private:
  using xtype      = uint64_t;
  using itype      = unsigned __int128;
  using bitcount_t = uint8_t;

  constexpr itype multiplier = ((itype) 2549297995355413924ULL << 64) | 4865540595714422341ULL;
  constexpr itype increment  = ((itype) 6364136223846793005ULL << 64) | 1442695040888963407ULL;

  __device__ xtype rotr(xtype value, bitcount_t rot)
  {
    constexpr bitcount_t bits = sizeof(xtype) * 8;
    constexpr bitcount_t mask = bits - 1;
    return (value >> rot) | (value << ((-rot) & mask));
  }
  __device__ xtype OutputTransform(itype internal)
  {
    bitcount_t rot = bitcount_t(internal >> 122);
    internal ^= internal >> 64;
    return rotr(xtype(internal), rot);
  }

  constexpr __device__ auto power_mod(itype delta)
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
    return std::make_pair(acc_mult, acc_plus);
  }
  result_type __x_;
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_LINEAR_CONGRUENTIAL_ENGINE_H
