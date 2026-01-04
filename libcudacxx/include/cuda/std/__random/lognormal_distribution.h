//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___LOGNORMAL_DISTRIBUTION_H
#define _CUDA_STD___LOGNORMAL_DISTRIBUTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/exponential_functions.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__random/is_valid.h>
#include <cuda/std/__random/normal_distribution.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <iosfwd>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _RealType = double>
class lognormal_distribution
{
  static_assert(__cccl_random_is_valid_realtype<_RealType>, "RealType must be a supported floating-point type");

public:
  // types
  using result_type = _RealType;

  class param_type
  {
  private:
    result_type __m_ = result_type{0};
    result_type __s_ = result_type{1};

  public:
    using distribution_type = lognormal_distribution;

    _CCCL_API constexpr explicit param_type(result_type __m = result_type{0}, result_type __s = result_type{1}) noexcept
        : __m_{__m}
        , __s_{__s}
    {}

    [[nodiscard]] _CCCL_API constexpr result_type m() const noexcept
    {
      return __m_;
    }
    [[nodiscard]] _CCCL_API constexpr result_type s() const noexcept
    {
      return __s_;
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const param_type& __x, const param_type& __y) noexcept
    {
      return __x.__m_ == __y.__m_ && __x.__s_ == __y.__s_;
    }
#if _CCCL_STD_VER <= 2017
    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const param_type& __x, const param_type& __y) noexcept
    {
      return !(__x == __y);
    }
#endif // _CCCL_STD_VER <= 2017
  };

private:
  normal_distribution<result_type> __nd_;

public:
  // constructor and reset functions
  constexpr lognormal_distribution() noexcept = default;
  _CCCL_API constexpr explicit lognormal_distribution(result_type __m, result_type __s = result_type{1}) noexcept
      : __nd_{__m, __s}
  {}
  _CCCL_API constexpr explicit lognormal_distribution(const param_type& __p) noexcept
      : __nd_{__p.m(), __p.s()}
  {}
  _CCCL_API constexpr void reset() noexcept
  {
    __nd_.reset();
  }

  // generating functions
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g)
  {
    return ::cuda::std::exp(__nd_(__g));
  }

  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g, const param_type& __p)
  {
    typename normal_distribution<result_type>::param_type __pn{__p.m(), __p.s()};
    return ::cuda::std::exp(__nd_(__g, __pn));
  }

  // property functions
  [[nodiscard]] _CCCL_API constexpr result_type m() const noexcept
  {
    return __nd_.mean();
  }
  [[nodiscard]] _CCCL_API constexpr result_type s() const noexcept
  {
    return __nd_.stddev();
  }

  [[nodiscard]] _CCCL_API constexpr param_type param() const noexcept
  {
    return param_type{__nd_.mean(), __nd_.stddev()};
  }
  _CCCL_API constexpr void param(const param_type& __p) noexcept
  {
    typename normal_distribution<result_type>::param_type __pn{__p.m(), __p.s()};
    __nd_.param(__pn);
  }

  [[nodiscard]] _CCCL_API static constexpr result_type min() noexcept
  {
    return result_type{0};
  }
  [[nodiscard]] _CCCL_API static constexpr result_type max() noexcept
  {
    return numeric_limits<result_type>::infinity();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const lognormal_distribution& __x, const lognormal_distribution& __y) noexcept
  {
    return __x.__nd_ == __y.__nd_;
  }
#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const lognormal_distribution& __x, const lognormal_distribution& __y) noexcept
  {
    return !(__x == __y);
  }
#endif // _CCCL_STD_VER <= 2017

#if !_CCCL_COMPILER(NVRTC)
  template <class _CharT, class _Traits>
  friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const lognormal_distribution& __x)
  {
    return __os << __x.__nd_;
  }

  template <class _CharT, class _Traits>
  friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, lognormal_distribution& __x)
  {
    return __is >> __x.__nd_;
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___LOGNORMAL_DISTRIBUTION_H
