//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------====//

#ifndef _CUDA_STD___NEGATIVE_BINOMIAL_DISTRIBUTION_H
#define _CUDA_STD___NEGATIVE_BINOMIAL_DISTRIBUTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__random/bernoulli_distribution.h>
#include <cuda/std/__random/gamma_distribution.h>
#include <cuda/std/__random/is_valid.h>
#include <cuda/std/__random/poisson_distribution.h>
#if !_CCCL_COMPILER(NVRTC)
#  include <ios>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _IntType = int>
class negative_binomial_distribution
{
  static_assert(__cccl_random_is_valid_inttype<_IntType>, "IntType must be a supported integer type");

public:
  // types
  using result_type = _IntType;

  class param_type
  {
    result_type __k_ = result_type{1};
    double __p_      = 0.5;

  public:
    using distribution_type = negative_binomial_distribution;

    _CCCL_API constexpr explicit param_type(result_type __k = 1, double __p = 0.5) noexcept
        : __k_{__k}
        , __p_{__p}
    {}

    [[nodiscard]] _CCCL_API constexpr result_type k() const noexcept
    {
      return __k_;
    }
    [[nodiscard]] _CCCL_API constexpr double p() const noexcept
    {
      return __p_;
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const param_type& __x, const param_type& __y) noexcept
    {
      return __x.__k_ == __y.__k_ && __x.__p_ == __y.__p_;
    }
#if _CCCL_STD_VER <= 2017
    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const param_type& __x, const param_type& __y) noexcept
    {
      return !(__x == __y);
    }
#endif // _CCCL_STD_VER <= 2017
  };

private:
  param_type __p_{};

public:
  // constructor and reset functions
  constexpr negative_binomial_distribution() noexcept = default;

  _CCCL_API constexpr explicit negative_binomial_distribution(result_type __k, double __p = 0.5) noexcept
      : __p_{__k, __p}
  {}
  _CCCL_API constexpr explicit negative_binomial_distribution(const param_type& __p) noexcept
      : __p_{__p}
  {}
  _CCCL_API constexpr void reset() noexcept {}

  // generating functions
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g)
  {
    return (*this)(__g, __p_);
  }
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __urng, const param_type& __pr)
  {
    static_assert(__cccl_random_is_valid_urng<_URng>, "URng must meet the UniformRandomBitGenerator requirements");
    const result_type __k = __pr.k();
    const double __p      = __pr.p();
    // When the number of bits in _IntType is small, we are too likely to
    // overflow __f below to use this technique.
    if (__k <= 21 * __p && sizeof(_IntType) > 1)
    {
      bernoulli_distribution __gen(__p);
      result_type __f = 0;
      result_type __s = 0;
      while (__s < __k)
      {
        if (__gen(__urng))
        {
          ++__s;
        }
        else
        {
          ++__f;
        }
      }
      return __f;
    }
    return poisson_distribution<result_type>(gamma_distribution<double>(__k, (1 - __p) / __p)(__urng))(__urng);
  }

  // property functions
  [[nodiscard]] _CCCL_API constexpr result_type k() const noexcept
  {
    return __p_.k();
  }
  [[nodiscard]] _CCCL_API constexpr double p() const noexcept
  {
    return __p_.p();
  }

  [[nodiscard]] _CCCL_API constexpr param_type param() const noexcept
  {
    return __p_;
  }
  _CCCL_API constexpr void param(const param_type& __p) noexcept
  {
    __p_ = __p;
  }

  [[nodiscard]] _CCCL_API static constexpr result_type min() noexcept
  {
    return 0;
  }
  [[nodiscard]] _CCCL_API static constexpr result_type max() noexcept
  {
    return numeric_limits<result_type>::max();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const negative_binomial_distribution& __x, const negative_binomial_distribution& __y) noexcept
  {
    return __x.__p_ == __y.__p_;
  }
#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const negative_binomial_distribution& __x, const negative_binomial_distribution& __y) noexcept
  {
    return !(__x == __y);
  }
#endif // _CCCL_STD_VER <= 2017

#if !_CCCL_COMPILER(NVRTC)
  template <class _CharT, class _Traits>
  friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const negative_binomial_distribution& __x)
  {
    using _Ostream = ::std::basic_ostream<_CharT, _Traits>;
    auto __flags   = __os.flags();
    __os.flags(_Ostream::dec | _Ostream::left | _Ostream::scientific);
    _CharT __sp      = __os.widen(' ');
    _CharT __fill    = __os.fill(__sp);
    auto __precision = __os.precision(numeric_limits<double>::max_digits10);
    __os << __x.k() << __sp << __x.p();
    __os.precision(__precision);
    __os.fill(__fill);
    __os.flags(__flags);
    return __os;
  }

  template <class _CharT, class _Traits>
  friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, negative_binomial_distribution& __x)
  {
    using _Istream = ::std::basic_istream<_CharT, _Traits>;
    auto __flags   = __is.flags();
    __is.flags(_Istream::skipws);
    result_type __k;
    double __p;
    __is >> __k >> __p;
    if (!__is.fail())
    {
      __x.param(param_type(__k, __p));
    }
    __is.flags(__flags);
    return __is;
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NEGATIVE_BINOMIAL_DISTRIBUTION_H
