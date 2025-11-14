//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANDOM_BINOMIAL_DISTRIBUTION_H
#define _CUDA_STD___RANDOM_BINOMIAL_DISTRIBUTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__random/is_valid.h>
#include <cuda/std/__random/uniform_real_distribution.h>
#include <cuda/std/cmath>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _IntType = int>
class binomial_distribution
{
  static_assert(__libcpp_random_is_valid_inttype<_IntType>, "IntType must be a supported integer type");

public:
  // types
  typedef _IntType result_type;

  class param_type
  {
    result_type __t_;
    double __p_;
    double __pr_;
    double __odds_ratio_;
    result_type __r0_;

  public:
    typedef binomial_distribution distribution_type;

    _CCCL_API explicit param_type(result_type __t = 1, double __p = 0.5);

    _CCCL_API result_type t() const
    {
      return __t_;
    }
    _CCCL_API double p() const
    {
      return __p_;
    }

    friend _CCCL_API bool operator==(const param_type& __x, const param_type& __y)
    {
      return __x.__t_ == __y.__t_ && __x.__p_ == __y.__p_;
    }
    friend _CCCL_API bool operator!=(const param_type& __x, const param_type& __y)
    {
      return !(__x == __y);
    }

    friend class binomial_distribution;
  };

private:
  param_type __p_;

public:
  // constructors and reset functions
  _CCCL_API binomial_distribution()
      : binomial_distribution(1)
  {}
  _CCCL_API explicit binomial_distribution(result_type __t, double __p = 0.5)
      : __p_(param_type(__t, __p))
  {}
  _CCCL_API explicit binomial_distribution(const param_type& __p)
      : __p_(__p)
  {}
  _CCCL_API void reset() {}

  // generating functions
  template <class _URNG>
  _CCCL_API result_type operator()(_URNG& __g)
  {
    return (*this)(__g, __p_);
  }
  template <class _URNG>
  _CCCL_API result_type operator()(_URNG& __g, const param_type& __p);

  // property functions
  _CCCL_API result_type t() const
  {
    return __p_.t();
  }
  _CCCL_API double p() const
  {
    return __p_.p();
  }

  _CCCL_API param_type param() const
  {
    return __p_;
  }
  _CCCL_API void param(const param_type& __p)
  {
    __p_ = __p;
  }

  _CCCL_API result_type min() const
  {
    return 0;
  }
  _CCCL_API result_type max() const
  {
    return t();
  }

  friend _CCCL_API bool operator==(const binomial_distribution& __x, const binomial_distribution& __y)
  {
    return __x.__p_ == __y.__p_;
  }
  friend _CCCL_API bool operator!=(const binomial_distribution& __x, const binomial_distribution& __y)
  {
    return !(__x == __y);
  }
};

// Kemp, C. D. "A modal method for generating binomial variables." Communications in Statistics-Theory and Methods 15.3
// (1986): 805-813.
template <class _IntType>
binomial_distribution<_IntType>::param_type::param_type(result_type __t, double __p)
    : __t_(__t)
    , __p_(__p)
{
  if (0 < __p_ && __p_ < 1)
  {
    __r0_ = static_cast<result_type>((__t_ + 1) * __p_);
    __pr_ = cuda::std::exp(
      cuda::std::lgamma(__t_ + 1.) - cuda::std::lgamma(__r0_ + 1.) - cuda::std::lgamma(__t_ - __r0_ + 1.)
      + __r0_ * cuda::std::log(__p_) + (__t_ - __r0_) * cuda::std::log(1 - __p_));
    __odds_ratio_ = __p_ / (1 - __p_);
  }
}

// Reference: Kemp, C. D. "A modal method for generating binomial variables." Communications in Statistics-Theory and
// Methods 15.3 (1986): 805-813.
template <class _IntType>
template <class _URNG>
_IntType binomial_distribution<_IntType>::operator()(_URNG& __g, const param_type& __pr)
{
  static_assert(__cccl_random_is_valid_urng<_URNG>);
  if (__pr.__t_ == 0 || __pr.__p_ == 0)
  {
    return 0;
  }
  if (__pr.__p_ == 1)
  {
    return __pr.__t_;
  }
  uniform_real_distribution<double> __gen;
  double __u = __gen(__g) - __pr.__pr_;
  if (__u < 0)
  {
    return __pr.__r0_;
  }
  double __pu      = __pr.__pr_;
  double __pd      = __pu;
  result_type __ru = __pr.__r0_;
  result_type __rd = __ru;
  while (true)
  {
    bool __break = true;
    if (__rd >= 1)
    {
      __pd *= __rd / (__pr.__odds_ratio_ * (__pr.__t_ - __rd + 1));
      __u -= __pd;
      __break = false;
      if (__u < 0)
      {
        return __rd - 1;
      }
    }
    if (__rd != 0)
    {
      --__rd;
    }
    ++__ru;
    if (__ru <= __pr.__t_)
    {
      __pu *= (__pr.__t_ - __ru + 1) * __pr.__odds_ratio_ / __ru;
      __u -= __pu;
      __break = false;
      if (__u < 0)
      {
        return __ru;
      }
    }
    if (__break)
    {
      return 0;
    }
  }
}

template <class _CharT, class _Traits, class _IntType>
basic_ostream<_CharT, _Traits>&
operator<<(basic_ostream<_CharT, _Traits>& __os, const binomial_distribution<_IntType>& __x)
{
  __save_flags<_CharT, _Traits> __lx(__os);
  typedef basic_ostream<_CharT, _Traits> _OStream;
  __os.flags(_OStream::dec | _OStream::left | _OStream::fixed | _OStream::scientific);
  _CharT __sp = __os.widen(' ');
  __os.fill(__sp);
  return __os << __x.t() << __sp << __x.p();
}

template <class _CharT, class _Traits, class _IntType>
basic_istream<_CharT, _Traits>& operator>>(basic_istream<_CharT, _Traits>& __is, binomial_distribution<_IntType>& __x)
{
  typedef binomial_distribution<_IntType> _Eng;
  typedef typename _Eng::result_type result_type;
  typedef typename _Eng::param_type param_type;
  __save_flags<_CharT, _Traits> __lx(__is);
  typedef basic_istream<_CharT, _Traits> _Istream;
  __is.flags(_Istream::dec | _Istream::skipws);
  result_type __t;
  double __p;
  __is >> __t >> __p;
  if (!__is.fail())
  {
    __x.param(param_type(__t, __p));
  }
  return __is;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_BINOMIAL_DISTRIBUTION_H
