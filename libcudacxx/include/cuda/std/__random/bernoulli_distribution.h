//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===////===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BERNOULLI_DISTRIBUTION_H
#define _CUDA_STD___BERNOULLI_DISTRIBUTION_H

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
#if !_CCCL_COMPILER(NVRTC)
#  include <ios>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

class bernoulli_distribution
{
public:
  // types
  typedef bool result_type;

  class param_type
  {
    double __p_;

  public:
    typedef bernoulli_distribution distribution_type;

    _CCCL_API explicit param_type(double __p = 0.5)
        : __p_(__p)
    {}

    _CCCL_API double p() const
    {
      return __p_;
    }

    friend _CCCL_API bool operator==(const param_type& __x, const param_type& __y)
    {
      return __x.__p_ == __y.__p_;
    }
    friend _CCCL_API bool operator!=(const param_type& __x, const param_type& __y)
    {
      return !(__x == __y);
    }
  };

private:
  param_type __p_;

public:
  // constructors and reset functions
  _CCCL_API bernoulli_distribution()
      : bernoulli_distribution(0.5)
  {}
  _CCCL_API explicit bernoulli_distribution(double __p)
      : __p_(param_type(__p))
  {}
  _CCCL_API explicit bernoulli_distribution(const param_type& __p)
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
  _CCCL_API result_type operator()(_URNG& __g, const param_type& __p)
  {
    static_assert(__cccl_random_is_valid_urng<_URNG>, "");
    uniform_real_distribution<double> __gen;
    return __gen(__g) < __p.p();
  }

  // property functions
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
    return false;
  }
  _CCCL_API result_type max() const
  {
    return true;
  }

  friend _CCCL_API bool operator==(const bernoulli_distribution& __x, const bernoulli_distribution& __y)
  {
    return __x.__p_ == __y.__p_;
  }
  friend _CCCL_API bool operator!=(const bernoulli_distribution& __x, const bernoulli_distribution& __y)
  {
    return !(__x == __y);
  }
#if !_CCCL_COMPILER(NVRTC)
  template <class _CharT, class _Traits>
  friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const bernoulli_distribution& __x)
  {
    using ostream_type                        = ::std::basic_ostream<_CharT, _Traits>;
    using ios_base                            = typename ostream_type::ios_base;
    const typename ios_base::fmtflags __flags = __os.flags();
    __os.flags(ios_base::dec | ios_base::left | ios_base::fixed | ios_base::scientific);
    _CharT __sp = __os.widen(' ');
    __os.fill(__sp);
    __os.flags(__flags);
    return __os << __x.p();
  }

  template <class _CharT, class _Traits>
  friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, bernoulli_distribution& __x)
  {
    using istream_type                        = ::std::basic_istream<_CharT, _Traits>;
    using ios_base                            = typename istream_type::ios_base;
    const typename ios_base::fmtflags __flags = __is.flags();
    typedef bernoulli_distribution _Eng;
    typedef typename _Eng::param_type param_type;
    __is.flags(ios_base::dec | ios_base::skipws);
    double __p;
    __is >> __p;
    if (!__is.fail())
    {
      __x.param(param_type(__p));
    }
    __is.flags(__flags);
    return __is;
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BERNOULLI_DISTRIBUTION_H
