//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANDOM_NORMAL_DISTRIBUTION_H
#define _CUDA_STD___RANDOM_NORMAL_DISTRIBUTION_H

#include <cuda/std/detail/__config>

#include <cuda/std/__cmath/logarithms.h>
#include <cuda/std/__cmath/roots.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__random/is_valid.h>
#include <cuda/std/__random/uniform_real_distribution.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)
#  include <ios>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _RealType = double>
class normal_distribution
{
  static_assert(__cccl_random_is_valid_realtype<_RealType>, "RealType must be a supported floating-point type");

public:
  // types
  using result_type = _RealType;

  class param_type
  {
    result_type __mean_;
    result_type __stddev_;

  public:
    using distribution_type = normal_distribution;

    _CCCL_API constexpr explicit param_type(result_type __mean = 0, result_type __stddev = 1) noexcept
        : __mean_{__mean}
        , __stddev_{__stddev}
    {}

    [[nodiscard]] _CCCL_API constexpr result_type mean() const noexcept
    {
      return __mean_;
    }
    [[nodiscard]] _CCCL_API constexpr result_type stddev() const noexcept
    {
      return __stddev_;
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const param_type& __x, const param_type& __y) noexcept
    {
      return __x.__mean_ == __y.__mean_ && __x.__stddev_ == __y.__stddev_;
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
  result_type __v_{};
  bool __v_hot_{false};

public:
  _CCCL_API constexpr normal_distribution() noexcept
      : normal_distribution{0}
  {}
  _CCCL_API constexpr explicit normal_distribution(result_type __mean, result_type __stddev = result_type{1}) noexcept
      : __p_{param_type(__mean, __stddev)}
  {}
  _CCCL_API constexpr explicit normal_distribution(const param_type& __p) noexcept
      : __p_{__p}
  {}
  _CCCL_API constexpr void reset() noexcept
  {
    __v_hot_ = false;
  }

  // generating functions
  template <class _URng>
  _CCCL_API constexpr result_type operator()(_URng& __g)
  {
    return (*this)(__g, __p_);
  }
  template <class _URng>
  _CCCL_API constexpr result_type operator()(_URng& __g, const param_type& __p)
  {
    static_assert(__cccl_random_is_valid_urng<_URng>, "URng must meet the UniformRandomBitGenerator requirements");
    result_type __up = 0;
    if (__v_hot_)
    {
      __v_hot_ = false;
      __up     = __v_;
    }
    else
    {
      uniform_real_distribution<result_type> __uni(-1, 1);
      result_type __u = __uni(__g);
      result_type __v = __uni(__g);
      result_type __s = __u * __u + __v * __v;
      while (__s > 1 || __s == 0)
      {
        __u = __uni(__g);
        __v = __uni(__g);
        __s = __u * __u + __v * __v;
      }
      const result_type __fp = ::cuda::std::sqrt(-2 * ::cuda::std::log(__s) / __s);
      __v_                   = __v * __fp;
      __v_hot_               = true;
      __up                   = __u * __fp;
    }
    return __up * __p.stddev() + __p.mean();
  }

  // property functions
  [[nodiscard]] _CCCL_API constexpr result_type mean() const noexcept
  {
    return __p_.mean();
  }
  [[nodiscard]] _CCCL_API constexpr result_type stddev() const noexcept
  {
    return __p_.stddev();
  }

  [[nodiscard]] _CCCL_API constexpr param_type param() const noexcept
  {
    return __p_;
  }
  _CCCL_API constexpr void param(const param_type& __p) noexcept
  {
    __p_ = __p;
  }

  [[nodiscard]] _CCCL_API constexpr result_type min() const noexcept
  {
    return -numeric_limits<result_type>::infinity();
  }
  [[nodiscard]] _CCCL_API constexpr result_type max() const noexcept
  {
    return numeric_limits<result_type>::infinity();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const normal_distribution& __x, const normal_distribution& __y) noexcept
  {
    return __x.__p_ == __y.__p_ && __x.__v_hot_ == __y.__v_hot_ && (!__x.__v_hot_ || __x.__v_ == __y.__v_);
  }
#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const normal_distribution& __x, const normal_distribution& __y) noexcept
  {
    return !(__x == __y);
  }
#endif // _CCCL_STD_VER <= 2017

#if !_CCCL_COMPILER(NVRTC)
  template <class _CharT, class _Traits>
  friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const normal_distribution& __x)
  {
    _CharT __sp                       = __os.widen(' ');
    ::std::ios_base::fmtflags __flags = __os.flags();
    __os.flags(::std::ios_base::dec | ::std::ios_base::left | ::std::ios_base::fixed);
    _CharT __fill                 = __os.fill(__sp);
    ::std::streamsize __precision = __os.precision(17); // Max precision for double
    __os << __x.mean() << __sp << __x.stddev() << __sp << __x.__v_hot_;
    if (__x.__v_hot_)
    {
      __os << __sp << __x.__v_;
    }
    __os.precision(__precision);
    __os.fill(__fill);
    __os.flags(__flags);
    return __os;
  }

  template <class _CharT, class _Traits>
  friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, normal_distribution& __x)
  {
    using _Istream = ::std::basic_istream<_CharT, _Traits>;
    auto __flags   = __is.flags();
    __is.flags(_Istream::skipws);
    result_type __mean;
    result_type __stddev;
    result_type __vp = 0;
    int __v_hot_int  = 0;
    __is >> __mean >> __stddev >> __v_hot_int;
    bool __v_hot = __v_hot_int != 0;
    if (__v_hot)
    {
      __is >> __vp;
    }
    if (!__is.fail())
    {
      __x.param(param_type(__mean, __stddev));
      __x.__v_hot_ = __v_hot;
      __x.__v_     = __vp;
    }
    __is.flags(__flags);
    return __is;
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_NORMAL_DISTRIBUTION_H
