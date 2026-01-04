//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------====//

#ifndef _CUDA_STD___POISSON_DISTRIBUTION_H
#define _CUDA_STD___POISSON_DISTRIBUTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__random/generate_canonical.h>
#include <cuda/std/__random/is_valid.h>
#include <cuda/std/__random/normal_distribution.h>
#include <cuda/std/__random/uniform_real_distribution.h>
#include <cuda/std/cmath>
#if !_CCCL_COMPILER(NVRTC)
#  include <ios>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _IntType = int>
class poisson_distribution
{
  static_assert(__cccl_random_is_valid_inttype<_IntType>, "IntType must be a supported integer type");

public:
  // types
  using result_type = _IntType;

  class param_type
  {
    double __mean_  = 1.0;
    double __s_     = 0.0;
    double __d_     = 0.0;
    double __l_     = 0.0;
    double __omega_ = 0.0;
    double __c0_    = 0.0;
    double __c1_    = 0.0;
    double __c2_    = 0.0;
    double __c3_    = 0.0;
    double __c_     = 0.0;

  public:
    using distribution_type = poisson_distribution;

    _CCCL_API explicit param_type(double __mean = 1.0) noexcept
        // According to the standard `inf` is a valid input, but it causes the
        // distribution to hang, so we replace it with the maximum representable
        // mean.
        : __mean_{isinf(__mean) ? numeric_limits<double>::max() : __mean}
    {
      if (__mean_ < 10)
      {
        __l_ = ::cuda::std::exp(-__mean_);
      }
      else
      {
        __s_        = ::cuda::std::sqrt(__mean_);
        __d_        = 6 * __mean_ * __mean_;
        __l_        = ::cuda::std::trunc(__mean_ - 1.1484);
        __omega_    = .3989423 / __s_;
        double __b1 = .4166667E-1 / __mean_;
        double __b2 = .3 * __b1 * __b1;
        __c3_       = .1428571 * __b1 * __b2;
        __c2_       = __b2 - 15. * __c3_;
        __c1_       = __b1 - 6. * __b2 + 45. * __c3_;
        __c0_       = 1. - __b1 + 3. * __b2 - 15. * __c3_;
        __c_        = .1069 / __mean_;
      }
    }

    [[nodiscard]] _CCCL_API constexpr double mean() const noexcept
    {
      return __mean_;
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const param_type& __x, const param_type& __y) noexcept
    {
      return __x.__mean_ == __y.__mean_;
    }
#if _CCCL_STD_VER <= 2017
    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const param_type& __x, const param_type& __y) noexcept
    {
      return !(__x == __y);
    }
#endif // _CCCL_STD_VER <= 2017

    friend class poisson_distribution;
  };

private:
  param_type __p_{};

  template <class _IntT,
            class _FloatT,
            bool _FloatBigger = (numeric_limits<_FloatT>::digits > numeric_limits<_IntT>::digits)>
  [[nodiscard]] _CCCL_API static constexpr _IntT __max_representable_int_for_float() noexcept
  {
    static_assert(::cuda::std::is_floating_point<_FloatT>::value, "must be a floating point type");
    static_assert(::cuda::std::is_integral<_IntT>::value, "must be an integral type");
    static_assert(numeric_limits<_FloatT>::radix == 2, "FloatT has incorrect radix");
    constexpr int _bits = cuda::std::max(numeric_limits<_IntT>::digits - numeric_limits<_FloatT>::digits, 0);
    return _FloatBigger ? numeric_limits<_IntT>::max() : (numeric_limits<_IntT>::max() >> _bits << _bits);
  }

  template <class _IntT, class _RealT>
  [[nodiscard]] _CCCL_API static _IntT __clamp_to_integral(_RealT __r) noexcept
  {
    using _Limits         = numeric_limits<_IntT>;
    const _IntT __max_val = __max_representable_int_for_float<_IntT, _RealT>();
    if (__r >= ::cuda::std::nextafter(static_cast<_RealT>(__max_val), INFINITY))
    {
      return _Limits::max();
    }
    else if (__r <= _Limits::lowest())
    {
      return _Limits::min();
    }
    return static_cast<_IntT>(__r);
  }

public:
  // constructors and reset functions
  _CCCL_API constexpr poisson_distribution() noexcept
      : poisson_distribution{1.0}
  {}
  _CCCL_API constexpr explicit poisson_distribution(double __mean) noexcept
      : __p_{__mean}
  {}
  _CCCL_API constexpr explicit poisson_distribution(const param_type& __p) noexcept
      : __p_{__p}
  {}
  _CCCL_API constexpr void reset() noexcept {}

  // generating functions
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g)
  {
    return (*this)(__g, __p_);
  }
  template <class _URNG>
  [[nodiscard]] _CCCL_API result_type operator()(_URNG& __urng, const param_type& __pr)
  {
    static_assert(__cccl_random_is_valid_urng<_URNG>, "");
    double __tx = 0;
    uniform_real_distribution<double> __urd{};
    if (__pr.__mean_ < 10)
    {
      for (double __p = __urd(__urng); __p > __pr.__l_; ++__tx)
      {
        __p *= __urd(__urng);
      }
    }
    else
    {
      double __difmuk = 0;
      double __g      = __pr.__mean_ + __pr.__s_ * normal_distribution<double>()(__urng);
      double __u      = 0;
      if (__g > 0)
      {
        __tx = ::cuda::std::trunc(__g);
        if (__tx >= __pr.__l_)
        {
          return __clamp_to_integral<result_type>(__tx);
        }
        __difmuk = __pr.__mean_ - __tx;
        __u      = __urd(__urng);
        if (__pr.__d_ * __u >= __difmuk * __difmuk * __difmuk)
        {
          return __clamp_to_integral<result_type>(__tx);
        }
      }
      for (bool __using_exp_dist = false; true; __using_exp_dist = true)
      {
        double __e = 0;
        if (__using_exp_dist || __g <= 0)
        {
          double __t = 0;
          do
          {
            // Inline exponential distribution: -log(1 - U) where U ~ Uniform(0,1)
            __e =
              -::cuda::std::log(1.0 - ::cuda::std::generate_canonical<double, numeric_limits<double>::digits>(__urng));
            __u = __urd(__urng);
            __u += __u - 1;
            __t = 1.8 + (__u < 0 ? -__e : __e);
          } while (__t <= -.6744);
          __tx             = ::cuda::std::trunc(__pr.__mean_ + __pr.__s_ * __t);
          __difmuk         = __pr.__mean_ - __tx;
          __using_exp_dist = true;
        }
        double __px = 0;
        double __py = 0;
        if (__tx < 10 && __tx >= 0)
        {
          const double __fac[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880};
          __px                 = -__pr.__mean_;
          __py                 = ::cuda::std::pow(__pr.__mean_, (double) __tx) / __fac[static_cast<int>(__tx)];
        }
        else
        {
          double __del = .8333333E-1 / __tx;
          __del -= 4.8 * __del * __del * __del;
          double __v = __difmuk / __tx;
          if (::cuda::std::abs(__v) > 0.25)
          {
            __px = __tx * ::cuda::std::log(1 + __v) - __difmuk - __del;
          }
          else
          {
            __px = __tx * __v * __v
                   * (((((((.1250060 * __v + -.1384794) * __v + .1421878) * __v + -.1661269) * __v + .2000118) * __v
                        + -.2500068)
                         * __v
                       + .3333333)
                        * __v
                      + -.5)
                 - __del;
          }
          __py = .3989423 / ::cuda::std::sqrt(__tx);
        }
        double __r  = (0.5 - __difmuk) / __pr.__s_;
        double __r2 = __r * __r;
        double __fx = -0.5 * __r2;
        double __fy = __pr.__omega_ * (((__pr.__c3_ * __r2 + __pr.__c2_) * __r2 + __pr.__c1_) * __r2 + __pr.__c0_);
        if (__using_exp_dist)
        {
          if (__pr.__c_ * ::cuda::std::abs(__u)
              <= __py * ::cuda::std::exp(__px + __e) - __fy * ::cuda::std::exp(__fx + __e))
          {
            break;
          }
        }
        else
        {
          if (__fy - __u * __fy <= __py * ::cuda::std::exp(__px - __fx))
          {
            break;
          }
        }
      }
    }
    return __clamp_to_integral<result_type>(__tx);
  }

  // property functions
  [[nodiscard]] _CCCL_API constexpr double mean() const noexcept
  {
    return __p_.mean();
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
  operator==(const poisson_distribution& __x, const poisson_distribution& __y) noexcept
  {
    return __x.__p_ == __y.__p_;
  }
#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const poisson_distribution& __x, const poisson_distribution& __y) noexcept
  {
    return !(__x == __y);
  }
#endif // _CCCL_STD_VER <= 2017

#if !_CCCL_COMPILER(NVRTC)
  template <class _CharT, class _Traits>
  friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const poisson_distribution& __x)
  {
    using ostream_type                        = ::std::basic_ostream<_CharT, _Traits>;
    using ios_base                            = typename ostream_type::ios_base;
    const typename ios_base::fmtflags __flags = __os.flags();
    __os.flags(::std::ios_base::dec | ::std::ios_base::left | ::std::ios_base::fixed);
    __os << __x.mean();
    __os.flags(__flags);
    return __os;
  }

  template <class _CharT, class _Traits>
  friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, poisson_distribution& __x)
  {
    using istream_type                        = ::std::basic_istream<_CharT, _Traits>;
    using ios_base                            = typename istream_type::ios_base;
    const typename ios_base::fmtflags __flags = __is.flags();
    using param_type                          = typename poisson_distribution::param_type;
    __is.flags(ios_base::dec | ios_base::skipws);
    double __mean = 0.0;
    __is >> __mean;
    if (!__is.fail())
    {
      __x.param(param_type(__mean));
    }
    __is.flags(__flags);
    return __is;
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___POISSON_DISTRIBUTION_H
