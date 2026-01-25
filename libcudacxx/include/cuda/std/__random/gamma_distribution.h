//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___GAMMA_DISTRIBUTION_H
#define _CUDA_STD___GAMMA_DISTRIBUTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/logarithms.h>
#include <cuda/std/__cmath/roots.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__random/exponential_distribution.h>
#include <cuda/std/__random/is_valid.h>
#include <cuda/std/__random/uniform_real_distribution.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <iosfwd>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _RealType = double>
class gamma_distribution
{
  static_assert(__cccl_random_is_valid_realtype<_RealType>, "RealType must be a supported floating-point type");

public:
  // types
  using result_type = _RealType;

  class param_type
  {
  private:
    result_type __alpha_ = result_type{1};
    result_type __beta_  = result_type{1};

  public:
    using distribution_type = gamma_distribution;

    _CCCL_API constexpr explicit param_type(result_type __alpha = result_type{1}, result_type __beta = result_type{1})
        : __alpha_{__alpha}
        , __beta_{__beta}
    {}

    [[nodiscard]] _CCCL_API constexpr result_type alpha() const noexcept
    {
      return __alpha_;
    }
    [[nodiscard]] _CCCL_API constexpr result_type beta() const noexcept
    {
      return __beta_;
    }

    [[nodiscard]] friend _CCCL_API constexpr bool operator==(const param_type& __x, const param_type& __y) noexcept
    {
      return __x.__alpha_ == __y.__alpha_ && __x.__beta_ == __y.__beta_;
    }
#if _CCCL_STD_VER <= 2017
    [[nodiscard]] friend _CCCL_API constexpr bool operator!=(const param_type& __x, const param_type& __y) noexcept
    {
      return !(__x == __y);
    }
#endif // _CCCL_STD_VER <= 2017
  };

private:
  param_type __p_;

public:
  // constructors and reset functions
  constexpr gamma_distribution() = default;
  _CCCL_API constexpr explicit gamma_distribution(result_type __alpha, result_type __beta = result_type{1}) noexcept
      : __p_{param_type{__alpha, __beta}}
  {}
  _CCCL_API constexpr explicit gamma_distribution(const param_type& __p) noexcept
      : __p_{__p}
  {}
  _CCCL_API void reset() noexcept {}

  // generating functions
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g)
  {
    return (*this)(__g, __p_);
  }
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g, const param_type& __p)
  {
    static_assert(__cccl_random_is_valid_urng<_URng>, "URng must meet the UniformRandomBitGenerator requirements");
    const result_type __a = __p.alpha();
    uniform_real_distribution<result_type> __gen(result_type{0}, result_type{1});
    exponential_distribution<result_type> __egen;
    result_type __x;
    if (__a == result_type{1})
    {
      __x = __egen(__g);
    }
    else if (__a > result_type{1})
    {
      const result_type __b = __a - result_type{1};
      const result_type __c = result_type{3} * __a - result_type{0.75};
      while (true)
      {
        const result_type __u = __gen(__g);
        const result_type __v = __gen(__g);
        const result_type __w = __u * (result_type{1} - __u);
        if (__w != result_type{0})
        {
          const result_type __y = ::cuda::std::sqrt(__c / __w) * (__u - result_type{0.5});
          __x                   = __b + __y;
          if (__x >= result_type{0})
          {
            const result_type __z = result_type{64} * __w * __w * __w * __v * __v;
            if (__z <= result_type{1} - result_type{2} * __y * __y / __x)
            {
              break;
            }
            if (::cuda::std::log(__z) <= result_type{2} * (__b * ::cuda::std::log(__x / __b) - __y))
            {
              break;
            }
          }
        }
      }
    }
    else // __a < 1
    {
      while (true)
      {
        const result_type __u  = __gen(__g);
        const result_type __es = __egen(__g);
        if (__u <= result_type{1} - __a)
        {
          __x = ::cuda::std::pow(__u, result_type{1} / __a);
          if (__x <= __es)
          {
            break;
          }
        }
        else
        {
          const result_type __e = -::cuda::std::log((result_type{1} - __u) / __a);
          __x                   = ::cuda::std::pow(result_type{1} - __a + __a * __e, result_type{1} / __a);
          if (__x <= __e + __es)
          {
            break;
          }
        }
      }
    }
    return __x * __p.beta();
  }

  // property functions
  [[nodiscard]] _CCCL_API constexpr result_type alpha() const noexcept
  {
    return __p_.alpha();
  }
  [[nodiscard]] _CCCL_API constexpr result_type beta() const noexcept
  {
    return __p_.beta();
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
    return result_type{0};
  }
  [[nodiscard]] _CCCL_API static constexpr result_type max() noexcept
  {
    return numeric_limits<result_type>::infinity();
  }

  [[nodiscard]] friend _CCCL_API constexpr bool
  operator==(const gamma_distribution& __x, const gamma_distribution& __y) noexcept
  {
    return __x.__p_ == __y.__p_;
  }
#if _CCCL_STD_VER <= 2017
  [[nodiscard]] friend _CCCL_API constexpr bool
  operator!=(const gamma_distribution& __x, const gamma_distribution& __y) noexcept
  {
    return !(__x == __y);
  }
#endif // _CCCL_STD_VER <= 2017

#if !_CCCL_COMPILER(NVRTC)
  template <class _CharT, class _Traits>
  friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const gamma_distribution& __x)
  {
    using ostream_type                        = ::std::basic_ostream<_CharT, _Traits>;
    using ios_base                            = typename ostream_type::ios_base;
    const typename ios_base::fmtflags __flags = __os.flags();
    const _CharT __fill                       = __os.fill();
    const ::std::streamsize __precision       = __os.precision();
    __os.flags(ios_base::dec | ios_base::left | ios_base::scientific);
    _CharT __sp = __os.widen(' ');
    __os.fill(__sp);
    __os.precision(numeric_limits<result_type>::max_digits10);
    __os << __x.alpha() << __sp << __x.beta();
    __os.flags(__flags);
    __os.fill(__fill);
    __os.precision(__precision);
    return __os;
  }

  template <class _CharT, class _Traits>
  friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, gamma_distribution& __x)
  {
    using istream_type                        = ::std::basic_istream<_CharT, _Traits>;
    using ios_base                            = typename istream_type::ios_base;
    const typename ios_base::fmtflags __flags = __is.flags();
    __is.flags(ios_base::dec | ios_base::skipws);
    result_type __alpha;
    result_type __beta;
    __is >> __alpha >> __beta;
    if (!__is.fail())
    {
      __x.param(param_type{__alpha, __beta});
    }
    __is.flags(__flags);
    return __is;
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___GAMMA_DISTRIBUTION_H
