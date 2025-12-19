//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FISHER_F_DISTRIBUTION_H
#define _CUDA_STD___FISHER_F_DISTRIBUTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__random/gamma_distribution.h>
#include <cuda/std/__random/is_valid.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <iosfwd>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _RealType = double>
class fisher_f_distribution
{
  static_assert(__cccl_random_is_valid_realtype<_RealType>, "RealType must be a supported floating-point type");

public:
  // types
  using result_type = _RealType;

  class param_type
  {
  private:
    result_type __m_ = result_type{1};
    result_type __n_ = result_type{1};

  public:
    using distribution_type = fisher_f_distribution;

    _CCCL_API constexpr explicit param_type(result_type __m = result_type{1}, result_type __n = result_type{1}) noexcept
        : __m_{__m}
        , __n_{__n}
    {}

    [[nodiscard]] _CCCL_API constexpr result_type m() const noexcept
    {
      return __m_;
    }
    [[nodiscard]] _CCCL_API constexpr result_type n() const noexcept
    {
      return __n_;
    }

    [[nodiscard]] friend _CCCL_API constexpr bool operator==(const param_type& __x, const param_type& __y) noexcept
    {
      return __x.__m_ == __y.__m_ && __x.__n_ == __y.__n_;
    }
#if _CCCL_STD_VER <= 2017
    [[nodiscard]] friend _CCCL_API constexpr bool operator!=(const param_type& __x, const param_type& __y) noexcept
    {
      return !(__x == __y);
    }
#endif // _CCCL_STD_VER <= 2017
  };

private:
  param_type __p_{};

public:
  // constructor and reset functions
  constexpr fisher_f_distribution() = default;

  _CCCL_API constexpr explicit fisher_f_distribution(result_type __m, result_type __n = result_type{1})
      : __p_{param_type{__m, __n}}
  {}
  _CCCL_API constexpr explicit fisher_f_distribution(const param_type& __p)
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
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g, const param_type& __p)
  {
    static_assert(__cccl_random_is_valid_urng<_URng>);
    gamma_distribution<result_type> __gdm{__p.m() * result_type{.5}};
    gamma_distribution<result_type> __gdn{__p.n() * result_type{.5}};
    return __p.n() * __gdm(__g) / (__p.m() * __gdn(__g));
  }

  // property functions
  [[nodiscard]] _CCCL_API constexpr result_type m() const noexcept
  {
    return __p_.m();
  }
  [[nodiscard]] _CCCL_API constexpr result_type n() const noexcept
  {
    return __p_.n();
  }

  [[nodiscard]] _CCCL_API constexpr param_type param() const noexcept
  {
    return __p_;
  }
  _CCCL_API _CCCL_CONSTEXPR_CXX20 void param(const param_type& __p) noexcept
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
  operator==(const fisher_f_distribution& __x, const fisher_f_distribution& __y) noexcept
  {
    return __x.__p_ == __y.__p_;
  }
#if _CCCL_STD_VER <= 2017
  [[nodiscard]] friend _CCCL_API constexpr bool
  operator!=(const fisher_f_distribution& __x, const fisher_f_distribution& __y) noexcept
  {
    return !(__x == __y);
  }
#endif // _CCCL_STD_VER <= 2017

#if !_CCCL_COMPILER(NVRTC)
  template <class _CharT, class _Traits>
  friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const fisher_f_distribution& __x)
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
    __os << __x.m() << __sp << __x.n();
    __os.flags(__flags);
    __os.fill(__fill);
    __os.precision(__precision);
    return __os;
  }

  template <class _CharT, class _Traits>
  friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, fisher_f_distribution& __x)
  {
    using istream_type                        = ::std::basic_istream<_CharT, _Traits>;
    using ios_base                            = typename istream_type::ios_base;
    const typename ios_base::fmtflags __flags = __is.flags();
    __is.flags(ios_base::dec | ios_base::skipws);
    result_type __m;
    result_type __n;
    __is >> __m >> __n;
    if (!__is.fail())
    {
      __x.param(param_type{__m, __n});
    }
    __is.flags(__flags);
    return __is;
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FISHER_F_DISTRIBUTION_H
