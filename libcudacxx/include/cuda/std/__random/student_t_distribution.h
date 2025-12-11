//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------====//

#ifndef _CUDA_STD___STUDENT_T_DISTRIBUTION_H
#define _CUDA_STD___STUDENT_T_DISTRIBUTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/roots.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__random/gamma_distribution.h>
#include <cuda/std/__random/is_valid.h>
#include <cuda/std/__random/normal_distribution.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <iosfwd>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _RealType = double>
class student_t_distribution
{
  static_assert(__cccl_random_is_valid_realtype<_RealType>, "RealType must be a supported floating-point type");

public:
  // types
  using result_type = _RealType;

  class param_type
  {
  private:
    result_type __n_ = result_type{1};

  public:
    using distribution_type = student_t_distribution;

    _CCCL_API constexpr explicit param_type(result_type __n = result_type{1}) noexcept
        : __n_{__n}
    {}

    [[nodiscard]] _CCCL_API constexpr result_type n() const noexcept
    {
      return __n_;
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const param_type& __x, const param_type& __y) noexcept
    {
      return __x.__n_ == __y.__n_;
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
  normal_distribution<result_type> __nd_{};

public:
  // constructor and reset functions
  constexpr student_t_distribution() noexcept = default;

  _CCCL_API constexpr explicit student_t_distribution(result_type __n) noexcept
      : __p_{param_type{__n}}
  {}
  _CCCL_API constexpr explicit student_t_distribution(const param_type& __p) noexcept
      : __p_{__p}
  {}
  _CCCL_API constexpr void reset() noexcept
  {
    __nd_.reset();
  }

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
    gamma_distribution<result_type> __gd{__p.n() * result_type{.5}, result_type{2}};
    return __nd_(__g) * ::cuda::std::sqrt(__p.n() / __gd(__g));
  }

  // property functions
  [[nodiscard]] _CCCL_API constexpr result_type n() const noexcept
  {
    return __p_.n();
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
    return -numeric_limits<result_type>::infinity();
  }
  [[nodiscard]] _CCCL_API static constexpr result_type max() noexcept
  {
    return numeric_limits<result_type>::infinity();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const student_t_distribution& __x, const student_t_distribution& __y) noexcept
  {
    return __x.__p_ == __y.__p_;
  }
#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const student_t_distribution& __x, const student_t_distribution& __y) noexcept
  {
    return !(__x == __y);
  }
#endif // _CCCL_STD_VER <= 2017

#if !_CCCL_COMPILER(NVRTC)
  template <class _CharT, class _Traits>
  friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const student_t_distribution& __x)
  {
    using _Ostream = ::std::basic_ostream<_CharT, _Traits>;
    auto __flags   = __os.flags();
    __os.flags(_Ostream::dec | _Ostream::left | _Ostream::scientific);
    _CharT __sp      = __os.widen(' ');
    _CharT __fill    = __os.fill(__sp);
    auto __precision = __os.precision(numeric_limits<result_type>::max_digits10);
    __os << __x.n();
    __os.precision(__precision);
    __os.fill(__fill);
    __os.flags(__flags);
    return __os;
  }

  template <class _CharT, class _Traits>
  friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, student_t_distribution& __x)
  {
    using _Istream = ::std::basic_istream<_CharT, _Traits>;
    auto __flags   = __is.flags();
    __is.flags(_Istream::skipws);
    result_type __n;
    __is >> __n;
    if (!__is.fail())
    {
      __x.param(param_type{__n});
    }
    __is.flags(__flags);
    return __is;
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___STUDENT_T_DISTRIBUTION_H
