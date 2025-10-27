//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

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

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

class bernoulli_distribution
{
public:
  // types
  using result_type = bool;

  class param_type
  {
    double __p_;

  public:
    using distribution_type = bernoulli_distribution;
    _CCCL_API explicit param_type(double __p = 0.5) noexcept
        : __p_(__p)
    {}

    [[nodiscard]] _CCCL_API result_type p() const noexcept
    {
      return __p_;
    }

    [[nodiscard]] _CCCL_API friend bool operator==(const param_type& __x, const param_type& __y) noexcept
    {
      return __x.__p_ == __y.__p_;
    }
    [[nodiscard]] _CCCL_API friend bool operator!=(const param_type& __x, const param_type& __y) noexcept
    {
      return !(__x == __y);
    }
  };

private:
  param_type __p_;

public:
  // constructors and reset functions
  _CCCL_API bernoulli_distribution() noexcept
      : bernoulli_distribution(0.5)
  {}
  _CCCL_API explicit bernoulli_distribution(double __p) noexcept
      : __p_(param_type(__p))
  {}
  _CCCL_API explicit bernoulli_distribution(const param_type& __p) noexcept
      : __p_(__p)
  {}
  _CCCL_API void reset() noexcept {}

  // generating functions
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g) noexcept
  {
    static_assert(__cccl_random_is_valid_urng<_URng>, "");
    return (*this)(__g, __p_);
  }
  _CCCL_EXEC_CHECK_DISABLE
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g, const param_type& __p) noexcept
  {
    static_assert(__cccl_random_is_valid_urng<_URng>, "");
    uniform_real_distribution<double> __gen;
    return __gen(__g) < __p.p();
  }

  // property functions
  [[nodiscard]] _CCCL_API result_type p() const noexcept
  {
    return __p_.p();
  }

  [[nodiscard]] _CCCL_API param_type param() const noexcept
  {
    return __p_;
  }
  _CCCL_API void param(const param_type& __p) noexcept
  {
    __p_ = __p;
  }

  [[nodiscard]] _CCCL_API result_type min() const noexcept
  {
    return false;
  }
  [[nodiscard]] _CCCL_API result_type max() const noexcept
  {
    return true;
  }

  [[nodiscard]] _CCCL_API friend bool
  operator==(const bernoulli_distribution& __x, const bernoulli_distribution& __y) noexcept
  {
    return __x.__p_ == __y.__p_;
  }
  [[nodiscard]] _CCCL_API friend bool
  operator!=(const bernoulli_distribution& __x, const bernoulli_distribution& __y) noexcept
  {
    return !(__x == __y);
  }
};

#if 0 // Implement stream operators
template <class _CharT, class _Traits, class _IT>
_CCCL_API basic_ostream<_CharT, _Traits>&
operator<<(basic_ostream<_CharT, _Traits>& __os, const bernoulli_distribution& __x)
{
  __save_flags<_CharT, _Traits> __lx(__os);
  using _Ostream = basic_ostream<_CharT, _Traits>;
  __os.flags(_Ostream::dec | _Ostream::left);
  _CharT __sp = __os.widen(' ');
  __os.fill(__sp);
  return __os << __x.p();
}

template <class _CharT, class _Traits, class _IT>
_CCCL_API basic_istream<_CharT, _Traits>&
operator>>(basic_istream<_CharT, _Traits>& __is, bernoulli_distribution& __x)
{
  using _Eng = uniform_int_distribution<_IT>;
  using result_type = typename _Eng::result_type;
  using param_type = typename _Eng::param_type;
  __save_flags<_CharT, _Traits> __lx(__is);
  using _Istream = basic_istream<_CharT, _Traits>;
  __is.flags(_Istream::dec | _Istream::skipws);
  result_type __a;
  result_type __b;
  __is >> __a >> __b;
  if (!__is.fail())
  {
    __x.param(param_type(__a, __b));
  }
  return __is;
}
#endif // Implement stream operators

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BERNOULLI_DISTRIBUTION_H
