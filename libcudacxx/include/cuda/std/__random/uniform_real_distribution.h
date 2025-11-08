//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANDOM_UNIFORM_REAL_DISTRIBUTION_H
#define _CUDA_STD___RANDOM_UNIFORM_REAL_DISTRIBUTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__random/generate_canonical.h>
#include <cuda/std/__random/is_valid.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _RealType = double>
class uniform_real_distribution
{
  static_assert(__libcpp_random_is_valid_realtype<_RealType>, "RealType must be a supported floating-point type");

public:
  // types
  using result_type = _RealType;

  class param_type
  {
    result_type __a_;
    result_type __b_;

  public:
    using distribution_type = uniform_real_distribution;

    _CCCL_API explicit param_type(result_type __a = 0, result_type __b = 1) noexcept
        : __a_(__a)
        , __b_(__b)
    {}

    [[nodiscard]] _CCCL_API result_type a() const noexcept
    {
      return __a_;
    }
    [[nodiscard]] _CCCL_API result_type b() const noexcept
    {
      return __b_;
    }

    [[nodiscard]] _CCCL_API friend bool operator==(const param_type& __x, const param_type& __y) noexcept
    {
      return __x.__a_ == __y.__a_ && __x.__b_ == __y.__b_;
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

  _CCCL_API uniform_real_distribution() noexcept
      : uniform_real_distribution(0)
  {}
  _CCCL_API explicit uniform_real_distribution(result_type __a, result_type __b = 1) noexcept
      : __p_(param_type(__a, __b))
  {}
  _CCCL_API explicit uniform_real_distribution(const param_type& __p) noexcept
      : __p_(__p)
  {}
  _CCCL_API void reset() noexcept {}

  // generating functions
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g) noexcept
  {
    return (*this)(__g, __p_);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g, const param_type& __p) noexcept
  {
    static_assert(__cccl_random_is_valid_urng<_URng>, "");
    return (__p.b() - __p.a()) * ::cuda::std::generate_canonical<_RealType, numeric_limits<_RealType>::digits>(__g)
         + __p.a();
  }

  // property functions
  [[nodiscard]] _CCCL_API result_type a() const noexcept
  {
    return __p_.a();
  }
  [[nodiscard]] _CCCL_API result_type b() const noexcept
  {
    return __p_.b();
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
    return a();
  }
  [[nodiscard]] _CCCL_API result_type max() const noexcept
  {
    return b();
  }

  [[nodiscard]] _CCCL_API friend bool
  operator==(const uniform_real_distribution& __x, const uniform_real_distribution& __y) noexcept
  {
    return __x.__p_ == __y.__p_;
  }
  [[nodiscard]] _CCCL_API friend bool
  operator!=(const uniform_real_distribution& __x, const uniform_real_distribution& __y) noexcept
  {
    return !(__x == __y);
  }
};

#if 0 // Implement streaming
template <class _CharT, class _Traits, class _RT>
_CCCL_API basic_ostream<_CharT, _Traits>&
operator<<(basic_ostream<_CharT, _Traits>& __os, const uniform_real_distribution<_RT>& __x)
{
  __save_flags<_CharT, _Traits> __lx(__os);
  using _OStream = basic_ostream<_CharT, _Traits>;
  __os.flags(_OStream::dec | _OStream::left | _OStream::fixed | _OStream::scientific);
  _CharT __sp = __os.widen(' ');
  __os.fill(__sp);
  return __os << __x.a() << __sp << __x.b();
}

template <class _CharT, class _Traits, class _RT>
_CCCL_API basic_istream<_CharT, _Traits>&
operator>>(basic_istream<_CharT, _Traits>& __is, uniform_real_distribution<_RT>& __x)
{
  using _Eng        = uniform_real_distribution<_RT>;
  using result_type = typename _Eng::result_type;
  using             = param_typetypename _Eng::param_type;
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
#endif // Not implemented

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_UNIFORM_REAL_DISTRIBUTION_H
