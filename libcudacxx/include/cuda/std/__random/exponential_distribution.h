//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANDOM_EXPONENTIAL_DISTRIBUTION_H
#define _CUDA_STD___RANDOM_EXPONENTIAL_DISTRIBUTION_H

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
#include <cuda/std/__random/uniform_real_distribution.h>
#include <cuda/std/cmath>

#if !_CCCL_COMPILER(NVRTC)
#  include <iosfwd>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _RealType = double>
class exponential_distribution
{
  static_assert(__cccl_random_is_valid_realtype<_RealType>, "RealType must be a supported floating-point type");

public:
  // types
  using result_type = _RealType;

  class param_type
  {
  private:
    result_type __lambda_;

  public:
    using distribution_type = exponential_distribution;

    _CCCL_API explicit param_type(result_type __lambda = 1)
        : __lambda_{__lambda}
    {}

    [[nodiscard]] _CCCL_API constexpr result_type lambda() const noexcept
    {
      return __lambda_;
    }

    [[nodiscard]] friend _CCCL_API constexpr bool operator==(const param_type& __x, const param_type& __y) noexcept
    {
      return __x.__lambda_ == __y.__lambda_;
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
  _CCCL_API exponential_distribution() noexcept
      : exponential_distribution{1}
  {}
  _CCCL_API explicit exponential_distribution(result_type __lambda) noexcept
      : __p_{param_type{__lambda}}
  {}
  _CCCL_API explicit exponential_distribution(const param_type& __p) noexcept
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
    static_assert(__cccl_random_is_valid_urng<_URng>, "URng must meet the UniformRandomBitGenerator requirements");
    return -::cuda::std::log(
             result_type{1} - ::cuda::std::generate_canonical<result_type, numeric_limits<result_type>::digits>(__g))
         / __p.lambda();
  }

  // property functions
  [[nodiscard]] _CCCL_API constexpr result_type lambda() const noexcept
  {
    return __p_.lambda();
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
  operator==(const exponential_distribution& __x, const exponential_distribution& __y) noexcept
  {
    return __x.__p_ == __y.__p_;
  }
#if _CCCL_STD_VER <= 2017
  [[nodiscard]] friend _CCCL_API constexpr bool
  operator!=(const exponential_distribution& __x, const exponential_distribution& __y) noexcept
  {
    return !(__x == __y);
  }
#endif // _CCCL_STD_VER <= 2017

#if !_CCCL_COMPILER(NVRTC)
  template <class _CharT, class _Traits>
  friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const exponential_distribution& __x)
  {
    using ostream_type                        = ::std::basic_ostream<_CharT, _Traits>;
    using ios_base                            = typename ostream_type::ios_base;
    const typename ios_base::fmtflags __flags = __os.flags();
    const _CharT __fill                       = __os.fill();
    const ::std::streamsize __precision       = __os.precision();
    __os.flags(ios_base::dec | ios_base::left | ios_base::scientific);
    __os.precision(numeric_limits<result_type>::max_digits10);
    __os << __x.lambda();
    __os.flags(__flags);
    __os.fill(__fill);
    __os.precision(__precision);
    return __os;
  }

  template <class _CharT, class _Traits>
  friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, exponential_distribution& __x)
  {
    using istream_type                        = ::std::basic_istream<_CharT, _Traits>;
    using ios_base                            = typename istream_type::ios_base;
    using param_type                          = typename exponential_distribution::param_type;
    const typename ios_base::fmtflags __flags = __is.flags();
    __is.flags(ios_base::dec | ios_base::skipws);
    result_type __lambda;
    __is >> __lambda;
    if (!__is.fail())
    {
      __x.param(param_type{__lambda});
    }
    __is.flags(__flags);
    return __is;
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_EXPONENTIAL_DISTRIBUTION_H
