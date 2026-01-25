//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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

#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__random/generate_canonical.h>
#include <cuda/std/__random/is_valid.h>
#if !_CCCL_COMPILER(NVRTC)
#  include <ios>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

class bernoulli_distribution
{
public:
  // types
  using result_type = bool;

  class param_type
  {
    double __p_{};

  public:
    using distribution_type = bernoulli_distribution;

    _CCCL_API constexpr explicit param_type(double __p = 0.5) noexcept
        : __p_{__p}
    {}

    [[nodiscard]] _CCCL_API constexpr double p() const noexcept
    {
      return __p_;
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const param_type& __x, const param_type& __y) noexcept
    {
      return __x.__p_ == __y.__p_;
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

public:
  // constructors and reset functions
  _CCCL_API constexpr bernoulli_distribution() noexcept
      : bernoulli_distribution{0.5}
  {}
  _CCCL_API constexpr explicit bernoulli_distribution(double __p) noexcept
      : __p_{param_type(__p)}
  {}
  _CCCL_API constexpr explicit bernoulli_distribution(const param_type& __p) noexcept
      : __p_{__p}
  {}
  _CCCL_API constexpr void reset() noexcept {}

  // generating functions
  template <class _URng>
  [[nodiscard]] _CCCL_API constexpr result_type operator()(_URng& __g) noexcept
  {
    return (*this)(__g, __p_);
  }
  template <class _URng>
  [[nodiscard]] _CCCL_API constexpr result_type operator()(_URng& __g, const param_type& __p) noexcept
  {
    static_assert(__cccl_random_is_valid_urng<_URng>, "URng must meet the UniformRandomBitGenerator requirements");
    return ::cuda::std::generate_canonical<double, numeric_limits<double>::digits>(__g) < __p.p();
  }

  // property functions
  [[nodiscard]] _CCCL_API constexpr double p() const noexcept
  {
    return __p_.p();
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
    return false;
  }
  [[nodiscard]] _CCCL_API constexpr result_type max() const noexcept
  {
    return true;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const bernoulli_distribution& __x, const bernoulli_distribution& __y) noexcept
  {
    return __x.__p_ == __y.__p_;
  }
#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const bernoulli_distribution& __x, const bernoulli_distribution& __y) noexcept
  {
    return !(__x == __y);
  }
#endif // _CCCL_STD_VER <= 2017
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
    using _Eng                                = bernoulli_distribution;
    using param_type                          = typename _Eng::param_type;
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
