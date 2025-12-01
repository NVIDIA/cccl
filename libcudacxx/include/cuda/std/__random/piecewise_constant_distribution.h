//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------====//

#ifndef _CUDA_STD___PIECEWISE_CONSTANT_DISTRIBUTION_H
#define _CUDA_STD___PIECEWISE_CONSTANT_DISTRIBUTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/upper_bound.h>
#include <cuda/std/__memory_>
#include <cuda/std/__random/is_valid.h>
#include <cuda/std/__random/uniform_real_distribution.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/initializer_list>
#include <cuda/std/numeric>
#include <cuda/std/span>

#if !_CCCL_COMPILER(NVRTC)
#  include <iosfwd>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _RealType = double>
class piecewise_constant_distribution
{
  static_assert(__libcpp_random_is_valid_realtype<_RealType>, "RealType must be a supported floating-point type");

public:
  // types
  using result_type = _RealType;

  class param_type
  {
  private:
    unique_ptr<result_type[]> __b_;
    unique_ptr<result_type[]> __densities_;
    unique_ptr<result_type[]> __areas_;
    size_t __size_b_         = 0;
    size_t __size_densities_ = 0;
    size_t __size_areas_     = 0;

    _CCCL_API void __init()
    {
      // __densities_ contains non-normalized areas
      result_type __total_area =
        ::cuda::std::accumulate(__densities_.get(), __densities_.get() + __size_densities_, result_type());
      for (size_t __i = 0; __i < __size_densities_; ++__i)
      {
        __densities_[__i] /= __total_area;
      }
      // __densities_ contains normalized areas
      __areas_.reset(new result_type[__size_densities_]);
      __size_areas_ = __size_densities_;
      if (__size_areas_ > 0)
      {
        __areas_[0] = result_type{0};
      }
      ::cuda::std::partial_sum(__densities_.get(), __densities_.get() + __size_densities_ - 1, __areas_.get() + 1);
      // __areas_ contains partial sums of normalized areas: [0, __densities_ - 1]
      __densities_[__size_densities_ - 1] = 1 - __areas_[__size_areas_ - 1]; // correct round off error
      for (size_t __i = 0; __i < __size_densities_; ++__i)
      {
        __densities_[__i] /= (__b_[__i + 1] - __b_[__i]);
      }
      // __densities_ now contains __densities_
    }

    friend class piecewise_constant_distribution;

#if !_CCCL_COMPILER(NVRTC)
    template <class _CharT, class _Traits>
    friend ::std::basic_ostream<_CharT, _Traits>&
    operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const piecewise_constant_distribution& __x);

    template <class _CharT, class _Traits>
    friend ::std::basic_istream<_CharT, _Traits>&
    operator>>(::std::basic_istream<_CharT, _Traits>& __is, piecewise_constant_distribution& __x);
#endif // !_CCCL_COMPILER(NVRTC)

  public:
    using distribution_type = piecewise_constant_distribution;

    _CCCL_API param_type()
        : __b_(new result_type[2])
        , __densities_(new result_type[1]{1.0})
        , __areas_(new result_type[1]{0.0})
    {
      __b_[1] = 1;
    }
    template <class _InputIteratorB, class _InputIteratorW>
    _CCCL_API param_type(_InputIteratorB __f_b, _InputIteratorB __l_b, _InputIteratorW __f_w)
    {
      if (::cuda::std::distance(__f_b, __l_b) < 2)
      {
        __b_.reset(new result_type[2]);
        __size_b_ = 2;
        __b_[0]   = 0;
        __b_[1]   = 1;
        __densities_.reset(new result_type[1]{1.0});
        __size_densities_ = 1;
        __areas_.reset(new result_type[1]{0.0});
        __size_areas_ = 1;
      }
      else
      {
        __b_.reset(new result_type[::cuda::std::distance(__f_b, __l_b)]);
        __size_b_ = ::cuda::std::distance(__f_b, __l_b);
        ::cuda::std::copy(__f_b, __l_b, __b_.get());
        __densities_.reset(new result_type[__size_b_ - 1]);
        __size_densities_ = __size_b_ - 1;
        ::cuda::std::copy_n(__f_w, __size_densities_, __densities_.get());
        __init();
      }
    }

    template <class _UnaryOperation>
    _CCCL_API param_type(initializer_list<result_type> __bl, _UnaryOperation __fw)
    {
      if (__bl.size() < 2)
      {
        __b_.reset(new result_type[2]);
        __size_b_ = 2;
        __b_[0]   = 0;
        __b_[1]   = 1;
        __densities_.reset(new result_type[1]{1.0});
        __size_densities_ = 1;
        __areas_.reset(new result_type[1]{0.0});
        __size_areas_ = 1;
      }
      else
      {
        __b_.reset(new result_type[__bl.size()]);
        __size_b_ = __bl.size();
        ::cuda::std::copy(__bl.begin(), __bl.end(), __b_.get());
        __densities_.reset(new result_type[__size_b_ - 1]);
        __size_densities_ = __size_b_ - 1;
        for (size_t __i = 0; __i < __size_b_ - 1; ++__i)
        {
          __densities_[__i] = __fw((__b_[__i + 1] + __b_[__i]) * .5);
        }
        __init();
      }
    }

    template <class _UnaryOperation>
    _CCCL_API param_type(size_t __nw, result_type __xmin, result_type __xmax, _UnaryOperation __fw)
    {
      __nw              = __nw == 0 ? 1 : __nw;
      __size_b_         = __nw + 1;
      __size_densities_ = __nw;
      __b_.reset(new result_type[__size_b_]);
      __densities_.reset(new result_type[__size_densities_]);
      result_type __d = (__xmax - __xmin) / __nw;
      for (size_t __i = 0; __i < __nw; ++__i)
      {
        __b_[__i]         = __xmin + __i * __d;
        __densities_[__i] = __fw(__b_[__i] + __d * .5);
      }
      __b_[__nw] = __xmax;
      __init();
    }

    _CCCL_API param_type(param_type const&)            = delete;
    _CCCL_API param_type& operator=(const param_type&) = delete;

    [[nodiscard]] _CCCL_API constexpr span<const result_type> intervals() const noexcept
    {
      return span<const result_type>(__b_.get(), __size_b_);
    }
    [[nodiscard]] _CCCL_API constexpr span<const result_type> densities() const noexcept
    {
      return span<const result_type>(__densities_.get(), __size_densities_);
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const param_type& __x, const param_type& __y) noexcept
    {
      if (__x.__size_b_ != __y.__size_b_ || __x.__size_densities_ != __y.__size_densities_)
      {
        return false;
      }
      for (size_t __i = 0; __i < __x.__size_b_; ++__i)
      {
        if (__x.__b_[__i] != __y.__b_[__i])
        {
          return false;
        }
      }
      for (size_t __i = 0; __i < __x.__size_densities_; ++__i)
      {
        if (__x.__densities_[__i] != __y.__densities_[__i])
        {
          return false;
        }
      }
      return true;
    }
#if _CCCL_STD_VER <= 2017
    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const param_type& __x, const param_type& __y) noexcept
    {
      return !(__x == __y);
    }
#endif // _CCCL_STD_VER <= 2017
  };

private:
  unique_ptr<param_type> __p_{};

public:
  // constructor and reset functions
  _CCCL_API piecewise_constant_distribution()
      : piecewise_constant_distribution(param_type())
  {}
  template <class _InputIteratorB, class _InputIteratorW>
  _CCCL_API piecewise_constant_distribution(_InputIteratorB __f_b, _InputIteratorB __l_b, _InputIteratorW __f_w)
      : __p_{new param_type{__f_b, __l_b, __f_w}}
  {}

  template <class _UnaryOperation>
  _CCCL_API piecewise_constant_distribution(initializer_list<result_type> __bl, _UnaryOperation __fw)
      : __p_{new param_type{__bl, __fw}}
  {}

  template <class _UnaryOperation>
  _CCCL_API piecewise_constant_distribution(size_t __nw, result_type __xmin, result_type __xmax, _UnaryOperation __fw)
      : __p_{new param_type{__nw, __xmin, __xmax, __fw}}
  {}

  _CCCL_API explicit piecewise_constant_distribution(const param_type& __p)
  {
    this->param(__p);
  }

  _CCCL_API constexpr void reset() noexcept {}

  // generating functions
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g)
  {
    return (*this)(__g, *__p_);
  }
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g, const param_type& __p)
  {
    static_assert(__cccl_random_is_valid_urng<_URng>, "URng must meet the UniformRandomBitGenerator requirements");
    using _Gen      = uniform_real_distribution<result_type>;
    result_type __u = _Gen{}(__g);
    ptrdiff_t __k =
      cuda::std::upper_bound(__p.__areas_.get(), __p.__areas_.get() + __p.__size_areas_, __u) - __p.__areas_.get() - 1;
    return (__u - __p.__areas_[__k]) / __p.__densities_[__k] + __p.__b_[__k];
  }

  // property functions
  [[nodiscard]] _CCCL_API constexpr span<const result_type> intervals() const noexcept
  {
    return __p_->intervals();
  }
  [[nodiscard]] _CCCL_API constexpr span<const result_type> densities() const noexcept
  {
    return __p_->densities();
  }

  _CCCL_API void param(const param_type& __p)
  {
    __p_.reset(new param_type());
    __p_->__b_.reset(new result_type[__p.__size_b_]);
    ::cuda::std::copy(__p.__b_.get(), __p.__b_.get() + __p.__size_b_, __p_->__b_.get());
    __p_->__size_b_ = __p.__size_b_;
    __p_->__densities_.reset(new result_type[__p.__size_densities_]);
    ::cuda::std::copy(__p.__densities_.get(), __p.__densities_.get() + __p.__size_densities_, __p_->__densities_.get());
    __p_->__size_densities_ = __p.__size_densities_;
    __p_->__areas_.reset(new result_type[__p.__size_areas_]);
    ::cuda::std::copy(__p.__areas_.get(), __p.__areas_.get() + __p.__size_areas_, __p_->__areas_.get());
    __p_->__size_areas_ = __p.__size_areas_;
  }

  [[nodiscard]] _CCCL_API constexpr const param_type& param() const noexcept
  {
    return *__p_;
  }

  [[nodiscard]] _CCCL_API constexpr result_type min() const noexcept
  {
    return __p_->__b_[0];
  }
  [[nodiscard]] _CCCL_API constexpr result_type max() const noexcept
  {
    return __p_->__b_[__p_->__size_b_ - 1];
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const piecewise_constant_distribution& __x, const piecewise_constant_distribution& __y) noexcept
  {
    return *__x.__p_ == *__y.__p_;
  }
#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const piecewise_constant_distribution& __x, const piecewise_constant_distribution& __y) noexcept
  {
    return !(__x == __y);
  }
#endif // _CCCL_STD_VER <= 2017

#if !_CCCL_COMPILER(NVRTC)
  template <class _CharT, class _Traits>
  friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const piecewise_constant_distribution& __x)
  {
    using _Ostream = ::std::basic_ostream<_CharT, _Traits>;
    auto __flags   = __os.flags();
    __os.flags(_Ostream::dec | _Ostream::left | _Ostream::scientific);
    _CharT __sp      = __os.widen(' ');
    _CharT __fill    = __os.fill(__sp);
    auto __precision = __os.precision(numeric_limits<result_type>::max_digits10);

    size_t __n = __x.__p_->__size_b_;
    __os << __n;
    for (size_t __i = 0; __i < __n; ++__i)
    {
      __os << __sp << __x.__p_->__b_[__i];
    }
    __n = __x.__p_->__size_densities_;
    __os << __sp << __n;
    for (size_t __i = 0; __i < __n; ++__i)
    {
      __os << __sp << __x.__p_->__densities_[__i];
    }
    __n = __x.__p_->__size_areas_;
    __os << __sp << __n;
    for (size_t __i = 0; __i < __n; ++__i)
    {
      __os << __sp << __x.__p_->__areas_[__i];
    }

    __os.precision(__precision);
    __os.fill(__fill);
    __os.flags(__flags);
    return __os;
  }

  template <class _CharT, class _Traits>
  friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, piecewise_constant_distribution& __x)
  {
    using _Istream = ::std::basic_istream<_CharT, _Traits>;
    auto __flags   = __is.flags();
    __is.flags(_Istream::skipws);

    size_t __n;
    __is >> __n;
    __x.__p_->__b_.reset(new result_type[__n]);
    __x.__p_->__size_b_ = __n;
    for (size_t __i = 0; __i < __n; ++__i)
    {
      __is >> __x.__p_->__b_[__i];
    }

    size_t __n_densities;
    __is >> __n_densities;
    __x.__p_->__densities_.reset(new result_type[__n_densities]);
    __x.__p_->__size_densities_ = __n_densities;
    for (size_t __i = 0; __i < __n_densities; ++__i)
    {
      __is >> __x.__p_->__densities_[__i];
    }

    size_t __n_areas;
    __is >> __n_areas;
    __x.__p_->__areas_.reset(new result_type[__n_areas]);
    __x.__p_->__size_areas_ = __n_areas;
    for (size_t __i = 0; __i < __n_areas; ++__i)
    {
      __is >> __x.__p_->__areas_[__i];
    }

    __is.flags(__flags);
    return __is;
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___PIECEWISE_CONSTANT_DISTRIBUTION_H
