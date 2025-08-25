// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_REFERENCE_H
#define _CUDA_STD___BIT_REFERENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/copy_n.h>
#include <cuda/std/__algorithm/fill_n.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__utility/swap.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Cp>
class __bit_const_reference;

template <class _Cp, bool _IsConst>
class __bit_iterator;

template <class _Cp>
class __bit_reference
{
  using __storage_type    = typename _Cp::__storage_type;
  using __storage_pointer = typename _Cp::__storage_pointer;

  __storage_pointer __seg_;
  __storage_type __mask_;

  friend typename _Cp::__self;

  friend class __bit_const_reference<_Cp>;
  friend class __bit_iterator<_Cp, false>;

public:
  using __container = typename _Cp::__self;

  _CCCL_HIDE_FROM_ABI constexpr __bit_reference(const __bit_reference&) = default;

  _CCCL_API constexpr operator bool() const noexcept
  {
    return static_cast<bool>(*__seg_ & __mask_);
  }
  _CCCL_API constexpr bool operator~() const noexcept
  {
    return !static_cast<bool>(*this);
  }

  _CCCL_API constexpr __bit_reference& operator=(bool __x) noexcept
  {
    if (__x)
    {
      *__seg_ |= __mask_;
    }
    else
    {
      *__seg_ &= ~__mask_;
    }
    return *this;
  }

#if _CCCL_STD_VER >= 2023
  _CCCL_API constexpr const __bit_reference& operator=(bool __x) const noexcept
  {
    if (__x)
    {
      *__seg_ |= __mask_;
    }
    else
    {
      *__seg_ &= ~__mask_;
    }
    return *this;
  }
#endif // C++23+

  _CCCL_API constexpr __bit_reference& operator=(const __bit_reference& __x) noexcept
  {
    return operator=(static_cast<bool>(__x));
  }

  _CCCL_API constexpr void flip() noexcept
  {
    *__seg_ ^= __mask_;
  }
  _CCCL_API constexpr __bit_iterator<_Cp, false> operator&() const noexcept
  {
    return __bit_iterator<_Cp, false>(__seg_, static_cast<unsigned>(::cuda::std::countr_zero(__mask_)));
  }

  friend _CCCL_API constexpr void swap(__bit_reference<_Cp> __x, __bit_reference<_Cp> __y) noexcept
  {
    bool __t = __x;
    __x      = __y;
    __y      = __t;
  }

  template <class _Dp>
  friend _CCCL_API constexpr void swap(__bit_reference<_Cp> __x, __bit_reference<_Dp> __y) noexcept
  {
    bool __t = __x;
    __x      = __y;
    __y      = __t;
  }

  friend _CCCL_API constexpr void swap(__bit_reference<_Cp> __x, bool& __y) noexcept
  {
    bool __t = __x;
    __x      = __y;
    __y      = __t;
  }

  friend _CCCL_API constexpr void swap(bool& __x, __bit_reference<_Cp> __y) noexcept
  {
    bool __t = __x;
    __x      = __y;
    __y      = __t;
  }

private:
  _CCCL_API constexpr explicit __bit_reference(__storage_pointer __s, __storage_type __m) noexcept
      : __seg_(__s)
      , __mask_(__m)
  {}
};

template <class _Cp>
class __bit_const_reference
{
  using __storage_type    = typename _Cp::__storage_type;
  using __storage_pointer = typename _Cp::__const_storage_pointer;

  __storage_pointer __seg_;
  __storage_type __mask_;

  friend typename _Cp::__self;
  friend class __bit_iterator<_Cp, true>;

public:
  using __container = typename _Cp::__self;

  _CCCL_HIDE_FROM_ABI __bit_const_reference(const __bit_const_reference&) = default;

  _CCCL_API constexpr __bit_const_reference(const __bit_reference<_Cp>& __x) noexcept
      : __seg_(__x.__seg_)
      , __mask_(__x.__mask_)
  {}

  _CCCL_API constexpr operator bool() const noexcept
  {
    return static_cast<bool>(*__seg_ & __mask_);
  }

  _CCCL_API constexpr __bit_iterator<_Cp, true> operator&() const noexcept
  {
    return __bit_iterator<_Cp, true>(__seg_, static_cast<unsigned>(::cuda::std::countr_zero(__mask_)));
  }

private:
  _CCCL_API constexpr explicit __bit_const_reference(__storage_pointer __s, __storage_type __m) noexcept
      : __seg_(__s)
      , __mask_(__m)
  {}

  __bit_const_reference& operator=(const __bit_const_reference&) = delete;
};

// fill_n

template <bool _FillVal, class _Cp>
_CCCL_API constexpr void __fill_n_impl(__bit_iterator<_Cp, false> __first, typename _Cp::size_type __n)
{
  using _It            = __bit_iterator<_Cp, false>;
  using __storage_type = typename _It::__storage_type;

  const int __bits_per_word = _It::__bits_per_word;
  // do first partial word
  if (__first.__ctz_ != 0)
  {
    __storage_type __clz_f = static_cast<__storage_type>(__bits_per_word - __first.__ctz_);
    __storage_type __dn    = (::cuda::std::min) (__clz_f, static_cast<__storage_type>(__n));
    __storage_type __m     = (~__storage_type(0) << __first.__ctz_) & (~__storage_type(0) >> (__clz_f - __dn));
    if (_FillVal)
    {
      *__first.__seg_ |= __m;
    }
    else
    {
      *__first.__seg_ &= ~__m;
    }
    __n -= __dn.__data;
    ++__first.__seg_;
  }
  // do middle whole words
  __storage_type __nw = __n / __bits_per_word;
  ::cuda::std::fill_n(::cuda::std::__to_address(__first.__seg_), __nw, _FillVal ? ~static_cast<__storage_type>(0) : 0);
  __n -= (__nw * __bits_per_word).__data;
  // do last partial word
  if (__n > 0)
  {
    __first.__seg_ += __nw.__data;
    __storage_type __m = ~__storage_type(0) >> (__bits_per_word - __n);
    if (_FillVal)
    {
      *__first.__seg_ |= __m;
    }
    else
    {
      *__first.__seg_ &= ~__m;
    }
  }
}

template <class _Cp>
_CCCL_API constexpr void fill_n(__bit_iterator<_Cp, false> __first, typename _Cp::size_type __n, bool __value)
{
  if (__n > 0)
  {
    if (__value)
    {
      ::cuda::std::__fill_n_impl<true>(__first, __n);
    }
    else
    {
      ::cuda::std::__fill_n_impl<false>(__first, __n);
    }
  }
}

// fill

template <class _Cp>
_CCCL_API constexpr void fill(__bit_iterator<_Cp, false> __first, __bit_iterator<_Cp, false> __last, bool __value)
{
  ::cuda::std::fill_n(__first, static_cast<typename _Cp::size_type>(__last - __first), __value);
}

// copy

template <class _Cp, bool _IsConst>
_CCCL_API constexpr __bit_iterator<_Cp, false> __copy_aligned(
  __bit_iterator<_Cp, _IsConst> __first, __bit_iterator<_Cp, _IsConst> __last, __bit_iterator<_Cp, false> __result)
{
  using _In             = __bit_iterator<_Cp, _IsConst>;
  using difference_type = typename _In::difference_type;
  using __storage_type  = typename _In::__storage_type;

  const int __bits_per_word = _In::__bits_per_word;
  difference_type __n       = __last - __first;
  if (__n > 0)
  {
    // do first word
    if (__first.__ctz_ != 0)
    {
      unsigned __clz       = __bits_per_word - __first.__ctz_;
      difference_type __dn = ::cuda::std::min(static_cast<difference_type>(__clz), __n);
      __n -= __dn;
      __storage_type __m = (~__storage_type(0) << __first.__ctz_) & (~__storage_type(0) >> (__clz - __dn));
      __storage_type __b = *__first.__seg_ & __m;
      *__result.__seg_ &= ~__m;
      *__result.__seg_ |= __b;
      __result.__seg_ += (__dn + __result.__ctz_) / __bits_per_word;
      __result.__ctz_ = static_cast<unsigned>((__dn + __result.__ctz_) % __bits_per_word);
      ++__first.__seg_;
      // __first.__ctz_ = 0;
    }
    // __first.__ctz_ == 0;
    // do middle words
    __storage_type __nw = __n / __bits_per_word;
    ::cuda::std::copy_n(
      ::cuda::std::__to_address(__first.__seg_), __nw.__data, ::cuda::std::__to_address(__result.__seg_));
    __result.__seg_ += __nw.__data;
    __n -= (__nw * __bits_per_word).__data;
    // do last word
    if (__n > 0)
    {
      __first.__seg_ += __nw.__data;
      __storage_type __m = ~__storage_type(0) >> (__bits_per_word - __n);
      __storage_type __b = *__first.__seg_ & __m;
      *__result.__seg_ &= ~__m;
      *__result.__seg_ |= __b;
      __result.__ctz_ = static_cast<unsigned>(__n);
    }
  }
  return __result;
}

template <class _Cp, bool _IsConst>
_CCCL_API constexpr __bit_iterator<_Cp, false> __copy_unaligned(
  __bit_iterator<_Cp, _IsConst> __first, __bit_iterator<_Cp, _IsConst> __last, __bit_iterator<_Cp, false> __result)
{
  using _In             = __bit_iterator<_Cp, _IsConst>;
  using difference_type = typename _In::difference_type;
  using __storage_type  = typename _In::__storage_type;

  const int __bits_per_word = _In::__bits_per_word;
  difference_type __n       = __last - __first;
  if (__n > 0)
  {
    // do first word
    if (__first.__ctz_ != 0)
    {
      unsigned __clz_f     = __bits_per_word - __first.__ctz_;
      difference_type __dn = ::cuda::std::min(static_cast<difference_type>(__clz_f), __n);
      __n -= __dn;
      __storage_type __m   = (~__storage_type(0) << __first.__ctz_) & (~__storage_type(0) >> (__clz_f - __dn));
      __storage_type __b   = *__first.__seg_ & __m;
      unsigned __clz_r     = __bits_per_word - __result.__ctz_;
      __storage_type __ddn = ::cuda::std::min<__storage_type>(__dn, __clz_r);
      __m                  = (~__storage_type(0) << __result.__ctz_) & (~__storage_type(0) >> (__clz_r - __ddn));
      *__result.__seg_ &= ~__m;
      if (__result.__ctz_ > __first.__ctz_)
      {
        *__result.__seg_ |= __b << (__result.__ctz_ - __first.__ctz_);
      }
      else
      {
        *__result.__seg_ |= __b >> (__first.__ctz_ - __result.__ctz_);
      }
      __result.__seg_ += ((__ddn + __result.__ctz_) / __bits_per_word).__data;
      __result.__ctz_ = static_cast<unsigned>(((__ddn + __result.__ctz_) % __bits_per_word).__data);
      __dn -= __ddn.__data;
      if (__dn > 0)
      {
        __m = ~__storage_type(0) >> (__bits_per_word - __dn);
        *__result.__seg_ &= ~__m;
        *__result.__seg_ |= __b >> (__first.__ctz_ + __ddn);
        __result.__ctz_ = static_cast<unsigned>(__dn);
      }
      ++__first.__seg_;
      // __first.__ctz_ = 0;
    }
    // __first.__ctz_ == 0;
    // do middle words
    unsigned __clz_r   = __bits_per_word - __result.__ctz_;
    __storage_type __m = ~__storage_type(0) << __result.__ctz_;
    for (; __n >= __bits_per_word; __n -= __bits_per_word, ++__first.__seg_)
    {
      __storage_type __b = *__first.__seg_;
      *__result.__seg_ &= ~__m;
      *__result.__seg_ |= __b << __result.__ctz_;
      ++__result.__seg_;
      *__result.__seg_ &= __m;
      *__result.__seg_ |= __b >> __clz_r;
    }
    // do last word
    if (__n > 0)
    {
      __m                 = ~__storage_type(0) >> (__bits_per_word - __n);
      __storage_type __b  = *__first.__seg_ & __m;
      __storage_type __dn = ::cuda::std::min(__n, static_cast<difference_type>(__clz_r));
      __m                 = (~__storage_type(0) << __result.__ctz_) & (~__storage_type(0) >> (__clz_r - __dn));
      *__result.__seg_ &= ~__m;
      *__result.__seg_ |= __b << __result.__ctz_;
      __result.__seg_ += ((__dn + __result.__ctz_) / __bits_per_word).__data;
      __result.__ctz_ = static_cast<unsigned>(((__dn + __result.__ctz_) % __bits_per_word).__data);
      __n -= __dn.__data;
      if (__n > 0)
      {
        __m = ~__storage_type(0) >> (__bits_per_word - __n);
        *__result.__seg_ &= ~__m;
        *__result.__seg_ |= __b >> __dn;
        __result.__ctz_ = static_cast<unsigned>(__n);
      }
    }
  }
  return __result;
}

template <class _Cp, bool _IsConst>
_CCCL_API constexpr __bit_iterator<_Cp, false>
copy(__bit_iterator<_Cp, _IsConst> __first, __bit_iterator<_Cp, _IsConst> __last, __bit_iterator<_Cp, false> __result)
{
  if (__first.__ctz_ == __result.__ctz_)
  {
    return ::cuda::std::__copy_aligned(__first, __last, __result);
  }
  return ::cuda::std::__copy_unaligned(__first, __last, __result);
}

// copy_backward

template <class _Cp, bool _IsConst>
_CCCL_API constexpr __bit_iterator<_Cp, false> __copy_backward_aligned(
  __bit_iterator<_Cp, _IsConst> __first, __bit_iterator<_Cp, _IsConst> __last, __bit_iterator<_Cp, false> __result)
{
  using _In             = __bit_iterator<_Cp, _IsConst>;
  using difference_type = typename _In::difference_type;
  using __storage_type  = typename _In::__storage_type;

  const int __bits_per_word = _In::__bits_per_word;
  difference_type __n       = __last - __first;
  if (__n > 0)
  {
    // do first word
    if (__last.__ctz_ != 0)
    {
      difference_type __dn = ::cuda::std::min(static_cast<difference_type>(__last.__ctz_), __n);
      __n -= __dn;
      unsigned __clz     = __bits_per_word - __last.__ctz_;
      __storage_type __m = (~__storage_type(0) << (__last.__ctz_ - __dn)) & (~__storage_type(0) >> __clz);
      __storage_type __b = *__last.__seg_ & __m;
      *__result.__seg_ &= ~__m;
      *__result.__seg_ |= __b;
      __result.__ctz_ = static_cast<unsigned>(((-__dn & (__bits_per_word - 1)) + __result.__ctz_) % __bits_per_word);
      // __last.__ctz_ = 0
    }
    // __last.__ctz_ == 0 || __n == 0
    // __result.__ctz_ == 0 || __n == 0
    // do middle words
    __storage_type __nw = __n / __bits_per_word;
    __result.__seg_ -= __nw.__data;
    __last.__seg_ -= __nw.__data;
    ::cuda::std::copy_n(
      ::cuda::std::__to_address(__last.__seg_), __nw.__data, ::cuda::std::__to_address(__result.__seg_));
    __n -= (__nw * __bits_per_word).__data;
    // do last word
    if (__n > 0)
    {
      __storage_type __m = ~__storage_type(0) << (__bits_per_word - __n);
#if _CCCL_COMPILER(GCC, <, 9)
      // workaround for GCC pre-9 being really bad at tracking one-past-the-end pointers at constexpr
      // can't check for is-constant-evaluated, because GCC pre-9 also lacks _that_.
      if (__last.__seg_ == __first.__seg_ + 1)
      {
        __last.__seg_ = __first.__seg_;
      }
      else
      {
        --__last.__seg_;
      }
#else // ^^ GCC < 9 ^^ | vv !GCC || GCC >= 9 vv
      --__last.__seg_;
#endif // !GCC || GCC >= 9
      __storage_type __b = *__last.__seg_ & __m;
#if _CCCL_COMPILER(GCC, <, 9)
      // workaround for GCC pre-9 being really bad at tracking one-past-the-end pointers at constexpr
      // can't check for is-constant-evaluated, because GCC pre-9 also lacks _that_.
      if (__result.__seg_ == __first.__seg_ + 1)
      {
        __result.__seg_ = __first.__seg_;
      }
      else
      {
        --__result.__seg_;
      }
#else // ^^ GCC < 9 ^^ | vv !GCC || GCC >= 9 vv
      --__result.__seg_;
#endif // !GCC || GCC >= 9
      *__result.__seg_ &= ~__m;
      *__result.__seg_ |= __b;
      __result.__ctz_ = static_cast<unsigned>(-__n & (__bits_per_word - 1));
    }
  }
  return __result;
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4146) // unary minus applied to unsigned type

template <class _Cp, bool _IsConst>
_CCCL_API constexpr __bit_iterator<_Cp, false> __copy_backward_unaligned(
  __bit_iterator<_Cp, _IsConst> __first, __bit_iterator<_Cp, _IsConst> __last, __bit_iterator<_Cp, false> __result)
{
  using _In             = __bit_iterator<_Cp, _IsConst>;
  using difference_type = typename _In::difference_type;
  using __storage_type  = typename _In::__storage_type;

  const int __bits_per_word = _In::__bits_per_word;
  difference_type __n       = __last - __first;
  if (__n > 0)
  {
    // do first word
    if (__last.__ctz_ != 0)
    {
      difference_type __dn = ::cuda::std::min(static_cast<difference_type>(__last.__ctz_), __n);
      __n -= __dn;
      unsigned __clz_l     = __bits_per_word - __last.__ctz_;
      __storage_type __m   = (~__storage_type(0) << (__last.__ctz_ - __dn)) & (~__storage_type(0) >> __clz_l);
      __storage_type __b   = *__last.__seg_ & __m;
      unsigned __clz_r     = __bits_per_word - __result.__ctz_;
      __storage_type __ddn = ::cuda::std::min(__dn, static_cast<difference_type>(__result.__ctz_));
      if (__ddn > 0)
      {
        __m = (~__storage_type(0) << (__result.__ctz_ - __ddn)) & (~__storage_type(0) >> __clz_r);
        *__result.__seg_ &= ~__m;
        if (__result.__ctz_ > __last.__ctz_)
        {
          *__result.__seg_ |= __b << (__result.__ctz_ - __last.__ctz_);
        }
        else
        {
          *__result.__seg_ |= __b >> (__last.__ctz_ - __result.__ctz_);
        }
        __result.__ctz_ =
          static_cast<unsigned>((((-__ddn & (__bits_per_word - 1)) + __result.__ctz_) % __bits_per_word).__data);
        __dn -= __ddn.__data;
      }
      if (__dn > 0)
      {
        // __result.__ctz_ == 0
#if _CCCL_COMPILER(GCC, <, 9)
        // workaround for GCC pre-9 being really bad at tracking one-past-the-end pointers at constexpr
        // can't check for is-constant-evaluated, because GCC pre-9 also lacks _that_.
        if (__result.__seg_ == __first.__seg_ + 1)
        {
          __result.__seg_ = __first.__seg_;
        }
        else
        {
          --__result.__seg_;
        }
#else // ^^ GCC < 9 ^^ | vv !GCC || GCC >= 9 vv
        --__result.__seg_;
#endif // !GCC || GCC >= 9
        _CCCL_DIAG_PUSH
        _CCCL_DIAG_SUPPRESS_MSVC(4146) // unary minus applied to unsigned type
        __result.__ctz_ = static_cast<unsigned>(-__dn & (__bits_per_word - 1));
        _CCCL_DIAG_POP
        __m = ~__storage_type(0) << __result.__ctz_;
        *__result.__seg_ &= ~__m;
        __last.__ctz_ -= (__dn + __ddn).__data;
        *__result.__seg_ |= __b << (__result.__ctz_ - __last.__ctz_);
      }
      // __last.__ctz_ = 0
    }
    // __last.__ctz_ == 0 || __n == 0
    // __result.__ctz_ != 0 || __n == 0
    // do middle words
    unsigned __clz_r   = __bits_per_word - __result.__ctz_;
    __storage_type __m = ~__storage_type(0) >> __clz_r;
    for (; __n >= __bits_per_word; __n -= __bits_per_word)
    {
      __storage_type __b = *--__last.__seg_;
      *__result.__seg_ &= ~__m;
      *__result.__seg_ |= __b >> __clz_r;
      *--__result.__seg_ &= __m;
      *__result.__seg_ |= __b << __result.__ctz_;
    }
    // do last word
    if (__n > 0)
    {
      __m                 = ~__storage_type(0) << (__bits_per_word - __n);
      __storage_type __b  = *--__last.__seg_ & __m;
      __clz_r             = __bits_per_word - __result.__ctz_;
      __storage_type __dn = ::cuda::std::min(__n, static_cast<difference_type>(__result.__ctz_));
      __m                 = (~__storage_type(0) << (__result.__ctz_ - __dn)) & (~__storage_type(0) >> __clz_r);
      *__result.__seg_ &= ~__m;
      *__result.__seg_ |= __b >> (__bits_per_word - __result.__ctz_);
      __result.__ctz_ =
        static_cast<unsigned>((((-__dn & (__bits_per_word - 1)) + __result.__ctz_) % __bits_per_word).__data);
      __n -= __dn.__data;
      if (__n > 0)
      {
        // __result.__ctz_ == 0
        --__result.__seg_;
        __result.__ctz_ = static_cast<unsigned>(-__n & (__bits_per_word - 1));
        __m             = ~__storage_type(0) << __result.__ctz_;
        *__result.__seg_ &= ~__m;
        *__result.__seg_ |= __b << (__result.__ctz_ - (__bits_per_word - __n - __dn));
      }
    }
  }
  return __result;
}

_CCCL_DIAG_POP

template <class _Cp, bool _IsConst>
_CCCL_API constexpr __bit_iterator<_Cp, false> copy_backward(
  __bit_iterator<_Cp, _IsConst> __first, __bit_iterator<_Cp, _IsConst> __last, __bit_iterator<_Cp, false> __result)
{
  if (__last.__ctz_ == __result.__ctz_)
  {
    return ::cuda::std::__copy_backward_aligned(__first, __last, __result);
  }
  return ::cuda::std::__copy_backward_unaligned(__first, __last, __result);
}

// move

template <class _Cp, bool _IsConst>
_CCCL_API inline __bit_iterator<_Cp, false>
move(__bit_iterator<_Cp, _IsConst> __first, __bit_iterator<_Cp, _IsConst> __last, __bit_iterator<_Cp, false> __result)
{
  return ::cuda::std::copy(__first, __last, __result);
}

// move_backward

template <class _Cp, bool _IsConst>
_CCCL_API inline __bit_iterator<_Cp, false> move_backward(
  __bit_iterator<_Cp, _IsConst> __first, __bit_iterator<_Cp, _IsConst> __last, __bit_iterator<_Cp, false> __result)
{
  return ::cuda::std::copy_backward(__first, __last, __result);
}

// swap_ranges

template <class _Cl, class _Cr>
_CCCL_API inline __bit_iterator<_Cr, false> __swap_ranges_aligned(
  __bit_iterator<_Cl, false> __first, __bit_iterator<_Cl, false> __last, __bit_iterator<_Cr, false> __result)
{
  using _I1             = __bit_iterator<_Cl, false>;
  using difference_type = typename _I1::difference_type;
  using __storage_type  = typename _I1::__storage_type;

  const int __bits_per_word = _I1::__bits_per_word;
  difference_type __n       = __last - __first;
  if (__n > 0)
  {
    // do first word
    if (__first.__ctz_ != 0)
    {
      unsigned __clz       = __bits_per_word - __first.__ctz_;
      difference_type __dn = ::cuda::std::min(static_cast<difference_type>(__clz), __n);
      __n -= __dn;
      __storage_type __m  = (~__storage_type(0) << __first.__ctz_) & (~__storage_type(0) >> (__clz - __dn));
      __storage_type __b1 = *__first.__seg_ & __m;
      *__first.__seg_ &= ~__m;
      __storage_type __b2 = *__result.__seg_ & __m;
      *__result.__seg_ &= ~__m;
      *__result.__seg_ |= __b1;
      *__first.__seg_ |= __b2;
      __result.__seg_ += (__dn + __result.__ctz_) / __bits_per_word;
      __result.__ctz_ = static_cast<unsigned>((__dn + __result.__ctz_) % __bits_per_word);
      ++__first.__seg_;
      // __first.__ctz_ = 0;
    }
    // __first.__ctz_ == 0;
    // do middle words
    for (; __n >= __bits_per_word; __n -= __bits_per_word, ++__first.__seg_, ++__result.__seg_)
    {
      swap(*__first.__seg_, *__result.__seg_);
    }
    // do last word
    if (__n > 0)
    {
      __storage_type __m  = ~__storage_type(0) >> (__bits_per_word - __n);
      __storage_type __b1 = *__first.__seg_ & __m;
      *__first.__seg_ &= ~__m;
      __storage_type __b2 = *__result.__seg_ & __m;
      *__result.__seg_ &= ~__m;
      *__result.__seg_ |= __b1;
      *__first.__seg_ |= __b2;
      __result.__ctz_ = static_cast<unsigned>(__n);
    }
  }
  return __result;
}

template <class _Cl, class _Cr>
_CCCL_API inline __bit_iterator<_Cr, false> __swap_ranges_unaligned(
  __bit_iterator<_Cl, false> __first, __bit_iterator<_Cl, false> __last, __bit_iterator<_Cr, false> __result)
{
  using _I1             = __bit_iterator<_Cl, false>;
  using difference_type = typename _I1::difference_type;
  using __storage_type  = typename _I1::__storage_type;

  const int __bits_per_word = _I1::__bits_per_word;
  difference_type __n       = __last - __first;
  if (__n > 0)
  {
    // do first word
    if (__first.__ctz_ != 0)
    {
      unsigned __clz_f     = __bits_per_word - __first.__ctz_;
      difference_type __dn = ::cuda::std::min(static_cast<difference_type>(__clz_f), __n);
      __n -= __dn;
      __storage_type __m  = (~__storage_type(0) << __first.__ctz_) & (~__storage_type(0) >> (__clz_f - __dn));
      __storage_type __b1 = *__first.__seg_ & __m;
      *__first.__seg_ &= ~__m;
      unsigned __clz_r     = __bits_per_word - __result.__ctz_;
      __storage_type __ddn = ::cuda::std::min<__storage_type>(__dn, __clz_r);
      __m                  = (~__storage_type(0) << __result.__ctz_) & (~__storage_type(0) >> (__clz_r - __ddn));
      __storage_type __b2  = *__result.__seg_ & __m;
      *__result.__seg_ &= ~__m;
      if (__result.__ctz_ > __first.__ctz_)
      {
        unsigned __s = __result.__ctz_ - __first.__ctz_;
        *__result.__seg_ |= __b1 << __s;
        *__first.__seg_ |= __b2 >> __s;
      }
      else
      {
        unsigned __s = __first.__ctz_ - __result.__ctz_;
        *__result.__seg_ |= __b1 >> __s;
        *__first.__seg_ |= __b2 << __s;
      }
      __result.__seg_ += (__ddn + __result.__ctz_) / __bits_per_word;
      __result.__ctz_ = static_cast<unsigned>((__ddn + __result.__ctz_) % __bits_per_word);
      __dn -= __ddn;
      if (__dn > 0)
      {
        __m  = ~__storage_type(0) >> (__bits_per_word - __dn);
        __b2 = *__result.__seg_ & __m;
        *__result.__seg_ &= ~__m;
        unsigned __s = __first.__ctz_ + __ddn;
        *__result.__seg_ |= __b1 >> __s;
        *__first.__seg_ |= __b2 << __s;
        __result.__ctz_ = static_cast<unsigned>(__dn);
      }
      ++__first.__seg_;
      // __first.__ctz_ = 0;
    }
    // __first.__ctz_ == 0;
    // do middle words
    __storage_type __m = ~__storage_type(0) << __result.__ctz_;
    unsigned __clz_r   = __bits_per_word - __result.__ctz_;
    for (; __n >= __bits_per_word; __n -= __bits_per_word, ++__first.__seg_)
    {
      __storage_type __b1 = *__first.__seg_;
      __storage_type __b2 = *__result.__seg_ & __m;
      *__result.__seg_ &= ~__m;
      *__result.__seg_ |= __b1 << __result.__ctz_;
      *__first.__seg_ = __b2 >> __result.__ctz_;
      ++__result.__seg_;
      __b2 = *__result.__seg_ & ~__m;
      *__result.__seg_ &= __m;
      *__result.__seg_ |= __b1 >> __clz_r;
      *__first.__seg_ |= __b2 << __clz_r;
    }
    // do last word
    if (__n > 0)
    {
      __m                 = ~__storage_type(0) >> (__bits_per_word - __n);
      __storage_type __b1 = *__first.__seg_ & __m;
      *__first.__seg_ &= ~__m;
      __storage_type __dn = ::cuda::std::min<__storage_type>(__n, __clz_r);
      __m                 = (~__storage_type(0) << __result.__ctz_) & (~__storage_type(0) >> (__clz_r - __dn));
      __storage_type __b2 = *__result.__seg_ & __m;
      *__result.__seg_ &= ~__m;
      *__result.__seg_ |= __b1 << __result.__ctz_;
      *__first.__seg_ |= __b2 >> __result.__ctz_;
      __result.__seg_ += (__dn + __result.__ctz_) / __bits_per_word;
      __result.__ctz_ = static_cast<unsigned>((__dn + __result.__ctz_) % __bits_per_word);
      __n -= __dn;
      if (__n > 0)
      {
        __m  = ~__storage_type(0) >> (__bits_per_word - __n);
        __b2 = *__result.__seg_ & __m;
        *__result.__seg_ &= ~__m;
        *__result.__seg_ |= __b1 >> __dn;
        *__first.__seg_ |= __b2 << __dn;
        __result.__ctz_ = static_cast<unsigned>(__n);
      }
    }
  }
  return __result;
}

template <class _Cl, class _Cr>
_CCCL_API inline __bit_iterator<_Cr, false> swap_ranges(
  __bit_iterator<_Cl, false> __first1, __bit_iterator<_Cl, false> __last1, __bit_iterator<_Cr, false> __first2)
{
  if (__first1.__ctz_ == __first2.__ctz_)
  {
    return ::cuda::std::__swap_ranges_aligned(__first1, __last1, __first2);
  }
  return ::cuda::std::__swap_ranges_unaligned(__first1, __last1, __first2);
}

// rotate

template <class _Cp>
struct __bit_array
{
  using difference_type   = typename _Cp::difference_type;
  using __storage_type    = typename _Cp::__storage_type;
  using __storage_pointer = typename _Cp::__storage_pointer;
  using iterator          = typename _Cp::iterator;

  static const unsigned __bits_per_word = _Cp::__bits_per_word;
  static const unsigned _Np             = 4;

  difference_type __size_;
  __storage_type __word_[_Np];

  _CCCL_API constexpr static difference_type capacity()
  {
    return static_cast<difference_type>(_Np * __bits_per_word);
  }
  _CCCL_API constexpr explicit __bit_array(difference_type __s)
      : __size_(__s)
  {
    if (::cuda::std::is_constant_evaluated())
    {
      for (size_t __i = 0; __i != __bit_array<_Cp>::_Np; ++__i)
      {
        ::cuda::std::__construct_at(__word_ + __i, 0);
      }
    }
  }
  _CCCL_API constexpr iterator begin()
  {
    return iterator(pointer_traits<__storage_pointer>::pointer_to(__word_[0]), 0);
  }
  _CCCL_API constexpr iterator end()
  {
    return iterator(pointer_traits<__storage_pointer>::pointer_to(__word_[0]) + __size_ / __bits_per_word,
                    static_cast<unsigned>(__size_ % __bits_per_word));
  }
};

template <class _Cp>
_CCCL_API constexpr __bit_iterator<_Cp, false>
rotate(__bit_iterator<_Cp, false> __first, __bit_iterator<_Cp, false> __middle, __bit_iterator<_Cp, false> __last)
{
  using _I1             = __bit_iterator<_Cp, false>;
  using difference_type = typename _I1::difference_type;

  difference_type __d1 = __middle - __first;
  difference_type __d2 = __last - __middle;
  _I1 __r              = __first + __d2;
  while (__d1 != 0 && __d2 != 0)
  {
    if (__d1 <= __d2)
    {
      if (__d1 <= __bit_array<_Cp>::capacity())
      {
        __bit_array<_Cp> __b(__d1);
        ::cuda::std::copy(__first, __middle, __b.begin());
        ::cuda::std::copy(__b.begin(), __b.end(), ::cuda::std::copy(__middle, __last, __first));
        break;
      }
      else
      {
        __bit_iterator<_Cp, false> __mp = ::cuda::std::swap_ranges(__first, __middle, __middle);
        __first                         = __middle;
        __middle                        = __mp;
        __d2 -= __d1;
      }
    }
    else
    {
      if (__d2 <= __bit_array<_Cp>::capacity())
      {
        __bit_array<_Cp> __b(__d2);
        ::cuda::std::copy(__middle, __last, __b.begin());
        ::cuda::std::copy_backward(__b.begin(), __b.end(), ::cuda::std::copy_backward(__first, __middle, __last));
        break;
      }
      else
      {
        __bit_iterator<_Cp, false> __mp = __first + __d2;
        ::cuda::std::swap_ranges(__first, __mp, __middle);
        __first = __mp;
        __d1 -= __d2;
      }
    }
  }
  return __r;
}

// equal

template <class _Cp, bool _IC1, bool _IC2>
_CCCL_API constexpr bool __equal_unaligned(
  __bit_iterator<_Cp, _IC1> __first1, __bit_iterator<_Cp, _IC1> __last1, __bit_iterator<_Cp, _IC2> __first2)
{
  using _It             = __bit_iterator<_Cp, _IC1>;
  using difference_type = typename _It::difference_type;
  using __storage_type  = typename _It::__storage_type;

  const int __bits_per_word = _It::__bits_per_word;
  difference_type __n       = __last1 - __first1;
  if (__n > 0)
  {
    // do first word
    if (__first1.__ctz_ != 0)
    {
      unsigned __clz_f     = __bits_per_word - __first1.__ctz_;
      difference_type __dn = ::cuda::std::min(static_cast<difference_type>(__clz_f), __n);
      __n -= __dn;
      __storage_type __m   = (~__storage_type(0) << __first1.__ctz_) & (~__storage_type(0) >> (__clz_f - __dn));
      __storage_type __b   = *__first1.__seg_ & __m;
      unsigned __clz_r     = __bits_per_word - __first2.__ctz_;
      __storage_type __ddn = ::cuda::std::min<__storage_type>(__dn, __clz_r);
      __m                  = (~__storage_type(0) << __first2.__ctz_) & (~__storage_type(0) >> (__clz_r - __ddn));
      if (__first2.__ctz_ > __first1.__ctz_)
      {
        if ((*__first2.__seg_ & __m) != (__b << (__first2.__ctz_ - __first1.__ctz_)))
        {
          return false;
        }
      }
      else
      {
        if ((*__first2.__seg_ & __m) != (__b >> (__first1.__ctz_ - __first2.__ctz_)))
        {
          return false;
        }
      }
      __first2.__seg_ += ((__ddn + __first2.__ctz_) / __bits_per_word).__data;
      __first2.__ctz_ = static_cast<unsigned>(((__ddn + __first2.__ctz_) % __bits_per_word).__data);
      __dn -= __ddn.__data;
      if (__dn > 0)
      {
        __m = ~__storage_type(0) >> (__bits_per_word - __dn);
        if ((*__first2.__seg_ & __m) != (__b >> (__first1.__ctz_ + __ddn)))
        {
          return false;
        }
        __first2.__ctz_ = static_cast<unsigned>(__dn);
      }
      ++__first1.__seg_;
      // __first1.__ctz_ = 0;
    }
    // __first1.__ctz_ == 0;
    // do middle words
    unsigned __clz_r   = __bits_per_word - __first2.__ctz_;
    __storage_type __m = ~__storage_type(0) << __first2.__ctz_;
    for (; __n >= __bits_per_word; __n -= __bits_per_word, ++__first1.__seg_)
    {
      __storage_type __b = *__first1.__seg_;
      if ((*__first2.__seg_ & __m) != (__b << __first2.__ctz_))
      {
        return false;
      }
      ++__first2.__seg_;
      if ((*__first2.__seg_ & ~__m) != (__b >> __clz_r))
      {
        return false;
      }
    }
    // do last word
    if (__n > 0)
    {
      __m                 = ~__storage_type(0) >> (__bits_per_word - __n);
      __storage_type __b  = *__first1.__seg_ & __m;
      __storage_type __dn = ::cuda::std::min(__n, static_cast<difference_type>(__clz_r));
      __m                 = (~__storage_type(0) << __first2.__ctz_) & (~__storage_type(0) >> (__clz_r - __dn));
      if ((*__first2.__seg_ & __m) != (__b << __first2.__ctz_))
      {
        return false;
      }
      __first2.__seg_ += ((__dn + __first2.__ctz_) / __bits_per_word).__data;
      __first2.__ctz_ = static_cast<unsigned>(((__dn + __first2.__ctz_) % __bits_per_word).__data);
      __n -= __dn.__data;
      if (__n > 0)
      {
        __m = ~__storage_type(0) >> (__bits_per_word - __n);
        if ((*__first2.__seg_ & __m) != (__b >> __dn))
        {
          return false;
        }
      }
    }
  }
  return true;
}

template <class _Cp, bool _IC1, bool _IC2>
_CCCL_API constexpr bool __equal_aligned(
  __bit_iterator<_Cp, _IC1> __first1, __bit_iterator<_Cp, _IC1> __last1, __bit_iterator<_Cp, _IC2> __first2)
{
  using _It             = __bit_iterator<_Cp, _IC1>;
  using difference_type = typename _It::difference_type;
  using __storage_type  = typename _It::__storage_type;

  const int __bits_per_word = _It::__bits_per_word;
  difference_type __n       = __last1 - __first1;
  if (__n > 0)
  {
    // do first word
    if (__first1.__ctz_ != 0)
    {
      unsigned __clz       = __bits_per_word - __first1.__ctz_;
      difference_type __dn = ::cuda::std::min(static_cast<difference_type>(__clz), __n);
      __n -= __dn;
      __storage_type __m = (~__storage_type(0) << __first1.__ctz_) & (~__storage_type(0) >> (__clz - __dn));
      if ((*__first2.__seg_ & __m) != (*__first1.__seg_ & __m))
      {
        return false;
      }
      ++__first2.__seg_;
      ++__first1.__seg_;
      // __first1.__ctz_ = 0;
      // __first2.__ctz_ = 0;
    }
    // __first1.__ctz_ == 0;
    // __first2.__ctz_ == 0;
    // do middle words
    for (; __n >= __bits_per_word; __n -= __bits_per_word, ++__first1.__seg_, ++__first2.__seg_)
    {
      if (*__first2.__seg_ != *__first1.__seg_)
      {
        return false;
      }
    }
    // do last word
    if (__n > 0)
    {
      __storage_type __m = ~__storage_type(0) >> (__bits_per_word - __n);
      if ((*__first2.__seg_ & __m) != (*__first1.__seg_ & __m))
      {
        return false;
      }
    }
  }
  return true;
}

template <class _Cp, bool _IC1, bool _IC2>
_CCCL_API constexpr bool
equal(__bit_iterator<_Cp, _IC1> __first1, __bit_iterator<_Cp, _IC1> __last1, __bit_iterator<_Cp, _IC2> __first2)
{
  if (__first1.__ctz_ == __first2.__ctz_)
  {
    return ::cuda::std::__equal_aligned(__first1, __last1, __first2);
  }
  return ::cuda::std::__equal_unaligned(__first1, __last1, __first2);
}

template <class _Cp, bool _IsConst>
class __bit_iterator
{
public:
  using difference_type   = typename _Cp::difference_type;
  using value_type        = bool;
  using pointer           = __bit_iterator;
  using reference         = conditional_t<_IsConst, __bit_const_reference<_Cp>, __bit_reference<_Cp>>;
  using iterator_category = random_access_iterator_tag;

private:
  using __storage_type = typename _Cp::__storage_type;
  using __storage_pointer =
    conditional_t<_IsConst, typename _Cp::__const_storage_pointer, typename _Cp::__storage_pointer>;

  static const unsigned __bits_per_word = _Cp::__bits_per_word;

  __storage_pointer __seg_;
  unsigned __ctz_;

public:
  _CCCL_API constexpr __bit_iterator() noexcept
      : __seg_(nullptr)
      , __ctz_(0)
  {}

  _CCCL_HIDE_FROM_ABI constexpr __bit_iterator(const __bit_iterator<_Cp, _IsConst>& __it) = default;

  template <bool _OtherIsConst, class = enable_if_t<_IsConst == true && _OtherIsConst == false>>
  _CCCL_API constexpr __bit_iterator(const __bit_iterator<_Cp, _OtherIsConst>& __it) noexcept
      : __seg_(__it.__seg_)
      , __ctz_(__it.__ctz_)
  {}

  _CCCL_API constexpr reference operator*() const noexcept
  {
    return reference(__seg_, __storage_type(1) << __ctz_);
  }

  _CCCL_API constexpr __bit_iterator& operator++()
  {
    if (__ctz_ != __bits_per_word - 1)
    {
      ++__ctz_;
    }
    else
    {
      __ctz_ = 0;
      ++__seg_;
    }
    return *this;
  }

  _CCCL_API constexpr __bit_iterator operator++(int)
  {
    __bit_iterator __tmp = *this;
    ++(*this);
    return __tmp;
  }

  _CCCL_API constexpr __bit_iterator& operator--()
  {
    if (__ctz_ != 0)
    {
      --__ctz_;
    }
    else
    {
      __ctz_ = __bits_per_word - 1;
      --__seg_;
    }
    return *this;
  }

  _CCCL_API constexpr __bit_iterator operator--(int)
  {
    __bit_iterator __tmp = *this;
    --(*this);
    return __tmp;
  }

  _CCCL_API constexpr __bit_iterator& operator+=(difference_type __n)
  {
    if (__n >= 0)
    {
      __seg_ += (__n + __ctz_) / __bits_per_word;
    }
    else
    {
      __seg_ += static_cast<difference_type>(__n - __bits_per_word + __ctz_ + 1)
              / static_cast<difference_type>(__bits_per_word);
    }
    __n &= (__bits_per_word - 1);
    __ctz_ = static_cast<unsigned>((__n + __ctz_) % __bits_per_word);
    return *this;
  }

  _CCCL_API constexpr __bit_iterator& operator-=(difference_type __n)
  {
    return *this += -__n;
  }

  _CCCL_API constexpr __bit_iterator operator+(difference_type __n) const
  {
    __bit_iterator __t(*this);
    __t += __n;
    return __t;
  }

  _CCCL_API constexpr __bit_iterator operator-(difference_type __n) const
  {
    __bit_iterator __t(*this);
    __t -= __n;
    return __t;
  }

  _CCCL_API constexpr friend __bit_iterator operator+(difference_type __n, const __bit_iterator& __it)
  {
    return __it + __n;
  }

  _CCCL_API constexpr friend difference_type operator-(const __bit_iterator& __x, const __bit_iterator& __y)
  {
#if _CCCL_COMPILER(GCC, >=, 8) && _CCCL_COMPILER(GCC, <, 9)
    if (__y.__seg_ && __y.__seg_ != __x.__seg_)
    {
      return (__x.__seg_ == __y.__seg_ + 1 ? 1 : __x.__seg_ - __y.__seg_) * __bits_per_word + __x.__ctz_ - __y.__ctz_;
    }
#endif // GCC [8, 9)
    return (__x.__seg_ - __y.__seg_) * __bits_per_word + __x.__ctz_ - __y.__ctz_;
  }

  _CCCL_API constexpr reference operator[](difference_type __n) const
  {
    return *(*this + __n);
  }

  _CCCL_API constexpr friend bool operator==(const __bit_iterator& __x, const __bit_iterator& __y)
  {
    return __x.__seg_ == __y.__seg_ && __x.__ctz_ == __y.__ctz_;
  }

  _CCCL_API constexpr friend bool operator!=(const __bit_iterator& __x, const __bit_iterator& __y)
  {
    return !(__x == __y);
  }

  _CCCL_API constexpr friend bool operator<(const __bit_iterator& __x, const __bit_iterator& __y)
  {
    return __x.__seg_ < __y.__seg_ || (__x.__seg_ == __y.__seg_ && __x.__ctz_ < __y.__ctz_);
  }

  _CCCL_API constexpr friend bool operator>(const __bit_iterator& __x, const __bit_iterator& __y)
  {
    return __y < __x;
  }

  _CCCL_API constexpr friend bool operator<=(const __bit_iterator& __x, const __bit_iterator& __y)
  {
    return !(__y < __x);
  }

  _CCCL_API constexpr friend bool operator>=(const __bit_iterator& __x, const __bit_iterator& __y)
  {
    return !(__x < __y);
  }

private:
  _CCCL_API constexpr explicit __bit_iterator(__storage_pointer __s, unsigned __ctz) noexcept
      : __seg_(__s)
      , __ctz_(__ctz)
  {}

  friend typename _Cp::__self;

  friend class __bit_reference<_Cp>;
  friend class __bit_const_reference<_Cp>;
  friend class __bit_iterator<_Cp, true>;
  template <class _Dp>
  friend struct __bit_array;
  template <bool _FillVal, class _Dp>
  constexpr _CCCL_API inline friend void __fill_n_impl(__bit_iterator<_Dp, false> __first, typename _Dp::size_type __n);

  template <class _Dp, bool _IC>
  constexpr _CCCL_API inline friend __bit_iterator<_Dp, false> __copy_aligned(
    __bit_iterator<_Dp, _IC> __first, __bit_iterator<_Dp, _IC> __last, __bit_iterator<_Dp, false> __result);
  template <class _Dp, bool _IC>
  constexpr _CCCL_API inline friend __bit_iterator<_Dp, false> __copy_unaligned(
    __bit_iterator<_Dp, _IC> __first, __bit_iterator<_Dp, _IC> __last, __bit_iterator<_Dp, false> __result);
  template <class _Dp, bool _IC>
  constexpr _CCCL_API inline friend __bit_iterator<_Dp, false>
  copy(__bit_iterator<_Dp, _IC> __first, __bit_iterator<_Dp, _IC> __last, __bit_iterator<_Dp, false> __result);
  template <class _Dp, bool _IC>
  constexpr _CCCL_API inline friend __bit_iterator<_Dp, false> __copy_backward_aligned(
    __bit_iterator<_Dp, _IC> __first, __bit_iterator<_Dp, _IC> __last, __bit_iterator<_Dp, false> __result);
  template <class _Dp, bool _IC>
  constexpr _CCCL_API inline friend __bit_iterator<_Dp, false> __copy_backward_unaligned(
    __bit_iterator<_Dp, _IC> __first, __bit_iterator<_Dp, _IC> __last, __bit_iterator<_Dp, false> __result);
  template <class _Dp, bool _IC>
  constexpr _CCCL_API inline friend __bit_iterator<_Dp, false>
  copy_backward(__bit_iterator<_Dp, _IC> __first, __bit_iterator<_Dp, _IC> __last, __bit_iterator<_Dp, false> __result);
  template <class _Cl, class _Cr>
  _CCCL_API inline friend __bit_iterator<_Cr, false>
    __swap_ranges_aligned(__bit_iterator<_Cl, false>, __bit_iterator<_Cl, false>, __bit_iterator<_Cr, false>);
  template <class _Cl, class _Cr>
  _CCCL_API inline friend __bit_iterator<_Cr, false>
    __swap_ranges_unaligned(__bit_iterator<_Cl, false>, __bit_iterator<_Cl, false>, __bit_iterator<_Cr, false>);
  template <class _Cl, class _Cr>
  _CCCL_API inline friend __bit_iterator<_Cr, false>
    swap_ranges(__bit_iterator<_Cl, false>, __bit_iterator<_Cl, false>, __bit_iterator<_Cr, false>);
  template <class _Dp>
  constexpr _CCCL_API inline friend __bit_iterator<_Dp, false>
    rotate(__bit_iterator<_Dp, false>, __bit_iterator<_Dp, false>, __bit_iterator<_Dp, false>);
  template <class _Dp, bool _IC1, bool _IC2>
  constexpr _CCCL_API inline friend bool
    __equal_aligned(__bit_iterator<_Dp, _IC1>, __bit_iterator<_Dp, _IC1>, __bit_iterator<_Dp, _IC2>);
  template <class _Dp, bool _IC1, bool _IC2>
  constexpr _CCCL_API inline friend bool
    __equal_unaligned(__bit_iterator<_Dp, _IC1>, __bit_iterator<_Dp, _IC1>, __bit_iterator<_Dp, _IC2>);
  template <class _Dp, bool _IC1, bool _IC2>
  constexpr
    _CCCL_API inline friend bool equal(__bit_iterator<_Dp, _IC1>, __bit_iterator<_Dp, _IC1>, __bit_iterator<_Dp, _IC2>);
  template <bool _ToFind, class _Dp, bool _IC>
  constexpr _CCCL_API inline friend __bit_iterator<_Dp, _IC>
    __find_bool(__bit_iterator<_Dp, _IC>, typename _Dp::size_type);
  template <bool _ToCount, class _Dp, bool _IC>
  constexpr _CCCL_API inline friend
    typename __bit_iterator<_Dp, _IC>::difference_type __count_bool(__bit_iterator<_Dp, _IC>, typename _Dp::size_type);
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_REFERENCE_H
