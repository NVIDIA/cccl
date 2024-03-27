// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ISTREAM_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_ISTREAM_ITERATOR_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__iterator/default_sentinel.h"
#include "../__iterator/iterator_traits.h"
#include "../__iterator/iterator.h"
#include "../__memory/addressof.h"
#include "../cstddef"
#include "../iosfwd"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Tp, class _CharT = char,
          class _Traits = char_traits<_CharT>, class _Distance = ptrdiff_t>
class _LIBCUDACXX_TEMPLATE_VIS istream_iterator
#if _CCCL_STD_VER <= 2014 || !defined(_LIBCUDACXX_ABI_NO_ITERATOR_BASES)
    : public iterator<input_iterator_tag, _Tp, _Distance, const _Tp*, const _Tp&>
#endif
{
_CCCL_SUPPRESS_DEPRECATED_POP
public:
    typedef input_iterator_tag iterator_category;
    typedef _Tp value_type;
    typedef _Distance difference_type;
    typedef const _Tp* pointer;
    typedef const _Tp& reference;
    typedef _CharT char_type;
    typedef _Traits traits_type;
    typedef basic_istream<_CharT,_Traits> istream_type;
private:
    istream_type* __in_stream_;
    _Tp __value_;
public:
    _LIBCUDACXX_INLINE_VISIBILITY constexpr istream_iterator() : __in_stream_(nullptr), __value_() {}
#if _CCCL_STD_VER > 2014
    _LIBCUDACXX_INLINE_VISIBILITY constexpr istream_iterator(default_sentinel_t) : istream_iterator() {}
#endif // _CCCL_STD_VER > 2014
    _LIBCUDACXX_INLINE_VISIBILITY istream_iterator(istream_type& __s) : __in_stream_(_CUDA_VSTD::addressof(__s))
        {
            if (!(*__in_stream_ >> __value_))
                __in_stream_ = nullptr;
        }

    _LIBCUDACXX_INLINE_VISIBILITY const _Tp& operator*() const {return __value_;}
    _LIBCUDACXX_INLINE_VISIBILITY const _Tp* operator->() const {return _CUDA_VSTD::addressof((operator*()));}
    _LIBCUDACXX_INLINE_VISIBILITY istream_iterator& operator++()
        {
            if (!(*__in_stream_ >> __value_))
                __in_stream_ = nullptr;
            return *this;
        }
    _LIBCUDACXX_INLINE_VISIBILITY istream_iterator  operator++(int)
        {istream_iterator __t(*this); ++(*this); return __t;}

    template <class _Up, class _CharU, class _TraitsU, class _DistanceU>
    friend _LIBCUDACXX_INLINE_VISIBILITY
    bool
    operator==(const istream_iterator<_Up, _CharU, _TraitsU, _DistanceU>& __x,
               const istream_iterator<_Up, _CharU, _TraitsU, _DistanceU>& __y);

#if _CCCL_STD_VER > 2014
    friend _LIBCUDACXX_INLINE_VISIBILITY bool operator==(const istream_iterator& __i, default_sentinel_t) {
      return __i.__in_stream_ == nullptr;
    }
#if _CCCL_STD_VER < 2020
    friend _LIBCUDACXX_INLINE_VISIBILITY bool operator==(default_sentinel_t, const istream_iterator& __i) {
      return __i.__in_stream_ == nullptr;
    }
    friend _LIBCUDACXX_INLINE_VISIBILITY bool operator!=(const istream_iterator& __i, default_sentinel_t) {
      return __i.__in_stream_ != nullptr;
    }
    friend _LIBCUDACXX_INLINE_VISIBILITY bool operator!=(default_sentinel_t, const istream_iterator& __i) {
      return __i.__in_stream_ != nullptr;
    }
#endif // _CCCL_STD_VER < 2020
#endif // _CCCL_STD_VER > 2014
};

template <class _Tp, class _CharT, class _Traits, class _Distance>
inline _LIBCUDACXX_INLINE_VISIBILITY
bool
operator==(const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __x,
           const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __y)
{
    return __x.__in_stream_ == __y.__in_stream_;
}

template <class _Tp, class _CharT, class _Traits, class _Distance>
inline _LIBCUDACXX_INLINE_VISIBILITY
bool
operator!=(const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __x,
           const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __y)
{
    return !(__x == __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_ISTREAM_ITERATOR_H
