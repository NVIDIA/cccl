// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ISTREAMBUF_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_ISTREAMBUF_ITERATOR_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

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
#include "../iosfwd"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_SUPPRESS_DEPRECATED_PUSH
template<class _CharT, class _Traits>
class _LIBCUDACXX_TEMPLATE_VIS istreambuf_iterator
#if _CCCL_STD_VER <= 2014 || !defined(_LIBCUDACXX_ABI_NO_ITERATOR_BASES)
    : public iterator<input_iterator_tag, _CharT,
                      typename _Traits::off_type, _CharT*,
                      _CharT>
#endif
{
_CCCL_SUPPRESS_DEPRECATED_POP
public:
    typedef input_iterator_tag              iterator_category;
    typedef _CharT                          value_type;
    typedef typename _Traits::off_type      difference_type;
    typedef _CharT*                         pointer;
    typedef _CharT                          reference;
    typedef _CharT                          char_type;
    typedef _Traits                         traits_type;
    typedef typename _Traits::int_type      int_type;
    typedef basic_streambuf<_CharT,_Traits> streambuf_type;
    typedef basic_istream<_CharT,_Traits>   istream_type;
private:
    mutable streambuf_type* __sbuf_;

    class __proxy
    {
        char_type __keep_;
        streambuf_type* __sbuf_;
        _LIBCUDACXX_INLINE_VISIBILITY
        explicit __proxy(char_type __c, streambuf_type* __s)
            : __keep_(__c), __sbuf_(__s) {}
        friend class istreambuf_iterator;
    public:
        _LIBCUDACXX_INLINE_VISIBILITY char_type operator*() const {return __keep_;}
    };

    _LIBCUDACXX_INLINE_VISIBILITY
    bool __test_for_eof() const
    {
        if (__sbuf_ && traits_type::eq_int_type(__sbuf_->sgetc(), traits_type::eof()))
            __sbuf_ = nullptr;
        return __sbuf_ == nullptr;
    }
public:
    _LIBCUDACXX_INLINE_VISIBILITY constexpr istreambuf_iterator() noexcept : __sbuf_(nullptr) {}
#if _CCCL_STD_VER > 2017
    _LIBCUDACXX_INLINE_VISIBILITY constexpr istreambuf_iterator(default_sentinel_t) noexcept
        : istreambuf_iterator() {}
#endif // _CCCL_STD_VER > 2017
    _LIBCUDACXX_INLINE_VISIBILITY istreambuf_iterator(istream_type& __s) noexcept
        : __sbuf_(__s.rdbuf()) {}
    _LIBCUDACXX_INLINE_VISIBILITY istreambuf_iterator(streambuf_type* __s) noexcept
        : __sbuf_(__s) {}
    _LIBCUDACXX_INLINE_VISIBILITY istreambuf_iterator(const __proxy& __p) noexcept
        : __sbuf_(__p.__sbuf_) {}

    _LIBCUDACXX_INLINE_VISIBILITY char_type  operator*() const
        {return static_cast<char_type>(__sbuf_->sgetc());}
    _LIBCUDACXX_INLINE_VISIBILITY istreambuf_iterator& operator++()
        {
            __sbuf_->sbumpc();
            return *this;
        }
    _LIBCUDACXX_INLINE_VISIBILITY __proxy              operator++(int)
        {
            return __proxy(__sbuf_->sbumpc(), __sbuf_);
        }

    _LIBCUDACXX_INLINE_VISIBILITY bool equal(const istreambuf_iterator& __b) const
        {return __test_for_eof() == __b.__test_for_eof();}

#if _CCCL_STD_VER > 2014
    friend _LIBCUDACXX_HIDE_FROM_ABI bool operator==(const istreambuf_iterator& __i, default_sentinel_t) {
      return __i.__test_for_eof();
    }
#if _CCCL_STD_VER < 2020
    friend _LIBCUDACXX_HIDE_FROM_ABI bool operator==(default_sentinel_t, const istreambuf_iterator& __i) {
      return __i.__test_for_eof();
    }
    friend _LIBCUDACXX_HIDE_FROM_ABI bool operator!=(const istreambuf_iterator& __i, default_sentinel_t) {
      return !__i.__test_for_eof();
    }
    friend _LIBCUDACXX_HIDE_FROM_ABI bool operator!=(default_sentinel_t, const istreambuf_iterator& __i) {
      return !__i.__test_for_eof();
    }
#endif // _CCCL_STD_VER < 2020
#endif // _CCCL_STD_VER > 2014
};

template <class _CharT, class _Traits>
inline _LIBCUDACXX_INLINE_VISIBILITY
bool operator==(const istreambuf_iterator<_CharT,_Traits>& __a,
                const istreambuf_iterator<_CharT,_Traits>& __b)
                {return __a.equal(__b);}

#if _CCCL_STD_VER <= 2017
template <class _CharT, class _Traits>
inline _LIBCUDACXX_INLINE_VISIBILITY
bool operator!=(const istreambuf_iterator<_CharT,_Traits>& __a,
                const istreambuf_iterator<_CharT,_Traits>& __b)
                {return !__a.equal(__b);}
#endif // _CCCL_STD_VER <= 2017

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_ISTREAMBUF_ITERATOR_H
