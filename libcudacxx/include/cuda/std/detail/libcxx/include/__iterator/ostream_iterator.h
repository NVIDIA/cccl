// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_OSTREAM_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_OSTREAM_ITERATOR_H

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

#include "../__iterator/iterator.h"
#include "../__iterator/iterator_traits.h"
#include "../__memory/addressof.h"
#include "../cstddef"
#include "../iosfwd"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Tp, class _CharT = char, class _Traits = char_traits<_CharT> >
class _LIBCUDACXX_TEMPLATE_VIS ostream_iterator
#if _CCCL_STD_VER <= 2014 || !defined(_LIBCUDACXX_ABI_NO_ITERATOR_BASES)
    : public iterator<output_iterator_tag, void, void, void, void>
#endif
{
_CCCL_SUPPRESS_DEPRECATED_POP
public:
    typedef output_iterator_tag             iterator_category;
    typedef void                            value_type;
#if _CCCL_STD_VER > 2017
    typedef ptrdiff_t                       difference_type;
#else
    typedef void                            difference_type;
#endif
    typedef void                            pointer;
    typedef void                            reference;
    typedef _CharT                          char_type;
    typedef _Traits                         traits_type;
    typedef basic_ostream<_CharT, _Traits>  ostream_type;

private:
    ostream_type* __out_stream_;
    const char_type* __delim_;
public:
    _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator(ostream_type& __s) noexcept
        : __out_stream_(_CUDA_VSTD::addressof(__s)), __delim_(nullptr) {}
    _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator(ostream_type& __s, const _CharT* __delimiter) noexcept
        : __out_stream_(_CUDA_VSTD::addressof(__s)), __delim_(__delimiter) {}
    _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator& operator=(const _Tp& __value)
        {
            *__out_stream_ << __value;
            if (__delim_)
                *__out_stream_ << __delim_;
            return *this;
        }

    _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator& operator*()     {return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator& operator++()    {return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator& operator++(int) {return *this;}
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_OSTREAM_ITERATOR_H
