// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_OSTREAMBUF_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_OSTREAMBUF_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/iterator.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/cstddef>
#include <cuda/std/detail/libcxx/include/iosfwd>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _CharT, class _Traits>
class _CCCL_TYPE_VISIBILITY_DEFAULT ostreambuf_iterator
#if _CCCL_STD_VER <= 2014 || !defined(_LIBCUDACXX_ABI_NO_ITERATOR_BASES)
    : public iterator<output_iterator_tag, void, void, void, void>
#endif
{
  _CCCL_SUPPRESS_DEPRECATED_POP

public:
  typedef output_iterator_tag iterator_category;
  typedef void value_type;
#if _CCCL_STD_VER > 2017
  typedef ptrdiff_t difference_type;
#else
  typedef void difference_type;
#endif
  typedef void pointer;
  typedef void reference;
  typedef _CharT char_type;
  typedef _Traits traits_type;
  typedef basic_streambuf<_CharT, _Traits> streambuf_type;
  typedef basic_ostream<_CharT, _Traits> ostream_type;

private:
  streambuf_type* __sbuf_;

public:
  _LIBCUDACXX_HIDE_FROM_ABI ostreambuf_iterator(ostream_type& __s) noexcept
      : __sbuf_(__s.rdbuf())
  {}
  _LIBCUDACXX_HIDE_FROM_ABI ostreambuf_iterator(streambuf_type* __s) noexcept
      : __sbuf_(__s)
  {}
  _LIBCUDACXX_HIDE_FROM_ABI ostreambuf_iterator& operator=(_CharT __c)
  {
    if (__sbuf_ && traits_type::eq_int_type(__sbuf_->sputc(__c), traits_type::eof()))
    {
      __sbuf_ = nullptr;
    }
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI ostreambuf_iterator& operator*()
  {
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI ostreambuf_iterator& operator++()
  {
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI ostreambuf_iterator& operator++(int)
  {
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI bool failed() const noexcept
  {
    return __sbuf_ == nullptr;
  }

  template <class _Ch, class _Tr>
  friend _LIBCUDACXX_HIDE_FROM_ABI ostreambuf_iterator<_Ch, _Tr> __pad_and_output(
    ostreambuf_iterator<_Ch, _Tr> __s, const _Ch* __ob, const _Ch* __op, const _Ch* __oe, ios_base& __iob, _Ch __fl);
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_OSTREAMBUF_ITERATOR_H
