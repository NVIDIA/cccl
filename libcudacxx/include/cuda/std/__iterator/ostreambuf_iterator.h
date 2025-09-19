// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ITERATOR_OSTREAMBUF_ITERATOR_H
#define _CUDA_STD___ITERATOR_OSTREAMBUF_ITERATOR_H

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

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _CharT, class _Traits>
class _CCCL_TYPE_VISIBILITY_DEFAULT ostreambuf_iterator
{
public:
  using iterator_category = output_iterator_tag;
  using value_type        = void;
#if _CCCL_STD_VER > 2017
  using difference_type = ptrdiff_t;
#else
  using difference_type = void;
#endif
  using pointer        = void;
  using reference      = void;
  using char_type      = _CharT;
  using traits_type    = _Traits;
  using streambuf_type = basic_streambuf<_CharT, _Traits>;
  using ostream_type   = basic_ostream<_CharT, _Traits>;

private:
  streambuf_type* __sbuf_;

public:
  _CCCL_API ostreambuf_iterator(ostream_type& __s) noexcept
      : __sbuf_(__s.rdbuf())
  {}

  _CCCL_API ostreambuf_iterator(streambuf_type* __s) noexcept
      : __sbuf_(__s)
  {}

  _CCCL_API ostreambuf_iterator& operator=(_CharT __c)
  {
    if (__sbuf_ && traits_type::eq_int_type(__sbuf_->sputc(__c), traits_type::eof()))
    {
      __sbuf_ = nullptr;
    }
    return *this;
  }

  [[nodiscard]] _CCCL_API ostreambuf_iterator& operator*() noexcept
  {
    return *this;
  }
  _CCCL_API ostreambuf_iterator& operator++() noexcept
  {
    return *this;
  }
  _CCCL_API ostreambuf_iterator& operator++(int) noexcept
  {
    return *this;
  }
  [[nodiscard]] _CCCL_API bool failed() const noexcept
  {
    return __sbuf_ == nullptr;
  }

  template <class _Ch, class _Tr>
  _CCCL_API friend ostreambuf_iterator<_Ch, _Tr> __pad_and_output(
    ostreambuf_iterator<_Ch, _Tr> __s, const _Ch* __ob, const _Ch* __op, const _Ch* __oe, ios_base& __iob, _Ch __fl);
};
_CCCL_SUPPRESS_DEPRECATED_POP

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ITERATOR_OSTREAMBUF_ITERATOR_H
