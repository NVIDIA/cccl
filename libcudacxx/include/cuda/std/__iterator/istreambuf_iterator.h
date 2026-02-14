// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ITERATOR_ISTREAMBUF_ITERATOR_H
#define _CUDA_STD___ITERATOR_ISTREAMBUF_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/ios.h>
#include <cuda/std/__iterator/default_sentinel.h>
#include <cuda/std/__iterator/iterator.h>
#include <cuda/std/__iterator/iterator_traits.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _CharT, class _Traits>
class _CCCL_TYPE_VISIBILITY_DEFAULT istreambuf_iterator
{
public:
  using iterator_category = input_iterator_tag;
  using value_type        = _CharT;
  using difference_type   = typename _Traits::off_type;
  using pointer           = _CharT*;
  using reference         = _CharT;
  using char_type         = _CharT;
  using traits_type       = _Traits;
  using int_type          = typename _Traits::int_type;
  using streambuf_type    = basic_streambuf<_CharT, _Traits>;
  using istream_type      = basic_istream<_CharT, _Traits>;

private:
  mutable streambuf_type* __sbuf_;

  class __proxy
  {
    char_type __keep_;
    streambuf_type* __sbuf_;
    _CCCL_API explicit __proxy(char_type __c, streambuf_type* __s)
        : __keep_(__c)
        , __sbuf_(__s)
    {}
    friend class istreambuf_iterator;

  public:
    [[nodiscard]] _CCCL_API char_type operator*() const
    {
      return __keep_;
    }
  };

  [[nodiscard]] _CCCL_API bool __test_for_eof() const
  {
    if (__sbuf_ && traits_type::eq_int_type(__sbuf_->sgetc(), traits_type::eof()))
    {
      __sbuf_ = nullptr;
    }
    return __sbuf_ == nullptr;
  }

public:
  _CCCL_API constexpr istreambuf_iterator() noexcept
      : __sbuf_(nullptr)
  {}

  _CCCL_API constexpr istreambuf_iterator(default_sentinel_t) noexcept
      : istreambuf_iterator()
  {}

  _CCCL_API istreambuf_iterator(istream_type& __s) noexcept
      : __sbuf_(__s.rdbuf())
  {}

  _CCCL_API istreambuf_iterator(streambuf_type* __s) noexcept
      : __sbuf_(__s)
  {}

  _CCCL_API istreambuf_iterator(const __proxy& __p) noexcept
      : __sbuf_(__p.__sbuf_)
  {}

  [[nodiscard]] _CCCL_API char_type operator*() const
  {
    return static_cast<char_type>(__sbuf_->sgetc());
  }

  _CCCL_API istreambuf_iterator& operator++()
  {
    __sbuf_->sbumpc();
    return *this;
  }

  _CCCL_API __proxy operator++(int)
  {
    return __proxy(__sbuf_->sbumpc(), __sbuf_);
  }

  [[nodiscard]] _CCCL_API bool equal(const istreambuf_iterator& __b) const
  {
    return __test_for_eof() == __b.__test_for_eof();
  }

  [[nodiscard]] _CCCL_API friend bool operator==(const istreambuf_iterator& __i, default_sentinel_t)
  {
    return __i.__test_for_eof();
  }
#if _CCCL_STD_VER < 2020
  [[nodiscard]] _CCCL_API friend bool operator==(default_sentinel_t, const istreambuf_iterator& __i)
  {
    return __i.__test_for_eof();
  }
  [[nodiscard]] _CCCL_API friend bool operator!=(const istreambuf_iterator& __i, default_sentinel_t)
  {
    return !__i.__test_for_eof();
  }
  [[nodiscard]] _CCCL_API friend bool operator!=(default_sentinel_t, const istreambuf_iterator& __i)
  {
    return !__i.__test_for_eof();
  }
#endif // _CCCL_STD_VER < 2020
};
_CCCL_SUPPRESS_DEPRECATED_POP

template <class _CharT, class _Traits>
[[nodiscard]] _CCCL_API bool
operator==(const istreambuf_iterator<_CharT, _Traits>& __a, const istreambuf_iterator<_CharT, _Traits>& __b)
{
  return __a.equal(__b);
}

#if _CCCL_STD_VER <= 2017
template <class _CharT, class _Traits>
[[nodiscard]] _CCCL_API bool
operator!=(const istreambuf_iterator<_CharT, _Traits>& __a, const istreambuf_iterator<_CharT, _Traits>& __b)
{
  return !__a.equal(__b);
}
#endif // _CCCL_STD_VER <= 2017

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ITERATOR_ISTREAMBUF_ITERATOR_H
