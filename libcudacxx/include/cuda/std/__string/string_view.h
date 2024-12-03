//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___STRING_STRING_VIEW_H
#define _LIBCUDACXX___STRING_STRING_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include <cuda/std/compare>
#endif
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/cstddef>
#include <cuda/std/detail/libcxx/include/__string>
#include <cuda/std/detail/libcxx/include/stdexcept>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __string_view
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view(char const* __str, size_t __len) noexcept
      : __str_(__str)
      , __len_(__len)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __string_view(char const* __str) noexcept
      : __str_(__str)
      , __len_(__strlen_(__str))
  {}

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr size_t size() const noexcept
  {
    return __len_;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr char const* data() const noexcept
  {
    return __str_;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr char const* begin() const noexcept
  {
    return __str_;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr char const* end() const noexcept
  {
    return __str_ + __len_;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr char const& operator[](ptrdiff_t __n) const noexcept
  {
    return __str_[__n];
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view substr(ptrdiff_t __start, ptrdiff_t __stop) const
  {
    return __string_view(__str_ + __check_offset(__start, __len_), __check_offset(__stop - __start, __len_));
  }

  // C++11 constexpr string comparison
#if _CCCL_STD_VER < 2014 || __cpp_constexpr < 201304L

private:
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr int
  __compare_(char const* __s1, size_t __len1, char const* __s2, size_t __len2, size_t __n) noexcept
  {
    return __n ? (*__s1 - *__s2 ? *__s1 - *__s2 : __compare_(__s1 + 1, __len1, __s2 + 1, __len2, __n - 1))
               : int(__len1) - int(__len2);
  }

  template <bool _Forward>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t __try_find(
    size_t __i,
    const char* __needle,
    size_t __needle_size,
    char const* __haystack_begin,
    char const* __haystack_end,
    char const* __it) noexcept
  {
    return __i == __needle_size //
           ? (_Forward ? __it - __haystack_begin : __it - __haystack_end - 1)
           : __needle[__i] != (_Forward ? __it : __it - 1)[__i]
               ? -1
               : __try_find<_Forward>(__i + 1, __needle, __needle_size, __haystack_begin, __haystack_end, __it);
  }

  template <bool _Forward>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t __find(
    ptrdiff_t __found,
    const char* __needle,
    size_t __needle_size,
    char const* __haystack_begin,
    char const* __haystack_end,
    char const* __it) noexcept
  {
    return __found != -1 //
           ? __found
           : __it == __haystack_end
               ? -1
               : __find<_Forward>(
                   __try_find<_Forward>(0, __needle, __needle_size, __haystack_begin, __haystack_end, __it),
                   __needle,
                   __needle_size,
                   __haystack_begin,
                   __haystack_end,
                   (_Forward ? __it + 1 : __it - 1));
  }

public:
  template <size_t _Np>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr ptrdiff_t find(const char (&__other)[_Np]) const noexcept
  {
    return (_Np - 1) > __len_
           ? -1
           : __find<true>(-1, __other, _Np - 1, __str_, __str_ + __len_ - (_Np - 1) + 1, __str_);
  }

  template <size_t _Np>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr ptrdiff_t find_end(const char (&__other)[_Np]) const noexcept
  {
    return (_Np - 1) > __len_
           ? -1
           : __find<false>(
               -1, __other, _Np - 1, __str_ + __len_ - (_Np - 1) + 1, __str_, __str_ + __len_ - (_Np - 1) + 1);
  }

#else // ^^^ C++11 ^^^ / vvv C++14 and beyond vvv

private:
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr int
  __compare_(char const* __s1, size_t __len1, char const* __s2, size_t __len2, size_t __n) noexcept
  {
    if (__n)
    {
      for (;; ++__s1, ++__s2)
      {
        if (*__s1 < *__s2)
        {
          return -1;
        }
        if (*__s2 < *__s1)
        {
          return 1;
        }
        if (0 == --__n)
        {
          break;
        }
      }
    }
    return int(__len1) - int(__len2);
  }

  template <bool _Forward>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t
  __find(const char* __needle, size_t __needle_size, const char* __haystack_begin, const char* __haystack_end) noexcept
  {
    char const* __it = __haystack_begin;
    for (; __it != __haystack_end; (_Forward ? ++__it : --__it))
    {
      size_t __i = 0;
      for (; __i != __needle_size; ++__i)
      {
        if ((_Forward ? __it : __it - 1)[__i] != __needle[__i])
        {
          break;
        }
      }
      if (__i == __needle_size)
      {
        return _Forward ? __it - __haystack_begin : __it - __haystack_end - 1;
      }
    }
    return -1;
  }

public:
  template <size_t _Np>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr ptrdiff_t find(const char (&__other)[_Np]) const noexcept
  {
    return ((_Np - 1) > __len_) ? -1 : __find<true>(__other, _Np - 1, __str_, __str_ + __len_ - (_Np - 1) + 1);
  }

  template <size_t _Np>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr ptrdiff_t find_end(const char (&__other)[_Np]) const noexcept
  {
    return ((_Np - 1) > __len_) ? -1 : __find<false>(__other, _Np - 1, __str_ + __len_ - (_Np - 1) + 1, __str_);
  }

#endif // ^^^ C++14 and beyond ^^^

private:
  // This overload is selected when we're not in a constant evaluated context.
  // Compare the two strings' addresses as a shortcut, and fall back to a string
  // comparison it they are not equal.
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int __compare(__string_view const& __other, false_type) const noexcept
  {
    return __str_ == __other.__str_
           ? int(__len_) - int(__other.__len_)
           : __compare_(__str_, __len_, __other.__str_, __other.__len_, (__min_) (__len_, __other.__len_));
  }

  // This overload is selected when we're in a constant evaluated context. We
  // cannot compare the two strings' addresses so fall back to a string
  // comparison.
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int
  __compare(__string_view const& __other, true_type) const noexcept
  {
    return __compare_(__str_, __len_, __other.__str_, __other.__len_, (__min_) (__len_, __other.__len_));
  }

public:
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int compare(__string_view const& __other) const noexcept
  {
    // If we're in a constant evaluated context, we cannot compare the __str_
    // members for equality.
    return __compare(__other, bool_constant<__cccl_default_is_constant_evaluated()>());
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator==(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.__len_ == __rhs.__len_ && __lhs.compare(__rhs) == 0;
  }

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator<=>(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.compare(__rhs) <=> 0;
  }

#else // ^^^ !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR ^^^ / vvv _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator!=(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator<(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.compare(__rhs) < 0;
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator<=(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.compare(__rhs) <= 0;
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator>(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.compare(__rhs) > 0;
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator>=(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.compare(__rhs) >= 0;
  }

#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

private:
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t __min_(size_t __x, size_t __y) noexcept
  {
    return __x < __y ? __x : __y;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t
  __strlen_0x_(char const* __str, size_t __len) noexcept
  {
    return *__str ? __strlen_0x_(__str + 1, __len + 1) : __len;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t __strlen_(char const* __str) noexcept
  {
#if _CCCL_STD_VER >= 2014
    return _CUDA_VSTD::char_traits<char>::length(__str);
#else
    return __strlen_0x_(__str, 0);
#endif
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t __check_offset(ptrdiff_t __diff, size_t __len)
  {
    return __diff < 0 || static_cast<size_t>(__diff) > __len
           ? (_CUDA_VSTD::__throw_out_of_range("__string_view index out of range"), size_t(0))
           : static_cast<size_t>(__diff);
  }

  char const* __str_;
  size_t __len_;
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___STRING_STRING_VIEW_H
