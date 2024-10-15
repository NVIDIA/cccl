//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_TYPEID_H
#define _LIBCUDACXX___UTILITY_TYPEID_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// BUGBUG
#define _CCCL_NO_TYPEID

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include <cuda/std/compare>
#endif
#include <cuda/std/cstddef>

#ifndef _CCCL_NO_TYPEID
#  include <typeinfo>
#endif

#ifndef _CCCL_NO_TYPEID

_LIBCUDACXX_BEGIN_NAMESPACE_STD
#  define _CCCL_TYPEID(...) typeid(__VA_ARGS__)
using __type_info = ::std::type_info;
_LIBCUDACXX_END_NAMESPACE_STD

#else // ^^^ !_CCCL_NO_TYPEID ^^^ / vvv _CCCL_NO_TYPEID

// We use an unversioned namespace here so that the versioning namespace doesn't
// appear in the __PRETTY_FUNCTION__ strings used below.
_LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION

// TODO: replace this with `cuda::std::string_view` when available.
struct __string_view
{
  template <size_t N>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __string_view(
    char const (&str)[N], size_t prefix = 0, size_t suffix = 0) noexcept
      : __str_{str + prefix}
      , __len_{N - 1 - prefix - suffix}
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

  // C++11 constexpr string comparison
#  if _CCCL_STD_VER < 2014 || __cpp_constexpr < 201304L

private:
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr int
  __compare(char const* __s1, size_t __len1, char const* __s2, size_t __len2, size_t __n) noexcept
  {
    return __n
           ? ((*__s1 < *__s2) //
                ? -1
                : ((*__s2 < *__s1) //
                     ? 1
                     : __compare(__s1 + 1, __len1, __s2 + 1, __len2, __n - 1)))
           : int(__len1) - int(__len2);
  }

public:
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int compare(__string_view const& __other) const noexcept
  {
    return __compare(__str_, __len_, __other.__str_, __other.__len_, __min_(__len_, __other.__len_));
  }

#  else // ^^^ C++11 ^^^ / vvv C++14 and beyond vvv

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int compare(__string_view const& __other) const noexcept
  {
    size_t __n       = __min_(__len_, __other.__len_);
    char const *__s1 = __str_, *__s2 = __other.__str_;
    for (; __n; --__n, ++__s1, ++__s2)
    {
      if (*__s1 < *__s2)
      {
        return -1;
      }
      if (*__s2 < *__s1)
      {
        return 1;
      }
    }
    return int(__len_) - int(__other.__len_);
  }

#  endif

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator==(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.__len_ == __rhs.__len_ && __lhs.compare(__rhs) == 0;
  }

#  ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator<=>(__string_view const& __lhs, __string_view const& __rhs) noexcept
  {
    return __lhs.compare(__rhs) <=> 0;
  }

#  else // ^^^ !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR ^^^ / vvv _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

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

#  endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

private:
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t __min_(size_t __x, size_t __y) noexcept
  {
    return __x < __y ? __x : __y;
  }

  _LIBCUDACXX_HIDE_FROM_ABI __string_view(char const* __str, size_t __len) noexcept
      : __str_(__str)
      , __len_(__len)
  {}

  char const* __str_;
  size_t __len_;
};

// For the known compilers, we can extract the type name from the compiler's
// built-in macros. This is a best-effort approach, and may not work for all
// compilers or configurations.
#  if defined(_CCCL_COMPILER_MSVC)
#    define _CCCL_PRETTY_FUNCTION __FUNCSIG__
#  else
#    define _CCCL_PRETTY_FUNCTION __PRETTY_FUNCTION__
#  endif

// The name of this function (__PRETTY_NAMEOF) must have the same number of
// characters as __pretty_nameof, the function used below for extracting the
// type name from the pretty function name. Otherwise, the trim counts computed
// below will be incorrect.
template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view __PRETTY_NAMEOF()
{
  return __string_view(_CCCL_PRETTY_FUNCTION);
}

struct __pretty_trim_counts
{
  size_t __front;
  size_t __back;
};

// These are the known pretty names for __PRETTY_NAMEOF<int>() for various compilers.
_CCCL_GLOBAL_CONSTANT __string_view __pretty_names[] = {
  __string_view{"constexpr cuda::std::__string_view cuda::std::__PRETTY_NAMEOF() [with _Tp = int]"},
  __string_view{"cuda::std::__string_view cuda::std::__PRETTY_NAMEOF() [with _Tp = int]"},
  __string_view{"constexpr cuda::std::__string_view cuda::std::__PRETTY_NAMEOF() [_Tp = int]"},
  __string_view{"cuda::std::__string_view cuda::std::__PRETTY_NAMEOF() [_Tp = int]"},
  __string_view{"constexpr __string_view cuda::std::__PRETTY_NAMEOF() [with _Tp = int]"},
  __string_view{"__string_view cuda::std::__PRETTY_NAMEOF() [with _Tp = int]"},
  __string_view{"constexpr __string_view cuda::std::__PRETTY_NAMEOF() [_Tp = int]"},
  __string_view{"__string_view cuda::std::__PRETTY_NAMEOF() [_Tp = int]"},
  __string_view{"cuda::std::__string_view cuda::std::__PRETTY_NAMEOF<int>()"},
  __string_view{"struct cuda::std::__string_view __cdecl cuda::std::__PRETTY_NAMEOF<int>(void)"},
};

// Break the pretty names into front and back parts to trim to get to the type name.
_CCCL_GLOBAL_CONSTANT __pretty_trim_counts __pretty_trim[] = {
  {sizeof("constexpr cuda::std::__string_view cuda::std::__PRETTY_NAMEOF() [with _Tp = ") - 1, sizeof("]") - 1}, //
  {sizeof("cuda::std::__string_view cuda::std::__PRETTY_NAMEOF() [with _Tp = ") - 1, sizeof("]") - 1}, //
  {sizeof("constexpr cuda::std::__string_view cuda::std::__PRETTY_NAMEOF() [_Tp = ") - 1, sizeof("]") - 1}, //
  {sizeof("cuda::std::__string_view cuda::std::__PRETTY_NAMEOF() [_Tp = ") - 1, sizeof("]") - 1}, //
  {sizeof("constexpr __string_view cuda::std::__PRETTY_NAMEOF() [with _Tp = ") - 1, sizeof("]") - 1}, //
  {sizeof("__string_view cuda::std::__PRETTY_NAMEOF() [with _Tp = ") - 1, sizeof("]") - 1}, //
  {sizeof("constexpr __string_view cuda::std::__PRETTY_NAMEOF() [_Tp = ") - 1, sizeof("]") - 1}, //
  {sizeof("__string_view cuda::std::__PRETTY_NAMEOF() [_Tp = ") - 1, sizeof("]") - 1}, //
  {sizeof("cuda::std::__string_view cuda::std::__PRETTY_NAMEOF<") - 1, sizeof(">()") - 1}, //
  {sizeof("struct cuda::std::__string_view __cdecl cuda::std::__PRETTY_NAMEOF<") - 1, sizeof(">(void)") - 1}, //
};

// Find the index of the pretty name for __PRETTY_NAMEOF<int>() in the known list.
constexpr size_t __pretty_index =
  __PRETTY_NAMEOF<int>() == __pretty_names[0]   ? 0
  : __PRETTY_NAMEOF<int>() == __pretty_names[1] ? 1
  : __PRETTY_NAMEOF<int>() == __pretty_names[2] ? 2
  : __PRETTY_NAMEOF<int>() == __pretty_names[3] ? 3
  : __PRETTY_NAMEOF<int>() == __pretty_names[4] ? 4
  : __PRETTY_NAMEOF<int>() == __pretty_names[5] ? 5
  : __PRETTY_NAMEOF<int>() == __pretty_names[6] ? 6
  : __PRETTY_NAMEOF<int>() == __pretty_names[7] ? 7
  : __PRETTY_NAMEOF<int>() == __pretty_names[8] ? 8
  : __PRETTY_NAMEOF<int>() == __pretty_names[9]
    ? 9
    : ~size_t(0);

// Assert that we found a match.
static_assert(__pretty_index != ~size_t(0), "Unrecognized __PRETTY_FUNCTION__ string format.");

// Get the type name from the pretty name by trimming the front and back.
template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view __pretty_nameof()
{
  return __string_view(_CCCL_PRETTY_FUNCTION, //
                       __pretty_trim[__pretty_index].__front,
                       __pretty_trim[__pretty_index].__back);
}

// A quick smoke test to ensure that the pretty name extraction is working.
static_assert(__pretty_nameof<int>() == __string_view("int"), "__pretty_nameof<int>() == __string_view(\"int\")");
static_assert(__pretty_nameof<float>() < __pretty_nameof<int>(), "__pretty_nameof<float>() < __pretty_nameof<int>()");

_LIBCUDACXX_END_NAMESPACE_STD_NOVERSION

_LIBCUDACXX_BEGIN_NAMESPACE_STD

/// @brief A minimal implementation of `std::type_info` for platforms that do
/// not support RTTI.
struct __type_info
{
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __type_info(__string_view __name) noexcept
      : __name_(__name)
  {}

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr char const* name() const noexcept
  {
    return __name_.begin();
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool before(const __type_info& __other) const noexcept
  {
    return __name_ < __other.__name_;
  }

  // Not yet implemented:
  // _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr size_t hash_code() const noexcept
  // {
  //   return ;
  // }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator==(const __type_info& __lhs, const __type_info& __rhs) noexcept
  {
    return &__lhs == &__rhs || __lhs.__name_ == __rhs.__name_;
  }

#  if _CCCL_STD_VER <= 2017
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator!=(const __type_info& __lhs, const __type_info& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#  endif

private:
  __string_view __name_;
};

#  if !defined(_CCCL_NO_INLINE_VARIABLES)

template <class _Tp>
_CCCL_INLINE_VAR _CCCL_CONSTEXPR_GLOBAL auto __typeid = __type_info(__pretty_nameof<_Tp>());

#    define _CCCL_TYPEID(...) _CUDA_VSTD::__typeid<__VA_ARGS__>

#  else

template <class _Tp>
struct __typeid_value
{
  static constexpr __type_info value = __type_info(__pretty_nameof<_Tp>());
};

// Before the addition of inline variables, it was necessary to
// provide a definition for constexpr class static data members.
template <class _Tp>
constexpr __type_info __typeid_value<_Tp>::value;

#    ifndef _CCCL_NO_VARIABLE_TEMPLATES

template <class _Tp>
_CCCL_CONSTEXPR_GLOBAL __type_info& __typeid = __typeid_value<_Tp>::value;

#      define _CCCL_TYPEID(...) _CUDA_VSTD::__typeid<__VA_ARGS__>

#    else // !_CCCL_NO_VARIABLE_TEMPLATES

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __type_info const& __typeid() noexcept
{
  return __typeid_value<_Tp>::value;
}

#      define _CCCL_TYPEID(...) _CUDA_VSTD::__typeid<__VA_ARGS__>()

#    endif // !_CCCL_NO_VARIABLE_TEMPLATES

#  endif // !_CCCL_NO_INLINE_VARIABLES

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _CCCL_NO_TYPEID

#endif // _LIBCUDACXX___UTILITY_TYPEID_H
