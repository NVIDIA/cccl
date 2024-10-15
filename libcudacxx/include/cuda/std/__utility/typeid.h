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

#ifndef _CCCL_NO_TYPEID
#  include <typeinfo>
#endif

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include <cuda/std/compare>
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifndef _CCCL_NO_TYPEID

#  define _CCCL_TYPEID(...) typeid(__VA_ARGS__)
using __type_info = ::std::type_info;

#else // ^^^ !_CCCL_NO_TYPEID ^^^ / vvv _CCCL_NO_TYPEID

// TODO: replace this with `cuda::std::string_view` when available.
struct __string_view
{
  template <size_t N>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __string_view(
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

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int compare(__string_view const& __other) const noexcept
  {
    size_t __n       = __len_ < __other.__len_ ? __len_ : __other.__len_;
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
  char const* __str_;
  size_t __len_;
};

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view __pretty_nameof()
{
#  if defined(_CCCL_COMPILER_MSVC)
  constexpr size_t __prefix =
    sizeof("struct cuda::std::__4::__string_view __cdecl cuda::std::__4::__pretty_nameof<") - 1;
  constexpr size_t __suffix = sizeof(">(void)") - 1;
  return __string_view(__FUNCSIG__, __prefix, __suffix);
#  elif defined(_CCCL_COMPILER_CLANG)
  constexpr size_t __nvcc_prefix =
    sizeof("cuda::std::__4::__string_view cuda::std::__4::__pretty_nameof() [with _Tp = ") - 1;
  constexpr size_t __clang_prefix = sizeof("__string_view cuda::std::__pretty_nameof() [_Tp = ") - 1;
  constexpr size_t __prefix       = __PRETTY_FUNCTION__[0] == 'c' ? __nvcc_prefix : __clang_prefix;
  constexpr size_t __suffix       = sizeof("]") - 1;
  return __string_view(__PRETTY_FUNCTION__, __prefix, __suffix);
#  elif defined(_CCCL_CUDA_COMPILER_NVCC) || defined(_CCCL_CUDA_COMPILER_NVHPC) || defined(_CCCL_COMPILER_NVRTC) \
    || defined(_CCCL_COMPILER_ICC) || defined(_CCCL_COMPILER_GCC)
  constexpr size_t __prefix =
    sizeof("constexpr cuda::std::__4::__string_view cuda::std::__4::__pretty_nameof() [with _Tp = ") - 1;
  constexpr size_t __suffix = sizeof("]") - 1;
  return __string_view(__PRETTY_FUNCTION__, __prefix, __suffix);
#  else
#    error "Unsupported compiler"
#  endif
}

static_assert(__pretty_nameof<int>() == __string_view("int"), "__pretty_nameof<int>() == __string_view(\"int\")");
static_assert(__pretty_nameof<float>() < __pretty_nameof<int>(), "__pretty_nameof<float>() < __pretty_nameof<int>()");

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
  static constexpr __type_info const value = __type_info(__pretty_nameof<_Tp>());
};

// Before the addition of inline variables, it was necessary to
// provide a definition for constexpr class static data members.
template <class _Ts>
constexpr __type_info const __typeid_value<_Ty>::value;

#    ifndef _CCCL_NO_VARIABLE_TEMPLATES

template <class _Tp>
_CCCL_CONSTEXPR_GLOBAL __type_info const& __typeid = __typeid_value<_Tp>::value;

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

#endif // _CCCL_NO_TYPEID

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_TYPEID_H
