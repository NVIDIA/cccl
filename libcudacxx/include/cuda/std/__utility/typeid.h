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

// This file provides a minimal implementation of `std::type_info` for platforms
// that do not support RTTI. It defines the following:
//
// `cuda::std::type_info`   - An alias for `std::type_info` if RTTI is available, or
//                            an alias for `cuda::std::__type_info` if RTTI is not
//                            available.
// `cuda::std::__type_info` - A class type that provides the `std::type_info` interface
//                            where the member functions are `constexpr`.
// `cuda::std::__type_info_ptr`
//                          - The result of taking the address of a
//                            `cuda::std::__type_info` object.
// `_CCCL_TYPEID(<type>)`   - A macro that returns a reference to the `const`
//                            `cuda::std::type_info` object for the specified type.
// `_CCCL_CONSTEXPR_TYPEID(<type>)`
//                          - A macro that returns a reference to the `const`
//                            `cuda::std::__type_info` object for the specified type.

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include <cuda/std/compare>
#endif
#include <cuda/std/__string/string_view.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/cstddef>

#ifndef _CCCL_NO_TYPEID
#  include <typeinfo>
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __type_info;

#ifndef _CCCL_NO_TYPEID

#  define _CCCL_TYPEID(...) typeid(_CUDA_VSTD::_CCCL_TYPEID_only_supports_types<__VA_ARGS__>)
using type_info = ::std::type_info;

#else // ^^^ !_CCCL_NO_TYPEID ^^^ / vvv _CCCL_NO_TYPEID

#  define _CCCL_TYPEID _CCCL_TYPEID_FALLBACK
using type_info = _CUDA_VSTD::__type_info;

#endif // _CCCL_NO_TYPEID

// We find a type _Tp's name as follows:
// 1. Use __PRETTY_FUNCTION__ in a function template parameterized by
//    __pretty_name_begin<_Tp>::__pretty_name_end.
// 2. Find the substrings "__pretty_name_begin<" and ">::__pretty_name_end".
//    Everything between them is the name of type _Tp.

template <class _Tp>
using _CCCL_TYPEID_only_supports_types = _Tp;

// Earlier versions of gcc (before gcc-9) do not treat __PRETTY_FUNCTION__ as a
// constexpr value after a reference to it has been returned from a function.
// Instead, arrange things so that the pretty name gets stored in a class static
// data member, where it can be referenced from other constexpr contexts.

#if defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION < 90000

template <size_t _Np>
struct __sstring
{
  char __str_[_Np];
  size_t __len_;
};

template <class _Tp, size_t _Np>
struct __static_nameof;

template <size_t _Np, size_t _Mp, size_t... _Is>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __sstring<_Np>
__make_pretty_name_impl(char const (&__s)[_Mp], index_sequence<_Is...>) noexcept
{
  static_assert(_Mp <= _Np, "Type name too long for __pretty_nameof");
  return __sstring<_Np>{{(_Is < _Mp ? __s[_Is] : '\0')...}, _Mp - 1};
}

template <class _Tp, size_t _Np>
_LIBCUDACXX_HIDE_FROM_ABI constexpr auto __make_pretty_name(integral_constant<size_t, _Np>) noexcept //
  -> __enable_if_t<_Np == size_t(-1), __string_view>
{
  using _TpName = __static_nameof<_Tp, sizeof(_CCCL_PRETTY_FUNCTION)>;
  return __string_view(_TpName::value.__str_, 0, sizeof(_CCCL_PRETTY_FUNCTION) - _TpName::value.__len_ - 1);
}

template <class _Tp, size_t _Np>
_LIBCUDACXX_HIDE_FROM_ABI constexpr auto __make_pretty_name(integral_constant<size_t, _Np>) noexcept //
  -> __enable_if_t<_Np != size_t(-1), __sstring<_Np>>
{
  return _CUDA_VSTD::__make_pretty_name_impl<_Np>(_CCCL_PRETTY_FUNCTION, make_index_sequence<_Np>{});
}

// TODO: class statics cannot be accessed from device code, so we need to use
// a variable template when that is available.
template <class _Tp, size_t _Np>
struct __static_nameof
{
  static constexpr __sstring<_Np> value = _CUDA_VSTD::__make_pretty_name<_Tp>(integral_constant<size_t, _Np>());
};

#  if defined(_CCCL_NO_INLINE_VARIABLES)
template <class _Tp, size_t _Np>
constexpr __sstring<_Np> __static_nameof<_Tp, _Np>::value;
#  endif // _CCCL_NO_INLINE_VARIABLES

#endif // _CCCL_GCC_VERSION < 90000

template <class _Tp>
struct __pretty_name_begin
{
  struct __pretty_name_end;
};

// If a position is -1, it is an invalid position. Return it unchanged.
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr ptrdiff_t
__add_string_view_position(ptrdiff_t __pos, ptrdiff_t __diff) noexcept
{
  return __pos == -1 ? -1 : __pos + __diff;
}

// Get the type name from the pretty name by trimming the front and back.
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view __pretty_nameof_3(__string_view __sv) noexcept
{
  return __sv.substr(
    __add_string_view_position(__sv.find("__pretty_name_begin<"), ptrdiff_t(sizeof("__pretty_name_begin<")) - 1),
    __sv.find_end(">::__pretty_name_end"));
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view __pretty_nameof_2() noexcept
{
#if defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION < 90000 && !defined(__CUDA_ARCH__)
  return _CUDA_VSTD::__pretty_nameof_3(_CUDA_VSTD::__make_pretty_name<_Tp>(integral_constant<size_t, size_t(-1)>{}));
#else // ^^^ gcc < 9 ^^^^/ vvv other compiler vvv
  return _CUDA_VSTD::__pretty_nameof_3(_CUDA_VSTD::__string_view(_CCCL_PRETTY_FUNCTION));
#endif // not gcc < 9
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view __pretty_nameof() noexcept
{
  return _CUDA_VSTD::__pretty_nameof_2<typename __pretty_name_begin<_Tp>::__pretty_name_end>();
}

// BUGBUG
#ifdef _CCCL_COMPILER_MSVC
template <char... _Chs>
struct __string_literal
{
  static constexpr char __str[sizeof...(_Chs) + 1] = {_Chs..., '\0'};
  [[deprecated]]
  static constexpr __string_view __get() noexcept
  {
    return __string_view(__str);
  }
};
template <class _Tp, size_t... _Is>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto __msvc_test(index_sequence<_Is...>) noexcept
{
  return __string_literal<(_Is < sizeof(_CCCL_PRETTY_FUNCTION) ? _CCCL_PRETTY_FUNCTION[_Is] : '\0')...>{};
}
using __pretty_name_int = typename __pretty_name_begin<int>::__pretty_name_end;
static_assert(-1 != _CUDA_VSTD::__msvc_test<__pretty_name_int>(make_index_sequence<100>()).__get().find("pretty"), "");
#endif

// In device code with old versions of gcc, we cannot have nice things.
#if defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION < 90000 && defined(__CUDA_ARCH__)
#  define _CCCL_NO_CONSTEXPR_PRETTY_NAMEOF
#endif

#ifndef _CCCL_NO_CONSTEXPR_PRETTY_NAMEOF
// A quick smoke test to ensure that the pretty name extraction is working.
static_assert(_CUDA_VSTD::__pretty_nameof<int>() == __string_view("int"), "");
static_assert(_CUDA_VSTD::__pretty_nameof<float>() < _CUDA_VSTD::__pretty_nameof<int>(), "");
#endif

// The preferred implementation of `__type_info` only works in device code when
// both variable templates are supported. Then, we can define a `__typeid<T>`
// variable template at namespace scope, an inline function to return it by
// reference, and have the linker select one of the inline definitions out of
// all the translation units that use it to yield a unique address.

#if defined(__CUDA_ARCH__) && defined(_CCCL_NO_VARIABLE_TEMPLATES)

struct __type_info_impl
{
  __string_view __name_;
};

struct __type_info_ptr
{
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __type_info operator*() const noexcept;

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator==(__type_info_ptr __a, __type_info_ptr __b) noexcept
  {
    return __a.__pfn_ == __b.__pfn_;
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator!=(__type_info_ptr __a, __type_info_ptr __b) noexcept
  {
    return !(__a == __b);
  }

  __type_info_impl (*__pfn_)() noexcept;
};

/// @brief A minimal implementation of `std::type_info` for device code that does
/// not depend on RTTI or variable templates.
struct __type_info
{
  __type_info() = delete;

  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __type_info(__type_info_impl (*__pfn)() noexcept) noexcept
      : __pfn_(__pfn)
  {}

  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __type_info_impl __get_ti_for() noexcept
  {
    return __type_info_impl{_CUDA_VSTD::__pretty_nameof<_Tp>()};
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr char const* name() const noexcept
  {
    return __pfn_().__name_.begin();
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view __name_view() const noexcept
  {
    return __pfn_().__name_;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool before(__type_info const& __other) const noexcept
  {
    return __pfn_().__name_ < __other.__pfn_().__name_;
  }

  // Not yet implemented:
  // _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr size_t hash_code() const noexcept
  // {
  //   return ;
  // }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __type_info_ptr operator&() const noexcept
  {
    return __type_info_ptr{__pfn_};
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator==(__type_info const& __a, __type_info const& __b) noexcept
  {
    return __a.__pfn_ == __b.__pfn_ || __a.__pfn_().__name_ == __b.__pfn_().__name_;
  }

  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  operator!=(__type_info const& __a, __type_info const& __b) noexcept
  {
    return !(__a == __b);
  }

private:
  friend struct __type_info_ptr;

  __type_info(__type_info const&)            = default; // needed by __type_info_ptr::operator*() before C++17
  __type_info& operator=(__type_info const&) = delete;

  __type_info_impl (*__pfn_)() noexcept;
};

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __type_info __type_info_ptr::operator*() const noexcept
{
  return __type_info(__pfn_);
}

using __type_info_ref = __type_info;

#  define _CCCL_TYPEID_FALLBACK(...) \
    _CUDA_VSTD::__type_info(&_CUDA_VSTD::__type_info::__get_ti_for<_CUDA_VSTD::__remove_cv_t<__VA_ARGS__>>)

#else // ^^^ defined(__CUDA_ARCH__) && defined(_CCCL_NO_VARIABLE_TEMPLATES) ^^^
      // vvv !defined(__CUDA_ARCH__) ||!defined(_CCCL_NO_VARIABLE_TEMPLATES) vvv

/// @brief A minimal implementation of `std::type_info` for platforms that do
/// not support RTTI.
struct __type_info
{
  __type_info()                              = delete;
  __type_info(__type_info const&)            = delete;
  __type_info& operator=(__type_info const&) = delete;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __type_info(__string_view __name) noexcept
      : __name_(__name)
  {}

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr char const* name() const noexcept
  {
    return __name_.begin();
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __string_view __name_view() const noexcept
  {
    return __name_;
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
#  endif // _CCCL_STD_VER <= 2017

private:
  __string_view __name_;
};

using __type_info_ptr = __type_info const*;
using __type_info_ref = __type_info const&;

#  ifndef _CCCL_NO_VARIABLE_TEMPLATES

template <class _Tp>
_CCCL_GLOBAL_CONSTANT __type_info __typeid_v{_CUDA_VSTD::__pretty_nameof<_Tp>()};

// When inline variables are available, this indirection through an inline function
// is not necessary, but it doesn't hurt either.
template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __type_info const& __typeid() noexcept
{
  return __typeid_v<_Tp>;
}

#    define _CCCL_TYPEID_FALLBACK(...) _CUDA_VSTD::__typeid<_CUDA_VSTD::__remove_cv_t<__VA_ARGS__>>()

#  else // ^^^ !_CCCL_NO_VARIABLE_TEMPLATES ^^^ / vvv _CCCL_NO_VARIABLE_TEMPLATES vvv

template <class _Tp>
struct __typeid_value
{
  static constexpr __type_info value{_CUDA_VSTD::__pretty_nameof<_Tp>()};
};

#    if defined(_CCCL_NO_INLINE_VARIABLES) || _CCCL_STD_VER < 2017
// Before the addition of inline variables, it was necessary to
// provide a definition for constexpr class static data members.
template <class _Tp>
constexpr __type_info __typeid_value<_Tp>::value;
#    endif // _CCCL_NO_INLINE_VARIABLES

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __type_info const& __typeid() noexcept
{
  return __typeid_value<_Tp>::value;
}

#    define _CCCL_TYPEID_FALLBACK(...) _CUDA_VSTD::__typeid<_CUDA_VSTD::__remove_cv_t<__VA_ARGS__>>()

#  endif // _CCCL_NO_VARIABLE_TEMPLATES

#endif // !defined(__CUDA_ARCH__) ||!defined(_CCCL_NO_VARIABLE_TEMPLATES)

// if `__pretty_nameof` is constexpr _CCCL_TYPEID_FALLBACK is also constexpr.
#ifndef _CCCL_NO_CONSTEXPR_PRETTY_NAMEOF
#  define _CCCL_TYPEOF_CONSTEXPR _CCCL_TYPEOF_FALLBACK
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_TYPEID_H