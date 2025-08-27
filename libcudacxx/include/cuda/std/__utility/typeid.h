//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_TYPEID_H
#define _CUDA_STD___UTILITY_TYPEID_H

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

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/compare>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/std/__cccl/preprocessor.h>
#include <cuda/std/__string/string_view.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/cstddef>

#if !defined(_CCCL_NO_TYPEID)
#  include <typeinfo>
#endif

// _CCCL_MSVC_BROKEN_FUNCSIG:
//
// When using MSVC as the host compiler, there is a bug in the EDG front-end of
// cudafe++ that causes the __FUNCSIG__ predefined macro to be expanded too
// early in the compilation process. The result is that within a function
// template the __FUNCSIG__ string does not mention the template arguments. This
// makes it impossible to extract the pretty name of a type from __FUNCSIG__,
// which in turn makes it impossible to implement a constexpr replacement for
// typeid. On MSVC v19.35 and higher, the __builtin_FUNCSIG() intrinsic is
// available and can be used in place of __FUNCSIG__, resolving the issue. For
// older versions of MSVC, we fall back to using the built-in typeid feature,
// which is always available on MSVC, even when RTTI is disabled.

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#define _CCCL_STD_TYPEID(...) typeid(::cuda::std::_CCCL_TYPEID_ONLY_SUPPORTS_TYPES<__VA_ARGS__>)

#if !defined(_CCCL_NO_TYPEID) && !defined(_CCCL_USE_TYPEID_FALLBACK)

#  define _CCCL_TYPEID _CCCL_STD_TYPEID
using type_info        = ::std::type_info;
using __type_info_ptr  = type_info const*;
using __type_info_ref  = type_info const&;
using __type_info_ref_ = type_info const&;

#else // ^^^ !_CCCL_NO_TYPEID ^^^ / vvv _CCCL_NO_TYPEID vvv

#  define _CCCL_TYPEID _CCCL_TYPEID_FALLBACK

#endif // _CCCL_NO_TYPEID

// We find a type _Tp's name as follows:
// 1. Use __PRETTY_FUNCTION__ in a function template parameterized by
//    __pretty_name_begin<_Tp>::__pretty_name_end.
// 2. Find the substrings "__pretty_name_begin<" and ">::__pretty_name_end".
//    Everything between them is the name of type _Tp.

template <class _Tp>
using _CCCL_TYPEID_ONLY_SUPPORTS_TYPES = _Tp;

// Earlier versions of gcc (before gcc-9) do not treat __PRETTY_FUNCTION__ as a
// constexpr value after a reference to it has been returned from a function.
// Instead, arrange things so that the pretty name gets stored in a class static
// data member, where it can be referenced from other constexpr contexts.

#if _CCCL_COMPILER(GCC, <, 9)

template <size_t _Np>
struct __sstring
{
  char __str_[_Np];
  size_t __len_;
};

template <class _Tp, size_t _Np>
struct __static_nameof;

template <size_t _Np, size_t _Mp, size_t... _Is>
_CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr __sstring<_Np>
__make_pretty_name_impl(char const (&__s)[_Mp], index_sequence<_Is...>) noexcept
{
  static_assert(_Mp <= _Np, "Type name too long for __pretty_nameof");
  return __sstring<_Np>{{(_Is < _Mp ? __s[_Is] : '\0')...}, _Mp - 1};
}

template <class _Tp, size_t _Np>
_CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr auto __make_pretty_name(integral_constant<size_t, _Np>) noexcept //
  -> enable_if_t<_Np == size_t(-1), __string_view>
{
  using _TpName = __static_nameof<_Tp, sizeof(_CCCL_BUILTIN_PRETTY_FUNCTION())>;
  return __string_view(_TpName::value.__str_, _TpName::value.__len_);
}

template <class _Tp, size_t _Np>
_CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr auto __make_pretty_name(integral_constant<size_t, _Np>) noexcept //
  -> enable_if_t<_Np != size_t(-1), __sstring<_Np>>
{
  return ::cuda::std::__make_pretty_name_impl<_Np>(_CCCL_BUILTIN_PRETTY_FUNCTION(), make_index_sequence<_Np>{});
}

// TODO: class statics cannot be accessed from device code, so we need to use
// a variable template when that is available.
template <class _Tp, size_t _Np>
struct __static_nameof
{
  static constexpr __sstring<_Np> value = ::cuda::std::__make_pretty_name<_Tp>(integral_constant<size_t, _Np>());
};

template <class _Tp, size_t _Np>
constexpr __sstring<_Np> __static_nameof<_Tp, _Np>::value;

#endif // _CCCL_COMPILER(GCC, <, 9)

template <class _Tp>
struct __pretty_name_begin
{
  struct __pretty_name_end;
};

// If a position is -1, it is an invalid position. Return it unchanged.
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr ptrdiff_t
__add_string_view_position(ptrdiff_t __pos, ptrdiff_t __diff) noexcept
{
  return __pos == -1 ? -1 : __pos + __diff;
}

// Get the type name from the pretty name by trimming the front and back.
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr __string_view
__find_pretty_name(__string_view __sv) noexcept
{
  return __sv.substr(::cuda::std::__add_string_view_position(
                       __sv.find("__pretty_name_begin<"), ptrdiff_t(sizeof("__pretty_name_begin<")) - 1),
                     __sv.find_end(">::__pretty_name_end"));
}

template <class _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr __string_view __pretty_nameof_helper() noexcept
{
#if _CCCL_COMPILER(GCC, <, 9) && !defined(__CUDA_ARCH__)
  return ::cuda::std::__find_pretty_name(::cuda::std::__make_pretty_name<_Tp>(integral_constant<size_t, size_t(-1)>{}));
#else // ^^^ gcc < 9 ^^^^/ vvv other compiler vvv
  return ::cuda::std::__find_pretty_name(::cuda::std::__string_view(_CCCL_BUILTIN_PRETTY_FUNCTION()));
#endif // not gcc < 9
}

template <class _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr __string_view __pretty_nameof() noexcept
{
  return ::cuda::std::__pretty_nameof_helper<typename __pretty_name_begin<_Tp>::__pretty_name_end>();
}

// In device code with old versions of gcc, we cannot have nice things.
#if _CCCL_COMPILER(GCC, <, 9) && defined(__CUDA_ARCH__)
#  define _CCCL_NO_CONSTEXPR_PRETTY_NAMEOF
#endif

#if !defined(_CCCL_NO_CONSTEXPR_PRETTY_NAMEOF) && !defined(_CCCL_BROKEN_MSVC_FUNCSIG)
// A quick smoke test to ensure that the pretty name extraction is working.
static_assert(::cuda::std::__pretty_nameof<int>() == __string_view("int"), "");
static_assert(::cuda::std::__pretty_nameof<float>() < ::cuda::std::__pretty_nameof<int>(), "");
#endif

// There are many complications with defining a unique constexpr global object
// for each type in device code, particularly on Windows. So rather than try,
// we use an alternate formulation of typeid that does not require such objects.
#if defined(__CUDA_ARCH__)

struct __type_info;
struct __type_info_ptr_;
struct __type_info_ref_;

struct __type_info_impl
{
  __string_view __name_;
};

struct __type_info_ptr_
{
  _CCCL_HIDE_FROM_ABI constexpr __type_info_ptr_() noexcept = default;

  _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr __type_info_ptr_(__type_info_impl (*__pfn_)() noexcept) noexcept
      : __pfn_(__pfn_)
  {}

  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr __type_info_ref_ operator*() const noexcept;

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(__type_info_ptr_ __a, __type_info_ptr_ __b) noexcept
  {
    return __a.__pfn_ == __b.__pfn_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(__type_info_ptr_ __a, __type_info_ptr_ __b) noexcept
  {
    return !(__a == __b);
  }

  __type_info_impl (*__pfn_)() noexcept = nullptr;
};

/// @brief A minimal implementation of `std::type_info` for device code that does
/// not depend on RTTI or variable templates.
struct __type_info
{
  __type_info() = delete;

  _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE explicit constexpr __type_info(__type_info_impl (*__pfn)() noexcept) noexcept
      : __pfn_(__pfn)
  {}

  template <class _Tp>
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE static constexpr __type_info_impl __get_ti_for() noexcept
  {
    return __type_info_impl{::cuda::std::__pretty_nameof<_Tp>()};
  }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr char const* name() const noexcept
  {
    return __pfn_().__name_.begin();
  }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr __string_view __name_view() const noexcept
  {
    return __pfn_().__name_;
  }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr bool before(__type_info const& __other) const noexcept
  {
    return __pfn_().__name_ < __other.__pfn_().__name_;
  }

  // Not yet implemented:
  // [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr size_t hash_code() const noexcept
  // {
  //   return ;
  // }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr __type_info_ptr_ operator&() const noexcept
  {
    return __type_info_ptr_{__pfn_};
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(__type_info const& __a, __type_info const& __b) noexcept
  {
    return __a.__pfn_ == __b.__pfn_ || __a.__pfn_().__name_ == __b.__pfn_().__name_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(__type_info const& __a, __type_info const& __b) noexcept
  {
    return !(__a == __b);
  }

private:
  friend struct __type_info_ptr_;
  friend struct __type_info_ref_;

  __type_info(__type_info const&)            = default; // needed by __type_info_ptr_::operator*() before C++17
  __type_info& operator=(__type_info const&) = delete;

  __type_info_impl (*__pfn_)() noexcept;
};

struct __type_info_ref_ : __type_info
{
  _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr __type_info_ref_(__type_info_impl (*__pfn)() noexcept) noexcept
      : __type_info(__pfn)
  {}

  _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr __type_info_ref_(__type_info const& __other) noexcept
      : __type_info(__other)
  {}

  _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr __type_info_ref_(__type_info_ref_ const& __other) noexcept
      : __type_info(__other)
  {}
};

[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr __type_info_ref_
__type_info_ptr_::operator*() const noexcept
{
  return __type_info_ref_(__pfn_);
}

#  if defined(_CCCL_NO_TYPEID) || defined(_CCCL_USE_TYPEID_FALLBACK)
using type_info       = __type_info;
using __type_info_ptr = __type_info_ptr_;
using __type_info_ref = __type_info_ref_;
#  endif // defined(_CCCL_NO_TYPEID) || defined(_CCCL_USE_TYPEID_FALLBACK)

#  define _CCCL_TYPEID_FALLBACK(...) \
    ::cuda::std::__type_info_ref(&::cuda::std::__type_info::__get_ti_for<::cuda::std::remove_cv_t<__VA_ARGS__>>)

#elif defined(_CCCL_BROKEN_MSVC_FUNCSIG) // ^^^ defined(__CUDA_ARCH__) ^^^

// See comment above about _CCCL_BROKEN_MSVC_FUNCSIG
#  define _CCCL_TYPEID_FALLBACK _CCCL_STD_TYPEID
#  if defined(_CCCL_NO_TYPEID) || defined(_CCCL_USE_TYPEID_FALLBACK)
using type_info       = ::std::type_info;
using __type_info_ptr = ::std::type_info const*;
using __type_info_ref = ::std::type_info const&;
#  endif // defined(_CCCL_NO_TYPEID) || defined(_CCCL_USE_TYPEID_FALLBACK)

#else // ^^^ _CCCL_BROKEN_MSVC_FUNCSIG ^^^ / vvv !__CUDA_ARCH__ && !_CCCL_BROKEN_MSVC_FUNCSIG vvv

/// @brief A minimal implementation of `std::type_info` for platforms that do
/// not support RTTI.
struct __type_info
{
  __type_info()                              = delete;
  __type_info(__type_info const&)            = delete;
  __type_info& operator=(__type_info const&) = delete;

  _CCCL_HIDE_FROM_ABI constexpr __type_info(__string_view __name) noexcept
      : __name_(__name)
  {}

  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr char const* name() const noexcept
  {
    return __name_.begin();
  }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr __string_view __name_view() const noexcept
  {
    return __name_;
  }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr bool before(const __type_info& __other) const noexcept
  {
    return __name_ < __other.__name_;
  }

  // Not yet implemented:
  // [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr size_t hash_code() const noexcept
  // {
  //   return ;
  // }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI friend constexpr bool
  operator==(const __type_info& __lhs, const __type_info& __rhs) noexcept
  {
    return &__lhs == &__rhs || __lhs.__name_ == __rhs.__name_;
  }

#  if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_HIDE_FROM_ABI friend constexpr bool
  operator!=(const __type_info& __lhs, const __type_info& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#  endif // _CCCL_STD_VER <= 2017

private:
  __string_view __name_;
};

#  if defined(_CCCL_NO_TYPEID) || defined(_CCCL_USE_TYPEID_FALLBACK)
using type_info       = __type_info;
using __type_info_ptr = __type_info const*;
using __type_info_ref = __type_info const&;
#  endif // defined(_CCCL_NO_TYPEID) || defined(_CCCL_USE_TYPEID_FALLBACK)

template <class _Tp>
_CCCL_GLOBAL_CONSTANT __type_info __typeid_v{::cuda::std::__pretty_nameof<_Tp>()};

// When inline variables are available, this indirection through an inline function
// is not necessary, but it doesn't hurt either.
template <class _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr __type_info const& __typeid() noexcept
{
  return __typeid_v<_Tp>;
}

#  define _CCCL_TYPEID_FALLBACK(...) ::cuda::std::__typeid<::cuda::std::remove_cv_t<__VA_ARGS__>>()

#endif // !defined(__CUDA_ARCH__) && !_CCCL_BROKEN_MSVC_FUNCSIG

// if `__pretty_nameof` is constexpr _CCCL_TYPEID_FALLBACK is also constexpr.
#if !defined(_CCCL_NO_CONSTEXPR_PRETTY_NAMEOF) && (!defined(_CCCL_BROKEN_MSVC_FUNCSIG) || defined(__CUDA_ARCH__))
#  define _CCCL_TYPEOF_CONSTEXPR _CCCL_TYPEOF_FALLBACK
#endif

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_TYPEID_H
