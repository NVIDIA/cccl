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
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_member_function_pointer.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_pointer.h>
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
_CCCL_API constexpr __sstring<_Np> __make_pretty_name_impl(char const (&__s)[_Mp], index_sequence<_Is...>) noexcept
{
  static_assert(_Mp <= _Np, "Type name too long for __pretty_nameof");
  return __sstring<_Np>{{(_Is < _Mp ? __s[_Is] : '\0')...}, _Mp - 1};
}

template <class _Tp, size_t _Np>
_CCCL_API constexpr auto __make_pretty_name(integral_constant<size_t, _Np>) noexcept //
  -> enable_if_t<_Np == size_t(-1), __string_view>
{
  using _TpName = __static_nameof<_Tp, sizeof(_CCCL_BUILTIN_PRETTY_FUNCTION())>;
  return __string_view(_TpName::value.__str_, _TpName::value.__len_);
}

template <class _Tp, size_t _Np>
_CCCL_API constexpr auto __make_pretty_name(integral_constant<size_t, _Np>) noexcept //
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
[[nodiscard]] _CCCL_API constexpr ptrdiff_t __add_string_view_position(ptrdiff_t __pos, ptrdiff_t __diff) noexcept
{
  return __pos == -1 ? -1 : __pos + __diff;
}

// Get the type name from the pretty name by trimming the front and back.
[[nodiscard]] _CCCL_API constexpr __string_view __find_pretty_name(__string_view __sv) noexcept
{
  return __sv.substr(::cuda::std::__add_string_view_position(
                       __sv.find("__pretty_name_begin<"), static_cast<ptrdiff_t>(sizeof("__pretty_name_begin<")) - 1),
                     __sv.find_end(">::__pretty_name_end"));
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr __string_view __pretty_nameof_helper() noexcept
{
#if _CCCL_COMPILER(GCC, <, 9) && !defined(__CUDA_ARCH__)
  return ::cuda::std::__find_pretty_name(::cuda::std::__make_pretty_name<_Tp>(integral_constant<size_t, size_t(-1)>{}));
#else // ^^^ gcc < 9 ^^^^/ vvv other compiler vvv
  return ::cuda::std::__find_pretty_name(::cuda::std::__string_view(_CCCL_BUILTIN_PRETTY_FUNCTION()));
#endif // not gcc < 9
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr __string_view __pretty_nameof() noexcept
{
  return ::cuda::std::__pretty_nameof_helper<typename __pretty_name_begin<_Tp>::__pretty_name_end>();
}

// In device code with old versions of gcc, we cannot have nice things.
#if _CCCL_COMPILER(GCC, <, 9) && defined(__CUDA_ARCH__)
#  define _CCCL_NO_CONSTEXPR_PRETTY_NAMEOF
#endif

#if !defined(_CCCL_NO_CONSTEXPR_PRETTY_NAMEOF) && !defined(_CCCL_BROKEN_MSVC_FUNCSIG) \
  && defined(_CCCL_ENABLE_DEBUG_MODE)
// A quick smoke test to ensure that the pretty name extraction is working.
static_assert(::cuda::std::__pretty_nameof<int>() == __string_view("int"));
static_assert(::cuda::std::__pretty_nameof<float>() < ::cuda::std::__pretty_nameof<int>());
#endif // !_CCCL_NO_CONSTEXPR_PRETTY_NAMEOF && !_CCCL_BROKEN_MSVC_FUNCSIG && _CCCL_ENABLE_DEBUG_MODE

// We find the spelling of a non-type template parameter value _Vp as follows:
// 1. Wrap the value in the class template __stringof_wrapper and obtain
//    the pretty name of that type via __pretty_nameof. This reuses all of the
//    compiler-specific machinery (and quirk handling) that __pretty_nameof
//    already implements.
// 2. The resulting string looks like "...__stringof_wrapper<42>...".
//    Trim the surrounding wrapper to recover the value's spelling.
//
// The exact spelling is whatever the compiler emits for the value and is not
// guaranteed to be identical across compilers (e.g. a char might be spelled
// 'A', an enumerator might be spelled by name or by a cast). Integral values
// such as `42` are spelled identically everywhere.

template <auto _Vp>
struct __stringof_wrapper
{};

// Extract the value's spelling from the pretty name of __stringof_wrapper<_Vp>.
[[nodiscard]] _CCCL_API constexpr __string_view __find_stringof(__string_view __sv) noexcept
{
  // Trim the surrounding "__stringof_wrapper<" ... ">".
  return __sv.substr(::cuda::std::__add_string_view_position(
                       __sv.find("__stringof_wrapper<"), static_cast<ptrdiff_t>(sizeof("__stringof_wrapper<")) - 1),
                     __sv.find_end(">"));
}

//! @brief Returns the compiler's spelling of a value passed as a non-type
//! template parameter, e.g. `__stringof<42>()` yields `"42"` and
//! `__stringof<cudaStreamSynchronize>()` yields `"cudaStreamSynchronize"`.
//!
//! @tparam _Vp The value whose compiler spelling is returned.
//! @return A string view containing the compiler's spelling of `_Vp`.
//!
//! This is the value counterpart of `__pretty_nameof` (which spells types). It
//! supports the same set of compilers and the same compiler quirks, because it
//! is implemented on top of `__pretty_nameof`.
//!
//! When the value is a function (or function pointer), the leading '&' that
//! clang and cudafe prepend to a function template argument is dropped, so the
//! result is just the function's (possibly qualified) name. The exact spelling
//! of other values is whatever the compiler emits and is not guaranteed to be
//! identical across compilers.
template <auto _Vp>
[[nodiscard]] _CCCL_API constexpr __string_view __stringof() noexcept
{
  __string_view __sv =
    ::cuda::std::__find_stringof(::cuda::std::__pretty_nameof<::cuda::std::__stringof_wrapper<_Vp>>());
  // For a function argument, clang and cudafe prepend a '&' (e.g. "&fn"); drop
  // it so the result is just the function's name.
  if constexpr (is_function_v<remove_pointer_t<decltype(_Vp)>> || is_member_function_pointer_v<decltype(_Vp)>)
  {
    if (__sv.size() != 0 && __sv[0] == '&')
    {
      __sv = __sv.substr(1, static_cast<ptrdiff_t>(__sv.size()));
    }
  }
  return __sv;
}

#if !defined(_CCCL_NO_CONSTEXPR_PRETTY_NAMEOF) && !defined(_CCCL_BROKEN_MSVC_FUNCSIG) \
  && defined(_CCCL_ENABLE_DEBUG_MODE)
// A quick smoke test to ensure that the value spelling extraction is working.
// An integer literal is spelled identically on every supported compiler.
static_assert(::cuda::std::__stringof<42>() == __string_view("42"));
static_assert(::cuda::std::__stringof<42>() != ::cuda::std::__stringof<43>());
// And that function arguments work (including the leading-'&' trim). We use a
// host/device function defined above so this works in both compilation passes
// without depending on any external symbols. The function lives in an inline
// namespace, whose spelling varies between compilers, so we only check that the
// unqualified name is present rather than matching it exactly.
static_assert(::cuda::std::__stringof<&::cuda::std::__add_string_view_position>().find("__add_string_view_position")
              != -1);
#endif // !_CCCL_NO_CONSTEXPR_PRETTY_NAMEOF && !_CCCL_BROKEN_MSVC_FUNCSIG && _CCCL_ENABLE_DEBUG_MODE

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

  _CCCL_API constexpr __type_info_ptr_(__type_info_impl (*__pfn_)() noexcept) noexcept
      : __pfn_(__pfn_)
  {}

  [[nodiscard]] _CCCL_API constexpr __type_info_ref_ operator*() const noexcept;

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

  _CCCL_API explicit constexpr __type_info(__type_info_impl (*__pfn)() noexcept) noexcept
      : __pfn_(__pfn)
  {}

  template <class _Tp>
  [[nodiscard]] _CCCL_API static constexpr __type_info_impl __get_ti_for() noexcept
  {
    return __type_info_impl{::cuda::std::__pretty_nameof<_Tp>()};
  }

  [[nodiscard]] _CCCL_API constexpr char const* name() const noexcept
  {
    return __pfn_().__name_.begin();
  }

  [[nodiscard]] _CCCL_API constexpr __string_view __name_view() const noexcept
  {
    return __pfn_().__name_;
  }

  [[nodiscard]] _CCCL_API constexpr bool before(__type_info const& __other) const noexcept
  {
    return __pfn_().__name_ < __other.__pfn_().__name_;
  }

  // Not yet implemented:
  // [[nodiscard]] _CCCL_API constexpr size_t hash_code() const noexcept
  // {
  //   return ;
  // }

  [[nodiscard]] _CCCL_API constexpr __type_info_ptr_ operator&() const noexcept
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

  __type_info& operator=(__type_info const&) = delete;

private:
  friend struct __type_info_ptr_;
  friend struct __type_info_ref_;

  __type_info(__type_info const&) = default; // needed by __type_info_ptr_::operator*() before C++17
  __type_info_impl (*__pfn_)() noexcept;
};

struct __type_info_ref_ : __type_info
{
  _CCCL_API constexpr __type_info_ref_(__type_info_impl (*__pfn)() noexcept) noexcept
      : __type_info(__pfn)
  {}

  _CCCL_API constexpr __type_info_ref_(__type_info const& __other) noexcept
      : __type_info(__other)
  {}

  _CCCL_HIDE_FROM_ABI constexpr __type_info_ref_(__type_info_ref_ const& __other) noexcept = default;
};

[[nodiscard]] _CCCL_API constexpr __type_info_ref_ __type_info_ptr_::operator*() const noexcept
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

  _CCCL_HOST_API constexpr __type_info(__string_view __name) noexcept
      : __name_(__name)
  {}

  [[nodiscard]] _CCCL_HOST_API constexpr char const* name() const noexcept
  {
    return __name_.begin();
  }

  [[nodiscard]] _CCCL_HOST_API constexpr __string_view __name_view() const noexcept
  {
    return __name_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr bool before(const __type_info& __other) const noexcept
  {
    return __name_ < __other.__name_;
  }

  // Not yet implemented:
  // [[nodiscard]] _CCCL_HOST_API constexpr size_t hash_code() const noexcept
  // {
  //   return ;
  // }

  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator==(const __type_info& __lhs, const __type_info& __rhs) noexcept
  {
    return &__lhs == &__rhs || __lhs.__name_ == __rhs.__name_;
  }

#  if _CCCL_STD_VER <= 2017
  [[nodiscard]]
  _CCCL_HOST_API friend constexpr bool operator!=(const __type_info& __lhs, const __type_info& __rhs) noexcept
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
[[nodiscard]] _CCCL_HOST_API constexpr __type_info const& __typeid() noexcept
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
