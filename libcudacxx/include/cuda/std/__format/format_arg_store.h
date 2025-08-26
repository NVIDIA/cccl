//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_FORMAT_ARG_STORE_H
#define _CUDA_STD___FORMAT_FORMAT_ARG_STORE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__format/concepts.h>
#include <cuda/std/__format/format_arg.h>
#include <cuda/std/__string/char_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/extent.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/cstdint>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Arr, class _Elem>
inline constexpr bool __is_bounded_array_of = false;

template <class _Elem, size_t _Len>
inline constexpr bool __is_bounded_array_of<_Elem[_Len], _Elem> = true;

template <class _Context, class _Tp>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __fmt_arg_t __fmt_determine_arg_t()
{
  using _CtxCharT = typename _Context::char_type;

  __fmt_arg_t __ret = __fmt_arg_t::__none;

  if constexpr (is_same_v<_Tp, bool>)
  {
    __ret = __fmt_arg_t::__boolean;
  }
  else if constexpr (is_same_v<_Tp, _CtxCharT>)
  {
    __ret = __fmt_arg_t::__char_type;
  }
#if _CCCL_HAS_WCHAR_T()
  else if (is_same_v<wchar_t, _CtxCharT> && is_same_v<_Tp, char>)
  {
    __ret = __fmt_arg_t::__char_type;
  }
#endif // _CCCL_HAS_WCHAR_T()
  else if constexpr (__cccl_is_integer_v<_Tp>)
  {
    if constexpr (sizeof(_Tp) <= sizeof(int))
    {
      __ret = (is_signed_v<_Tp>) ? __fmt_arg_t::__int : __fmt_arg_t::__unsigned;
    }
    else if constexpr (sizeof(_Tp) <= sizeof(long long))
    {
      __ret = (is_signed_v<_Tp>) ? __fmt_arg_t::__long_long : __fmt_arg_t::__unsigned_long_long;
    }
    else
    {
      // Extended integer types are handled as handles
      __ret = __fmt_arg_t::__handle;
    }
  }
  else if constexpr (is_same_v<_Tp, float>)
  {
    __ret = __fmt_arg_t::__float;
  }
  else if constexpr (is_same_v<_Tp, double>)
  {
    __ret = __fmt_arg_t::__double;
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (is_same_v<_Tp, long double>)
  {
    __ret = __fmt_arg_t::__long_double;
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (is_same_v<_Tp, _CtxCharT*> || is_same_v<_Tp, const _CtxCharT*>)
  {
    __ret = __fmt_arg_t::__const_char_type_ptr;
  }
  // todo: add std::string and std::string_view support
  // - add `|| __cccl_is_std_string_view_v<_Tp> || __cccl_is_std_string_v<_Tp>` to the condition
  else if constexpr (__cccl_is_string_view_v<_Tp>)
  {
    __ret = (is_same_v<_CtxCharT, typename _Tp::value_type>) ? __fmt_arg_t::__string_view : __ret;
  }
  else if constexpr (__is_bounded_array_of<_Tp, _CtxCharT>)
  {
    __ret = __fmt_arg_t::__string_view;
  }
  else if constexpr (is_same_v<_Tp, void*> || is_same_v<_Tp, const void*> || is_same_v<_Tp, nullptr_t>)
  {
    __ret = __fmt_arg_t::__ptr;
  }

  if constexpr (__formattable_with<_Tp, _Context>)
  {
    __ret = (__ret == __fmt_arg_t::__none) ? __fmt_arg_t::__handle : __ret;
  }

  return __ret;
}

template <class _Context, class _Tp>
[[nodiscard]] _CCCL_API basic_format_arg<_Context> __fmt_make_format_arg(_Tp& __value) noexcept
{
  using _Dp                   = remove_const_t<_Tp>;
  constexpr __fmt_arg_t __arg = __fmt_determine_arg_t<_Context, _Dp>();

  static_assert(__arg != __fmt_arg_t::__none, "the supplied type is not formattable");
  static_assert(__formattable_with<_Tp, _Context>);

  using _CtxCharT = typename _Context::char_type;
  // Not all types can be used to directly initialize the
  // __basic_format_arg_value.  First handle all types needing adjustment, the
  // final else requires no adjustment.
  if constexpr (__arg == __fmt_arg_t::__char_type)
  {
#if _CCCL_HAS_WCHAR_T()
    if constexpr (is_same_v<wchar_t, _CtxCharT> && is_same_v<_Dp, char>)
    {
      return basic_format_arg<_Context>{__arg, static_cast<wchar_t>(static_cast<unsigned char>(__value))};
    }
    else
#endif // _CCCL_HAS_WCHAR_T()
    {
      return basic_format_arg<_Context>{__arg, __value};
    }
  }
  else if constexpr (__arg == __fmt_arg_t::__int)
  {
    return basic_format_arg<_Context>{__arg, static_cast<int>(__value)};
  }
  else if constexpr (__arg == __fmt_arg_t::__long_long)
  {
    return basic_format_arg<_Context>{__arg, static_cast<long long>(__value)};
  }
  else if constexpr (__arg == __fmt_arg_t::__unsigned)
  {
    return basic_format_arg<_Context>{__arg, static_cast<unsigned>(__value)};
  }
  else if constexpr (__arg == __fmt_arg_t::__unsigned_long_long)
  {
    return basic_format_arg<_Context>{__arg, static_cast<unsigned long long>(__value)};
  }
  else if constexpr (__arg == __fmt_arg_t::__string_view)
  {
    // Using std::size on a character array will add the NUL-terminator to the size.
    if constexpr (__is_bounded_array_of<_Dp, _CtxCharT>)
    {
      const _CtxCharT* const __pbegin = begin(__value);
      const _CtxCharT* const __pzero  = char_traits<_CtxCharT>::find(__pbegin, extent_v<_Dp>, _CtxCharT{});
      _CCCL_ASSERT(__pzero != nullptr, "formatting a non-null-terminated array");
      return basic_format_arg<_Context>{
        __arg, basic_string_view<_CtxCharT>{__pbegin, static_cast<size_t>(__pzero - __pbegin)}};
    }
    else
    {
      // When the _Traits or _Allocator are different an implicit conversion will fail.
      return basic_format_arg<_Context>{__arg, basic_string_view<_CtxCharT>{__value.data(), __value.size()}};
    }
  }
  else if constexpr (__arg == __fmt_arg_t::__ptr)
  {
    return basic_format_arg<_Context>{__arg, static_cast<const void*>(__value)};
  }
  else if constexpr (__arg == __fmt_arg_t::__handle)
  {
    return basic_format_arg<_Context>{__arg, typename __basic_format_arg_value<_Context>::__handle{__value}};
  }
  else
  {
    return basic_format_arg<_Context>{__arg, __value};
  }
}

template <class _Context, class _Tp>
_CCCL_API void __fmt_make_packed_storage_impl(
  uint64_t& __types, __basic_format_arg_value<_Context>*& __values, int& __shift, _Tp& __v) noexcept
{
  basic_format_arg<_Context> __arg = ::cuda::std::__fmt_make_format_arg<_Context>(__v);
  if (__shift != 0)
  {
    __types |= static_cast<uint64_t>(__arg.__type_) << __shift;
  }
  else
  {
    // Assigns the initial value.
    __types = static_cast<uint64_t>(__arg.__type_);
  }
  __shift += __fmt_packed_arg_t_bits;
  *__values++ = __arg.__value_;
}

template <class _Context, class... _Args>
_CCCL_API void
__fmt_make_packed_storage(uint64_t& __types, __basic_format_arg_value<_Context>* __values, _Args&... __args) noexcept
{
  int __shift = 0;
  (::cuda::std::__fmt_make_packed_storage_impl(__types, __values, __shift, __args), ...);
}

template <class _Context, class... _Args>
_CCCL_API void __fmt_store_basic_format_arg(basic_format_arg<_Context>* __data, _Args&... __args) noexcept
{
  ((*__data++ = ::cuda::std::__fmt_make_format_arg<_Context>(__args)), ...);
}

template <class _Context, size_t _Np>
struct __packed_format_arg_store
{
  __basic_format_arg_value<_Context> __values_[_Np];
  uint64_t __types_ = 0;
};

template <class _Context>
struct __packed_format_arg_store<_Context, 0>
{
  uint64_t __types_ = 0;
};

template <class _Context, size_t _Np>
struct __unpacked_format_arg_store
{
  basic_format_arg<_Context> __args_[_Np];
};

template <class _Context, class... _Args>
struct __format_arg_store
{
  _CCCL_API __format_arg_store(_Args&... __args) noexcept
  {
    if constexpr (sizeof...(_Args) != 0)
    {
      if constexpr (::cuda::std::__fmt_use_packed_format_arg_store(sizeof...(_Args)))
      {
        ::cuda::std::__fmt_make_packed_storage(__storage.__types_, __storage.__values_, __args...);
      }
      else
      {
        ::cuda::std::__fmt_store_basic_format_arg(__storage.__args_, __args...);
      }
    }
  }

  using _Storage _CCCL_NODEBUG_ALIAS =
    conditional_t<::cuda::std::__fmt_use_packed_format_arg_store(sizeof...(_Args)),
                  __packed_format_arg_store<_Context, sizeof...(_Args)>,
                  __unpacked_format_arg_store<_Context, sizeof...(_Args)>>;

  _Storage __storage;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMAT_ARG_STORE_H
