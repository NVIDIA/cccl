//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_FORMAT_ARG_H
#define _CUDA_STD___FORMAT_FORMAT_ARG_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/arithmetic.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__format/concepts.h>
#include <cuda/std/__format/format_parse_context.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__fwd/format.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/monostate.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/unreachable.h>
#include <cuda/std/cstdint>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

enum class __fmt_arg_t : uint8_t
{
  __none,
  __boolean,
  __char_type,
  __int,
  __long_long,
  __unsigned,
  __unsigned_long_long,
  __float,
  __double,
  __long_double,
  __const_char_type_ptr,
  __string_view,
  __ptr,
  __handle,
};

inline constexpr unsigned __fmt_packed_arg_t_bits = 5;
inline constexpr uint8_t __fmt_packed_arg_t_mask  = 0x1f;

inline constexpr unsigned __fmt_packed_types_storage_bits = 64;
inline constexpr unsigned __fmt_packed_types_max          = __fmt_packed_types_storage_bits / __fmt_packed_arg_t_bits;

[[nodiscard]] _CCCL_API constexpr bool __fmt_use_packed_format_arg_store(size_t __size) noexcept
{
  return __size <= __fmt_packed_types_max;
}

[[nodiscard]] _CCCL_API constexpr __fmt_arg_t __fmt_get_packed_type(uint64_t __types, size_t __id) noexcept
{
  _CCCL_ASSERT(__id <= __fmt_packed_types_max, "");

  if (__id > 0)
  {
    __types >>= __id * __fmt_packed_arg_t_bits;
  }

  return static_cast<__fmt_arg_t>(__types & __fmt_packed_arg_t_mask);
}

/// Contains the values used in basic_format_arg.
///
/// This is a separate type so it's possible to store the values and types in
/// separate arrays.
template <class _Context>
class __basic_format_arg_value
{
  using _CharT _CCCL_NODEBUG_ALIAS = typename _Context::char_type;

  template <class _Tp>
  _CCCL_API static void
  __formatter_invoker(basic_format_parse_context<_CharT>& __parse_ctx, _Context& __ctx, const void* __ptr)
  {
    using _Dp = remove_const_t<_Tp>;
    using _Qp = conditional_t<__formattable_with<const _Dp, _Context>, const _Dp, _Dp>;
    static_assert(__formattable_with<_Qp, _Context>, "Mandated by [format.arg]/10");

    using _Formatter = typename _Context::template formatter_type<_Dp>;
    _Formatter __f;
    __parse_ctx.advance_to(__f.parse(__parse_ctx));
    __ctx.advance_to(__f.format(*const_cast<_Qp*>(static_cast<const _Dp*>(__ptr)), __ctx));
  }

public:
  /// Contains the implementation for basic_format_arg::handle.
  struct __handle
  {
    template <class _Tp>
    _CCCL_API explicit __handle(_Tp& __v) noexcept
        : __ptr_(::cuda::std::addressof(__v))
        , __format_(__formatter_invoker<_Tp>)
    {}

    const void* __ptr_;
    void (*__format_)(basic_format_parse_context<_CharT>&, _Context&, const void*);
  };

  union
  {
    monostate __monostate_;
    bool __boolean_;
    _CharT __char_type_;
    int __int_;
    unsigned __unsigned_;
    long long __long_long_;
    unsigned long long __unsigned_long_long_;
    float __float_;
    double __double_;
    long double __long_double_;
    const _CharT* __const_char_type_ptr_;
    basic_string_view<_CharT> __string_view_;
    const void* __ptr_;
    __handle __handle_;
  };

  _CCCL_API __basic_format_arg_value() noexcept
      : __monostate_()
  {}
  _CCCL_API __basic_format_arg_value(bool __value) noexcept
      : __boolean_(__value)
  {}
  _CCCL_API __basic_format_arg_value(_CharT __value) noexcept
      : __char_type_(__value)
  {}
  _CCCL_API __basic_format_arg_value(int __value) noexcept
      : __int_(__value)
  {}
  _CCCL_API __basic_format_arg_value(unsigned __value) noexcept
      : __unsigned_(__value)
  {}
  _CCCL_API __basic_format_arg_value(long long __value) noexcept
      : __long_long_(__value)
  {}
  _CCCL_API __basic_format_arg_value(unsigned long long __value) noexcept
      : __unsigned_long_long_(__value)
  {}
  _CCCL_API __basic_format_arg_value(float __value) noexcept
      : __float_(__value)
  {}
  _CCCL_API __basic_format_arg_value(double __value) noexcept
      : __double_(__value)
  {}
  _CCCL_API __basic_format_arg_value(long double __value) noexcept
      : __long_double_(__value)
  {}
  _CCCL_API __basic_format_arg_value(const _CharT* __value) noexcept
      : __const_char_type_ptr_(__value)
  {}
  _CCCL_API __basic_format_arg_value(basic_string_view<_CharT> __value) noexcept
      : __string_view_(__value)
  {}
  _CCCL_API __basic_format_arg_value(const void* __value) noexcept
      : __ptr_(__value)
  {}
  _CCCL_API __basic_format_arg_value(__handle&& __value) noexcept
      : __handle_(::cuda::std::move(__value))
  {}
};

template <class _Context>
class _CCCL_NO_SPECIALIZATIONS _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_arg
{
public:
  class handle;

  _CCCL_API basic_format_arg() noexcept
      : __type_{__fmt_arg_t::__none}
  {}

  _CCCL_API explicit operator bool() const noexcept
  {
    return __type_ != __fmt_arg_t::__none;
  }

private:
  using char_type = typename _Context::char_type;

  // TODO FMT Implement constrain [format.arg]/4
  // Constraints: The template specialization
  //   typename Context::template formatter_type<T>
  // meets the Formatter requirements ([formatter.requirements]).  The extent
  // to which an implementation determines that the specialization meets the
  // Formatter requirements is unspecified, except that as a minimum the
  // expression
  //   typename Context::template formatter_type<T>()
  //    .format(declval<const T&>(), declval<Context&>())
  // shall be well-formed when treated as an unevaluated operand.

public:
  __basic_format_arg_value<_Context> __value_;
  __fmt_arg_t __type_;

  _CCCL_API explicit basic_format_arg(__fmt_arg_t __type, __basic_format_arg_value<_Context> __value) noexcept
      : __value_(__value)
      , __type_(__type)
  {}
};

template <class _Context>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_arg<_Context>::handle
{
public:
  _CCCL_API void format(basic_format_parse_context<char_type>& __parse_ctx, _Context& __ctx) const
  {
    __handle_.__format_(__parse_ctx, __ctx, __handle_.__ptr_);
  }

  _CCCL_API explicit handle(typename __basic_format_arg_value<_Context>::__handle& __handle) noexcept
      : __handle_(__handle)
  {}

private:
  typename __basic_format_arg_value<_Context>::__handle& __handle_;
};

template <class _Visitor, class _Context>
_CCCL_API decltype(auto) visit_format_arg(_Visitor&& __vis, basic_format_arg<_Context> __arg)
{
  switch (__arg.__type_)
  {
    case __fmt_arg_t::__none:
      return ::cuda::std::invoke(::cuda::std::forward<_Visitor>(__vis), __arg.__value_.__monostate_);
    case __fmt_arg_t::__boolean:
      return ::cuda::std::invoke(::cuda::std::forward<_Visitor>(__vis), __arg.__value_.__boolean_);
    case __fmt_arg_t::__char_type:
      return ::cuda::std::invoke(::cuda::std::forward<_Visitor>(__vis), __arg.__value_.__char_type_);
    case __fmt_arg_t::__int:
      return ::cuda::std::invoke(::cuda::std::forward<_Visitor>(__vis), __arg.__value_.__int_);
    case __fmt_arg_t::__long_long:
      return ::cuda::std::invoke(::cuda::std::forward<_Visitor>(__vis), __arg.__value_.__long_long_);
    case __fmt_arg_t::__unsigned:
      return ::cuda::std::invoke(::cuda::std::forward<_Visitor>(__vis), __arg.__value_.__unsigned_);
    case __fmt_arg_t::__unsigned_long_long:
      return ::cuda::std::invoke(::cuda::std::forward<_Visitor>(__vis), __arg.__value_.__unsigned_long_long_);
    case __fmt_arg_t::__float:
      return ::cuda::std::invoke(::cuda::std::forward<_Visitor>(__vis), __arg.__value_.__float_);
    case __fmt_arg_t::__double:
      return ::cuda::std::invoke(::cuda::std::forward<_Visitor>(__vis), __arg.__value_.__double_);
    case __fmt_arg_t::__long_double:
      return ::cuda::std::invoke(::cuda::std::forward<_Visitor>(__vis), __arg.__value_.__long_double_);
    case __fmt_arg_t::__const_char_type_ptr:
      return ::cuda::std::invoke(::cuda::std::forward<_Visitor>(__vis), __arg.__value_.__const_char_type_ptr_);
    case __fmt_arg_t::__string_view:
      return ::cuda::std::invoke(::cuda::std::forward<_Visitor>(__vis), __arg.__value_.__string_view_);
    case __fmt_arg_t::__ptr:
      return ::cuda::std::invoke(::cuda::std::forward<_Visitor>(__vis), __arg.__value_.__ptr_);
    case __fmt_arg_t::__handle:
      return ::cuda::std::invoke(
        ::cuda::std::forward<_Visitor>(__vis), typename basic_format_arg<_Context>::handle{__arg.__value_.__handle_});
    default:
      _CCCL_UNREACHABLE();
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMAT_ARG_H
