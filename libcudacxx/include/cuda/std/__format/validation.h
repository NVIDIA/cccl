//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FORMAT_VALIDATION_H
#define _LIBCUDACXX___FORMAT_VALIDATION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__format/format_arg.h>
#include <cuda/std/__format/format_arg_store.h>
#include <cuda/std/__format/format_error.h>
#include <cuda/std/__format/format_parse_context.h>
#include <cuda/std/__format/formatter.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//! @brief Helper class parse and handle argument.
//!
//! When parsing a handle which is not enabled the code is ill-formed.
//! This helper uses the parser of the appropriate formatter for the stored type.
template <class _CharT>
struct __fmt_validation_format_arg_handle
{
  void (*__parse_)(basic_format_parse_context<_CharT>&);

  template <class _Tp>
  _CCCL_API static constexpr void __formatter_invoker(basic_format_parse_context<_CharT>& __parse_ctx)
  {
    formatter<_Tp, _CharT> __f;
    __parse_ctx.advance_to(__f.parse(__parse_ctx));
  }

  _CCCL_API static constexpr void __formatter_invoker_invalid(basic_format_parse_context<_CharT>& __parse_ctx)
  {
    ::cuda::std::__throw_format_error("Not a handle");
  }

  template <class _ParseCtx>
  _CCCL_API constexpr void __parse(_ParseCtx& __parse_ctx) const
  {
    __parse_(__parse_ctx);
  }
};

template <class _Context, class _Tp>
[[nodiscard]] _CCCL_API constexpr __fmt_validation_format_arg_handle<typename _Context::char_type>
__fmt_make_validation_format_arg_handle()
{
  using _CharT = typename _Context::char_type;
  if constexpr (::cuda::std::__fmt_determine_arg_t<_Context, _Tp>() == __fmt_arg_t::__handle)
  {
    return {__fmt_validation_format_arg_handle<_CharT>::template __formatter_invoker<_Tp>};
  }
  else
  {
    return {__fmt_validation_format_arg_handle<_CharT>::__formatter_invoker_invalid};
  }
}

//! @brief Dummy format_context only providing the parts used during constant validation of the basic_format_string.
template <class _CharT>
class __fmt_validation_format_context
{
  using _FmtArgHandle = __fmt_validation_format_arg_handle<_CharT>;

  const __fmt_arg_t* __args_;
  const _FmtArgHandle* __handles_;
  size_t __size_;

public:
  using char_type = _CharT;
  template <class _Tp>
  using formatter_type = formatter<_Tp, _CharT>;

  //! @brief Dummy iterator. During the compile-time validation nothing needs to be written. Therefore all operations of
  //!        this iterator are a NOP.
  struct iterator
  {
    _CCCL_API constexpr iterator& operator=(_CharT)
    {
      return *this;
    }
    _CCCL_API constexpr iterator& operator*()
    {
      return *this;
    }
    _CCCL_API constexpr iterator operator++(int)
    {
      return *this;
    }
  };

  _CCCL_API constexpr explicit __fmt_validation_format_context(
    const __fmt_arg_t* __args, const _FmtArgHandle* __handles, size_t __size)
      : __args_{__args}
      , __handles_{__handles}
      , __size_{__size}
  {}

  [[nodiscard]] _CCCL_API constexpr __fmt_arg_t arg(size_t __id) const
  {
    if (__id >= __size_)
    {
      ::cuda::std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
    }
    return __args_[__id];
  }

  [[nodiscard]] _CCCL_API constexpr const _FmtArgHandle& __handle(size_t __id) const
  {
    if (__id >= __size_)
    {
      ::cuda::std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
    }
    return __handles_[__id];
  }

  [[nodiscard]] _CCCL_API constexpr iterator out()
  {
    return {};
  }
  _CCCL_API constexpr void advance_to(iterator) {}
};

// [format.string.std]/8
// If { arg-idopt } is used in a width or precision, the value of the
// corresponding formatting argument is used in its place. If the
// corresponding formatting argument is not of standard signed or unsigned
// integer type, or its value is negative for precision or non-positive for
// width, an exception of type format_error is thrown.
//
// _HasPrecision does the formatter have a precision?
template <class _CharT, class _Tp, bool _HasPrecision = false>
_CCCL_API constexpr void __fmt_validate_format_arg(basic_format_parse_context<_CharT>& __parse_ctx,
                                                   __fmt_validation_format_context<_CharT>& __ctx)
{
  // LWG3720 originally allowed "signed or unsigned integer types", however
  // the final version explicitly changed it to "*standard* signed or unsigned
  // integer types". It's trivial to use 128-bit integrals in libc++'s
  // implementation, but other implementations may not implement it.
  // (Using a width or precision, that does not fit in 64-bits, sounds very
  // unlikely in real world code.)

  formatter<_Tp, _CharT> __formatter;
  __parse_ctx.advance_to(__formatter.parse(__parse_ctx));
  if (__formatter.__parser_.__width_as_arg_)
  {
    switch (__ctx.arg(__formatter.__parser_.__width_))
    {
      case __fmt_arg_t::__int:
      case __fmt_arg_t::__long_long:
      case __fmt_arg_t::__unsigned:
      case __fmt_arg_t::__unsigned_long_long:
        break;
      default:
        ::cuda::std::__throw_format_error("Replacement argument isn't a standard signed or unsigned integer type");
    }
  }

  if constexpr (_HasPrecision)
  {
    if (__formatter.__parser_.__precision_as_arg_)
    {
      switch (__ctx.arg(__formatter.__parser_.__precision_))
      {
        case __fmt_arg_t::__int:
        case __fmt_arg_t::__long_long:
        case __fmt_arg_t::__unsigned:
        case __fmt_arg_t::__unsigned_long_long:
          break;
        default:
          ::cuda::std::__throw_format_error("Replacement argument isn't a standard signed or unsigned integer type");
      }
    }
  }
}

template <class _CharT>
_CCCL_API constexpr void __fmt_validate_visit_format_arg(
  basic_format_parse_context<_CharT>& __parse_ctx, __fmt_validation_format_context<_CharT>& __ctx, __fmt_arg_t __type)
{
  switch (__type)
  {
    case __fmt_arg_t::__none:
      ::cuda::std::__throw_format_error("Invalid argument");
    case __fmt_arg_t::__boolean:
      return ::cuda::std::__fmt_validate_format_arg<_CharT, bool>(__parse_ctx, __ctx);
    case __fmt_arg_t::__char_type:
      return ::cuda::std::__fmt_validate_format_arg<_CharT, _CharT>(__parse_ctx, __ctx);
    case __fmt_arg_t::__int:
      return ::cuda::std::__fmt_validate_format_arg<_CharT, int>(__parse_ctx, __ctx);
    case __fmt_arg_t::__long_long:
      return ::cuda::std::__fmt_validate_format_arg<_CharT, long long>(__parse_ctx, __ctx);
    case __fmt_arg_t::__unsigned:
      return ::cuda::std::__fmt_validate_format_arg<_CharT, unsigned>(__parse_ctx, __ctx);
    case __fmt_arg_t::__unsigned_long_long:
      return ::cuda::std::__fmt_validate_format_arg<_CharT, unsigned long long>(__parse_ctx, __ctx);
    case __fmt_arg_t::__float:
      return ::cuda::std::__fmt_validate_format_arg<_CharT, float, true>(__parse_ctx, __ctx);
    case __fmt_arg_t::__double:
      return ::cuda::std::__fmt_validate_format_arg<_CharT, double, true>(__parse_ctx, __ctx);
    case __fmt_arg_t::__long_double:
#if _CCCL_HAS_LONG_DOUBLE()
      return ::cuda::std::__fmt_validate_format_arg<_CharT, long double, true>(__parse_ctx, __ctx);
#else // ^^^ _CCCL_HAS_LONG_DOUBLE() ^^^ / vvv _CCCL_HAS_LONG_DOUBLE() vvv
      ::cuda::std::__throw_format_error("long double is disabled");
#endif // ^^^ _CCCL_HAS_LONG_DOUBLE() ^^^
    case __fmt_arg_t::__const_char_type_ptr:
      return ::cuda::std::__fmt_validate_format_arg<_CharT, const _CharT*, true>(__parse_ctx, __ctx);
    case __fmt_arg_t::__string_view:
      return ::cuda::std::__fmt_validate_format_arg<_CharT, basic_string_view<_CharT>, true>(__parse_ctx, __ctx);
    case __fmt_arg_t::__ptr:
      return ::cuda::std::__fmt_validate_format_arg<_CharT, const void*>(__parse_ctx, __ctx);
    case __fmt_arg_t::__handle:
      ::cuda::std::__throw_format_error("Handle should use __fmt_validation_format_arg_handle");
    default:
      _CCCL_UNREACHABLE();
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FORMAT_VALIDATION_H
