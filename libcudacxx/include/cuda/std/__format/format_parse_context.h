//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_FORMAT_PARSE_CONTEXT_H
#define _CUDA_STD___FORMAT_FORMAT_PARSE_CONTEXT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__format/format_error.h>
#include <cuda/std/__fwd/format.h>
#include <cuda/std/__utility/ctad_support.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _CharT>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_parse_context
{
public:
  using char_type      = _CharT;
  using const_iterator = typename basic_string_view<_CharT>::const_iterator;
  using iterator       = const_iterator;

  _CCCL_API constexpr explicit basic_format_parse_context(
    basic_string_view<_CharT> __fmt, size_t __num_args = 0) noexcept
      : __begin_(__fmt.begin())
      , __end_(__fmt.end())
      , __indexing_(_Indexing::__unknown)
      , __next_arg_id_(0)
      , __num_args_(__num_args)
  {}

  basic_format_parse_context(const basic_format_parse_context&)            = delete;
  basic_format_parse_context(basic_format_parse_context&&)                 = delete;
  basic_format_parse_context& operator=(const basic_format_parse_context&) = delete;
  basic_format_parse_context& operator=(basic_format_parse_context&&)      = delete;

  [[nodiscard]] _CCCL_API constexpr const_iterator begin() const noexcept
  {
    return __begin_;
  }

  [[nodiscard]] _CCCL_API constexpr const_iterator end() const noexcept
  {
    return __end_;
  }

  _CCCL_API constexpr void advance_to(const_iterator __it)
  {
    __begin_ = __it;
  }

  [[nodiscard]] _CCCL_API constexpr size_t next_arg_id()
  {
    if (__indexing_ == _Indexing::__manual)
    {
      ::cuda::std::__throw_format_error("using automatic argument numbering in manual argument numbering mode");
    }
    if (__indexing_ == _Indexing::__unknown)
    {
      __indexing_ = _Indexing::__automatic;
    }
    _CCCL_IF_CONSTEVAL
    {
      _CCCL_VERIFY(__next_arg_id_ < __num_args_, "argument index outside the valid range");
    }
    return __next_arg_id_++;
  }

  _CCCL_API constexpr void check_arg_id(size_t __id)
  {
    if (__indexing_ == _Indexing::__automatic)
    {
      ::cuda::std::__throw_format_error("using manual argument numbering in automatic argument numbering mode");
    }
    if (__indexing_ == _Indexing::__unknown)
    {
      __indexing_ = _Indexing::__manual;
    }
    _CCCL_IF_CONSTEVAL
    {
      _CCCL_VERIFY(__id < __num_args_, "argument index outside the valid range");
    }
  }

private:
  enum class _Indexing
  {
    __unknown,
    __manual,
    __automatic
  };

  iterator __begin_;
  iterator __end_;
  _Indexing __indexing_;
  size_t __next_arg_id_;
  size_t __num_args_;
};

_CCCL_CTAD_SUPPORTED_FOR_TYPE(basic_format_parse_context);

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMAT_PARSE_CONTEXT_H
