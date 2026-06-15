//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_FORMATTED_SIZE_H
#define _CUDA_STD___FORMAT_FORMATTED_SIZE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__format/buffer.h>
#include <cuda/std/__format/format_args.h>
#include <cuda/std/__format/format_context.h>
#include <cuda/std/__format/format_parse_context.h>
#include <cuda/std/__format/format_string.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// A buffer that counts the number of insertions.
//
// Since formatted_size only needs to know the size, the output itself is
// discarded.
template <class _CharT>
class __fmt_formatted_size_buffer : __fmt_output_buffer<_CharT>
{
  __fmt_max_output_size __max_output_size_{0};

  _CCCL_API static constexpr void __prepare_write([[maybe_unused]] __fmt_output_buffer<_CharT>&, [[maybe_unused]] size_t)
  {
    // Note this function does not satisfy the requirement of giving a 1 code unit buffer.
    _CCCL_ASSERT(false, "Since __max_output_size_.__max_size_ == 0 there should never be call to this function.");
  }

public:
  using _Base _CCCL_NODEBUG_ALIAS = __fmt_output_buffer<_CharT>;

  _CCCL_API constexpr __fmt_formatted_size_buffer()
      : _Base{nullptr, 0, __prepare_write, ::cuda::std::addressof(__max_output_size_)}
  {}

  [[nodiscard]] _CCCL_API constexpr auto __make_output_iterator()
  {
    return _Base::__make_output_iterator();
  }

  // This function does not need to be r-value qualified, however this is
  // consistent with similar objects.
  [[nodiscard]] _CCCL_API constexpr size_t __result() &&
  {
    return __max_output_size_.__code_units_written();
  }
};

template <class _CharT, class _FmtArgs>
[[nodiscard]] _CCCL_API size_t __formatted_size_impl(basic_string_view<_CharT> __fmt, _FmtArgs __args)
{
  __fmt_formatted_size_buffer<_CharT> __buffer;
  (void) ::cuda::std::__fmt_vformat_to(
    basic_format_parse_context{__fmt, __args.__size()},
    ::cuda::std::__fmt_make_format_context(__buffer.__make_output_iterator(), __args));
  return ::cuda::std::move(__buffer).__result();
}

template <class... _Args>
[[nodiscard]] _CCCL_API size_t formatted_size(format_string<_Args...> __fmt, _Args&&... __args)
{
  return ::cuda::std::__formatted_size_impl(__fmt.get(), basic_format_args{::cuda::std::make_format_args(__args...)});
}

#if _CCCL_HAS_WCHAR_T()
template <class... _Args>
[[nodiscard]] _CCCL_API size_t formatted_size(wformat_string<_Args...> __fmt, _Args&&... __args)
{
  return ::cuda::std::__formatted_size_impl(__fmt.get(), basic_format_args{::cuda::std::make_wformat_args(__args...)});
}
#endif // _CCCL_HAS_WCHAR_T()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMATTED_SIZE_H
