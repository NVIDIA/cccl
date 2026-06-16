//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_FORMAT_TO_N_H
#define _CUDA_STD___FORMAT_FORMAT_TO_N_H

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
#include <cuda/std/__format/vformat.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/ctad_support.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _OutIt>
struct _CCCL_TYPE_VISIBILITY_DEFAULT format_to_n_result
{
  _OutIt out;
  iter_difference_t<_OutIt> size;
};
_CCCL_CTAD_SUPPORTED_FOR_TYPE(format_to_n_result);

// A buffer that counts and limits the number of insertions.
template <class _OutIt, class _CharT>
class __fmt_format_to_n_buffer : __fmt_buffer_select_t<_OutIt, _CharT>
{
  __fmt_max_output_size __max_output_size_;

public:
  using _Base _CCCL_NODEBUG_ALIAS = __fmt_buffer_select_t<_OutIt, _CharT>;

  _CCCL_API constexpr __fmt_format_to_n_buffer(_OutIt __out_it, iter_difference_t<_OutIt> __n)
      : _Base{::cuda::std::move(__out_it), ::cuda::std::addressof(__max_output_size_)}
      , __max_output_size_{::cuda::std::cmp_less(__n, 0) ? size_t{0} : static_cast<size_t>(__n)}
  {}

  [[nodiscard]] _CCCL_API constexpr auto __make_output_iterator()
  {
    return _Base::__make_output_iterator();
  }

  [[nodiscard]] _CCCL_API constexpr format_to_n_result<_OutIt> __result() &&
  {
    return {static_cast<_Base&&>(*this).__out_it(),
            static_cast<iter_difference_t<_OutIt>>(__max_output_size_.__code_units_written())};
  }
};

template <class _Context, class _OutIt, class _CharT>
[[nodiscard]] _CCCL_API format_to_n_result<_OutIt> __format_to_n_impl(
  _OutIt __out_it, iter_difference_t<_OutIt> __n, basic_string_view<_CharT> __fmt, basic_format_args<_Context> __args)
{
  __fmt_format_to_n_buffer<_OutIt, _CharT> __buffer{::cuda::std::move(__out_it), __n};
  (void) ::cuda::std::__fmt_vformat_to(
    basic_format_parse_context{__fmt, __args.__size()},
    ::cuda::std::__fmt_make_format_context(__buffer.__make_output_iterator(), __args));
  return ::cuda::std::move(__buffer).__result();
}

_CCCL_TEMPLATE(class _OutIt, class... _Args)
_CCCL_REQUIRES(output_iterator<_OutIt, const char&>)
/*discard*/ _CCCL_API format_to_n_result<_OutIt>
format_to_n(_OutIt __out_it, iter_difference_t<_OutIt> __n, format_string<_Args...> __fmt, _Args&&... __args)
{
  return ::cuda::std::__format_to_n_impl<format_context>(
    ::cuda::std::move(__out_it), __n, __fmt.get(), ::cuda::std::make_format_args(__args...));
}

#if _CCCL_HAS_WCHAR_T()
_CCCL_TEMPLATE(class _OutIt, class... _Args)
_CCCL_REQUIRES(output_iterator<_OutIt, const wchar_t&>)
/*discard*/ _CCCL_API format_to_n_result<_OutIt>
format_to_n(_OutIt __out_it, iter_difference_t<_OutIt> __n, wformat_string<_Args...> __fmt, _Args&&... __args)
{
  return ::cuda::std::__format_to_n_impl<wformat_context>(
    ::cuda::std::move(__out_it), __n, __fmt.get(), ::cuda::std::make_wformat_args(__args...));
}
#endif // _CCCL_HAS_WCHAR_T()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMAT_TO_N_H
