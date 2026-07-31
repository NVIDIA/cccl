//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_FORMAT_CONTEXT_H
#define _CUDA_STD___FORMAT_FORMAT_CONTEXT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__format/buffer.h>
#include <cuda/std/__format/format_arg.h>
#include <cuda/std/__format/format_arg_store.h>
#include <cuda/std/__format/format_error.h>
#include <cuda/std/__format/formatter.h>
#include <cuda/std/__fwd/format.h>
#include <cuda/std/__iterator/back_insert_iterator.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__utility/ctad_support.h>
#include <cuda/std/__utility/monostate.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Since CCCL doesn't support localization, we don't implement the locale() method
template <class _OutIt, class _CharT>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_context
{
  static_assert(output_iterator<_OutIt, const _CharT&>, "_OutIt must be an output iterator for const _CharT&");

public:
  using iterator  = _OutIt;
  using char_type = _CharT;
  template <class _Tp>
  using formatter_type = formatter<_Tp, _CharT>;

  basic_format_context(const basic_format_context&)            = delete;
  basic_format_context(basic_format_context&&)                 = delete;
  basic_format_context& operator=(const basic_format_context&) = delete;
  basic_format_context& operator=(basic_format_context&&)      = delete;

  [[nodiscard]] _CCCL_HOST_DEVICE_API basic_format_arg<basic_format_context> arg(size_t __id) const noexcept
  {
    return __args_.get(__id);
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API iterator out()
  {
    return ::cuda::std::move(__out_it_);
  }
  _CCCL_HOST_DEVICE_API void advance_to(iterator __it)
  {
    __out_it_ = ::cuda::std::move(__it);
  }

  template <class _OtherOutIt, class _OtherCharT>
  _CCCL_HOST_DEVICE_API friend basic_format_context<_OtherOutIt, _OtherCharT>
    __fmt_make_format_context(_OtherOutIt, basic_format_args<basic_format_context<_OtherOutIt, _OtherCharT>>);

private:
  _CCCL_HOST_DEVICE_API explicit basic_format_context(_OutIt __out_it, basic_format_args<basic_format_context> __args)
      : __out_it_(::cuda::std::move(__out_it))
      , __args_(__args)
  {}

  iterator __out_it_;
  basic_format_args<basic_format_context> __args_;
};

// A specialization for __retarget_buffer
//
// See __retarget_buffer for the motivation for this specialization.
//
// This context holds a reference to the instance of the basic_format_context
// that is retargeted. It converts a formatting argument when it is requested
// during formatting. It is expected that the usage of the arguments is rare so
// the lookups are not expected to be used often. An alternative would be to
// convert all elements during construction.
//
// The elements of the retargets context are only used when an underlying
// formatter uses a locale specific formatting or an formatting argument is
// part for the format spec. For example
//   format("{:256:{}}", input, 8);
// Here the width of an element in input is determined dynamically.
// Note when the top-level element has no width the retargeting is not needed.
template <class _CharT>
class basic_format_context<typename __fmt_retarget_buffer<_CharT>::__iterator, _CharT>
{
  typename __fmt_retarget_buffer<_CharT>::__iterator __out_it_;
  void* __ctx_;
  basic_format_arg<basic_format_context> (*__arg_)(void* __ctx, size_t __id);

  template <class _Context>
  [[nodiscard]] _CCCL_API static basic_format_arg<basic_format_context> __arg_impl(void* __ctx, size_t __id)
  {
    auto __visitor = [&](auto __arg) -> basic_format_arg<basic_format_context> {
      if constexpr (same_as<decltype(__arg), monostate>)
      {
        return {};
      }
      else if constexpr (same_as<decltype(__arg), typename basic_format_arg<_Context>::handle>)
      {
        // At the moment it's not possible for formatting to use a re-targeted handle.
        // TODO FMT add this when support is needed.
        ::cuda::std::__throw_format_error("Re-targeting handle not supported");
      }
      else
      {
        return basic_format_arg<basic_format_context>{
          ::cuda::std::__fmt_determine_arg_t<basic_format_context, decltype(__arg)>(),
          __basic_format_arg_value<basic_format_context>(__arg)};
      }
    };
    return ::cuda::std::visit_format_arg(::cuda::std::move(__visitor), static_cast<_Context*>(__ctx)->arg(__id));
  }

public:
  using iterator  = typename __fmt_retarget_buffer<_CharT>::__iterator;
  using char_type = _CharT;
  template <class _Tp>
  using formatter_type = formatter<_Tp, _CharT>;

  template <class _Context>
  _CCCL_API explicit basic_format_context(iterator __out_it, _Context& __ctx)
      : __out_it_{::cuda::std::move(__out_it)}
      , __ctx_{::cuda::std::addressof(__ctx)}
      , __arg_{__arg_impl<_Context>}
  {}

  [[nodiscard]] _CCCL_API basic_format_arg<basic_format_context> arg(size_t __id) const
  {
    return __arg_(__ctx_, __id);
  }
  [[nodiscard]] _CCCL_API iterator out()
  {
    return ::cuda::std::move(__out_it_);
  }
  _CCCL_API void advance_to(iterator __it)
  {
    __out_it_ = ::cuda::std::move(__it);
  }
};

_CCCL_CTAD_SUPPORTED_FOR_TYPE(basic_format_context);

template <class _OutIt, class _CharT>
[[nodiscard]] _CCCL_HOST_DEVICE_API basic_format_context<_OutIt, _CharT>
__fmt_make_format_context(_OutIt __out_it, basic_format_args<basic_format_context<_OutIt, _CharT>> __args)
{
  return basic_format_context{::cuda::std::move(__out_it), __args};
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMAT_CONTEXT_H
