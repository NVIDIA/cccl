//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_FORMAT_ARGS_H
#define _CUDA_STD___FORMAT_FORMAT_ARGS_H

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
#include <cuda/std/__fwd/format.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Context>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_args
{
public:
  template <class... _Args>
  _CCCL_API basic_format_args(const __format_arg_store<_Context, _Args...>& __store) noexcept
      : __size_(sizeof...(_Args))
  {
    if constexpr (sizeof...(_Args) != 0)
    {
      if constexpr (::cuda::std::__fmt_use_packed_format_arg_store(sizeof...(_Args)))
      {
        __packed_.__values = __store.__storage.__values_;
        __packed_.__types  = __store.__storage.__types_;
      }
      else
      {
        __unpacked_.__args = __store.__storage.__args_;
      }
    }
  }

  [[nodiscard]] _CCCL_API basic_format_arg<_Context> get(size_t __id) const noexcept
  {
    if (__id >= __size_)
    {
      return basic_format_arg<_Context>{};
    }

    if (::cuda::std::__fmt_use_packed_format_arg_store(__size_))
    {
      return basic_format_arg<_Context>{
        ::cuda::std::__fmt_get_packed_type(__packed_.__types, __id), __packed_.__values[__id]};
    }

    return __unpacked_.__args[__id];
  }

  [[nodiscard]] _CCCL_API size_t __size() const noexcept
  {
    return __size_;
  }

private:
  size_t __size_{0};
  // [format.args]/5
  // [Note 1: Implementations are encouraged to optimize the representation of
  // basic_format_args for small number of formatting arguments by storing
  // indices of type alternatives separately from values and packing the
  // former. - end note]

  struct _Packed
  {
    const __basic_format_arg_value<_Context>* __values;
    uint64_t __types;
  };
  struct _Unpacked
  {
    const basic_format_arg<_Context>* __args;
  };

  union
  {
    _Packed __packed_;
    _Unpacked __unpacked_;
  };
};

template <class _Context, class... _Args>
_CCCL_HOST_DEVICE basic_format_args(__format_arg_store<_Context, _Args...>) -> basic_format_args<_Context>;

template <class _Context = format_context, class... _Args>
[[nodiscard]] _CCCL_API __format_arg_store<_Context, _Args...> make_format_args(_Args&... __args)
{
  return __format_arg_store<_Context, _Args...>(__args...);
}

#if _CCCL_HAS_WCHAR_T()
template <class... _Args>
[[nodiscard]] _CCCL_API __format_arg_store<wformat_context, _Args...> make_wformat_args(_Args&... __args)
{
  return __format_arg_store<wformat_context, _Args...>(__args...);
}
#endif // _CCCL_HAS_WCHAR_T()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMAT_ARGS_H
