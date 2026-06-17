//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_FORMAT_H
#define _CUDA_STD___FORMAT_FORMAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__format/format_string.h>
#include <cuda/std/__format/vformat.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_TEMPLATE(class _OutIt, class... _Args)
_CCCL_REQUIRES(output_iterator<_OutIt, const char&>)
/*discard*/ _CCCL_API _OutIt format_to(_OutIt __out_it, format_string<_Args...> __fmt, _Args&&... __args)
{
  return ::cuda::std::vformat_to(::cuda::std::move(__out_it), __fmt.get(), ::cuda::std::make_format_args(__args...));
}

#if _CCCL_HAS_WCHAR_T()
_CCCL_TEMPLATE(class _OutIt, class... _Args)
_CCCL_REQUIRES(output_iterator<_OutIt, const wchar_t&>)
/*discard*/ _CCCL_API _OutIt format_to(_OutIt __out_it, wformat_string<_Args...> __fmt, _Args&&... __args)
{
  return ::cuda::std::vformat_to(::cuda::std::move(__out_it), __fmt.get(), ::cuda::std::make_wformat_args(__args...));
}
#endif // _CCCL_HAS_WCHAR_T()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMAT_H
