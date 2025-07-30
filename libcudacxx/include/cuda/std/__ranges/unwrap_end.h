// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_UNWRAP_SENTINEL_H
#define _LIBCUDACXX___RANGES_UNWRAP_SENTINEL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

_CCCL_TEMPLATE(class _Range)
_CCCL_REQUIRES(forward_range<_Range>)
[[nodiscard]] _CCCL_API constexpr iterator_t<_Range> __unwrap_end(_Range& __range)
{
  if constexpr (common_range<_Range>)
  {
    return _CUDA_VRANGES::end(__range);
  }
  else
  {
    auto __ret = _CUDA_VRANGES::begin(__range);
    _CUDA_VRANGES::advance(__ret, _CUDA_VRANGES::end(__range));
    return __ret;
  }
  _CCCL_UNREACHABLE();
}

_LIBCUDACXX_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANGES_UNWRAP_SENTINEL_H
