//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_ENSURE_CURRENT_CONTEXT_H
#define _CUDA_STD___PSTL_CUDA_ENSURE_CURRENT_CONTEXT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/__device/device_ref.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__runtime/ensure_current_context.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/std/__type_traits/is_callable.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

template <class _Policy>
[[nodiscard]] _CCCL_HOST_API __ensure_current_context __pstl_ensure_current_ctx_for(const _Policy& __policy)
{
  if constexpr (__is_callable_v<get_stream_t, const _Policy&>)
  {
    return __ensure_current_context{get_stream(__policy)};
  }
  else
  {
    int __curr_device{};
    _CCCL_TRY_CUDA_API(::cudaGetDevice, "Failed to get current device", &__curr_device);
    return __ensure_current_context{device_ref{__curr_device}};
  }
}

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif /// _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_ENSURE_CURRENT_CONTEXT_H
