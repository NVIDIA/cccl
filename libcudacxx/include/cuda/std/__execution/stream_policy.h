//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___EXECUTION_STREAM_POLICY_H
#define _CUDA_STD___EXECUTION_STREAM_POLICY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__fwd/policy.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_EXECUTION

//! @brief Wrapper around an execution policy to store a stream
template <class _Policy>
class __execution_policy_stream : public _Policy
{
private:
  ::cuda::stream_ref __stream_;

public:
  _CCCL_HOST_API __execution_policy_stream(::cuda::stream_ref __stream) noexcept
      : __stream_(__stream)
  {}

  [[nodiscard]] _CCCL_HOST_API ::cuda::stream_ref get_stream() const noexcept
  {
    return __stream_;
  }

  // We want to ensure that if there are nested `set_stream` calls we only wrap it once
  [[nodiscard]] _CCCL_HOST_API __execution_policy_stream set_stream(::cuda::stream_ref __stream) noexcept
  {
    __stream_ = __stream;
    return *this;
  }
};

_CCCL_END_NAMESPACE_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___EXECUTION_STREAM_POLICY_H
