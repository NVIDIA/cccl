//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STREAM_LET_VALUE
#define __CUDAX_EXECUTION_STREAM_LET_VALUE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/let_value.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
/////////////////////////////////////////////////////////////////////////////////
// let_value, let_error, let_stopped: customization for the stream scheduler
template <>
struct stream_domain::__apply_t<let_value_t>
{
  template <class _Sndr, class _Env>
  _CCCL_API auto operator()(_Sndr __sndr, const _Env& __env) const
  {
    static_assert(_CUDA_VSTD::__always_false_v<_Sndr>,
                  "The CUDA stream scheduler does not yet support the `let_value`, `let_error`, and `let_stopped` "
                  "algorithms.");
  }
};

template <>
struct stream_domain::__apply_t<let_error_t>
{
  template <class _Sndr, class _Env>
  _CCCL_API auto operator()(_Sndr __sndr, const _Env& __env) const
  {
    static_assert(_CUDA_VSTD::__always_false_v<_Sndr>,
                  "The CUDA stream scheduler does not yet support the `let_value`, `let_error`, and `let_stopped` "
                  "algorithms.");
  }
};

template <>
struct stream_domain::__apply_t<let_stopped_t>
{
  template <class _Sndr, class _Env>
  _CCCL_API auto operator()(_Sndr __sndr, const _Env& __env) const
  {
    static_assert(_CUDA_VSTD::__always_false_v<_Sndr>,
                  "The CUDA stream scheduler does not yet support the `let_value`, `let_error`, and `let_stopped` "
                  "algorithms.");
  }
};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_LET_VALUE
