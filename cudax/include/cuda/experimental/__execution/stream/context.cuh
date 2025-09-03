//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STREAM_CONTEXT_IMPL
#define __CUDAX_EXECUTION_STREAM_CONTEXT_IMPL

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/device_ref.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__utility/immovable.h>

#include <cuda/experimental/__execution/stream/scheduler.cuh>
#include <cuda/experimental/stream.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
//////////////////////////////////////////////////////////////////////////////////////////
// stream_context
struct _CCCL_TYPE_VISIBILITY_DEFAULT stream_context : private __immovable
{
  _CCCL_HOST_API explicit stream_context(device_ref __device)
      : __stream_{__device}
  {}

  _CCCL_HOST_API void sync() noexcept
  {
    __stream_.sync();
  }

  [[nodiscard]] _CCCL_HOST_API constexpr auto query(get_stream_t) const noexcept -> stream_ref
  {
    return __stream_;
  }

  [[nodiscard]] _CCCL_NODEBUG_HOST_API auto get_scheduler() noexcept -> stream_scheduler
  {
    return stream_scheduler{__stream_};
  }

private:
  stream __stream_;
};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_CONTEXT_IMPL
