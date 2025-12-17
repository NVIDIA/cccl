// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/device_memory_resource.cuh>
#include <cub/detail/temporary_storage.cuh>

#include <cuda/__execution/tune.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__execution/env.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! @cond
//! Generic environment-based algorithm dispatch wrapper
//!
//! Handles common boilerplate for all env-based algorithms:
//! - Query stream, memory resource, and tuning from environment
//! - Two-phase call (query temp storage size, then execute)
//! - Temporary storage allocation/deallocation
//! - Memory resource querying from environment
//!
//! @param env The execution environment
//! @param algorithm_callable Callable that invokes the algorithm implementation with determinism specified
template <typename EnvT, typename AlgorithmCallable>
CUB_RUNTIME_FUNCTION static cudaError_t dispatch_with_env(EnvT env, AlgorithmCallable&& algorithm_callable)
{
  // Query stream from environment
  auto stream = ::cuda::std::execution::__query_or(env, ::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}});

  // Query memory resource from environment
  auto mr =
    ::cuda::std::execution::__query_or(env, ::cuda::mr::__get_memory_resource, detail::device_memory_resource{});

  // Query tuning from environment

  // clang 15-19 ICE when using __query_result_or_t, so we just use the underlying machinery as a workaround
#if _CCCL_COMPILER(CLANG, >=, 15) && _CCCL_COMPILER(CLANG, <, 19)
  const auto tuning = ::cuda::std::execution::__detail::__query_or_t{}(
    env, ::cuda::execution::__get_tuning_t{}, ::cuda::std::execution::env<>{});
#else
  const auto tuning =
    ::cuda::std::execution::__query_result_or_t<EnvT, ::cuda::execution::__get_tuning_t, ::cuda::std::execution::env<>>{};
#endif

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;

  // Phase 1: Query temporary storage size
  if (const auto error = algorithm_callable(tuning, d_temp_storage, temp_storage_bytes, stream.get()))
  {
    return error;
  }

  // Allocate temporary storage
  if (const auto error = CubDebug(detail::temporary_storage::allocate(stream, d_temp_storage, temp_storage_bytes, mr)))
  {
    return error;
  }

  // Phase 2: Execute algorithm
  const auto error = algorithm_callable(tuning, d_temp_storage, temp_storage_bytes, stream.get());

  // Deallocate temporary storage (always attempt, even on error)
  const auto deallocate_error =
    CubDebug(detail::temporary_storage::deallocate(stream, d_temp_storage, temp_storage_bytes, mr));

  // Algorithm error takes precedence over deallocation error
  return (error != cudaSuccess) ? error : deallocate_error;
}
//! @endcond
} // namespace detail

CUB_NAMESPACE_END
