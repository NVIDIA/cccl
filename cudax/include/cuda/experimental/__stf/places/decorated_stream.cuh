//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief CUDA stream with metadata: decorated_stream, stream ID helpers.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/assert.h>

#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda::experimental::stf
{
/** Sentinel for "no stream" / empty slot. Distinct from any value returned by cuStreamGetId. */
inline constexpr unsigned long long k_no_stream_id = static_cast<unsigned long long>(-1);

/**
 * @brief Returns the unique stream ID from the CUDA driver (cuStreamGetId).
 * @param stream A valid CUDA stream, or nullptr.
 * @return The stream's unique ID, or k_no_stream_id if stream is nullptr.
 */
inline unsigned long long get_stream_id(cudaStream_t stream)
{
  unsigned long long id = 0;
  cuda_safe_call(cuStreamGetId(reinterpret_cast<CUstream>(stream), &id));
  _CCCL_ASSERT(id != k_no_stream_id, "Internal error: cuStreamGetId returned k_no_stream_id");
  return id;
}

/**
 * @brief A class to store a CUDA stream along with metadata
 *
 * It contains
 *  - the stream itself,
 *  - the stream's unique ID from the CUDA driver (cuStreamGetId), or k_no_stream_id if no stream,
 *  - the device index in which the stream resides
 */
struct decorated_stream
{
  decorated_stream() = default;

  decorated_stream(cudaStream_t stream, unsigned long long id, int dev_id = -1)
      : stream(stream)
      , id(id)
      , dev_id(dev_id)
  {}

  /** Construct from stream only; id is from cuStreamGetId, dev_id is -1 (filled lazily when needed). */
  explicit decorated_stream(cudaStream_t stream)
      : stream(stream)
      , id(get_stream_id(stream))
      , dev_id(-1)
  {}

  cudaStream_t stream = nullptr;
  // Unique ID from cuStreamGetId (k_no_stream_id if no stream)
  unsigned long long id = k_no_stream_id;
  // Device in which this stream resides
  int dev_id = -1;
};
} // namespace cuda::experimental::stf
