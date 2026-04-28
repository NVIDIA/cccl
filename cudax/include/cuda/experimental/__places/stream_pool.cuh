//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Stream pool and decorated stream types used by places.
 *
 * Definitions of stream_pool::next() live in places.cuh (because we need to
 * activate the place to create a stream in the appropriate CUDA context).
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif

#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

#include <memory>
#include <mutex>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda::experimental::places
{
using ::cuda::experimental::stf::cuda_try;
using ::cuda::experimental::stf::mv;

class exec_place;

/**
 * @brief Computes the CUDA device in which the stream was created
 */
inline int get_device_from_stream(cudaStream_t stream)
{
  if (stream == nullptr)
  {
    int device = 0;
    cuda_try(cudaGetDevice(&device));
    return device;
  }

  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  const cudaError_t cap_err              = cudaStreamIsCapturing(stream, &capture_status);
  if (cap_err == cudaSuccess && capture_status != cudaStreamCaptureStatusNone)
  {
    // cudaStreamGetDevice/cuStreamGetCtx are not permitted while the stream is
    // participating in capture. Use the active device, which is the device on
    // which the capture is being constructed.
    int device = 0;
    cuda_try(cudaGetDevice(&device));
    return device;
  }

#if _CCCL_CTK_AT_LEAST(12, 8)
  int device = 0;
  cuda_try(cudaStreamGetDevice(stream, &device));
  return device;
#else
  auto stream_driver = CUstream(stream);

  CUcontext ctx;
  cuda_try(cuStreamGetCtx(stream_driver, &ctx));

  cuda_try(cuCtxPushCurrent(ctx));
  CUdevice stream_dev = cuda_try<cuCtxGetDevice>();
  static_cast<void>(cuda_try<cuCtxPopCurrent>());

  return static_cast<int>(stream_dev);
#endif
}

/** Sentinel for "no stream" / empty slot. Distinct from any value returned by cuStreamGetId. */
inline constexpr unsigned long long k_no_stream_id = static_cast<unsigned long long>(-1);

/**
 * @brief Returns the unique stream ID from the CUDA driver (cuStreamGetId).
 * @param stream A valid CUDA stream, or nullptr.
 * @return The stream's unique ID, or k_no_stream_id if stream is nullptr.
 *
 * When @p stream is participating in CUDA graph capture, querying the driver
 * stream ID is not permitted (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED). In that case
 * this returns k_no_stream_id; callers that cache syncs treat unknown ids as
 * ``never skip cudaStreamWaitEvent`` (see async_resources_handle).
 */
inline unsigned long long get_stream_id(cudaStream_t stream)
{
  if (stream == nullptr)
  {
    return k_no_stream_id;
  }
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  const cudaError_t cap_err              = cudaStreamIsCapturing(stream, &capture_status);
  if (cap_err == cudaSuccess && capture_status != cudaStreamCaptureStatusNone)
  {
    return k_no_stream_id;
  }
  unsigned long long id = 0;
  cuda_try(cuStreamGetId(reinterpret_cast<CUstream>(stream), &id));
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

  cudaStream_t stream   = nullptr;
  unsigned long long id = k_no_stream_id;
  int dev_id            = -1;
};

/**
 * @brief A stream_pool object stores a set of streams associated to a specific
 * CUDA context (device, green context, ...)
 *
 * This class uses a PIMPL idiom so that it is copyable and movable with shared
 * semantics: copies refer to the same underlying pool of streams.
 *
 * When a slot is empty, next(place) activates the place (RAII guard) and calls
 * place.create_stream(). Defined in places.cuh.
 */
class stream_pool
{
  struct impl
  {
    explicit impl(size_t n)
        : payload(n, decorated_stream(nullptr, k_no_stream_id, -1))
    {}

    // Construct from a decorated stream, this is used to create a stream pool with a single stream.
    explicit impl(decorated_stream ds)
        : payload(1, mv(ds))
    {}

    mutable ::std::mutex mtx;
    ::std::vector<decorated_stream> payload;
    size_t index = 0;
  };

  ::std::shared_ptr<impl> pimpl;

public:
  stream_pool() = default;

  explicit stream_pool(size_t n)
      : pimpl(::std::make_shared<impl>(n))
  {}

  // Construct from a decorated stream, this is used to create a stream pool with a single stream.
  explicit stream_pool(decorated_stream ds)
      : pimpl(::std::make_shared<impl>(mv(ds)))
  {}

  stream_pool(const stream_pool&)            = default;
  stream_pool(stream_pool&&)                 = default;
  stream_pool& operator=(const stream_pool&) = default;
  stream_pool& operator=(stream_pool&&)      = default;

  /**
   * @brief Get the next stream in the pool; when a slot is empty, activate the place (RAII guard) and call
   * place.create_stream(). Defined in places.cuh so the pool can use exec_place_scope and exec_place::create_stream().
   */
  decorated_stream next(const exec_place& place);

  using iterator = ::std::vector<decorated_stream>::iterator;
  iterator begin()
  {
    return pimpl->payload.begin();
  }
  iterator end()
  {
    return pimpl->payload.end();
  }

  /**
   * @brief Number of streams in the pool
   *
   * CUDA streams are initialized lazily, so this gives the number of slots
   * available in the pool, not the number of streams initialized.
   */
  size_t size() const
  {
    ::std::lock_guard<::std::mutex> locker(pimpl->mtx); // NOLINT(modernize-use-scoped-lock)
    return pimpl->payload.size();
  }

  explicit operator bool() const
  {
    return pimpl != nullptr;
  }

  bool operator==(const stream_pool& other) const
  {
    return pimpl == other.pimpl;
  }

  bool operator<(const stream_pool& other) const
  {
    return pimpl < other.pimpl;
  }
};
} // namespace cuda::experimental::places
