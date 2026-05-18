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
 * @brief Is this stream currently in graph-capture mode?
 *
 * Most CUDA driver metadata queries (``cuStreamGetId``, ``cudaStreamGetDevice``,
 * ...) are not capture-safe: under ``cudaStreamCaptureModeThreadLocal`` /
 * ``Global`` the driver both rejects them with
 * ``cudaErrorStreamCaptureUnsupported`` *and* invalidates the in-progress
 * capture. We therefore gate such queries on this probe, which is itself
 * capture-safe.
 */
inline bool is_stream_capturing(cudaStream_t stream)
{
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cuda_try(cudaStreamIsCapturing(stream, &status));
  return status != cudaStreamCaptureStatusNone;
}

/**
 * @brief Computes the CUDA device in which the stream was created
 */
inline int get_device_from_stream(cudaStream_t stream)
{
  if (stream == nullptr)
  {
    return cuda_try<cudaGetDevice>();
  }

  // If the stream is currently capturing, ``cudaStreamGetDevice`` /
  // ``cuStreamGetCtx`` are not allowed and would invalidate the capture.
  // Fall back to the current device: STF's own stream pool is allocated
  // against the caller's active device context, and a user stream passed in
  // while that context is captured is assumed to live on that same device.
  if (is_stream_capturing(stream))
  {
    return cuda_try<cudaGetDevice>();
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

  // ``cuStreamGetId`` is not capture-safe: during
  // ``cudaStreamCaptureModeThreadLocal`` / ``Global`` it rejects the query
  // *and* invalidates the capture itself. Gate on ``cudaStreamIsCapturing``
  // (which is safe) and use the ``cudaStream_t`` pointer value as a stable,
  // unique-per-process stream identifier while capture is in flight. STF only
  // uses this ID to key its internal per-stream tracking; the pointer is just
  // as suitable as ``cuStreamGetId``'s nonce for that purpose, and cannot
  // collide with a valid ID because ``k_no_stream_id`` is ``~0ULL``.
  if (is_stream_capturing(stream))
  {
    // return static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(stream));
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
        , externally_owned(true)
    {}

    // Release every stream the pool has lazily created. We intentionally
    // skip entries that came from an externally-owned `decorated_stream`
    // (single-stream pool built from a user-supplied stream, used for
    // `exec_place::cuda_stream(s)`); those are not ours to destroy.
    //
    // `cudaStreamDestroy` is documented to be asynchronous when work is
    // still pending on the stream: the call returns immediately and CUDA
    // releases the stream's resources once the device has completed its
    // pending work. That contract is what makes it safe to tear the pool
    // down at the end of an STF context without blocking on a caller stream
    // synchronize, as long as the outbound event chain has already been
    // recorded back onto the user stream.
    ~impl()
    {
      if (externally_owned)
      {
        return;
      }
      for (auto& ds : payload)
      {
        if (ds.stream != nullptr)
        {
          // Best-effort: never throw from a destructor, and never crash a
          // process that has already torn down its CUDA primary context
          // (e.g. after `cudaDeviceReset()`).
          [[maybe_unused]] cudaError_t err = cudaStreamDestroy(ds.stream);
          ds.stream                        = nullptr;
        }
      }
    }

    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

    mutable ::std::mutex mtx;
    ::std::vector<decorated_stream> payload;
    size_t index          = 0;
    bool externally_owned = false;
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
