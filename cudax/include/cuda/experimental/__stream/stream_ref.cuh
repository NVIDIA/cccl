//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__STREAM_STREAM_REF
#define _CUDAX__STREAM_STREAM_REF

#include <cuda/std/detail/__config>
#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/all_devices.h>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/stream_ref>

#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__event/timed_event.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>

#include <cuda_runtime_api.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

namespace __detail
{
// 0 is a valid stream in CUDA, so we need some other invalid stream representation
// Can't make it constexpr, because cudaStream_t is a pointer type
static const ::cudaStream_t __invalid_stream = reinterpret_cast<cudaStream_t>(~0ULL);
} // namespace __detail

//! @brief A type representing a stream ID.
enum class stream_id : unsigned long long
{
};

//! @brief A non-owning wrapper for cudaStream_t.
//!
//! @note It is undefined behavior to use a `stream_ref` object beyond the lifetime of the stream it was created from,
//! except for the `get()` member function.
struct stream_ref : ::cuda::stream_ref
{
  using scheduler_concept = execution::scheduler_t;

  stream_ref() = delete;

  //! @brief Wrap a native \c ::cudaStream_t in a \c stream_ref
  //!
  //! @post `this->get() == __stream`
  _CCCL_HOST_API constexpr stream_ref(value_type __stream) noexcept
      : ::cuda::stream_ref{__stream}
  {}

  //! @brief Converting constructor from \c ::cuda::stream_ref
  //!
  //! @post `*this == __other`
  _CCCL_HOST_API constexpr stream_ref(const ::cuda::stream_ref& __other) noexcept
      : ::cuda::stream_ref(__other)
  {}

  /// Disallow construction from an `int`, e.g., `0`.
  stream_ref(int) = delete;

  /// Disallow construction from `nullptr`.
  stream_ref(_CUDA_VSTD::nullptr_t) = delete;

  //! \brief Queries if all operations on the stream have completed.
  //!
  //! \throws cuda::cuda_error if the query fails.
  //!
  //! \return `true` if all operations have completed, or `false` if not.
  [[nodiscard]] bool is_done() const
  {
    const auto __result = _CUDA_DRIVER::__streamQueryNoThrow(__stream);
    switch (__result)
    {
      case ::cudaErrorNotReady:
        return false;
      case ::cudaSuccess:
        return true;
      default:
        ::cuda::__throw_cuda_error(__result, "Failed to query stream.");
    }
  }

  //! @brief Deprecated. Use is_done() instead.
  [[deprecated("Use is_done() instead.")]] [[nodiscard]] bool ready() const
  {
    return is_done();
  }

  //! @brief Get the priority of the stream
  //!
  //! @return The priority of the stream
  //!
  //! @throws cuda_error if the priority query fails
  [[nodiscard]] _CCCL_HOST_API int priority() const
  {
    return _CUDA_DRIVER::__streamGetPriority(__stream);
  }

  //! @brief Get the unique ID of the stream
  //!
  //! Stream handles are sometimes reused, but ID is guaranteed to be unique.
  //!
  //! @return The unique ID of the stream
  //!
  //! @throws cuda_error if the ID query fails
  [[nodiscard]] _CCCL_HOST_API stream_id id() const
  {
    return stream_id{_CUDA_DRIVER::__streamGetId(__stream)};
  }

  //! @brief Create a new event and record it into this stream
  //!
  //! @return A new event that was recorded into this stream
  //!
  //! @throws cuda_error if event creation or record failed
  [[nodiscard]] _CCCL_HOST_API event record_event(event::flags __flags = event::flags::none) const
  {
    return event(*this, __flags);
  }

  //! @brief Create a new timed event and record it into this stream
  //!
  //! @return A new timed event that was recorded into this stream
  //!
  //! @throws cuda_error if event creation or record failed
  [[nodiscard]] _CCCL_HOST_API timed_event record_timed_event(event::flags __flags = event::flags::none) const
  {
    return timed_event(*this, __flags);
  }

  //! @brief Synchronize the stream
  //!
  //! @throws cuda_error if synchronization fails
  _CCCL_HOST_API void sync() const
  {
    _CUDA_DRIVER::__streamSynchronize(__stream);
  }

  //! @brief Make all future work submitted into this stream depend on completion of the specified event
  //!
  //! @param __ev Event that this stream should wait for
  //!
  //! @throws cuda_error if inserting the dependency fails
  _CCCL_HOST_API void wait(event_ref __ev) const
  {
    _CCCL_ASSERT(__ev.get() != nullptr, "cuda::experimental::stream_ref::wait invalid event passed");
    // Need to use driver API, cudaStreamWaitEvent would push dev 0 if stack was empty
    _CUDA_DRIVER::__streamWaitEvent(get(), __ev.get());
  }

  //! @brief Returns a \c execution::sender that completes on this stream.
  //!
  //! @note Equivalent to `execution::schedule(execution::stream_scheduler{*this})`.
  _CCCL_HOST_API auto schedule() const noexcept;

  //! @brief Make all future work submitted into this stream depend on completion of all work from the specified
  //! stream
  //!
  //! @param __other Stream that this stream should wait for
  //!
  //! @throws cuda_error if inserting the dependency fails
  _CCCL_HOST_API void wait(stream_ref __other) const
  {
    // TODO consider an optimization to not create an event every time and instead have one persistent event or one
    // per stream
    _CCCL_ASSERT(__stream != __detail::__invalid_stream, "cuda::experimental::stream_ref::wait invalid stream passed");
    if (*this != __other)
    {
      event __tmp(__other);
      wait(__tmp);
    }
  }

  //! @brief Get the logical device under which this stream was created.
  //!
  //! Compared to `device()` member function the returned \c logical_device will
  //! hold a green context for streams created under one.
  _CCCL_HOST_API logical_device logical_device() const
  {
    CUcontext __stream_ctx;
    ::cuda::experimental::logical_device::kinds __ctx_kind = ::cuda::experimental::logical_device::kinds::device;
#if _CCCL_CTK_AT_LEAST(12, 5)
    if (__driver::__getVersion() >= 12050)
    {
      auto __ctx = _CUDA_DRIVER::__streamGetCtx_v2(__stream);
      if (__ctx.__ctx_kind_ == _CUDA_DRIVER::__ctx_from_stream::__kind::__green)
      {
        __stream_ctx = _CUDA_DRIVER::__ctxFromGreenCtx(__ctx.__ctx_green_);
        __ctx_kind   = ::cuda::experimental::logical_device::kinds::green_context;
      }
      else
      {
        __stream_ctx = __ctx.__ctx_device_;
        __ctx_kind   = ::cuda::experimental::logical_device::kinds::device;
      }
    }
    else
#endif // _CCCL_CTK_AT_LEAST(12, 5)
    {
      __stream_ctx = _CUDA_DRIVER::__streamGetCtx(__stream);
      __ctx_kind   = ::cuda::experimental::logical_device::kinds::device;
    }
    // Because the stream can come from_native_handle, we can't just loop over devices comparing contexts,
    // lower to CUDART for this instead
    __ensure_current_device __setter(__stream_ctx);
    int __id;
    _CCCL_TRY_CUDA_API(cudaGetDevice, "Could not get device from a stream", &__id);
    return __logical_device_access::make_logical_device(__id, __stream_ctx, __ctx_kind);
  }

  //! @brief Get device under which this stream was created.
  //!
  //! Note: In case of a stream created under a `green_context` the device on which that `green_context` was created is
  //! returned
  //!
  //! @throws cuda_error if device check fails
  _CCCL_HOST_API device_ref device() const
  {
    return logical_device().underlying_device();
  }

  [[nodiscard]] _CCCL_API constexpr auto query(const get_stream_t&) const noexcept -> stream_ref
  {
    return *this;
  }

  [[nodiscard]] _CCCL_API static constexpr auto query(const execution::get_forward_progress_guarantee_t&) noexcept
    -> execution::forward_progress_guarantee
  {
    return execution::forward_progress_guarantee::weakly_parallel;
  }

  [[nodiscard]] _CCCL_API static constexpr auto query(const execution::get_domain_t&) noexcept
    -> execution::stream_domain;
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__STREAM_STREAM_REF
