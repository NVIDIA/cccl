//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda_runtime_api.h>

#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/stream_ref>

#include <cuda/experimental/__device/device_ref.cuh>
#include <cuda/experimental/__event/timed_event.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>

namespace cuda::experimental
{

namespace detail
{
// 0 is a valid stream in CUDA, so we need some other invalid stream representation
// Can't make it constexpr, because cudaStream_t is a pointer type
static const ::cudaStream_t __invalid_stream = reinterpret_cast<cudaStream_t>(~0ULL);
} // namespace detail

//! @brief A non-owning wrapper for cudaStream_t.
struct stream_ref : ::cuda::stream_ref
{
  using ::cuda::stream_ref::stream_ref;

  stream_ref() = delete;

  //! @brief Create a new event and record it into this stream
  //!
  //! @return A new event that was recorded into this stream
  //!
  //! @throws cuda_error if event creation or record failed
  _CCCL_NODISCARD event record_event(event::flags __flags = event::flags::none) const
  {
    return event(*this, __flags);
  }

  //! @brief Create a new timed event and record it into this stream
  //!
  //! @return A new timed event that was recorded into this stream
  //!
  //! @throws cuda_error if event creation or record failed
  _CCCL_NODISCARD timed_event record_timed_event(event::flags __flags = event::flags::none) const
  {
    return timed_event(*this, __flags);
  }

  using ::cuda::stream_ref::wait;

  //! @brief Make all future work submitted into this stream depend on completion of the specified event
  //!
  //! @param __ev Event that this stream should wait for
  //!
  //! @throws cuda_error if inserting the dependency fails
  void wait(event_ref __ev) const
  {
    assert(__ev.get() != nullptr);
    // Need to use driver API, cudaStreamWaitEvent would push dev 0 if stack was empty
    detail::driver::streamWaitEvent(get(), __ev.get());
  }

  //! @brief Make all future work submitted into this stream depend on completion of all work from the specified
  //! stream
  //!
  //! @param __other Stream that this stream should wait for
  //!
  //! @throws cuda_error if inserting the dependency fails
  void wait(stream_ref __other) const
  {
    // TODO consider an optimization to not create an event every time and instead have one persistent event or one
    // per stream
    assert(__stream != detail::__invalid_stream);
    event __tmp(__other);
    wait(__tmp);
  }

  //! @brief Get device under which this stream was created.
  //!
  //! @throws cuda_error if device check fails
  device_ref device() const
  {
    // Because the stream can come from_native_handle, we can't just loop over devices comparing contexts,
    // lower to CUDART for this instead
    __ensure_current_device __dev_setter(*this);
    int result;
    _CCCL_TRY_CUDA_API(cudaGetDevice, "Could not get device from a stream", &result);
    return result;
  }
};

} // namespace cuda::experimental

#endif // _CUDAX__STREAM_STREAM_REF
