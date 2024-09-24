//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__UTILITY_ENSURE_CURRENT_DEVICE
#define _CUDAX__UTILITY_ENSURE_CURRENT_DEVICE

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/stream_ref>

#include <cuda/experimental/__device/all_devices.cuh>
#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__utility/driver_api.cuh>

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

namespace cuda::experimental
{
//! TODO we might want to change the comments to indicate it operates on contexts for certains differences
//! with green context, but it depends on how exactly green context internals end up being

//! @brief RAII helper which on construction sets the current device to the specified one or one a
//! stream was created under. It sets the state back on destruction.
//!
struct [[maybe_unused]] __ensure_current_device
{
  //! @brief Construct a new `__ensure_current_device` object and switch to the specified
  //!        device.
  //!
  //! @param new_device The device to switch to
  //!
  //! @throws cuda_error if the device switch fails
  explicit __ensure_current_device(device_ref __new_device)
  {
    auto __ctx = devices[__new_device.get()].primary_context();
    detail::driver::ctxPush(__ctx);
  }

  //! @brief Construct a new `__ensure_current_device` object and switch to the specified
  //!        device.
  //!
  //! Note: if this logical device contains a green_context the device under which the green
  //! context was created will be set to current
  //!
  //! @param new_device The device to switch to
  //!
  //! @throws cuda_error if the device switch fails
  explicit __ensure_current_device(logical_device __new_device)
  {
    detail::driver::ctxPush(__new_device.context());
  }

  // Doesn't really fit into the type description, we might consider changing it once
  // green ctx design is more finalized
  explicit __ensure_current_device(CUcontext __ctx)
  {
    detail::driver::ctxPush(__ctx);
  }

  //! @brief Construct a new `__ensure_current_device` object and switch to the device
  //!        under which the specified stream was created.
  //!
  //! @param stream Stream indicating the device to switch to
  //!
  //! @throws cuda_error if the device switch fails
  explicit __ensure_current_device(stream_ref __stream)
  {
    auto __ctx = detail::driver::streamGetCtx(__stream.get());
    detail::driver::ctxPush(__ctx);
  }

  __ensure_current_device(__ensure_current_device&&)                 = delete;
  __ensure_current_device(__ensure_current_device const&)            = delete;
  __ensure_current_device& operator=(__ensure_current_device&&)      = delete;
  __ensure_current_device& operator=(__ensure_current_device const&) = delete;

  //! @brief Destroy the `__ensure_current_device` object and switch back to the original
  //!        device.
  //!
  //! @throws cuda_error if the device switch fails. If the destructor is called
  //!         during stack unwinding, the program is automatically terminated.
  ~__ensure_current_device() noexcept(false)
  {
    // TODO would it make sense to assert here that we pushed and popped the same thing?
    detail::driver::ctxPop();
  }
};
} // namespace cuda::experimental
#endif // DOXYGEN_SHOULD_SKIP_THIS
#endif // _CUDAX__UTILITY_ENSURE_CURRENT_DEVICE
