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
#include <cuda/experimental/__utility/driver_api.cuh>

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

namespace cuda::experimental::detail
{
//! @brief RAII helper which saves the current device and switches to the
//!        specified device on construction and switches to the saved device on
//!        destruction.
//!
struct __ensure_current_device
{
  //! @brief Construct a new `__ensure_current_device` object and switch to the specified
  //!        device.
  //!
  //! @param new_device The device to switch to
  //!
  //! @throws cuda_error if the device switch fails
  explicit __ensure_current_device(device_ref new_device)
  {
    auto ctx = devices[new_device.get()].primary_context();
    detail::driver::ctxPush(ctx);
  }

  explicit __ensure_current_device(stream_ref stream)
  {
    auto ctx = detail::driver::streamGetCtx(stream.get());
    detail::driver::ctxPush(ctx);
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
    auto ctx = detail::driver::ctxPop();
  }
};
} // namespace cuda::experimental::detail
#endif // DOXYGEN_SHOULD_SKIP_THIS
#endif // _CUDAX__UTILITY_ENSURE_CURRENT_DEVICE
