//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__DEVICE_DEVICE
#define _CUDAX__DEVICE_DEVICE

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__device/device_ref.cuh>

namespace cuda::experimental
{
namespace __detail
{
struct __emplace_device; // see all_devices.cuh
} // namespace __detail

// This is the element type of the the global `devices` array. In the future, we
// can cache device properties here.
//
//! @brief An immovable "owning" representation of a CUDA device.
class device : public device_ref
{
  device_ref ref() const noexcept
  {
    return *this;
  }

private:
  // TODO: put a mutable thread-safe (or thread_local) cache of device
  // properties here.

  friend class device_ref;
  friend struct __detail::__emplace_device;

  explicit constexpr device(int __id) noexcept
      : device_ref(__id)
  {}

  // `device` objects are not movable or copyable.
  device(device&&)                 = delete;
  device(const device&)            = delete;
  device& operator=(device&&)      = delete;
  device& operator=(const device&) = delete;
};

} // namespace cuda::experimental

#endif // _CUDAX__DEVICE_DEVICE
