//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__DEVICE_LOGICAL_DEVICE
#define _CUDAX__DEVICE_LOGICAL_DEVICE

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__device/all_devices.cuh>
#include <cuda/experimental/__green_context/green_ctx.cuh>

namespace cuda::experimental
{
namespace detail
{
struct logical_device_access;
}

//! @brief A non-owning representation of a CUDA device or a green context
class logical_device
{
  //! @brief Enum to indicate the kind of logical device stored
  enum class kinds
  {
    // Indicates logical device is a full device
    device,
    // Indicated logical device is a green context
    green_context
  };

  // We might want to make this private depending on how this type ends up looking like long term,
  // not documenting it for now
  _CCCL_NODISCARD constexpr CUcontext context() const noexcept
  {
    return __ctx;
  }

  //! @brief Retrieve the device on which this logical device resides
  _CCCL_NODISCARD constexpr device_ref underlying_device() const noexcept
  {
    return __dev_id;
  }

  //! @brief Retrieve the kind of logical device stored in this object
  //! The kind indicates if this logical_device holds a device or green_context
  _CCCL_NODISCARD constexpr kinds get_kind() const noexcept
  {
    return __kind;
  }

  //! @brief Construct logical_device from a device ordinal
  explicit logical_device(int __id)
      : __dev_id(__id)
      , __ctx(devices[__id].primary_context())
      , __kind(kinds::device)
  {}

  //! @brief Construct logical_device from a device_ref
  explicit logical_device(device_ref __dev)
      : logical_device(__dev.get())
  {}

  // More of a micro-optimization, we can also remove this (depending if we keep device_ref)
  logical_device(const ::cuda::experimental::device& __dev)
      : __dev_id(__dev.get())
      , __ctx(__dev.primary_context())
      , __kind(kinds::device)
  {}

#if CUDART_VERSION >= 12050
  //! @brief Construct logical_device from a green_context
  logical_device(const green_context& __gctx)
      : __dev_id(__gctx.__dev_id)
      , __ctx(__gctx.__transformed)
      , __kind(kinds::green_context)
  {}
#endif // CUDART_VERSION >= 12050

  //! @brief Compares two logical_devices for equality
  //!
  //! @param __lhs The first `logical_device` to compare
  //! @param __rhs The second `logical_device` to compare
  //! @return true if `lhs` and `rhs` refer to the same logical device
  _CCCL_NODISCARD_FRIEND bool operator==(logical_device __lhs, logical_device __rhs) noexcept
  {
    return __lhs.__ctx == __rhs.__ctx;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two logical_devices for inequality
  //!
  //! @param __lhs The first `logical_device` to compare
  //! @param __rhs The second `logical_device` to compare
  //! @return true if `lhs` and `rhs` refer to the different logical device
  _CCCL_NODISCARD_FRIEND bool operator!=(logical_device __lhs, logical_device __rhs) noexcept
  {
    return __lhs.__ctx != __rhs.__ctx;
  }
#endif // _CCCL_STD_VER <= 2017

private:
  friend detail::logical_device_access;
  // This might be a CUdevice as well
  int __dev_id    = 0;
  kinds __kind;
  CUcontext __ctx = nullptr;

  logical_device(int __id, CUcontext __context, kinds __k)
      : __dev_id(__id)
      , __ctx(__context)
      , __kind(__k)
  {}
};

namespace detail
{
struct logical_device_access
{
  static logical_device make_logical_device(int __id, CUcontext __context, logical_device::kinds __k)
  {
    return logical_device(__id, __context, __k);
  }
};
} // namespace detail

} // namespace cuda::experimental

#endif // _CUDAX__DEVICE_DEVICE_REF
