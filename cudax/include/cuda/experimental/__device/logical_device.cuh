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

//! @brief A non-owning representation of a CUDA device or a green context
struct logical_device
{
  enum class kinds
  {
    device,
    green_context
  };

  //! @brief Retrieve the native ordinal of the `device_ref`
  //!
  //! @return int The native device ordinal held by the `device_ref` object
  _CCCL_NODISCARD constexpr CUcontext context() const noexcept
  {
    return __ctx;
  }

  _CCCL_NODISCARD constexpr device_ref underlying_device() const noexcept
  {
    return __dev_id;
  }

  _CCCL_NODISCARD constexpr kinds kind() const noexcept
  {
    return __kind;
  }

  explicit logical_device(int __id)
      : __dev_id(__id)
      , __ctx(devices[__id].primary_context())
      , __kind(kinds::device)
  {}

  explicit logical_device(device_ref __dev)
      : logical_device(__dev.get())
  {}

  logical_device(const ::cuda::experimental::device& __dev)
      : __dev_id(__dev.get())
      , __ctx(__dev.primary_context())
      , __kind(kinds::device)
  {}

#if CUDART_VERSION >= 12050
  logical_device(const green_context& __gctx)
      : __dev_id(__gctx.__dev_id)
      , __ctx(__gctx.__transformed)
      , __kind(kinds::green_context)
  {}
#endif

  //! @brief Compares two `device_ref`s for equality
  //!
  //! @note Allows comparison with `int` due to implicit conversion to
  //! `device_ref`.
  //!
  //! @param __lhs The first `device_ref` to compare
  //! @param __rhs The second `device_ref` to compare
  //! @return true if `lhs` and `rhs` refer to the same device ordinal
  _CCCL_NODISCARD_FRIEND bool operator==(logical_device __lhs, logical_device __rhs) noexcept
  {
    return __lhs.__ctx == __rhs.__ctx;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two `device_ref`s for inequality
  //!
  //! @note Allows comparison with `int` due to implicit conversion to
  //! `device_ref`.
  //!
  //! @param __lhs The first `device_ref` to compare
  //! @param __rhs The second `device_ref` to compare
  //! @return true if `lhs` and `rhs` refer to different device ordinal
  _CCCL_NODISCARD_FRIEND bool operator!=(logical_device __lhs, logical_device __rhs) noexcept
  {
    return __lhs.__ctx != __rhs.__ctx;
  }
#endif // _CCCL_STD_VER <= 2017

private:
  friend struct stream_ref;
  // This might be a CUdevice as well
  int __dev_id    = 0;
  CUcontext __ctx = nullptr;
  // If we ever move to a variant instead of CUcontext we can probably remove the kind member
  kinds __kind;

  logical_device(int __id, CUcontext __context, kinds __k)
      : __dev_id(__id)
      , __ctx(__context)
      , __kind(__k)
  {}
};

} // namespace cuda::experimental

#endif // _CUDAX__DEVICE_DEVICE_REF
