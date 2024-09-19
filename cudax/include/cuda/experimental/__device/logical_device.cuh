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

//! @brief A non-owning representation of a CUDA device
class logical_device
{
  friend struct stream_ref;
  // This might be a CUdevice as well
  int __dev_id    = 0;
  CUcontext __ctx = nullptr;

public:
  //! @brief Retrieve the native ordinal of the `device_ref`
  //!
  //! @return int The native device ordinal held by the `device_ref` object
  _CCCL_NODISCARD constexpr CUcontext context() const noexcept
  {
    return __ctx;
  }

  _CCCL_NODISCARD constexpr device_ref device() const noexcept
  {
    return __dev_id;
  }

  explicit logical_device(int __id)
      : __dev_id(__id)
      , __ctx(devices[__id].primary_context())
  {}

  logical_device(device_ref __dev)
      : logical_device(__dev.get())
  {}

  logical_device(const ::cuda::experimental::device& __dev)
      : __dev_id(__dev.get())
      , __ctx(__dev.primary_context())
  {}

#if CUDART_VERSION >= 12050
  logical_device(const green_context& __gctx)
      : __dev_id(__gctx.__dev_id)
      , __ctx(__gctx.__transformed)
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
    return __lhs.__dev_id == __rhs.__dev_id && __lhs.__ctx == __rhs.__ctx;
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
    return __lhs.__dev_id != __rhs.__dev_id || __lhs.__ctx != __rhs.__ctx;
  }
#endif // _CCCL_STD_VER <= 2017

  /*
    //! @brief Retrieve the specified attribute for the device
    //!
    //! @param __attr The attribute to query. See `device::attrs` for the available
    //!        attributes.
    //!
    //! @throws cuda_error if the attribute query fails
    //!
    //! @sa device::attrs
    template <typename _Attr>
    _CCCL_NODISCARD auto attr(_Attr __attr) const
    {
      return __attr(*this);
    }

    //! @overload
    template <::cudaDeviceAttr _Attr>
    _CCCL_NODISCARD auto attr() const
    {
      return attr(detail::__dev_attr<_Attr>());
    }

    const arch_traits_t& arch_traits() const;*/

private:
  logical_device(int __id, CUcontext __context)
      : __dev_id(__id)
      , __ctx(__context)
  {}
};

} // namespace cuda::experimental

#endif // _CUDAX__DEVICE_DEVICE_REF
