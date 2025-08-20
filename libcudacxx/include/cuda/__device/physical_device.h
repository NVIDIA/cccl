//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_PHYSICAL_DEVICE_H
#define _CUDA___DEVICE_PHYSICAL_DEVICE_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__device/arch_traits.h>
#  include <cuda/__device/attributes.h>
#  include <cuda/__device/device_ref.h>
#  include <cuda/__driver/driver_api.h>

#  include <cassert>
#  include <mutex>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
namespace __detail
{
//! @brief A proxy object used to in-place construct a `device` object from an
//! integer ID. Used in __detail/all_devices.cuh.
struct __emplace_device
{
  int __id_;

  [[nodiscard]] operator physical_device() const;

  [[nodiscard]] constexpr const __emplace_device* operator->() const;
};
} // namespace __detail

//! @brief For a given attribute, type of the attribute value.
//!
//! @par Example
//! @code
//! using threads_per_block_t = device::attr_result_t<device_attributes::max_threads_per_block>;
//! static_assert(std::is_same_v<threads_per_block_t, int>);
//! @endcode
//!
//! @sa device_attributes
template <::cudaDeviceAttr _Attr>
using device_attribute_result_t = typename __detail::__dev_attr<_Attr>::type;

// This is the element type of the the global `devices` array. In the future, we
// can cache device properties here.
//
//! @brief An immovable "owning" representation of a CUDA device.
class physical_device : public device_ref
{
public:
#  ifndef _CCCL_DOXYGEN_INVOKED // Do not document
#    if _CCCL_COMPILER(MSVC)
  // When __EDG__ is defined, std::construct_at will not permit constructing
  // a device object from an __emplace_device object. This is a workaround.
  physical_device(__detail::__emplace_device __ed)
      : physical_device(__ed.__id_)
  {}
#    endif // _CCCL_COMPILER(MSVC)
#  endif // _CCCL_COMPILER(MSVC)

  //! @brief Retrieve architecture traits of this device.
  //!
  //! Architecture traits object contains information about certain traits
  //! that are shared by all devices belonging to given architecture.
  //!
  //! @return A reference to `arch_traits_t` object containing architecture traits of this device
  const arch::traits_t& arch_traits() const noexcept
  {
    return __traits;
  }

  //! @brief Retrieve the primary context for this device.
  //!
  //! @return A reference to the primary context for this device.
  ::CUcontext primary_context() const
  {
    ::std::call_once(__init_once, [this]() {
      __device      = ::cuda::__driver::__deviceGet(__id_);
      __primary_ctx = ::cuda::__driver::__primaryCtxRetain(__device);
    });
    _CCCL_ASSERT(__primary_ctx != nullptr, "cuda::primary_context failed to get context");

    return __primary_ctx;
  }

  ~physical_device()
  {
    if (__primary_ctx)
    {
      ::cuda::__driver::__primaryCtxRelease(__device);
    }
  }

private:
  // TODO: put a mutable thread-safe (or thread_local) cache of device
  // properties here.

  friend class device_ref;
  friend struct __detail::__emplace_device;

  mutable ::CUcontext __primary_ctx = nullptr;
  mutable ::CUdevice __device{};
  mutable ::std::once_flag __init_once;

  // TODO should this be a reference/pointer to the constexpr traits instances?
  //  Do we care about lazy init?
  //  We should have some of the attributes just return from the arch traits
  arch::traits_t __traits;

  explicit physical_device(int __id)
      : device_ref(__id)
      , __traits(arch::__arch_traits_might_be_unknown(__id, device_attributes::compute_capability(__id)))
  {}

  // `device` objects are not movable or copyable.
  physical_device(physical_device&&)                 = delete;
  physical_device(const physical_device&)            = delete;
  physical_device& operator=(physical_device&&)      = delete;
  physical_device& operator=(const physical_device&) = delete;

  friend bool operator==(const physical_device& __lhs, int __rhs) = delete;
  friend bool operator==(int __lhs, const physical_device& __rhs) = delete;

#  if _CCCL_STD_VER <= 2017
  friend bool operator!=(const physical_device& __lhs, int __rhs) = delete;
  friend bool operator!=(int __lhs, const physical_device& __rhs) = delete;
#  endif // _CCCL_STD_VER <= 2017
};

namespace __detail
{
[[nodiscard]] inline __emplace_device::operator physical_device() const
{
  return physical_device(__id_);
}

[[nodiscard]] inline constexpr const __emplace_device* __emplace_device::operator->() const
{
  return this;
}
} // namespace __detail

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DEVICE_PHYSICAL_DEVICE_H
