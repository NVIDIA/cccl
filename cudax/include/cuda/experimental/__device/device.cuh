//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/experimental/__device/arch_traits.cuh>
#include <cuda/experimental/__device/attributes.cuh>
#include <cuda/experimental/__device/device_ref.cuh>
#include <cuda/experimental/__utility/driver_api.cuh>

#include <cassert>
#include <mutex>

#include <cuda.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
namespace __detail
{
//! @brief A proxy object used to in-place construct a `device` object from an
//! integer ID. Used in __detail/all_devices.cuh.
struct __emplace_device
{
  int __id_;

  [[nodiscard]] operator device() const;

  [[nodiscard]] constexpr const __emplace_device* operator->() const;
};
} // namespace __detail

// This is the element type of the the global `devices` array. In the future, we
// can cache device properties here.
//
//! @brief An immovable "owning" representation of a CUDA device.
class device : public device_ref
{
public:
  using attributes = __detail::__device_attrs;

  //! @brief For a given attribute, returns the type of the attribute value.
  //!
  //! @par Example
  //! @code
  //! using threads_per_block_t = device::attr_result_t<device::attributes::max_threads_per_block>;
  //! static_assert(std::is_same_v<threads_per_block_t, int>);
  //! @endcode
  //!
  //! @sa device::attributes
  template <::cudaDeviceAttr _Attr>
  using attribute_result_t = typename __detail::__dev_attr<_Attr>::type;

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
#  if _CCCL_COMPILER(MSVC)
  // When __EDG__ is defined, std::construct_at will not permit constructing
  // a device object from an __emplace_device object. This is a workaround.
  device(__detail::__emplace_device __ed)
      : device(__ed.__id_)
  {}
#  endif
#endif

  //! @brief Retrieve architecture traits of this device.
  //!
  //! Architecture traits object contains information about certain traits
  //! that are shared by all devices belonging to given architecture.
  //!
  //! @return A reference to `arch_traits_t` object containing architecture traits of this device
  const arch_traits_t& arch_traits() const noexcept
  {
    return __traits;
  }

  CUcontext primary_context() const
  {
    ::std::call_once(__init_once, [this]() {
      __device      = __detail::driver::deviceGet(__id_);
      __primary_ctx = __detail::driver::primaryCtxRetain(__device);
    });
    _CCCL_ASSERT(__primary_ctx != nullptr, "cuda::experimental::primary_context failed to get context");
    return __primary_ctx;
  }

  ~device()
  {
    if (__primary_ctx)
    {
      __detail::driver::primaryCtxRelease(__device);
    }
  }

private:
  // TODO: put a mutable thread-safe (or thread_local) cache of device
  // properties here.

  friend class device_ref;
  friend struct __detail::__emplace_device;

  mutable CUcontext __primary_ctx = nullptr;
  mutable CUdevice __device{};
  mutable ::std::once_flag __init_once;

  // TODO should this be a reference/pointer to the constexpr traits instances?
  //  Do we care about lazy init?
  //  We should have some of the attributes just return from the arch traits
  arch_traits_t __traits;

  explicit device(int __id)
      : device_ref(__id)
      , __traits(__detail::__arch_traits_might_be_unknown(__id, attributes::compute_capability(__id)))
  {}

  // `device` objects are not movable or copyable.
  device(device&&)                 = delete;
  device(const device&)            = delete;
  device& operator=(device&&)      = delete;
  device& operator=(const device&) = delete;

  friend bool operator==(const device& __lhs, int __rhs) = delete;
  friend bool operator==(int __lhs, const device& __rhs) = delete;

#if _CCCL_STD_VER <= 2017
  friend bool operator!=(const device& __lhs, int __rhs) = delete;
  friend bool operator!=(int __lhs, const device& __rhs) = delete;
#endif // _CCCL_STD_VER <= 2017
};

namespace __detail
{
[[nodiscard]] inline __emplace_device::operator device() const
{
  return device(__id_);
}

[[nodiscard]] inline constexpr const __emplace_device* __emplace_device::operator->() const
{
  return this;
}
} // namespace __detail

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__DEVICE_DEVICE
