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

#include <cuda.h>

#include <cuda/experimental/__device/arch_traits.cuh>
#include <cuda/experimental/__device/attributes.cuh>
#include <cuda/experimental/__device/device_ref.cuh>
#include <cuda/experimental/__utility/driver_api.cuh>

#include <cassert>
#include <mutex>

namespace cuda::experimental
{
namespace detail
{
//! @brief A proxy object used to in-place construct a `device` object from an
//! integer ID. Used in __detail/all_devices.cuh.
struct __emplace_device
{
  int __id_;

  _CCCL_NODISCARD operator device() const;

  _CCCL_NODISCARD constexpr const __emplace_device* operator->() const;
};
} // namespace detail

// This is the element type of the the global `devices` array. In the future, we
// can cache device properties here.
//
//! @brief An immovable "owning" representation of a CUDA device.
class device : public device_ref
{
public:
  using attrs = detail::__device_attrs;

  //! @brief For a given attribute, returns the type of the attribute value.
  //!
  //! @par Example
  //! @code
  //! using threads_per_block_t = device::attr_result_t<device::attrs::max_threads_per_block>;
  //! static_assert(std::is_same_v<threads_per_block_t, int>);
  //! @endcode
  //!
  //! @sa device::attrs
  template <::cudaDeviceAttr _Attr>
  using attr_result_t = typename detail::__dev_attr<_Attr>::type;

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
#  if defined(_CCCL_COMPILER_MSVC)
  // When __EDG__ is defined, std::construct_at will not permit constructing
  // a device object from an __emplace_device object. This is a workaround.
  device(detail::__emplace_device __ed)
      : device(__ed.__id_)
  {}
#  endif
#endif

  const arch_traits_t& arch_traits() const noexcept
  {
    return __traits;
  }

  CUcontext primary_context() const
  {
    ::std::call_once(__init_once, [this]() {
      __device      = detail::driver::deviceGet(__id_);
      __primary_ctx = detail::driver::primaryCtxRetain(__device);
    });
    assert(__primary_ctx != nullptr);
    return __primary_ctx;
  }

  ~device()
  {
    if (__primary_ctx)
    {
      detail::driver::primaryCtxRelease(__device);
    }
  }

private:
  // TODO: put a mutable thread-safe (or thread_local) cache of device
  // properties here.

  friend class device_ref;
  friend struct detail::__emplace_device;

  mutable CUcontext __primary_ctx = nullptr;
  mutable CUdevice __device{};
  mutable ::std::once_flag __init_once;

  // TODO should this be a reference/pointer to the constexpr traits instances?
  //  Do we care about lazy init?
  //  We should have some of the attributes just return from the arch traits
  arch_traits_t __traits;

  explicit device(int __id)
      : device_ref(__id)
      , __traits(detail::__arch_traits_might_be_unknown(__id, attrs::compute_capability(__id)))
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

namespace detail
{
_CCCL_NODISCARD inline __emplace_device::operator device() const
{
  return device(__id_);
}

_CCCL_NODISCARD inline constexpr const __emplace_device* __emplace_device::operator->() const
{
  return this;
}
} // namespace detail

} // namespace cuda::experimental

#endif // _CUDAX__DEVICE_DEVICE
