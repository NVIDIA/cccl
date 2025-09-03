//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_DEVICE_REF_H
#define _CUDA___DEVICE_DEVICE_REF_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
#  include <cuda/__driver/driver_api.h>
#  include <cuda/std/__cuda/api_wrapper.h>

#  include <string>
#  include <vector>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
class physical_device;
namespace arch
{
struct traits_t;
} // namespace arch

namespace __detail
{
template <::cudaDeviceAttr _Attr>
struct __dev_attr;
} // namespace __detail

//! @brief A non-owning representation of a CUDA device
class device_ref
{
  friend class physical_device;

  int __id_ = 0;

public:
  //! @brief Create a `device_ref` object from a native device ordinal.
  /*implicit*/ constexpr device_ref(int __id) noexcept
      : __id_(__id)
  {}

  //! @brief Retrieve the native ordinal of the `device_ref`
  //!
  //! @return int The native device ordinal held by the `device_ref` object
  [[nodiscard]] constexpr int get() const noexcept
  {
    return __id_;
  }

  //! @brief Compares two `device_ref`s for equality
  //!
  //! @note Allows comparison with `int` due to implicit conversion to
  //! `device_ref`.
  //!
  //! @param __lhs The first `device_ref` to compare
  //! @param __rhs The second `device_ref` to compare
  //! @return true if `lhs` and `rhs` refer to the same device ordinal
  [[nodiscard]] friend constexpr bool operator==(device_ref __lhs, device_ref __rhs) noexcept
  {
    return __lhs.__id_ == __rhs.__id_;
  }

#  if _CCCL_STD_VER <= 2017
  //! @brief Compares two `device_ref`s for inequality
  //!
  //! @note Allows comparison with `int` due to implicit conversion to
  //! `device_ref`.
  //!
  //! @param __lhs The first `device_ref` to compare
  //! @param __rhs The second `device_ref` to compare
  //! @return true if `lhs` and `rhs` refer to different device ordinal
  [[nodiscard]] constexpr friend bool operator!=(device_ref __lhs, device_ref __rhs) noexcept
  {
    return __lhs.__id_ != __rhs.__id_;
  }
#  endif // _CCCL_STD_VER <= 2017

  //! @brief Retrieve the specified attribute for the device
  //!
  //! @param __attr The attribute to query. See `device::attrs` for the available
  //!        attributes.
  //!
  //! @throws cuda_error if the attribute query fails
  //!
  //! @sa device::attrs
  template <typename _Attr>
  [[nodiscard]] auto attribute(_Attr __attr) const
  {
    return __attr(*this);
  }

  //! @overload
  template <::cudaDeviceAttr _Attr>
  [[nodiscard]] auto attribute() const
  {
    return attribute(__detail::__dev_attr<_Attr>());
  }

  //! @brief Retrieve string with the name of this device.
  //!
  //! @return String containing the name of this device.
  [[nodiscard]] ::std::string name() const
  {
    constexpr int __max_name_length = 256;
    ::std::string __name(256, 0);

    // For some reason there is no separate name query in CUDA runtime
    ::cuda::__driver::__deviceGetName(__name.data(), __max_name_length, get());
    return __name;
  }

  //! @brief Queries if its possible for this device to directly access specified device's memory.
  //!
  //! If this function returns true, device supplied to this call can be passed into enable_peer_access
  //! on memory resource or pool that manages memory on this device. It will make allocations from that
  //! pool accessible by this device.
  //!
  //! @param __other_dev Device to query the peer access
  //! @return true if its possible for this device to access the specified device's memory
  bool has_peer_access_to(device_ref __other_dev) const
  {
    int __can_access;
    _CCCL_TRY_CUDA_API(
      ::cudaDeviceCanAccessPeer,
      "Could not query if device can be peer accessed",
      &__can_access,
      get(),
      __other_dev.get());
    return __can_access;
  }

  //! @brief Retrieve architecture traits of this device.
  //!
  //! Architecture traits object contains information about certain traits
  //! that are shared by all devices belonging to given architecture.
  //!
  //! @return A reference to `arch_traits_t` object containing architecture traits of this device
  const arch::traits_t& arch_traits() const;

  // TODO this might return some more complex type in the future
  // TODO we might want to include the calling device, depends on what we decide
  // peer access APIs

  //! @brief Retrieve a vector of `device_ref`s that are peers of this device
  //!
  //! The device on which this API is called is not included in the vector,
  //! if a full group of peer devices is needed, it needs to be pushed_back separately.
  //!
  //! @throws cuda_error if any peer access query fails
  ::std::vector<device_ref> peer_devices() const;
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DEVICE_DEVICE_REF_H
