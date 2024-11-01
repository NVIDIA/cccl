//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__DEVICE_DEVICE_REF
#define _CUDAX__DEVICE_DEVICE_REF

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cuda/api_wrapper.h>

#include <cuda/experimental/__utility/driver_api.cuh>

#include <string>
#include <vector>

namespace cuda::experimental
{
class device;
struct arch_traits_t;

namespace detail
{
template <::cudaDeviceAttr _Attr>
struct __dev_attr;
} // namespace detail

//! @brief A non-owning representation of a CUDA device
class device_ref
{
  friend class device;

  int __id_ = 0;

public:
  //! @brief Create a `device_ref` object from a native device ordinal.
  /*implicit*/ constexpr device_ref(int __id) noexcept
      : __id_(__id)
  {}

  //! @brief Retrieve the native ordinal of the `device_ref`
  //!
  //! @return int The native device ordinal held by the `device_ref` object
  _CCCL_NODISCARD constexpr int get() const noexcept
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
  _CCCL_NODISCARD_FRIEND constexpr bool operator==(device_ref __lhs, device_ref __rhs) noexcept
  {
    return __lhs.__id_ == __rhs.__id_;
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
  _CCCL_NODISCARD_FRIEND constexpr bool operator!=(device_ref __lhs, device_ref __rhs) noexcept
  {
    return __lhs.__id_ != __rhs.__id_;
  }
#endif // _CCCL_STD_VER <= 2017

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

  _CCCL_NODISCARD ::std::string get_name() const
  {
    constexpr int __max_name_length = 256;
    ::std::string __name(256, 0);

    // For some reason there is no separate name query in CUDA runtime
    detail::driver::getName(__name.data(), __max_name_length, get());
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

  const arch_traits_t& arch_traits() const;

  // TODO this might return some more complex type in the future
  // TODO we might want to include the calling device, depends on what we decide
  // peer access APIs

  //! @brief Retrieve a vector of `device_ref`s that are peers of this device
  //!
  //! The device on which this API is called is not included in the vector,
  //! if a full group of peer devices is needed, it needs to be pushed_back separately.
  //!
  //! @throws cuda_error if any peer access query fails
  ::std::vector<device_ref> get_peers() const;
};

} // namespace cuda::experimental

#endif // _CUDAX__DEVICE_DEVICE_REF
