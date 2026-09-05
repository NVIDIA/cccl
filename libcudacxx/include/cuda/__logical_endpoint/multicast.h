//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___LOGICAL_ENDPOINT_MULTICAST_H
#define _CUDA___LOGICAL_ENDPOINT_MULTICAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && _CCCL_CTK_AT_LEAST(13, 3) && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__logical_endpoint/common.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Size-independent configuration for a multicast CUDA logical endpoint.
class multicast_logical_endpoint_spec
{
  unsigned int __num_devices_{};
  logical_endpoint_flag __flags_          = logical_endpoint_flag::none;
  logical_endpoint_ipc_handle_type __ipc_ = logical_endpoint_ipc_handle_type::fabric;

public:
  //! @brief Creates a multicast endpoint specification.
  //!
  //! @param[in] __num_devices The number of CUDA devices that will be added to the endpoint.
  //! @param[in] __flags Logical endpoint creation flags.
  //! @param[in] __ipc The IPC handle type requested for this endpoint.
  _CCCL_HOST_API explicit multicast_logical_endpoint_spec(
    unsigned int __num_devices,
    logical_endpoint_flag __flags          = logical_endpoint_flag::none,
    logical_endpoint_ipc_handle_type __ipc = logical_endpoint_ipc_handle_type::fabric) noexcept
      : __num_devices_(__num_devices)
      , __flags_(__flags)
      , __ipc_(__ipc)
  {}

  //! @brief Returns the number of devices expected by the multicast endpoint.
  //!
  //! @return The endpoint device count.
  [[nodiscard]] _CCCL_HOST_API constexpr unsigned int num_devices() const noexcept
  {
    return __num_devices_;
  }

  //! @brief Returns the creation flags in this specification.
  //!
  //! @return The logical endpoint flags.
  [[nodiscard]] _CCCL_HOST_API constexpr logical_endpoint_flag flags() const noexcept
  {
    return __flags_;
  }

  //! @brief Returns the requested IPC handle type.
  //!
  //! @return The logical endpoint IPC handle type.
  [[nodiscard]] _CCCL_HOST_API constexpr logical_endpoint_ipc_handle_type ipc_handle_type() const noexcept
  {
    return __ipc_;
  }

  //! @brief Checks whether a device supports this multicast endpoint configuration.
  //!
  //! @param[in] __device Device used for support attribute checks.
  //! @return `true` if the requested configuration is supported by the device.
  [[nodiscard]] _CCCL_HOST_API bool is_supported(::cuda::device_ref __device) const;

  //! @brief Queries CUDA driver limits for this endpoint configuration.
  //!
  //! @return The required bind alignment and maximum endpoint size.
  [[nodiscard]] _CCCL_HOST_API logical_endpoint_limits limits() const;
};

namespace __detail
{
[[nodiscard]] _CCCL_HOST_API inline ::CUlogicalEndpointProp __make_multicast_logical_endpoint_prop(
  const ::cuda::multicast_logical_endpoint_spec& __spec, ::cuda::std::uint64_t __bytes)
{
  ::CUlogicalEndpointProp __prop{};
  __prop.type                 = ::CU_LOGICAL_ENDPOINT_TYPE_MULTICAST;
  __prop.multicast.numDevices = __spec.num_devices();
  __prop.size                 = __bytes;
  __prop.ipcHandleTypes       = static_cast<unsigned>(__spec.ipc_handle_type());
  __prop.flags                = static_cast<unsigned>(__spec.flags());
  return __prop;
}
} // namespace __detail

[[nodiscard]] _CCCL_HOST_API inline bool multicast_logical_endpoint_spec::is_supported(::cuda::device_ref __device) const
{
  return ::cuda::__detail::__is_logical_endpoint_supported(
    __device, ::CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED, ipc_handle_type(), flags());
}

[[nodiscard]] _CCCL_HOST_API inline logical_endpoint_limits multicast_logical_endpoint_spec::limits() const
{
  const auto __prop = ::cuda::__detail::__make_multicast_logical_endpoint_prop(*this, 0);
  return ::cuda::__detail::__get_logical_endpoint_limits(__prop);
}

//! @brief Non-owning reference to a multicast CUDA logical endpoint.
//!
//! This type is trivially copyable and can be passed to device code directly, including raw `<<<>>>` kernel launches.
//! Owning `multicast_logical_endpoint` objects are also valid kernel arguments when using `cuda::launch`, which
//! transforms them to this ref type before invoking the kernel.
class multicast_logical_endpoint_ref
    : public ::cuda::__detail::__logical_endpoint_ref_base<::cuda::__detail::__logical_endpoint_type::__multicast>
{
  using __base = ::cuda::__detail::__logical_endpoint_ref_base<::cuda::__detail::__logical_endpoint_type::__multicast>;

public:
  //! @brief Creates a multicast endpoint reference from a logical endpoint ID.
  //!
  //! @param[in] __id The logical endpoint ID.
  _CCCL_HOST_DEVICE_API explicit constexpr multicast_logical_endpoint_ref(logical_endpoint_id __id) noexcept
      : __base(__id)
  {}

  //! @brief Adds a CUDA device to the multicast logical endpoint.
  //!
  //! @param[in] __device The device to add to the endpoint.
  _CCCL_HOST_API void add_device(::cuda::device_ref __device) const
  {
    ::cuda::__driver::__logicalEndpointAddDevice(this->native_handle(), ::cuda::__driver::__deviceGet(__device.get()));
  }
};

//! @brief Move-only owning RAII wrapper for a multicast CUDA logical endpoint.
//!
//! This type owns endpoint creation and destruction. It can be passed as a kernel argument through `cuda::launch`; CCCL
//! launch argument transformation converts it to `multicast_logical_endpoint_ref`. Raw `<<<>>>` launches do not perform
//! that transformation, so they should pass `multicast_logical_endpoint_ref` explicitly.
class multicast_logical_endpoint
    : public ::cuda::__detail::__logical_endpoint_owner_base<multicast_logical_endpoint_ref,
                                                             ::cuda::__detail::__logical_endpoint_type::__multicast>
{
  using __base =
    ::cuda::__detail::__logical_endpoint_owner_base<multicast_logical_endpoint_ref,
                                                    ::cuda::__detail::__logical_endpoint_type::__multicast>;

public:
  //! @brief Creates an empty logical endpoint owner.
  _CCCL_HOST_API multicast_logical_endpoint() noexcept
      : __base()
  {}

  _CCCL_HOST_API multicast_logical_endpoint(multicast_logical_endpoint&& __other) noexcept
      : __base(::cuda::std::move(__other))
  {}

  _CCCL_HOST_API multicast_logical_endpoint& operator=(multicast_logical_endpoint&& __other) noexcept
  {
    static_cast<__base&>(*this) = ::cuda::std::move(static_cast<__base&>(__other));
    return *this;
  }

  multicast_logical_endpoint(const multicast_logical_endpoint&)            = delete;
  multicast_logical_endpoint& operator=(const multicast_logical_endpoint&) = delete;

  //! @brief Reserves one ID and creates a multicast logical endpoint.
  //!
  //! @param[in] __spec The endpoint specification.
  //! @param[in] __bytes The endpoint size in bytes.
  _CCCL_HOST_API explicit multicast_logical_endpoint(const multicast_logical_endpoint_spec& __spec,
                                                     ::cuda::std::uint64_t __bytes)
      : multicast_logical_endpoint(logical_endpoint_id_range{1}, 0, __spec, __bytes)
  {}

  //! @brief Creates a multicast logical endpoint from a caller-managed ID.
  //!
  //! @param[in] __id The caller-managed logical endpoint ID.
  //! @param[in] __spec The endpoint specification.
  //! @param[in] __bytes The endpoint size in bytes.
  _CCCL_HOST_API multicast_logical_endpoint(
    logical_endpoint_id __id, const multicast_logical_endpoint_spec& __spec, ::cuda::std::uint64_t __bytes)
      : __base(__id)
  {
    const auto __prop = ::cuda::__detail::__make_multicast_logical_endpoint_prop(__spec, __bytes);
    this->__create_endpoint(__prop);
  }

  //! @brief Creates a multicast logical endpoint from an ID in a retained range.
  //!
  //! @param[in] __range The logical endpoint ID range to retain.
  //! @param[in] __index The ID index in the range.
  //! @param[in] __spec The endpoint specification.
  //! @param[in] __bytes The endpoint size in bytes.
  _CCCL_HOST_API multicast_logical_endpoint(
    const logical_endpoint_id_range& __range,
    ::cuda::std::uint32_t __index,
    const multicast_logical_endpoint_spec& __spec,
    ::cuda::std::uint64_t __bytes)
      : __base(__range[__index])
  {
    this->__retain_id_range(__range);
    const auto __prop = ::cuda::__detail::__make_multicast_logical_endpoint_prop(__spec, __bytes);
    this->__create_endpoint(__prop);
  }

  //! @brief Adds a CUDA device to the multicast logical endpoint.
  //!
  //! @param[in] __device The device to add to the endpoint.
  _CCCL_HOST_API void add_device(::cuda::device_ref __device) const
  {
    _CCCL_ASSERT(this->has_value(), "Cannot add a device to an empty logical endpoint");
    multicast_logical_endpoint_ref{this->id()}.add_device(__device);
  }
};

//! @brief Converts a multicast logical endpoint owner to the ref passed to kernels by `cuda::launch`.
//!
//! @param[in] __endpoint The non-empty owning endpoint.
//! @return A non-owning multicast endpoint ref for the kernel argument list.
[[nodiscard]] _CCCL_HOST_API constexpr multicast_logical_endpoint_ref
transform_launch_argument(::cuda::stream_ref, const multicast_logical_endpoint& __endpoint) noexcept
{
  _CCCL_ASSERT(__endpoint.has_value(), "Cannot pass an empty logical endpoint to a kernel");
  return multicast_logical_endpoint_ref{__endpoint.id()};
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && _CCCL_CTK_AT_LEAST(13, 3) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___LOGICAL_ENDPOINT_MULTICAST_H
