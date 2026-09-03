//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___LOGICAL_ENDPOINT_UNICAST_H
#define _CUDA___LOGICAL_ENDPOINT_UNICAST_H

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

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Size-independent configuration for a unicast CUDA logical endpoint.
class unicast_logical_endpoint_spec
{
  ::cuda::device_ref __device_;
  logical_endpoint_flag __flags_          = logical_endpoint_flag::none;
  logical_endpoint_ipc_handle_type __ipc_ = logical_endpoint_ipc_handle_type::fabric;

public:
  //! @brief Creates a unicast endpoint specification.
  //!
  //! @param __device The CUDA device for the unicast endpoint.
  //! @param __flags Logical endpoint creation flags.
  //! @param __ipc The IPC handle type requested for this endpoint.
  _CCCL_HOST_API explicit unicast_logical_endpoint_spec(
    ::cuda::device_ref __device,
    logical_endpoint_flag __flags          = logical_endpoint_flag::none,
    logical_endpoint_ipc_handle_type __ipc = logical_endpoint_ipc_handle_type::fabric) noexcept
      : __device_(__device)
      , __flags_(__flags)
      , __ipc_(__ipc)
  {}

  //! @brief Returns the CUDA device for the unicast endpoint.
  //!
  //! @return The endpoint device.
  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::device_ref device() const noexcept
  {
    return __device_;
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

  //! @brief Checks whether a device supports this unicast endpoint configuration.
  //!
  //! Construction still uses `device()`. The optional checker device is only used for capability queries.
  //!
  //! @param __checker Optional device used for support attribute checks.
  //! @return `true` if the requested configuration is supported by the checked device.
  [[nodiscard]] _CCCL_HOST_API bool
  is_supported(::cuda::std::optional<::cuda::device_ref> __checker = ::cuda::std::nullopt) const;

  //! @brief Queries CUDA driver limits for this endpoint configuration.
  //!
  //! @return The required bind alignment and maximum endpoint size.
  [[nodiscard]] _CCCL_HOST_API logical_endpoint_limits limits() const;
};

namespace __detail
{
[[nodiscard]] _CCCL_HOST_API inline ::CUlogicalEndpointProp
__make_unicast_logical_endpoint_prop(const ::cuda::unicast_logical_endpoint_spec& __spec, ::cuda::std::uint64_t __bytes)
{
  ::CUlogicalEndpointProp __prop{};
  __prop.type           = ::CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
  __prop.unicast.device = ::cuda::__driver::__deviceGet(__spec.device().get());
  __prop.size           = __bytes;
  __prop.ipcHandleTypes = static_cast<unsigned>(__spec.ipc_handle_type());
  __prop.flags          = static_cast<unsigned>(__spec.flags());
  return __prop;
}
} // namespace __detail

[[nodiscard]] _CCCL_HOST_API inline bool
unicast_logical_endpoint_spec::is_supported(::cuda::std::optional<::cuda::device_ref> __checker) const
{
  const auto __device = __checker.has_value() ? *__checker : device();
  return ::cuda::__detail::__is_logical_endpoint_supported(
    __device, ::CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED, ipc_handle_type(), flags());
}

[[nodiscard]] _CCCL_HOST_API inline logical_endpoint_limits unicast_logical_endpoint_spec::limits() const
{
  const auto __prop = ::cuda::__detail::__make_unicast_logical_endpoint_prop(*this, 0);
  return ::cuda::__detail::__get_logical_endpoint_limits(__prop);
}

//! @brief Non-owning reference to a unicast CUDA logical endpoint.
//!
//! This type is trivially copyable and can be passed to device code directly, including raw `<<<>>>` kernel launches.
//! Owning `unicast_logical_endpoint` objects are also valid kernel arguments when using `cuda::launch`, which
//! transforms them to this ref type before invoking the kernel.
class unicast_logical_endpoint_ref
    : public ::cuda::__detail::__logical_endpoint_ref_base<::cuda::__detail::__logical_endpoint_type::__unicast>
{
  using __base = ::cuda::__detail::__logical_endpoint_ref_base<::cuda::__detail::__logical_endpoint_type::__unicast>;

public:
  //! @brief Creates a unicast endpoint reference from a logical endpoint ID.
  //!
  //! @param __id The logical endpoint ID.
  _CCCL_HOST_DEVICE_API explicit constexpr unicast_logical_endpoint_ref(logical_endpoint_id __id) noexcept
      : __base(__id)
  {}
};

//! @brief Move-only owning RAII wrapper for a unicast CUDA logical endpoint.
//!
//! This type owns endpoint creation/import and destruction. It can be passed as a kernel argument through
//! `cuda::launch`; CCCL launch argument transformation converts it to `unicast_logical_endpoint_ref`. Raw `<<<>>>`
//! launches do not perform that transformation, so they should pass `unicast_logical_endpoint_ref` explicitly.
class unicast_logical_endpoint
    : public ::cuda::__detail::__logical_endpoint_owner_base<unicast_logical_endpoint_ref,
                                                             ::cuda::__detail::__logical_endpoint_type::__unicast>
{
  using __base = ::cuda::__detail::__logical_endpoint_owner_base<unicast_logical_endpoint_ref,
                                                                 ::cuda::__detail::__logical_endpoint_type::__unicast>;

public:
  //! @brief Creates an empty logical endpoint owner.
  _CCCL_HOST_API unicast_logical_endpoint() noexcept
      : __base()
  {}

  _CCCL_HOST_API unicast_logical_endpoint(unicast_logical_endpoint&& __other) noexcept
      : __base(::cuda::std::move(__other))
  {}

  _CCCL_HOST_API unicast_logical_endpoint& operator=(unicast_logical_endpoint&& __other) noexcept
  {
    static_cast<__base&>(*this) = ::cuda::std::move(static_cast<__base&>(__other));
    return *this;
  }

  unicast_logical_endpoint(const unicast_logical_endpoint&)            = delete;
  unicast_logical_endpoint& operator=(const unicast_logical_endpoint&) = delete;

  //! @brief Reserves one ID and creates a unicast logical endpoint.
  //!
  //! @param __spec The endpoint specification.
  //! @param __bytes The endpoint size in bytes.
  _CCCL_HOST_API explicit unicast_logical_endpoint(const unicast_logical_endpoint_spec& __spec,
                                                   ::cuda::std::uint64_t __bytes)
      : unicast_logical_endpoint(logical_endpoint_id_range{1}, 0, __spec, __bytes)
  {}

  //! @brief Reserves one ID and imports a unicast logical endpoint.
  //!
  //! @param __handle The exported logical endpoint handle.
  _CCCL_HOST_API explicit unicast_logical_endpoint(const logical_endpoint_handle& __handle)
      : unicast_logical_endpoint(logical_endpoint_id_range{1}, 0, __handle)
  {}

  //! @brief Creates a unicast logical endpoint from a caller-managed ID.
  //!
  //! @param __id The caller-managed logical endpoint ID.
  //! @param __spec The endpoint specification.
  //! @param __bytes The endpoint size in bytes.
  _CCCL_HOST_API unicast_logical_endpoint(
    logical_endpoint_id __id, const unicast_logical_endpoint_spec& __spec, ::cuda::std::uint64_t __bytes)
      : __base(__id)
  {
    const auto __prop = ::cuda::__detail::__make_unicast_logical_endpoint_prop(__spec, __bytes);
    this->__create_endpoint(__prop);
  }

  //! @brief Imports a unicast logical endpoint into a caller-managed ID.
  //!
  //! @param __id The caller-managed logical endpoint ID.
  //! @param __handle The exported logical endpoint handle.
  _CCCL_HOST_API unicast_logical_endpoint(logical_endpoint_id __id, const logical_endpoint_handle& __handle)
      : __base(__id)
  {
    this->__import_endpoint(__handle);
  }

  //! @brief Creates a unicast logical endpoint from an ID in a retained range.
  //!
  //! @param __range The logical endpoint ID range to retain.
  //! @param __index The ID index in the range.
  //! @param __spec The endpoint specification.
  //! @param __bytes The endpoint size in bytes.
  _CCCL_HOST_API unicast_logical_endpoint(
    const logical_endpoint_id_range& __range,
    ::cuda::std::uint32_t __index,
    const unicast_logical_endpoint_spec& __spec,
    ::cuda::std::uint64_t __bytes)
      : __base(__range[__index])
  {
    this->__retain_id_range(__range);
    const auto __prop = ::cuda::__detail::__make_unicast_logical_endpoint_prop(__spec, __bytes);
    this->__create_endpoint(__prop);
  }

  //! @brief Imports a unicast logical endpoint into an ID in a retained range.
  //!
  //! @param __range The logical endpoint ID range to retain.
  //! @param __index The ID index in the range.
  //! @param __handle The exported logical endpoint handle.
  _CCCL_HOST_API unicast_logical_endpoint(
    const logical_endpoint_id_range& __range, ::cuda::std::uint32_t __index, const logical_endpoint_handle& __handle)
      : __base(__range[__index])
  {
    this->__retain_id_range(__range);
    this->__import_endpoint(__handle);
  }
};

//! @brief Converts a unicast logical endpoint owner to the ref passed to kernels by `cuda::launch`.
//!
//! @param __endpoint The non-empty owning endpoint.
//! @return A non-owning unicast endpoint ref for the kernel argument list.
[[nodiscard]] _CCCL_HOST_API constexpr unicast_logical_endpoint_ref
transform_launch_argument(::cuda::stream_ref, const unicast_logical_endpoint& __endpoint) noexcept
{
  _CCCL_ASSERT(__endpoint.has_value(), "Cannot pass an empty logical endpoint to a kernel");
  return unicast_logical_endpoint_ref{__endpoint.id()};
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && _CCCL_CTK_AT_LEAST(13, 3) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___LOGICAL_ENDPOINT_UNICAST_H
