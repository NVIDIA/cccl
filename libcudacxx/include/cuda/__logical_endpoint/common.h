//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___LOGICAL_ENDPOINT_COMMON_H
#define _CUDA___LOGICAL_ENDPOINT_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && _CCCL_CTK_AT_LEAST(13, 3) && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__device/device_ref.h>
#  include <cuda/__driver/driver_api.h>
#  include <cuda/__memory_resource/shared_block_ptr.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__thread/threading_support.h>
#  include <cuda/std/__utility/exchange.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/__utility/pair.h>
#  include <cuda/std/chrono>
#  include <cuda/std/cstdint>
#  include <cuda/std/optional>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

class logical_endpoint_id;
class logical_endpoint_id_range;
class multicast_logical_endpoint;
class multicast_logical_endpoint_ref;
class multicast_logical_endpoint_spec;
class unicast_logical_endpoint;
class unicast_logical_endpoint_ref;
class unicast_logical_endpoint_spec;

namespace __detail
{
enum class __logical_endpoint_type
{
  __invalid   = ::CU_LOGICAL_ENDPOINT_TYPE_INVALID,
  __unicast   = ::CU_LOGICAL_ENDPOINT_TYPE_UNICAST,
  __multicast = ::CU_LOGICAL_ENDPOINT_TYPE_MULTICAST
};

struct __logical_endpoint_id_range_state;

template <class _Ref, __logical_endpoint_type _Type>
class __logical_endpoint_owner_base;

template <class _IsReady>
[[nodiscard]] _CCCL_HOST_API bool
__wait_until_ready_with_backoff(_IsReady __is_ready, ::cuda::std::chrono::nanoseconds __timeout);

[[nodiscard]] _CCCL_HOST_API constexpr bool
__handle_types_include_fabric(::CUmemAllocationHandleType __handle_types) noexcept
{
  return (static_cast<unsigned>(__handle_types) & static_cast<unsigned>(::CU_MEM_HANDLE_TYPE_FABRIC)) != 0;
}

[[nodiscard]] _CCCL_HOST_API inline bool __allocation_is_known_not_fabric_exportable(const void* __ptr) noexcept
{
  ::CUmemoryPool __mempool{};
  const auto __mempool_status =
    ::cuda::__driver::__pointerGetAttributeNoThrow<::CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE>(__mempool, __ptr);
  if (__mempool_status == ::cudaSuccess && __mempool != nullptr)
  {
    ::CUmemAllocationHandleType __pool_handle_types{};
    const auto __handle_types_status = ::cuda::__driver::__mempoolGetAttributeNoThrow(
      __mempool, ::CU_MEMPOOL_ATTR_EXPORT_HANDLE_TYPES, &__pool_handle_types);
    if (__handle_types_status == ::cudaSuccess)
    {
      return !::cuda::__detail::__handle_types_include_fabric(__pool_handle_types);
    }
  }

  ::CUmemAllocationHandleType __allowed_handle_types{};
  const auto __allowed_handle_types_status =
    ::cuda::__driver::__pointerGetAttributeNoThrow<::CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES>(
      __allowed_handle_types, __ptr);
  return __allowed_handle_types_status == ::cudaSuccess
      && !::cuda::__detail::__handle_types_include_fabric(__allowed_handle_types);
}
} // namespace __detail

//! @brief CUDA logical endpoint creation flags.
enum class logical_endpoint_flag : unsigned
{
  none        = ::CU_LOGICAL_ENDPOINT_FLAG_NONE,
  counted_ops = ::CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS
};

//! @brief Combines CUDA logical endpoint creation flags.
//!
//! @param __lhs The first flag set to combine.
//! @param __rhs The second flag set to combine.
//! @return The combined flag set.
[[nodiscard]] _CCCL_HOST_API constexpr logical_endpoint_flag
operator|(logical_endpoint_flag __lhs, logical_endpoint_flag __rhs) noexcept
{
  return static_cast<logical_endpoint_flag>(static_cast<unsigned>(__lhs) | static_cast<unsigned>(__rhs));
}

//! @brief CUDA logical endpoint IPC handle kinds.
enum class logical_endpoint_ipc_handle_type : unsigned
{
  none   = ::CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_NONE,
  fabric = ::CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC
};

namespace __detail
{
_CCCL_HOST_API inline void __throw_logical_endpoint_bind_addr_error(
  ::cudaError_t __status, logical_endpoint_ipc_handle_type __ipc, const void* __ptr)
{
  if (__status == ::cudaErrorInvalidValue && __ipc == logical_endpoint_ipc_handle_type::fabric
      && ::cuda::__detail::__allocation_is_known_not_fabric_exportable(__ptr))
  {
    _CCCL_THROW(::cuda::cuda_error,
                __status,
                "Failed to bind a virtual address to a fabric logical endpoint. Fabric logical endpoints require "
                "backing memory from a memory pool or allocation created with cudaMemHandleTypeFabric");
  }

  _CCCL_THROW(::cuda::cuda_error, __status, "Failed to bind a virtual address to a logical endpoint");
}
} // namespace __detail

//! @brief CUDA logical endpoint size and binding limits.
struct logical_endpoint_limits
{
  ::cuda::std::uint64_t bind_alignment{};
  ::cuda::std::uint64_t max_size{};
};

//! @brief A CUDA logical endpoint ID.
//!
//! `logical_endpoint_id` is an endpoint-kind-agnostic value type. It identifies a logical endpoint slot, but it does
//! not encode whether that slot names a unicast endpoint, a multicast endpoint, an imported endpoint, or no endpoint.
//! Constructing `unicast_logical_endpoint_ref` or `multicast_logical_endpoint_ref` from an ID creates the corresponding
//! typed non-owning view and asserts that the ID refers to an endpoint of that kind.
//!
//! ID arithmetic is intended for IDs obtained from a contiguous `logical_endpoint_id_range`. The ID type itself does
//! not own or retain the reservation.
class logical_endpoint_id
{
  ::CUlogicalEndpointId __id_{};

public:
  using native_handle_type = ::CUlogicalEndpointId;

  //! @brief Creates an ID wrapper from a native CUDA logical endpoint ID.
  //!
  //! This constructor is intentionally implicit so APIs can accept native IDs without duplicating overloads.
  //!
  //! @param __id The CUDA logical endpoint ID.
  _CCCL_HOST_DEVICE_API constexpr logical_endpoint_id(native_handle_type __id) noexcept
      : __id_(__id)
  {}

  //! @brief Returns the native CUDA logical endpoint ID.
  //!
  //! @return The wrapped CUDA logical endpoint ID.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr native_handle_type native_handle() const noexcept
  {
    return __id_;
  }

  _CCCL_HOST_DEVICE_API constexpr logical_endpoint_id& operator+=(native_handle_type __offset) noexcept
  {
    __id_ = static_cast<native_handle_type>(__id_ + __offset);
    return *this;
  }

  _CCCL_HOST_DEVICE_API constexpr logical_endpoint_id& operator-=(native_handle_type __offset) noexcept
  {
    __id_ = static_cast<native_handle_type>(__id_ - __offset);
    return *this;
  }

  [[nodiscard]] friend _CCCL_HOST_DEVICE_API constexpr logical_endpoint_id
  operator+(logical_endpoint_id __id, native_handle_type __offset) noexcept
  {
    __id += __offset;
    return __id;
  }

  [[nodiscard]] friend _CCCL_HOST_DEVICE_API constexpr logical_endpoint_id
  operator+(native_handle_type __offset, logical_endpoint_id __id) noexcept
  {
    return __id + __offset;
  }

  [[nodiscard]] friend _CCCL_HOST_DEVICE_API constexpr logical_endpoint_id
  operator-(logical_endpoint_id __id, native_handle_type __offset) noexcept
  {
    __id -= __offset;
    return __id;
  }

  [[nodiscard]] friend _CCCL_HOST_DEVICE_API constexpr bool
  operator==(logical_endpoint_id __lhs, logical_endpoint_id __rhs) noexcept
  {
    return __lhs.__id_ == __rhs.__id_;
  }

#  if _CCCL_STD_VER <= 2017
  [[nodiscard]] friend _CCCL_HOST_DEVICE_API constexpr bool
  operator!=(logical_endpoint_id __lhs, logical_endpoint_id __rhs) noexcept
  {
    return __lhs.__id_ != __rhs.__id_;
  }
#  endif // _CCCL_STD_VER <= 2017
};

namespace __detail
{
struct __logical_endpoint_id_range_state
{
  logical_endpoint_id __base_id_{0};
  ::cuda::std::uint32_t __count_{};

  _CCCL_HOST_API explicit __logical_endpoint_id_range_state(::cuda::std::uint32_t __count)
      : __count_(__count)
  {
    if (__count_ == 0)
    {
      _CCCL_THROW(::std::invalid_argument, "Cannot reserve an empty logical endpoint ID range");
    }
    __base_id_ = logical_endpoint_id{::cuda::__driver::__logicalEndpointIdReserve(__count_)};
  }

  __logical_endpoint_id_range_state(const __logical_endpoint_id_range_state&)            = delete;
  __logical_endpoint_id_range_state& operator=(const __logical_endpoint_id_range_state&) = delete;

  _CCCL_HOST_API ~__logical_endpoint_id_range_state()
  {
    this->__release_reserved_ids_no_throw();
  }

  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::uint32_t size() const noexcept
  {
    return __count_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr logical_endpoint_id base_id() const noexcept
  {
    return __base_id_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr logical_endpoint_id operator[](::cuda::std::uint32_t __index) const noexcept
  {
    _CCCL_ASSERT(__index < __count_, "logical endpoint ID range index is out of bounds");
    return __base_id_ + __index;
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::pair<logical_endpoint_id, ::cuda::std::uint32_t> release() noexcept
  {
    return {__base_id_, ::cuda::std::exchange(__count_, 0)};
  }

private:
  _CCCL_HOST_API void __release_reserved_ids_no_throw() noexcept
  {
    if (__count_ != 0)
    {
      [[maybe_unused]] const auto __status =
        ::cuda::__driver::__logicalEndpointIdReleaseNoThrow(__base_id_.native_handle(), __count_);
      __count_ = 0;
    }
  }
};
} // namespace __detail

//! @brief An owning, ref-counted reservation of contiguous CUDA logical endpoint IDs.
class logical_endpoint_id_range
{
  // TODO: __shared_block_ptr is not memory-resource-specific; move it out of ::cuda::mr.
  ::cuda::mr::__shared_block_ptr<::cuda::__detail::__logical_endpoint_id_range_state> __range_{};

public:
  //! @brief Reserves a contiguous range of CUDA logical endpoint IDs.
  //!
  //! @param __count The number of endpoint IDs to reserve.
  _CCCL_HOST_API explicit logical_endpoint_id_range(::cuda::std::uint32_t __count)
      : __range_(__count)
  {}

  _CCCL_HOST_API logical_endpoint_id_range(const logical_endpoint_id_range& __other) noexcept
      : __range_(__other.__range_)
  {}

  _CCCL_HOST_API logical_endpoint_id_range(logical_endpoint_id_range&& __other) noexcept
      : __range_(::cuda::std::move(__other.__range_))
  {}

  _CCCL_HOST_API logical_endpoint_id_range& operator=(const logical_endpoint_id_range& __other) noexcept
  {
    __range_ = __other.__range_;
    return *this;
  }

  _CCCL_HOST_API logical_endpoint_id_range& operator=(logical_endpoint_id_range&& __other) noexcept
  {
    __range_ = ::cuda::std::move(__other.__range_);
    return *this;
  }

  _CCCL_HOST_API ~logical_endpoint_id_range() {}

  //! @brief Returns the number of IDs still owned by this reservation.
  //!
  //! @return The reservation size, or zero after release/move.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::uint32_t size() const noexcept
  {
    return __range_ ? __range_.__payload().size() : 0;
  }

  //! @brief Returns the first ID in the reserved range.
  //!
  //! @return The base logical endpoint ID.
  [[nodiscard]] _CCCL_HOST_API logical_endpoint_id base_id() const noexcept
  {
    _CCCL_ASSERT(static_cast<bool>(__range_), "logical endpoint ID range has no reservation");
    return __range_.__payload().base_id();
  }

  //! @brief Returns an ID from the reserved contiguous range.
  //!
  //! @param __index The zero-based index into the reserved range.
  //! @return `base_id() + __index`.
  [[nodiscard]] _CCCL_HOST_API logical_endpoint_id operator[](::cuda::std::uint32_t __index) const noexcept
  {
    _CCCL_ASSERT(static_cast<bool>(__range_), "logical endpoint ID range has no reservation");
    return __range_.__payload()[__index];
  }

  //! @brief Releases ownership of the reserved ID range without releasing it to the CUDA driver.
  //!
  //! @return The base ID and number of released IDs.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::pair<logical_endpoint_id, ::cuda::std::uint32_t> release() noexcept
  {
    if (!__range_)
    {
      return {logical_endpoint_id{0}, 0};
    }
    return __range_.__payload().release();
  }

  [[nodiscard]] _CCCL_HOST_API auto __shared_state() const noexcept
    -> ::cuda::mr::__shared_block_ptr<::cuda::__detail::__logical_endpoint_id_range_state>
  {
    return __range_;
  }

private:
  _CCCL_HOST_API explicit logical_endpoint_id_range(
    ::cuda::mr::__shared_block_ptr<::cuda::__detail::__logical_endpoint_id_range_state>&& __range) noexcept
      : __range_(::cuda::std::move(__range))
  {}

  template <class, ::cuda::__detail::__logical_endpoint_type>
  friend class ::cuda::__detail::__logical_endpoint_owner_base;
};

namespace __detail
{
[[nodiscard]] _CCCL_HOST_API inline logical_endpoint_limits
__get_logical_endpoint_limits(const ::CUlogicalEndpointProp& __prop)
{
  const auto __limits = ::cuda::__driver::__logicalEndpointGetLimits(&__prop);
  return {static_cast<::cuda::std::uint64_t>(__limits.first), static_cast<::cuda::std::uint64_t>(__limits.second)};
}

[[nodiscard]] _CCCL_HOST_API inline bool __is_logical_endpoint_supported(
  ::cuda::device_ref __device,
  ::CUdevice_attribute __endpoint_attr,
  logical_endpoint_ipc_handle_type __ipc,
  logical_endpoint_flag __flags)
{
  ::CUdevice __native_device{};
  if (::cuda::__driver::__deviceGetNoThrow(&__native_device, __device.get()) != ::cudaSuccess)
  {
    return false;
  }

  int __attr_value{};
  if (::cuda::__driver::__deviceGetAttributeNoThrow(&__attr_value, __endpoint_attr, __native_device) != ::cudaSuccess
      || __attr_value == 0)
  {
    return false;
  }

  if (__ipc == logical_endpoint_ipc_handle_type::fabric)
  {
    if (::cuda::__driver::__deviceGetAttributeNoThrow(
          &__attr_value, ::CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, __native_device)
          != ::cudaSuccess
        || __attr_value == 0)
    {
      return false;
    }
  }

  if ((static_cast<unsigned>(__flags) & static_cast<unsigned>(logical_endpoint_flag::counted_ops)) != 0)
  {
    if (::cuda::__driver::__deviceGetAttributeNoThrow(
          &__attr_value, ::CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED, __native_device)
          != ::cudaSuccess
        || __attr_value == 0)
    {
      return false;
    }
  }

  return true;
}

template <class _IsReady>
[[nodiscard]] _CCCL_HOST_API bool
__wait_until_ready_with_backoff(_IsReady __is_ready, ::cuda::std::chrono::nanoseconds __timeout)
{
  constexpr int __polling_count = 16;
  const auto __start            = ::cuda::std::chrono::high_resolution_clock::now();

  for (int __count = 0;;)
  {
    if (__is_ready())
    {
      return true;
    }

    if (__count < __polling_count)
    {
      if (__count > (__polling_count / 2))
      {
        ::cuda::std::__cccl_thread_yield_processor();
      }
      ++__count;
      continue;
    }

    const auto __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
    if (__timeout != ::cuda::std::chrono::nanoseconds::zero() && __timeout < __elapsed)
    {
      return false;
    }

    const auto __step = __elapsed / 4;
    if (__step >= ::cuda::std::chrono::milliseconds(1))
    {
      ::cuda::std::__cccl_thread_sleep_for(::cuda::std::chrono::milliseconds(1));
    }
    else if (__step >= ::cuda::std::chrono::microseconds(10))
    {
      ::cuda::std::__cccl_thread_sleep_for(__step);
    }
    else
    {
      ::cuda::std::__cccl_thread_yield();
    }
  }
}
} // namespace __detail

namespace __detail
{
template <__logical_endpoint_type _Type>
class __logical_endpoint_ref_base
{
protected:
  logical_endpoint_id __id_;

public:
  //! @brief Creates a logical endpoint reference base from a logical endpoint ID.
  //!
  //! @param __id The logical endpoint ID.
  _CCCL_HOST_DEVICE_API explicit constexpr __logical_endpoint_ref_base(logical_endpoint_id __id) noexcept
      : __id_(__id)
  {}

  //! @brief Returns the referenced logical endpoint ID.
  //!
  //! @return The logical endpoint ID.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr logical_endpoint_id id() const noexcept
  {
    return __id_;
  }

  //! @brief Returns the native CUDA logical endpoint ID.
  //!
  //! @return The native CUDA logical endpoint ID.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::CUlogicalEndpointId native_handle() const noexcept
  {
    return __id_.native_handle();
  }

  //! @brief Queries whether the referenced endpoint is ready.
  //!
  //! @return `true` if the endpoint is ready.
  [[nodiscard]] _CCCL_HOST_API bool is_ready() const
  {
    return ::cuda::__driver::__logicalEndpointQuery(native_handle(), 1);
  }

  //! @brief Waits until the referenced endpoint is ready or a timeout expires.
  //!
  //! @param __timeout The timeout duration; zero means wait indefinitely.
  //! @return `true` if the endpoint became ready before timeout.
  [[nodiscard]] _CCCL_HOST_API bool
  wait_until_ready(::cuda::std::chrono::nanoseconds __timeout = ::cuda::std::chrono::nanoseconds::zero()) const
  {
    return ::cuda::__detail::__wait_until_ready_with_backoff(
      [this] {
        return this->is_ready();
      },
      __timeout);
  }

  //! @brief Requires the referenced endpoint to be ready.
  _CCCL_HOST_API void require_ready() const
  {
    if (!is_ready())
    {
      _CCCL_THROW(::cuda::cuda_error, ::cudaErrorNotReady, "Logical endpoint is not ready");
    }
  }

  //! @brief Binds a device pointer range to an endpoint offset.
  //!
  //! @param __device The device whose memory is being bound.
  //! @param __endpoint_offset The byte offset in the logical endpoint.
  //! @param __ptr The device pointer to bind.
  //! @param __bytes The number of bytes to bind.
  _CCCL_HOST_API void bind(::cuda::device_ref __device,
                           ::cuda::std::uint64_t __endpoint_offset,
                           void* __ptr,
                           ::cuda::std::uint64_t __bytes) const
  {
    if (__ptr == nullptr)
    {
      _CCCL_THROW(::std::invalid_argument, "Cannot bind a null pointer to a logical endpoint");
    }
    ::cuda::__driver::__logicalEndpointBindAddr(
      native_handle(),
      ::cuda::__driver::__deviceGet(__device.get()),
      static_cast<::cuuint64_t>(__endpoint_offset),
      __ptr,
      static_cast<::cuuint64_t>(__bytes));
  }

  //! @brief Binds a generic allocation handle range to an endpoint offset.
  //!
  //! @param __device The device whose memory is being bound.
  //! @param __endpoint_offset The byte offset in the logical endpoint.
  //! @param __handle The CUDA generic allocation handle to bind.
  //! @param __handle_offset The byte offset in the generic allocation handle.
  //! @param __bytes The number of bytes to bind.
  //! @param __bind_flags CUDA logical endpoint bind flags.
  _CCCL_HOST_API void
  bind(::cuda::device_ref __device,
       ::cuda::std::uint64_t __endpoint_offset,
       ::CUmemGenericAllocationHandle __handle,
       ::cuda::std::uint64_t __handle_offset,
       ::cuda::std::uint64_t __bytes,
       unsigned int __bind_flags = 0) const
  {
    ::cuda::__driver::__logicalEndpointBindMem(
      native_handle(),
      ::cuda::__driver::__deviceGet(__device.get()),
      static_cast<::cuuint64_t>(__endpoint_offset),
      __handle,
      static_cast<::cuuint64_t>(__handle_offset),
      static_cast<::cuuint64_t>(__bytes),
      __bind_flags);
  }

  //! @brief Unbinds an endpoint byte range for a device.
  //!
  //! @param __device The device whose binding is being removed.
  //! @param __endpoint_offset The byte offset in the logical endpoint.
  //! @param __bytes The number of bytes to unbind.
  _CCCL_HOST_API void
  unbind(::cuda::device_ref __device, ::cuda::std::uint64_t __endpoint_offset, ::cuda::std::uint64_t __bytes) const
  {
    ::cuda::__driver::__logicalEndpointUnbind(
      native_handle(), ::cuda::__driver::__deviceGet(__device.get()), __endpoint_offset, __bytes);
  }
};
} // namespace __detail

namespace __detail
{
template <class _Ref, __logical_endpoint_type _Type>
class __logical_endpoint_owner_base : public _Ref
{
  friend class ::cuda::unicast_logical_endpoint;
  friend class ::cuda::multicast_logical_endpoint;

  static constexpr ::cuda::std::uint64_t __default_bind_alignment = 0;

  ::cuda::mr::__shared_block_ptr<__logical_endpoint_id_range_state> __range_ref_{};
  // Lifetime state is independent from endpoint metadata. A future driver may allow useful zero-size endpoints.
  bool __owns_endpoint_ = false;
  ::cuda::std::uint64_t __size_{};
  ::cuda::std::uint64_t __bind_alignment_             = __default_bind_alignment;
  logical_endpoint_ipc_handle_type __ipc_handle_type_ = logical_endpoint_ipc_handle_type::none;

protected:
  _CCCL_HOST_API __logical_endpoint_owner_base() noexcept
      : _Ref{logical_endpoint_id{0}}
  {}

  _CCCL_HOST_API explicit __logical_endpoint_owner_base(logical_endpoint_id __id) noexcept
      : _Ref{__id}
  {}

  _CCCL_HOST_API void __retain_id_range(const logical_endpoint_id_range& __range) noexcept
  {
    __range_ref_ = __range.__shared_state();
  }

  _CCCL_HOST_API void __create_endpoint(const ::CUlogicalEndpointProp& __prop)
  {
    if (static_cast<__logical_endpoint_type>(__prop.type) != _Type)
    {
      _CCCL_THROW(::std::invalid_argument, "Logical endpoint property type does not match the endpoint type");
    }

    const auto __limits = ::cuda::__detail::__get_logical_endpoint_limits(__prop);
    if (__limits.max_size != 0 && __prop.size > __limits.max_size)
    {
      _CCCL_THROW(::std::invalid_argument, "Logical endpoint size exceeds the device limit");
    }

    ::cuda::__driver::__logicalEndpointCreate(this->native_handle(), &__prop);

    __owns_endpoint_   = true;
    __size_            = __prop.size;
    __bind_alignment_  = __limits.bind_alignment;
    __ipc_handle_type_ = static_cast<logical_endpoint_ipc_handle_type>(__prop.ipcHandleTypes);
  }

public:
  using release_type = ::cuda::std::pair<logical_endpoint_id, ::cuda::std::optional<logical_endpoint_id_range>>;

  _CCCL_HOST_API __logical_endpoint_owner_base(__logical_endpoint_owner_base&& __other) noexcept
      : _Ref{__other.id()}
      , __range_ref_(::cuda::std::move(__other.__range_ref_))
      , __owns_endpoint_(::cuda::std::exchange(__other.__owns_endpoint_, false))
      , __size_(::cuda::std::exchange(__other.__size_, 0))
      , __bind_alignment_(::cuda::std::exchange(__other.__bind_alignment_, __default_bind_alignment))
      , __ipc_handle_type_(::cuda::std::exchange(__other.__ipc_handle_type_, logical_endpoint_ipc_handle_type::none))
  {
    static_cast<_Ref&>(__other) = _Ref{logical_endpoint_id{0}};
  }

  _CCCL_HOST_API __logical_endpoint_owner_base& operator=(__logical_endpoint_owner_base&& __other) noexcept
  {
    if (this != &__other)
    {
      this->__reset_no_throw();
      static_cast<_Ref&>(*this)   = _Ref{__other.id()};
      static_cast<_Ref&>(__other) = _Ref{logical_endpoint_id{0}};
      __range_ref_                = ::cuda::std::move(__other.__range_ref_);
      __owns_endpoint_            = ::cuda::std::exchange(__other.__owns_endpoint_, false);
      __size_                     = ::cuda::std::exchange(__other.__size_, 0);
      __bind_alignment_           = ::cuda::std::exchange(__other.__bind_alignment_, __default_bind_alignment);
      __ipc_handle_type_ = ::cuda::std::exchange(__other.__ipc_handle_type_, logical_endpoint_ipc_handle_type::none);
    }
    return *this;
  }

  __logical_endpoint_owner_base(const __logical_endpoint_owner_base&)            = delete;
  __logical_endpoint_owner_base& operator=(const __logical_endpoint_owner_base&) = delete;

  _CCCL_HOST_API ~__logical_endpoint_owner_base()
  {
    this->__reset_no_throw();
  }

  //! @brief Checks whether this object owns a created logical endpoint.
  //!
  //! @return `true` if this object owns an endpoint.
  [[nodiscard]] _CCCL_HOST_API constexpr bool has_value() const noexcept
  {
    return __owns_endpoint_;
  }

  //! @brief Queries whether the owned endpoint is ready.
  //!
  //! @return `true` if the endpoint is ready.
  [[nodiscard]] _CCCL_HOST_API bool is_ready() const
  {
    _CCCL_ASSERT(has_value(), "Cannot query an empty logical endpoint");
    return _Ref::is_ready();
  }

  //! @brief Waits until the owned endpoint is ready or a timeout expires.
  //!
  //! @param __timeout The timeout duration; zero means wait indefinitely.
  //! @return `true` if the endpoint became ready before timeout.
  [[nodiscard]] _CCCL_HOST_API bool
  wait_until_ready(::cuda::std::chrono::nanoseconds __timeout = ::cuda::std::chrono::nanoseconds::zero()) const
  {
    _CCCL_ASSERT(has_value(), "Cannot query an empty logical endpoint");
    return _Ref::wait_until_ready(__timeout);
  }

  //! @brief Requires the owned endpoint to be ready.
  _CCCL_HOST_API void require_ready() const
  {
    _CCCL_ASSERT(has_value(), "Cannot query an empty logical endpoint");
    _Ref::require_ready();
  }

  using _Ref::bind;

  //! @brief Binds a virtual address range and adds endpoint-aware diagnostics on failure.
  //!
  //! For fabric IPC endpoints, failures caused by non-fabric-exportable backing allocations are diagnosed with a
  //! targeted error message when the allocation metadata is available from the CUDA driver.
  _CCCL_HOST_API void bind(::cuda::device_ref __device,
                           ::cuda::std::uint64_t __endpoint_offset,
                           void* __ptr,
                           ::cuda::std::uint64_t __bytes) const
  {
    _CCCL_ASSERT(has_value(), "Cannot bind memory to an empty logical endpoint");
    if (__ptr == nullptr)
    {
      _CCCL_THROW(::std::invalid_argument, "Cannot bind a null pointer to a logical endpoint");
    }
    const auto __status = ::cuda::__driver::__logicalEndpointBindAddrNoThrow(
      this->native_handle(),
      ::cuda::__driver::__deviceGet(__device.get()),
      static_cast<::cuuint64_t>(__endpoint_offset),
      __ptr,
      static_cast<::cuuint64_t>(__bytes));
    if (__status != ::cudaSuccess)
    {
      ::cuda::__detail::__throw_logical_endpoint_bind_addr_error(__status, __ipc_handle_type_, __ptr);
    }
  }

  _CCCL_HOST_API void
  bind(::cuda::device_ref __device,
       ::cuda::std::uint64_t __endpoint_offset,
       ::CUmemGenericAllocationHandle __handle,
       ::cuda::std::uint64_t __handle_offset,
       ::cuda::std::uint64_t __bytes,
       unsigned int __bind_flags = 0) const
  {
    _CCCL_ASSERT(has_value(), "Cannot bind memory to an empty logical endpoint");
    _Ref::bind(__device, __endpoint_offset, __handle, __handle_offset, __bytes, __bind_flags);
  }

  _CCCL_HOST_API void
  unbind(::cuda::device_ref __device, ::cuda::std::uint64_t __endpoint_offset, ::cuda::std::uint64_t __bytes) const
  {
    _CCCL_ASSERT(has_value(), "Cannot unbind memory from an empty logical endpoint");
    _Ref::unbind(__device, __endpoint_offset, __bytes);
  }

  //! @brief Releases endpoint ownership without destroying the CUDA logical endpoint.
  //!
  //! @return The endpoint ID and an optional retained ID range reservation.
  [[nodiscard]] _CCCL_HOST_API release_type release() noexcept
  {
    _CCCL_ASSERT(has_value(), "Cannot release an empty logical endpoint");
    if (!has_value())
    {
      return {logical_endpoint_id{0}, ::cuda::std::optional<logical_endpoint_id_range>{}};
    }

    __owns_endpoint_ = false;
    if (__range_ref_)
    {
      auto __id_range =
        ::cuda::std::optional<logical_endpoint_id_range>{logical_endpoint_id_range{::cuda::std::move(__range_ref_)}};
      return {this->id(), ::cuda::std::move(__id_range)};
    }
    return {this->id(), ::cuda::std::optional<logical_endpoint_id_range>{}};
  }

  //! @brief Returns the endpoint size captured at creation.
  //!
  //! @return The logical endpoint size in bytes.
  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::uint64_t size() const noexcept
  {
    return __size_;
  }

  //! @brief Returns the bind alignment captured at creation.
  //!
  //! @return The logical endpoint bind alignment in bytes.
  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::uint64_t bind_alignment() const noexcept
  {
    return __bind_alignment_;
  }

private:
  _CCCL_HOST_API void __reset_no_throw() noexcept
  {
    if (__owns_endpoint_)
    {
      [[maybe_unused]] const auto __destroy_status =
        ::cuda::__driver::__logicalEndpointDestroyNoThrow(this->native_handle());
      __owns_endpoint_ = false;
    }
  }
};
} // namespace __detail

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && _CCCL_CTK_AT_LEAST(13, 3) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___LOGICAL_ENDPOINT_COMMON_H
