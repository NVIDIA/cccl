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
#  include <cuda/__logical_endpoint/fwd.h>
#  include <cuda/__memory_resource/shared_block_ptr.h>
#  include <cuda/std/__cccl/unreachable.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__thread/threading_support.h>
#  include <cuda/std/__type_traits/underlying_type.h>
#  include <cuda/std/__utility/exchange.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/__utility/pair.h>
#  include <cuda/std/chrono>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

namespace __detail
{
enum class __logical_endpoint_type : ::cuda::std::underlying_type_t<::CUlogicalEndpointType>
{
  __invalid   = ::CU_LOGICAL_ENDPOINT_TYPE_INVALID,
  __unicast   = ::CU_LOGICAL_ENDPOINT_TYPE_UNICAST,
  __multicast = ::CU_LOGICAL_ENDPOINT_TYPE_MULTICAST
};

struct __logical_endpoint_id_range_state;

template <class _IsReady>
[[nodiscard]] _CCCL_HOST_API bool
__wait_until_ready_with_backoff(_IsReady __is_ready, ::cuda::std::chrono::nanoseconds __timeout);
} // namespace __detail

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
  //! @param[in] __id The CUDA logical endpoint ID.
  _CCCL_API constexpr explicit logical_endpoint_id(native_handle_type __id) noexcept
      : __id_(__id)
  {}

  //! @brief Returns the native CUDA logical endpoint ID.
  //!
  //! @return The wrapped CUDA logical endpoint ID.
  [[nodiscard]] _CCCL_API constexpr native_handle_type native_handle() const noexcept
  {
    return __id_;
  }

  _CCCL_API constexpr logical_endpoint_id& operator+=(native_handle_type __offset) noexcept
  {
    __id_ = static_cast<native_handle_type>(__id_ + __offset);
    return *this;
  }

  _CCCL_API constexpr logical_endpoint_id& operator-=(native_handle_type __offset) noexcept
  {
    __id_ = static_cast<native_handle_type>(__id_ - __offset);
    return *this;
  }

  [[nodiscard]] friend _CCCL_API constexpr logical_endpoint_id
  operator+(logical_endpoint_id __id, native_handle_type __offset) noexcept
  {
    __id += __offset;
    return __id;
  }

  [[nodiscard]] friend _CCCL_API constexpr logical_endpoint_id
  operator+(native_handle_type __offset, logical_endpoint_id __id) noexcept
  {
    return __id + __offset;
  }

  [[nodiscard]] friend _CCCL_API constexpr logical_endpoint_id
  operator-(logical_endpoint_id __id, native_handle_type __offset) noexcept
  {
    __id -= __offset;
    return __id;
  }

  [[nodiscard]] friend _CCCL_API constexpr bool operator==(logical_endpoint_id __lhs, logical_endpoint_id __rhs) noexcept
  {
    return __lhs.__id_ == __rhs.__id_;
  }

#  if _CCCL_STD_VER <= 2017
  [[nodiscard]] friend
    _CCCL_API constexpr bool operator!=(logical_endpoint_id __lhs, logical_endpoint_id __rhs) noexcept
  {
    return __lhs.__id_ != __rhs.__id_;
  }
#  endif // _CCCL_STD_VER <= 2017
};

namespace __detail
{
struct __logical_endpoint_id_range_state
{
  ::cuda::std::uint32_t __count_{};
  logical_endpoint_id __base_id_{0};

  _CCCL_HOST_API explicit __logical_endpoint_id_range_state(::cuda::std::uint32_t __count)
      : __count_([__count] {
        if (__count == 0)
        {
          _CCCL_THROW(::std::invalid_argument, "Cannot reserve an empty logical endpoint ID range");
        }
        return __count;
      }())
      , __base_id_(::cuda::__driver::__logicalEndpointIdReserve(__count_))
  {}

  __logical_endpoint_id_range_state(const __logical_endpoint_id_range_state&)            = delete;
  __logical_endpoint_id_range_state& operator=(const __logical_endpoint_id_range_state&) = delete;
  __logical_endpoint_id_range_state(__logical_endpoint_id_range_state&&)                 = delete;
  __logical_endpoint_id_range_state& operator=(__logical_endpoint_id_range_state&&)      = delete;

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
    _CCCL_ASSERT(__count_ != 0, "logical endpoint ID range has no active reservation");
    return __base_id_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr logical_endpoint_id operator[](::cuda::std::uint32_t __index) const noexcept
  {
    _CCCL_ASSERT(__index < __count_, "logical endpoint ID range index is out of bounds");
    return __base_id_ + __index;
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::pair<logical_endpoint_id, ::cuda::std::uint32_t> release() noexcept
  {
    _CCCL_ASSERT(__count_ != 0, "logical endpoint ID range has no active reservation");
    return {__base_id_, ::cuda::std::exchange(__count_, 0)};
  }

private:
  _CCCL_HOST_API void __release_reserved_ids_no_throw() noexcept
  {
    [[maybe_unused]] const auto __status =
      ::cuda::__driver::__logicalEndpointIdReleaseNoThrow(__base_id_.native_handle(), __count_);
    __count_ = 0;
  }
};
} // namespace __detail

//! @brief An owning, ref-counted reservation of contiguous CUDA logical endpoint IDs.
class logical_endpoint_id_range
{
  // TODO: __shared_block_ptr is not memory-resource-specific; move it out of ::cuda::mr.
  ::cuda::mr::__shared_block_ptr<::cuda::__detail::__logical_endpoint_id_range_state> __range_{};

public:
  //! @brief Creates an empty logical endpoint ID range.
  _CCCL_HOST_API logical_endpoint_id_range() noexcept {}

  //! @brief Reserves a contiguous range of CUDA logical endpoint IDs.
  //!
  //! @param[in] __count The number of endpoint IDs to reserve.
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

  _CCCL_HOST_API ~logical_endpoint_id_range() {} // NOLINT(bugprone-exception-escape)

  //! @brief Returns the number of IDs still owned by this reservation.
  //!
  //! @return The reservation size, or zero for an empty, released, or moved-from range.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::uint32_t size() const noexcept
  {
    return __range_ ? __range_.__payload().size() : 0;
  }

  //! @brief Returns the first ID in the reserved range.
  //!
  //! Calling this function requires `size() != 0`.
  //!
  //! @return The base logical endpoint ID.
  [[nodiscard]] _CCCL_HOST_API logical_endpoint_id base_id() const noexcept
  {
    _CCCL_ASSERT(size() != 0, "logical endpoint ID range has no active reservation");
    return __range_.__payload().base_id();
  }

  //! @brief Returns an ID from the reserved contiguous range.
  //!
  //! Calling this function requires `__index < size()`.
  //!
  //! @param[in] __index The zero-based index into the reserved range.
  //! @return `base_id() + __index`.
  [[nodiscard]] _CCCL_HOST_API logical_endpoint_id operator[](::cuda::std::uint32_t __index) const noexcept
  {
    _CCCL_ASSERT(__index < size(), "logical endpoint ID range index is out of bounds");
    return __range_.__payload()[__index];
  }

  //! @brief Releases ownership of the reserved ID range without releasing it to the CUDA driver.
  //!
  //! Calling this function requires `size() != 0`.
  //!
  //! @return The base ID and number of released IDs.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::pair<logical_endpoint_id, ::cuda::std::uint32_t> release() noexcept
  {
    _CCCL_ASSERT(size() != 0, "logical endpoint ID range has no active reservation");
    return __range_.__payload().release();
  }
};

namespace __detail
{
template <class _IsReady>
[[nodiscard]] _CCCL_HOST_API bool
__wait_until_ready_with_backoff(_IsReady __is_ready, ::cuda::std::chrono::nanoseconds __timeout)
{
  constexpr int __polling_count = 16;
  const auto __start            = ::cuda::std::chrono::high_resolution_clock::now();

  for (int __count = 0;;)
  {
    const auto __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
    if (__timeout != ::cuda::std::chrono::nanoseconds::zero() && __timeout <= __elapsed)
    {
      return false;
    }

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

  _CCCL_UNREACHABLE();
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
  //! @param[in] __id The logical endpoint ID.
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
    return ::cuda::__driver::__logicalEndpointQuery(native_handle(), /*__count=*/1);
  }

  //! @brief Waits until the referenced endpoint is ready or a timeout expires.
  //!
  //! @param[in] __timeout The timeout duration; zero means wait indefinitely.
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
  //! @param[in] __device The device whose memory is being bound.
  //! @param[in] __endpoint_offset The byte offset in the logical endpoint.
  //! @param[in] __ptr The device pointer to bind.
  //! @param[in] __bytes The number of bytes to bind.
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
      native_handle(), ::cuda::__driver::__deviceGet(__device.get()), __endpoint_offset, __ptr, __bytes);
  }

  //! @brief Binds a generic allocation handle range to an endpoint offset.
  //!
  //! @param[in] __device The device whose memory is being bound.
  //! @param[in] __endpoint_offset The byte offset in the logical endpoint.
  //! @param[in] __handle The CUDA generic allocation handle to bind.
  //! @param[in] __handle_offset The byte offset in the generic allocation handle.
  //! @param[in] __bytes The number of bytes to bind.
  //! @param[in] __bind_flags CUDA logical endpoint bind flags.
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
      __endpoint_offset,
      __handle,
      __handle_offset,
      __bytes,
      __bind_flags);
  }

  //! @brief Unbinds an endpoint byte range for a device.
  //!
  //! @param[in] __device The device whose binding is being removed.
  //! @param[in] __endpoint_offset The byte offset in the logical endpoint.
  //! @param[in] __bytes The number of bytes to unbind.
  _CCCL_HOST_API void
  unbind(::cuda::device_ref __device, ::cuda::std::uint64_t __endpoint_offset, ::cuda::std::uint64_t __bytes) const
  {
    ::cuda::__driver::__logicalEndpointUnbind(
      native_handle(), ::cuda::__driver::__deviceGet(__device.get()), __endpoint_offset, __bytes);
  }
};
} // namespace __detail

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && _CCCL_CTK_AT_LEAST(13, 3) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___LOGICAL_ENDPOINT_COMMON_H
