// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/std/detail/__config>

#include <cuda/__memory_resource/memory_resource_base.h>
#include <cuda/__memory_resource/properties.h>
#include <cuda/__memory_resource/resource.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/initializer_list>
#include <cuda/stream>

#include <cstddef>

#include <c2h/detail/checked_memory.cuh>
#include <c2h/detail/current_device.cuh>

namespace c2h
{
//! @brief Device memory resource that rejects allocations when insufficient device memory is available.
//!
//! @pre The resource's device must be the current C2H test device when memory is allocated or deallocated.
class checked_device_memory_resource : public ::cuda::mr::memory_resource_base<checked_device_memory_resource>
{
public:
  _CCCL_HOST_API constexpr explicit checked_device_memory_resource(int device = 0) noexcept
      : m_device(device)
  {}

  _CCCL_HOST_API constexpr explicit checked_device_memory_resource(::cuda::device_ref device) noexcept
      : m_device(device.get())
  {}

  [[nodiscard]] _CCCL_HOST_API void*
  allocate_sync(std::size_t bytes, std::size_t alignment = ::cuda::mr::default_cuda_malloc_alignment)
  {
    return ::c2h::detail::checked_device_allocate(m_device, bytes, alignment);
  }

  _CCCL_HOST_API void deallocate_sync(
    void* ptr,
    [[maybe_unused]] std::size_t bytes,
    [[maybe_unused]] std::size_t alignment = ::cuda::mr::default_cuda_malloc_alignment) noexcept
  {
    ::c2h::detail::checked_device_deallocate(m_device, ptr);
  }

  _CCCL_HOST_API friend constexpr void
  get_property(checked_device_memory_resource const&, ::cuda::mr::device_accessible) noexcept
  {}

  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator==(checked_device_memory_resource lhs, checked_device_memory_resource rhs) noexcept
  {
    return lhs.m_device == rhs.m_device;
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator!=(checked_device_memory_resource lhs, checked_device_memory_resource rhs) noexcept
  {
    return !(lhs == rhs);
  }
#endif // _CCCL_STD_VER <= 2017

  using default_queries = ::cuda::mr::properties_list<::cuda::mr::device_accessible>;

private:
  int m_device = 0;
};

static_assert(::cuda::mr::synchronous_resource_with<checked_device_memory_resource, ::cuda::mr::device_accessible>);

//! @brief Creates a device buffer backed by the checked C2H memory resource.
//!
//! @pre @p stream and @p device must refer to the current C2H test device.
template <typename T, typename... Args>
[[nodiscard]] _CCCL_HOST_API ::cuda::device_buffer<T>
make_device_buffer(::cuda::stream_ref stream, ::cuda::device_ref device, Args&&... args)
{
  ::c2h::detail::assert_current_device(stream.device().get());
  ::c2h::detail::assert_current_device(device.get());
  return ::cuda::make_buffer<T>(stream, checked_device_memory_resource{device}, ::cuda::std::forward<Args>(args)...);
}

//! @brief Creates an initialized device buffer backed by the checked C2H memory resource.
//!
//! @pre @p stream and @p device must refer to the current C2H test device.
template <typename T>
[[nodiscard]] _CCCL_HOST_API ::cuda::device_buffer<T>
make_device_buffer(::cuda::stream_ref stream, ::cuda::device_ref device, ::cuda::std::initializer_list<T> values)
{
  ::c2h::detail::assert_current_device(stream.device().get());
  ::c2h::detail::assert_current_device(device.get());
  return ::cuda::make_buffer<T>(stream, checked_device_memory_resource{device}, values);
}

//! @brief Host memory resource that rejects allocations when insufficient integrated-device memory is available.
//!
//! @pre The resource's device must be the current C2H test device when memory is allocated.
class checked_host_buffer_memory_resource : public ::cuda::mr::memory_resource_base<checked_host_buffer_memory_resource>
{
public:
  _CCCL_HOST_API constexpr explicit checked_host_buffer_memory_resource(int device = 0) noexcept
      : m_device(device)
  {}

  _CCCL_HOST_API constexpr explicit checked_host_buffer_memory_resource(::cuda::device_ref device) noexcept
      : m_device(device.get())
  {}

  [[nodiscard]] _CCCL_HOST_API void*
  allocate_sync(std::size_t bytes, std::size_t alignment = ::cuda::mr::default_cuda_malloc_alignment)
  {
    return ::c2h::detail::checked_host_allocate(m_device, bytes, alignment);
  }

  _CCCL_HOST_API void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    ::c2h::detail::checked_host_deallocate(ptr, bytes, alignment);
  }

  _CCCL_HOST_API friend constexpr void
  get_property(checked_host_buffer_memory_resource const&, ::cuda::mr::host_accessible) noexcept
  {}

  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator==(checked_host_buffer_memory_resource lhs, checked_host_buffer_memory_resource rhs) noexcept
  {
    return lhs.m_device == rhs.m_device;
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator!=(checked_host_buffer_memory_resource lhs, checked_host_buffer_memory_resource rhs) noexcept
  {
    return !(lhs == rhs);
  }
#endif // _CCCL_STD_VER <= 2017

  using default_queries = ::cuda::mr::properties_list<::cuda::mr::host_accessible>;

private:
  int m_device = 0;
};

static_assert(::cuda::mr::synchronous_resource_with<checked_host_buffer_memory_resource, ::cuda::mr::host_accessible>);

//! @brief Creates a host buffer backed by the checked C2H memory resource.
//!
//! @pre @p stream and @p device must refer to the current C2H test device.
template <typename T, typename... Args>
[[nodiscard]] _CCCL_HOST_API ::cuda::host_buffer<T>
make_host_buffer(::cuda::stream_ref stream, ::cuda::device_ref device, Args&&... args)
{
  ::c2h::detail::assert_current_device(stream.device().get());
  ::c2h::detail::assert_current_device(device.get());
  return ::cuda::make_buffer<T>(
    stream, checked_host_buffer_memory_resource{device}, ::cuda::std::forward<Args>(args)...);
}
} // namespace c2h
