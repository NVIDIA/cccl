// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/std/detail/__config>

#include <thrust/device_allocator.h>
#include <thrust/mr/new.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/system/cuda/memory_resource.h>
#include <thrust/system/cuda/pointer.h>

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
#  include <cuda/buffer>
#  include <cuda/devices>
#  include <cuda/memory_resource>
#  include <cuda/std/initializer_list>
#  include <cuda/std/utility>
#  include <cuda/stream>
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <new>
#include <optional>
#include <string>

#include <cuda_runtime_api.h>

namespace c2h
{
namespace detail
{
inline std::optional<std::string> get_env(const char* name)
{
#ifdef _WIN32
  char* buf       = nullptr;
  std::size_t len = 0;
  if (_dupenv_s(&buf, &len, name) || !buf)
  {
    return std::nullopt;
  }
  std::string val(buf);
  free(buf);
  return val;
#else
  if (const char* v = std::getenv(name))
  {
    return std::string(v);
  }
  return std::nullopt;
#endif
}

struct memory_info
{
  std::size_t free{};
  std::size_t total{};
  bool override{false};
};

// If the environment variable C2H_DEVICE_MEMORY_LIMIT is set, the total device memory
// will be limited to this number of bytes.
inline std::size_t get_device_memory_limit()
{
  static std::optional<std::string> override_str = get_env("C2H_DEVICE_MEMORY_LIMIT");
  static std::size_t result =
    override_str ? static_cast<std::size_t>(std::strtoll(override_str->c_str(), nullptr, 10)) : 0;
  return result;
}

inline bool get_debug_checked_allocs()
{
  static std::optional<std::string> debug_checked_allocs = get_env("C2H_DEBUG_CHECKED_ALLOC_FAILURES");
  static bool result = debug_checked_allocs && (std::strtol(debug_checked_allocs->c_str(), nullptr, 10) != 0);
  return result;
}

inline cudaError_t get_device_memory(memory_info& info)
{
  static std::size_t device_memory_limit = get_device_memory_limit();

  cudaError_t status = cudaMemGetInfo(&info.free, &info.total);
  if (status != cudaSuccess)
  {
    return status;
  }

  if (device_memory_limit > 0)
  {
    info.free  = (std::max) (std::size_t{0}, static_cast<std::size_t>(info.free - (info.total - device_memory_limit)));
    info.total = device_memory_limit;
    info.override = true;
  }

  return cudaSuccess;
}

inline cudaError_t check_free_device_memory(std::size_t bytes)
{
  memory_info info;
  cudaError_t status = get_device_memory(info);
  if (status != cudaSuccess)
  {
    return status;
  }

  // Avoid allocating all available memory:
  constexpr std::size_t padding = 16 * 1024 * 1024; // 16 MiB
  if (info.free < (bytes + padding))
  {
    if (get_debug_checked_allocs())
    {
      const double total_GiB     = static_cast<double>(info.total) / (1024 * 1024 * 1024);
      const double free_GiB      = static_cast<double>(info.free) / (1024 * 1024 * 1024);
      const double requested_GiB = static_cast<double>(bytes) / (1024 * 1024 * 1024);
      const double padded_GiB    = static_cast<double>(bytes + padding) / (1024 * 1024 * 1024);

      std::cerr << "Device memory allocation failed due to insufficient free device memory.\n";

      if (info.override)
      {
        std::cerr
          << "Available device memory has been limited (env var C2H_DEVICE_MEMORY_LIMIT=" << get_device_memory_limit()
          << ").\n";
      }

      std::cerr
        << "Total device mem:     " << total_GiB << " GiB\n" //
        << "Free device mem:      " << free_GiB << " GiB\n" //
        << "Requested device mem: " << requested_GiB << " GiB\n" //
        << "Padded device mem:    " << padded_GiB << " GiB\n";
    }

    return cudaErrorMemoryAllocation;
  }

  return cudaSuccess;
}

// Check available memory prior to calling cudaMalloc.
// This avoids hangups and slowdowns from allocating swap / non-device memory
// on some platforms, namely tegra.
inline cudaError_t checked_cuda_malloc(void** ptr, std::size_t bytes)
{
  auto status = check_free_device_memory(bytes);
  if (status != cudaSuccess)
  {
    return status;
  }

  return cudaMalloc(ptr, bytes);
}

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
class scoped_current_device
{
public:
  explicit scoped_current_device(int device)
  {
    const cudaError_t get_status = cudaGetDevice(&m_previous_device);
    if (get_status != cudaSuccess)
    {
      throw std::bad_alloc{};
    }

    if (m_previous_device != device)
    {
      const cudaError_t set_status = cudaSetDevice(device);
      if (set_status != cudaSuccess)
      {
        throw std::bad_alloc{};
      }
      m_restore = true;
    }
  }

  scoped_current_device(const scoped_current_device&)            = delete;
  scoped_current_device& operator=(const scoped_current_device&) = delete;

  ~scoped_current_device() noexcept
  {
    if (m_restore)
    {
      (void) cudaSetDevice(m_previous_device);
    }
  }

private:
  int m_previous_device = 0;
  bool m_restore        = false;
};

[[nodiscard]] inline bool is_valid_cuda_malloc_alignment(std::size_t alignment) noexcept
{
  return alignment != 0 && alignment <= ::cuda::mr::default_cuda_malloc_alignment
      && (::cuda::mr::default_cuda_malloc_alignment % alignment == 0);
}

[[nodiscard]] inline void* checked_device_allocate(int device, std::size_t bytes, std::size_t alignment)
{
  if (!is_valid_cuda_malloc_alignment(alignment))
  {
    throw std::bad_alloc{};
  }

  if (bytes == 0)
  {
    return nullptr;
  }

  scoped_current_device guard{device};

  void* ptr                = nullptr;
  const cudaError_t status = checked_cuda_malloc(&ptr, bytes);
  if (status != cudaSuccess)
  {
    (void) cudaGetLastError();
    throw std::bad_alloc{};
  }

  return ptr;
}

inline void checked_device_deallocate(int device, void* ptr) noexcept
{
  if (ptr == nullptr)
  {
    return;
  }

  int previous_device       = 0;
  bool restore              = false;
  const auto get_status     = cudaGetDevice(&previous_device);
  const bool switch_current = (get_status == cudaSuccess) && (previous_device != device);
  if (switch_current)
  {
    restore = (cudaSetDevice(device) == cudaSuccess);
  }

  (void) cudaFree(ptr);

  if (restore)
  {
    (void) cudaSetDevice(previous_device);
  }
}
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
} // namespace detail

using checked_cuda_memory_resource = THRUST_NS_QUALIFIER::system::cuda::detail::
  cuda_memory_resource<detail::checked_cuda_malloc, cudaFree, THRUST_NS_QUALIFIER::cuda::pointer<void>>;

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
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

#  if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator!=(checked_device_memory_resource lhs, checked_device_memory_resource rhs) noexcept
  {
    return !(lhs == rhs);
  }
#  endif // _CCCL_STD_VER <= 2017

  using default_queries = ::cuda::mr::properties_list<::cuda::mr::device_accessible>;

private:
  int m_device = 0;
};

template <typename T, typename... Args>
[[nodiscard]] _CCCL_HOST_API ::cuda::device_buffer<T>
make_device_buffer(::cuda::stream_ref stream, ::cuda::device_ref device, Args&&... args)
{
  return ::cuda::make_buffer<T>(stream, checked_device_memory_resource{device}, ::cuda::std::forward<Args>(args)...);
}

template <typename T>
[[nodiscard]] _CCCL_HOST_API ::cuda::device_buffer<T>
make_device_buffer(::cuda::stream_ref stream, ::cuda::device_ref device, ::cuda::std::initializer_list<T> values)
{
  return ::cuda::make_buffer<T>(stream, checked_device_memory_resource{device}, values);
}
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

template <typename T>
class checked_cuda_allocator
    : public THRUST_NS_QUALIFIER::mr::
        stateless_resource_allocator<T, THRUST_NS_QUALIFIER::device_ptr_memory_resource<checked_cuda_memory_resource>>
{
  using base = THRUST_NS_QUALIFIER::mr::
    stateless_resource_allocator<T, THRUST_NS_QUALIFIER::device_ptr_memory_resource<checked_cuda_memory_resource>>;

public:
  template <typename U>
  struct rebind
  {
    using other = checked_cuda_allocator<U>;
  };

  checked_cuda_allocator() = default;

  _CCCL_HOST_DEVICE checked_cuda_allocator(const checked_cuda_allocator& other)
      : base(other)
  {}

  template <typename U>
  _CCCL_HOST_DEVICE checked_cuda_allocator(const checked_cuda_allocator<U>& other)
      : base(other)
  {}

  checked_cuda_allocator& operator=(const checked_cuda_allocator&) = default;

  ~checked_cuda_allocator() = default;
};

struct checked_host_memory_resource final : public THRUST_NS_QUALIFIER::mr::new_delete_resource_base
{
  void* do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) final
  {
    // Some systems with integrated host/device memory have issues with allocating more memory
    // than is available. Check the amount of free memory before attempting to allocate on
    // integrated systems.
    int device = 0;
    CubDebugExit(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CubDebugExit(cudaGetDeviceProperties(&prop, device));
    if (prop.integrated)
    {
      auto status = detail::check_free_device_memory(bytes + alignment + sizeof(std::size_t));
      if (status != cudaSuccess)
      {
        throw std::bad_alloc{};
      }
    }

    return this->new_delete_resource_base::do_allocate(bytes, alignment);
  }
};

template <typename T>
using checked_host_allocator = THRUST_NS_QUALIFIER::mr::stateless_resource_allocator<T, checked_host_memory_resource>;
} // namespace c2h
