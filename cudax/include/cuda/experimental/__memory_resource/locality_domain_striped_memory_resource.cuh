//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_LOCALITY_DOMAIN_STRIPED_MEMORY_RESOURCE_CUH
#define _CUDAX__MEMORY_RESOURCE_LOCALITY_DOMAIN_STRIPED_MEMORY_RESOURCE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && _CCCL_CTK_AT_LEAST(13, 4)
#  include <cuda/__device/device_ref.h>
#  include <cuda/__driver/driver_api.h>
#  include <cuda/__memory_resource/get_property.h>
#  include <cuda/__memory_resource/memory_resource_base.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__memory_resource/resource.h>
#  include <cuda/__memory_resource/shared_block_ptr.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__runtime/ensure_current_context.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__host_stdlib/new>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__limits/numeric_limits.h>
#  include <cuda/std/__utility/in_place.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/experimental/__driver/driver_api.cuh>

#  include <mutex>
#  include <unordered_map>
#  include <vector>

#  include <cuda.h>
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && _CCCL_CTK_AT_LEAST(13, 4)

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Query object for the byte stride used by a locality-domain striped memory resource.
struct locality_domain_stride_t
{
  using value_type = ::cuda::std::size_t;
};

_CCCL_GLOBAL_CONSTANT locality_domain_stride_t locality_domain_stride{};

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && _CCCL_CTK_AT_LEAST(13, 4)

struct __locality_domain_mapping
{
  ::CUdeviceptr __ptr_{};
  ::cuda::std::size_t __size_{};
  ::CUmemGenericAllocationHandle __handle_{};
  bool __mapped_{false};
  bool __has_handle_{false};
};

struct __locality_domain_allocation
{
  ::CUdeviceptr __ptr_{};
  ::cuda::std::size_t __size_{};
  bool __reserved_{false};
  ::std::vector<__locality_domain_mapping> __mappings_{};
};

_CCCL_HOST_API inline void __destroy_locality_domain_allocation(__locality_domain_allocation& __allocation) noexcept
{
  for (auto& __mapping : __allocation.__mappings_)
  {
    if (__mapping.__mapped_)
    {
      _CCCL_ASSERT_CUDA_API(
        ::cuda::experimental::__driver::__memUnmapNoThrow,
        "locality_domain_striped_memory_resource failed to unmap a memory allocation",
        __mapping.__ptr_,
        __mapping.__size_);
    }
    if (__mapping.__has_handle_)
    {
      _CCCL_ASSERT_CUDA_API(::cuda::experimental::__driver::__memReleaseNoThrow,
                            "locality_domain_striped_memory_resource failed to release a memory allocation",
                            __mapping.__handle_);
    }
  }

  if (__allocation.__reserved_)
  {
    _CCCL_ASSERT_CUDA_API(
      ::cuda::experimental::__driver::__memAddressFreeNoThrow,
      "locality_domain_striped_memory_resource failed to free a virtual address range",
      __allocation.__ptr_,
      __allocation.__size_);
  }
}

[[nodiscard]] _CCCL_HOST_API constexpr bool __locality_domain_is_power_of_two(::cuda::std::size_t __value) noexcept
{
  return __value != 0 && ((__value & (__value - 1)) == 0);
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::size_t
__locality_domain_round_up(::cuda::std::size_t __value, ::cuda::std::size_t __multiple)
{
  _CCCL_ASSERT(__multiple != 0, "Cannot round up to a zero multiple");
  const auto __remainder = __value % __multiple;
  if (__remainder == 0)
  {
    return __value;
  }

  const auto __delta = __multiple - __remainder;
  if (__value > ::cuda::std::numeric_limits<::cuda::std::size_t>::max() - __delta)
  {
    _CCCL_THROW(::std::bad_alloc);
  }
  return __value + __delta;
}

class __locality_domain_striped_memory_resource_state
{
  ::cuda::device_ref __device_;
  ::cuda::std::size_t __stride_{};
  ::cuda::std::size_t __granularity_{1};
  ::cuda::std::size_t __domain_count_{};
  ::std::mutex __mutex_{};
  ::std::unordered_map<void*, __locality_domain_allocation> __allocations_{};

  [[nodiscard]] _CCCL_HOST_API ::CUmemAllocationProp __allocation_prop(::cuda::std::size_t __domain_id) const noexcept
  {
    ::CUmemAllocationProp __prop{};
    __prop.type                                = ::CU_MEM_ALLOCATION_TYPE_PINNED;
    __prop.location.type                       = ::CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
    __prop.location.localized.deviceId         = static_cast<unsigned char>(__device_.get());
    __prop.location.localized.localityDomainId = static_cast<unsigned char>(__domain_id);
    return __prop;
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t __mapping_count(::cuda::std::size_t __size) const noexcept
  {
    if (__domain_count_ == 1)
    {
      return 1;
    }

    return (__size / __stride_) + (__size % __stride_ != 0 ? 1 : 0);
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t
  __chunk_size(::cuda::std::size_t __offset, ::cuda::std::size_t __size) const noexcept
  {
    if (__domain_count_ == 1)
    {
      return __size;
    }

    const auto __remaining = __size - __offset;
    return __remaining < __stride_ ? __remaining : __stride_;
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t __domain_for_offset(::cuda::std::size_t __offset) const noexcept
  {
    return __domain_count_ == 1 ? 0 : (__offset / __stride_) % __domain_count_;
  }

public:
  _CCCL_HOST_API
  __locality_domain_striped_memory_resource_state(::cuda::device_ref __device, ::cuda::std::size_t __stride)
      : __device_(__device)
      , __stride_(__stride)
  {
    if (__stride_ == 0)
    {
      _CCCL_THROW(::std::invalid_argument, "locality_domain_striped_memory_resource requires a non-zero stride");
    }

    ::cuda::__ensure_current_context __context_guard(__device_);
    const auto __cu_device = ::cuda::__driver::__deviceGet(__device_.get());
    const auto __vmm_supported =
      ::cuda::__driver::__deviceGetAttribute(::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, __cu_device);
    if (__vmm_supported == 0)
    {
      _CCCL_THROW(::cuda::cuda_error,
                  ::cudaErrorNotSupported,
                  "locality_domain_striped_memory_resource requires virtual memory management support");
    }

    const auto __domain_count =
      ::cuda::__driver::__deviceGetAttribute(::CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, __cu_device);
    if (__domain_count <= 0)
    {
      _CCCL_THROW(::cuda::cuda_error,
                  ::cudaErrorNotSupported,
                  "locality_domain_striped_memory_resource requires at least one locality domain");
    }

    __domain_count_ = static_cast<::cuda::std::size_t>(__domain_count);

    const auto __prop = __allocation_prop(0);
    __granularity_ =
      ::cuda::experimental::__driver::__memGetAllocationGranularity(&__prop, ::CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    if (!::cuda::experimental::__locality_domain_is_power_of_two(__granularity_))
    {
      _CCCL_THROW(::cuda::cuda_error,
                  ::cudaErrorInvalidValue,
                  "locality_domain_striped_memory_resource requires a power-of-two VMM allocation granularity");
    }
    if (__stride_ % __granularity_ != 0)
    {
      _CCCL_THROW(::std::invalid_argument,
                  "locality_domain_striped_memory_resource stride must be a multiple of the minimum VMM allocation "
                  "granularity");
    }
  }

  _CCCL_HOST_API ~__locality_domain_striped_memory_resource_state() noexcept
  {
    ::cuda::__ensure_current_context __context_guard(__device_);
    for (auto& __allocation : __allocations_)
    {
      ::cuda::experimental::__destroy_locality_domain_allocation(__allocation.second);
    }
  }

  [[nodiscard]] _CCCL_HOST_API void* __allocate(::cuda::std::size_t __bytes, ::cuda::std::size_t __alignment)
  {
    if (!::cuda::experimental::__locality_domain_is_power_of_two(__alignment))
    {
      _CCCL_THROW(::std::invalid_argument,
                  "Invalid alignment passed to locality_domain_striped_memory_resource::allocate_sync.");
    }
    if (__bytes == 0)
    {
      return nullptr;
    }

    ::cuda::__ensure_current_context __context_guard(__device_);

    const auto __reservation_alignment = __alignment > __granularity_ ? __alignment : __granularity_;
    ::cuda::experimental::__locality_domain_allocation __allocation{};
    __allocation.__size_ = ::cuda::experimental::__locality_domain_round_up(__bytes, __reservation_alignment);
    __allocation.__mappings_.reserve(__mapping_count(__allocation.__size_));
    __allocation.__ptr_ =
      ::cuda::experimental::__driver::__memAddressReserve(__allocation.__size_, __reservation_alignment);
    __allocation.__reserved_ = true;

    // Experimental shortcut: no rollback if a later VMM step throws.
    for (::cuda::std::size_t __offset = 0; __offset != __allocation.__size_;)
    {
      const auto __size      = __chunk_size(__offset, __allocation.__size_);
      const auto __domain_id = __domain_for_offset(__offset);
      const auto __prop      = __allocation_prop(__domain_id);
      auto __handle          = ::cuda::experimental::__driver::__memCreate(__size, &__prop);
      const auto __ptr       = __allocation.__ptr_ + __offset;

      ::cuda::experimental::__driver::__memMap(__ptr, __size, __handle);
      __allocation.__mappings_.push_back({__ptr, __size, __handle, true, true});
      __offset += __size;
    }

    // Same tradeoff here if access setup or registry insertion throws.
    ::CUmemAccessDesc __access_desc{};
    __access_desc.location.type = ::CU_MEM_LOCATION_TYPE_DEVICE;
    __access_desc.location.id   = __device_.get();
    __access_desc.flags         = ::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    ::cuda::experimental::__driver::__memSetAccess(__allocation.__ptr_, __allocation.__size_, &__access_desc, 1);

    void* __result = reinterpret_cast<void*>(__allocation.__ptr_);
    {
      ::std::lock_guard<::std::mutex> __lock(__mutex_);
      auto __entry = __allocations_.emplace(__result, ::cuda::std::move(__allocation));
      if (!__entry.second)
      {
        _CCCL_THROW(::cuda::cuda_error,
                    ::cudaErrorInvalidValue,
                    "locality_domain_striped_memory_resource created a duplicate virtual address range");
      }
    }

    return __result;
  }

  _CCCL_HOST_API void
  __deallocate(void* __ptr, [[maybe_unused]] ::cuda::std::size_t __bytes, ::cuda::std::size_t __alignment) noexcept
  {
    if (__ptr == nullptr)
    {
      return;
    }

    _CCCL_ASSERT(::cuda::experimental::__locality_domain_is_power_of_two(__alignment),
                 "Invalid alignment passed to locality_domain_striped_memory_resource::deallocate_sync.");

    ::cuda::experimental::__locality_domain_allocation __allocation{};
    {
      ::std::lock_guard<::std::mutex> __lock(__mutex_);
      auto __iter = __allocations_.find(__ptr);
      if (__iter == __allocations_.end())
      {
        _CCCL_ASSERT(false, "locality_domain_striped_memory_resource cannot deallocate an unknown pointer.");
        return;
      }

      __allocation = ::cuda::std::move(__iter->second);
      __allocations_.erase(__iter);
    }

    ::cuda::__ensure_current_context __context_guard(__device_);
    ::cuda::experimental::__destroy_locality_domain_allocation(__allocation);
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::device_ref __device() const noexcept
  {
    return __device_;
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t __stride() const noexcept
  {
    return __stride_;
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t __locality_domain_count() const noexcept
  {
    return __domain_count_;
  }
};

//! @brief Synchronous memory resource allocating VMM-backed memory striped across a device's locality domains.
//!
//! Allocations are split into consecutive chunks of `stride()` bytes. Chunk 0 is allocated in locality domain 0,
//! chunk 1 in domain 1, and so on, wrapping back to domain 0 after `locality_domain_count()` chunks. The resource is
//! copyable; copies share ownership of the allocation registry and can deallocate each other's pointers.
class locality_domain_striped_memory_resource
    : public ::cuda::mr::memory_resource_base<locality_domain_striped_memory_resource>
{
  ::cuda::mr::__shared_block_ptr<__locality_domain_striped_memory_resource_state> __state_;

public:
  //! @brief Constructs a resource for device 0 using the requested byte stride.
  _CCCL_HOST_API explicit locality_domain_striped_memory_resource(::cuda::std::size_t __stride)
      : locality_domain_striped_memory_resource(::cuda::device_ref{0}, __stride)
  {}

  //! @brief Constructs a resource for `device` using the requested byte stride.
  _CCCL_HOST_API explicit locality_domain_striped_memory_resource(
    ::cuda::device_ref __device, ::cuda::std::size_t __stride)
      : __state_(::cuda::std::in_place_type<__locality_domain_striped_memory_resource_state>, __device, __stride)
  {}

  [[nodiscard]] _CCCL_HOST_API void* allocate_sync(
    ::cuda::std::size_t __bytes, ::cuda::std::size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment)
  {
    return __state_.__payload().__allocate(__bytes, __alignment);
  }

  _CCCL_HOST_API void deallocate_sync(
    void* __ptr,
    ::cuda::std::size_t __bytes,
    ::cuda::std::size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment) noexcept
  {
    __state_.__payload().__deallocate(__ptr, __bytes, __alignment);
  }

  //! @brief Returns the device whose locality domains back this resource's allocations.
  [[nodiscard]] _CCCL_HOST_API ::cuda::device_ref device() const noexcept
  {
    return __state_.__payload().__device();
  }

  //! @brief Returns the requested byte stride used to rotate allocations across locality domains.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t stride() const noexcept
  {
    return __state_.__payload().__stride();
  }

  //! @brief Returns the number of locality domains used by this resource.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t locality_domain_count() const noexcept
  {
    return __state_.__payload().__locality_domain_count();
  }

  [[nodiscard]] _CCCL_HOST_API bool operator==(const locality_domain_striped_memory_resource& __other) const noexcept
  {
    return __state_ == __other.__state_;
  }

#  if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_HOST_API bool operator!=(const locality_domain_striped_memory_resource& __other) const noexcept
  {
    return __state_ != __other.__state_;
  }
#  endif // _CCCL_STD_VER <= 2017

  _CCCL_HOST_API friend constexpr void
  get_property(const locality_domain_striped_memory_resource&, ::cuda::mr::device_accessible) noexcept
  {}

  [[nodiscard]] _CCCL_HOST_API friend ::cuda::std::size_t
  get_property(const locality_domain_striped_memory_resource& __resource, locality_domain_stride_t) noexcept
  {
    return __resource.stride();
  }

  using default_queries =
    ::cuda::mr::properties_list<::cuda::mr::device_accessible, ::cuda::experimental::locality_domain_stride_t>;
};

static_assert(
  ::cuda::mr::synchronous_resource_with<locality_domain_striped_memory_resource, ::cuda::mr::device_accessible>);
static_assert(::cuda::mr::synchronous_resource_with<locality_domain_striped_memory_resource, locality_domain_stride_t>);
static_assert(
  ::cuda::has_property_with<locality_domain_striped_memory_resource, locality_domain_stride_t, ::cuda::std::size_t>);

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && _CCCL_CTK_AT_LEAST(13, 4)
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__MEMORY_RESOURCE_LOCALITY_DOMAIN_STRIPED_MEMORY_RESOURCE_CUH
