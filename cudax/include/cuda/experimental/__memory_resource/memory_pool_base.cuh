//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_MEMORY_POOL_BASE_CUH
#define _CUDAX__MEMORY_RESOURCE_MEMORY_POOL_BASE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <stdexcept>

#if _CCCL_CUDA_COMPILER(CLANG)
#  include <cuda_runtime.h>
#  include <cuda_runtime_api.h>
#endif // _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/span>

#include <cuda/experimental/__stream/internal_streams.cuh>

#include <cuda/std/__cccl/prologue.h>

//! @file
//! The \c __memory_pool_base class provides a wrapper around a `cudaMempool_t`.
namespace cuda::experimental
{

enum class __memory_location_type
{
  __device,
  __host,
};

//! @brief  Checks whether the current device supports \c cudaMallocAsync.
//! @param __device_id The id of the device for which to query support.
//! @throws cuda_error if \c cudaDeviceGetAttribute failed.
//! @returns true if \c cudaDevAttrMemoryPoolsSupported is not zero.
inline void __verify_device_supports_stream_ordered_allocations(const int __device_id)
{
  int __pool_is_supported = 0;
  _CCCL_TRY_CUDA_API(
    ::cudaDeviceGetAttribute,
    "Failed to call cudaDeviceGetAttribute",
    &__pool_is_supported,
    ::cudaDevAttrMemoryPoolsSupported,
    __device_id);
  if (__pool_is_supported == 0)
  {
    ::cuda::__throw_cuda_error(::cudaErrorNotSupported, "cudaMallocAsync is not supported on the given device");
  }
}

//! @brief Enable access to this memory pool from the supplied devices
//!
//! Device on which this pool resides can be included in the span.
//!
//! @param __pool The memory pool to set access for
//! @param __devices A span of `device_ref`s listing devices to enable access for
//! @param __flags The access flags to set
//! @throws cuda_error if ``cudaMemPoolSetAccess`` fails.
inline void
__mempool_set_access(cudaMemPool_t __pool, ::cuda::std::span<const device_ref> __devices, cudaMemAccessFlags __flags)
{
  ::std::vector<cudaMemAccessDesc> __descs;
  __descs.reserve(__devices.size());
  for (size_t __i = 0; __i < __devices.size(); ++__i)
  {
    __descs.push_back({cudaMemLocation{cudaMemLocationTypeDevice, __devices[__i].get()}, __flags});
  }
  _CCCL_TRY_CUDA_API(
    ::cudaMemPoolSetAccess, "Failed to set access of a memory pool", __pool, __descs.data(), __descs.size());
}

//! @brief Query if memory from a pool is accessible by the supplied device
//!
//! @param __pool The memory pool to query access for
//! @param __dev The device to query access for
//! @returns true if the memory pool is accessible from the device
[[nodiscard]] inline bool __mempool_get_access(cudaMemPool_t __pool, device_ref __dev)
{
  cudaMemAccessFlags __result;
  cudaMemLocation __loc;
  __loc.type = cudaMemLocationTypeDevice;
  __loc.id   = __dev.get();
  _CCCL_TRY_CUDA_API(::cudaMemPoolGetAccess, "failed to get access of a memory pool", &__result, __pool, &__loc);
  return __result == cudaMemAccessFlagsProtReadWrite;
}

//! @brief Internal redefinition of ``cudaMemAllocationHandleType``.
//! @note We need to define our own enum here because the earliest CUDA runtime version that supports asynchronous
//! memory pools (CUDA 11.2) did not support these flags. See the <a
//! href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html">cudaMemAllocationHandleType</a>
//! docs.
enum class cudaMemAllocationHandleType
{
  cudaMemHandleTypeNone                = 0x0, ///< Does not allow any export mechanism.
  cudaMemHandleTypePosixFileDescriptor = 0x1, ///< Allows a file descriptor to be used for exporting.
  cudaMemHandleTypeWin32               = 0x2, ///< Allows a Win32 NT handle to be used for exporting. (HANDLE)
  cudaMemHandleTypeWin32Kmt            = 0x4, ///< Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)
  cudaMemHandleTypeFabric = 0x8, ///< Allows a fabric handle to be used for exporting. (cudaMemFabricHandle_t)
};

//! @brief \c memory_pool_properties is a wrapper around properties passed to \c __memory_pool_base to create a
//! <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">cudaMemPool_t</a>.
struct memory_pool_properties
{
  size_t initial_pool_size                           = 0;
  size_t release_threshold                           = 0;
  cudaMemAllocationHandleType allocation_handle_type = cudaMemAllocationHandleType::cudaMemHandleTypeNone;
};

class __memory_pool_base
{
private:
  ::cudaMemPool_t __pool_handle_ = nullptr;

  //! @brief Check whether the specified `cudaMemAllocationHandleType` is supported on the present
  //! CUDA driver/runtime version.
  //! @param __device_id The id of the device to check for support.
  //! @param __handle_type An IPC export handle type to check for support.
  //! @throws cuda_error if the specified `cudaMemAllocationHandleType` is not supported on the specified device.
  static void __cuda_supports_export_handle_type(const int __device_id, cudaMemAllocationHandleType __handle_type)
  {
    int __supported_handles = static_cast<int>(cudaMemAllocationHandleType::cudaMemHandleTypeNone);
    if (__handle_type != cudaMemAllocationHandleType::cudaMemHandleTypeNone)
    {
      const ::cudaError_t __status =
        ::cudaDeviceGetAttribute(&__supported_handles, ::cudaDevAttrMemoryPoolSupportedHandleTypes, __device_id);
      // export handle is not supported at all
      switch (__status)
      {
        case ::cudaSuccess:
          break;
        case ::cudaErrorInvalidValue:
          ::cuda::__throw_cuda_error(
            ::cudaErrorNotSupported, "Requested IPC memory handle type not supported on given device");
        default:
          ::cuda::__throw_cuda_error(__status, "Failed to call cudaDeviceGetAttribute");
      }
    }
    if ((static_cast<int>(__handle_type) & __supported_handles) != static_cast<int>(__handle_type))
    {
      ::cuda::__throw_cuda_error(
        ::cudaErrorNotSupported, "Requested IPC memory handle type not supported on given device");
    }
  }

  //! @brief  Creates the CUDA memory pool from the passed in arguments.
  //! @throws cuda_error If the creation of the CUDA memory pool failed.
  //! @returns The created CUDA memory pool.
  [[nodiscard]] static cudaMemPool_t
  __create_cuda_mempool(__memory_location_type __kind, memory_pool_properties __properties, int __id) noexcept
  {
    ::cudaMemPoolProps __pool_properties{};
    __pool_properties.allocType   = ::cudaMemAllocationTypePinned;
    __pool_properties.handleTypes = ::cudaMemAllocationHandleType(__properties.allocation_handle_type);

    switch (__kind)
    {
      case __memory_location_type::__device: {
        ::cuda::experimental::__verify_device_supports_stream_ordered_allocations(__id);
        __memory_pool_base::__cuda_supports_export_handle_type(__id, __properties.allocation_handle_type);
        __pool_properties.location.type = ::cudaMemLocationTypeDevice;
        __pool_properties.location.id   = __id;
        break;
      }
      case __memory_location_type::__host: {
#if _CCCL_CTK_AT_LEAST(12, 6)
        // Construct on NUMA node 0 only for now
        __pool_properties.location.type = ::cudaMemLocationTypeHostNuma;
        __pool_properties.location.id   = __id;
#else // _CCCL_CTK_BELOW(12, 6)
        ::cuda::std::__throw_invalid_argument(
          "Host pinned memory pools are unavailable in this CUDA "
          "version");
#endif // _CCCL_CTK_AT_LEAST(12, 6)
        break;
      }
      default:
        ::cuda::std::__throw_invalid_argument("Invalid memory pool location type");
    }

    ::cudaMemPool_t __cuda_pool_handle{};
    _CCCL_TRY_CUDA_API(::cudaMemPoolCreate, "Failed to call cudaMemPoolCreate", &__cuda_pool_handle, &__pool_properties);

    _CCCL_TRY_CUDA_API(
      ::cudaMemPoolSetAttribute,
      "Failed to call cudaMemPoolSetAttribute with cudaMemPoolAttrReleaseThreshold",
      __cuda_pool_handle,
      ::cudaMemPoolAttrReleaseThreshold,
      &__properties.release_threshold);

    // allocate the requested initial size to prime the pool.
    // We need to use a new stream so we do not wait on other work
    if (__properties.initial_pool_size != 0)
    {
      void* __ptr{nullptr};
      _CCCL_TRY_CUDA_API(
        ::cudaMallocAsync,
        "__memory_pool_base failed to allocate the initial pool size",
        &__ptr,
        __properties.initial_pool_size,
        __cccl_allocation_stream().get());

      _CCCL_ASSERT_CUDA_API(
        ::cudaFreeAsync,
        "__memory_pool_base failed to free the initial pool allocation",
        __ptr,
        __cccl_allocation_stream().get());
    }
    return __cuda_pool_handle;
  }

protected:
  struct __from_handle_t
  {};

  //! @brief Constructs a \c __memory_pool_base from a handle taking ownership of the pool
  //! @param __handle The handle to the existing pool
  explicit __memory_pool_base(__from_handle_t, ::cudaMemPool_t __handle) noexcept
      : __pool_handle_(__handle)
  {}

public:
  //! @brief Constructs a \c __memory_pool_base with the optionally specified initial pool size and release threshold.
  //! If the pool size grows beyond the release threshold, unused memory held by the pool will be released at the next
  //! synchronization event.
  //! @throws cuda_error if the CUDA version does not support ``cudaMallocAsync``.
  //! @param __id The device id of the device the stream pool is constructed on.
  //! @param __pool_properties Optional, additional properties of the pool to be created.
  explicit __memory_pool_base(__memory_location_type __kind, memory_pool_properties __properties, int __id = -1)
      : __pool_handle_(__create_cuda_mempool(__kind, __properties, __id))
  {}

  __memory_pool_base(__memory_pool_base const&)            = delete;
  __memory_pool_base(__memory_pool_base&&)                 = delete;
  __memory_pool_base& operator=(__memory_pool_base const&) = delete;
  __memory_pool_base& operator=(__memory_pool_base&&)      = delete;

  ~__memory_pool_base() noexcept
  {
    _CCCL_ASSERT_CUDA_API(::cudaMemPoolDestroy, "~__memory_pool_base() failed to destroy pool", __pool_handle_);
  }

  //! @brief Tries to release memory.
  //! @param __min_bytes_to_keep the minimal guaranteed size of the pool.
  //! @note If the pool has less than \p __minBytesToKeep reserved, the trim_to operation is a no-op. Otherwise the
  //! pool will be guaranteed to have at least \p __minBytesToKeep bytes reserved after the operation.
  void trim_to(const size_t __min_bytes_to_keep)
  {
    _CCCL_TRY_CUDA_API(::cudaMemPoolTrimTo,
                       "Failed to call cudaMemPoolTrimTo in __memory_pool_base::trim_to ",
                       __pool_handle_,
                       __min_bytes_to_keep);
  }

  //! @brief Gets the value of an attribute of the pool.
  //! @param __attribute the attribute to be set.
  //! @return The value of the attribute. For boolean attributes any value not equal to 0 equates to true.
  // TODO rename to configuration
  size_t attribute(::cudaMemPoolAttr __attr) const
  {
    size_t __value = 0;
    _CCCL_TRY_CUDA_API(
      ::cudaMemPoolGetAttribute,
      "Failed to call cudaMemPoolSetAttribute in __memory_pool_base::attribute ",
      __pool_handle_,
      __attr,
      static_cast<void*>(&__value));
    return __value;
  }

  //! @brief Sets an attribute of the pool to a given value.
  //! @param __attribute the attribute to be set.
  //! @param __value the new value of that attribute.
  //! @note For boolean attributes any non-zero value equates to true.
  // TODO: rename to set_configuration
  void set_attribute(::cudaMemPoolAttr __attr, size_t __value)
  {
    switch (__attr)
    {
      case ::cudaMemPoolAttrReservedMemCurrent:
      case ::cudaMemPoolAttrUsedMemCurrent:
        ::cuda::std::__throw_invalid_argument("Invalid attribute passed to set_attribute.");
        break;
      case ::cudaMemPoolAttrReservedMemHigh:
      case ::cudaMemPoolAttrUsedMemHigh:
        if (__value != 0)
        {
          ::cuda::std::__throw_invalid_argument(
            "set_attribute: It is illegal to set this "
            "attribute to a non-zero value.");
        }
        break;
      default:
        break;
    }

    _CCCL_TRY_CUDA_API(
      ::cudaMemPoolSetAttribute,
      "Failed to call cudaMemPoolSetAttribute in __memory_pool_base::set_attribute ",
      __pool_handle_,
      __attr,
      static_cast<void*>(&__value));
  }

  //! @brief Enable access to this memory pool from the supplied devices
  //!
  //! Device on which this pool resides can be included in the span.
  //!
  //! @param __devices A span of `device_ref`s listing devices to enable access for
  void enable_access_from(::cuda::std::span<const device_ref> __devices)
  {
    ::cuda::experimental::__mempool_set_access(
      __pool_handle_, {__devices.data(), __devices.size()}, cudaMemAccessFlagsProtReadWrite);
  }

  //! @brief Enable access to this memory pool from the supplied device
  //!
  //! @param __device device_ref indicating for which device the access should be enabled
  void enable_access_from(device_ref __device)
  {
    ::cuda::experimental::__mempool_set_access(__pool_handle_, {&__device, 1}, cudaMemAccessFlagsProtReadWrite);
  }

  //! @brief Disable access to this memory pool from the supplied devices
  //!
  //! Device on which this pool resides can be included in the span.
  //!
  //! @param __devices A span of `device_ref`s listing devices to disable access for
  void disable_access_from(::cuda::std::span<const device_ref> __devices)
  {
    ::cuda::experimental::__mempool_set_access(
      __pool_handle_, {__devices.data(), __devices.size()}, cudaMemAccessFlagsProtNone);
  }

  //! @brief Disable access to this memory pool from the supplied device
  //!
  //! @param __device device_ref indicating for which device the access should be disable
  void disable_access_from(device_ref __device)
  {
    ::cuda::experimental::__mempool_set_access(__pool_handle_, {&__device, 1}, cudaMemAccessFlagsProtNone);
  }

  //! @brief Query if memory allocated through this memory resource is accessible by the supplied device
  //!
  //! @param __device device for which the access is queried
  [[nodiscard]] bool is_accessible_from(device_ref __device)
  {
    return ::cuda::experimental::__mempool_get_access(__pool_handle_, __device);
  }

  //! @brief Equality comparison with another \c __memory_pool_base.
  //! @returns true if the stored ``cudaMemPool_t`` are equal.
  [[nodiscard]] constexpr bool operator==(__memory_pool_base const& __rhs) const noexcept
  {
    return __pool_handle_ == __rhs.__pool_handle_;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison with another \c __memory_pool_base.
  //! @returns true if the stored ``cudaMemPool_t`` are not equal.
  [[nodiscard]] constexpr bool operator!=(__memory_pool_base const& __rhs) const noexcept
  {
    return __pool_handle_ != __rhs.__pool_handle_;
  }
#endif // _CCCL_STD_VER <= 2017

  //! @brief Equality comparison with a \c cudaMemPool_t.
  //! @param __rhs A \c cudaMemPool_t.
  //! @returns true if the stored ``cudaMemPool_t`` is equal to \p __rhs.
  [[nodiscard]] friend constexpr bool operator==(__memory_pool_base const& __lhs, ::cudaMemPool_t __rhs) noexcept
  {
    return __lhs.__pool_handle_ == __rhs;
  }

#if _CCCL_STD_VER <= 2017
  //! @copydoc __memory_pool_base::operator==(__memory_pool_base const&, ::cudaMemPool_t)
  [[nodiscard]] friend constexpr bool operator==(::cudaMemPool_t __lhs, __memory_pool_base const& __rhs) noexcept
  {
    return __rhs.__pool_handle_ == __lhs;
  }

  //! @copydoc __memory_pool_base::operator==(__memory_pool_base const&, ::cudaMemPool_t)
  [[nodiscard]] friend constexpr bool operator!=(__memory_pool_base const& __lhs, ::cudaMemPool_t __rhs) noexcept
  {
    return __lhs.__pool_handle_ != __rhs;
  }

  //! @copydoc __memory_pool_base::operator==(__memory_pool_base const&, ::cudaMemPool_t)
  [[nodiscard]] friend constexpr bool operator!=(::cudaMemPool_t __lhs, __memory_pool_base const& __rhs) noexcept
  {
    return __rhs.__pool_handle_ != __lhs;
  }
#endif // _CCCL_STD_VER <= 2017

  //! @brief Returns the underlying handle to the CUDA memory pool.
  [[nodiscard]] constexpr cudaMemPool_t get() const noexcept
  {
    return __pool_handle_;
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__MEMORY_RESOURCE_MEMORY_POOL_BASE_CUH
