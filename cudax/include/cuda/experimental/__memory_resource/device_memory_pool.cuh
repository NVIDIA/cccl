//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_DEVICE_MEMORY_POOL
#define _CUDAX__MEMORY_RESOURCE_DEVICE_MEMORY_POOL

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// cudaMallocAsync was introduced in CTK 11.2
#if !_CCCL_COMPILER(MSVC2017) && _CCCL_CUDACC_AT_LEAST(11, 2)

#  if defined(_CCCL_CUDA_COMPILER_CLANG)
#    include <cuda_runtime.h>
#    include <cuda_runtime_api.h>
#  endif // _CCCL_CUDA_COMPILER_CLANG

#  include <cuda/__memory_resource/get_property.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__memory_resource/resource_ref.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/__new_>
#  include <cuda/std/span>
#  include <cuda/stream_ref>

#  include <cuda/experimental/__stream/stream.cuh>

#  if _CCCL_STD_VER >= 2014

//! @file
//! The \c device_memory_pool class provides a wrapper around a `cudaMempool_t`.
namespace cuda::experimental
{

//! @brief  Checks whether the current device supports \c cudaMallocAsync.
//! @param __device_id The id of the device for which to query support.
//! @throws cuda_error if \c cudaDeviceGetAttribute failed.
//! @returns true if \c cudaDevAttrMemoryPoolsSupported is not zero.
inline void __device_supports_stream_ordered_allocations(const int __device_id)
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

inline void __mempool_switch_peer_access(
  cudaMemPool_t __pool, ::cuda::std::span<const device_ref> __devices, cudaMemAccessFlags __flags)
{
  ::std::vector<cudaMemAccessDesc> __descs;
  __descs.reserve(__devices.size());
  cudaMemAccessDesc __desc;
  __desc.flags         = __flags;
  __desc.location.type = cudaMemLocationTypeDevice;
  for (size_t __i = 0; __i < __devices.size(); ++__i)
  {
    __desc.location.id = __devices[__i].get();
    __descs.push_back(__desc);
  }
  _CCCL_TRY_CUDA_API(
    ::cudaMemPoolSetAccess, "Failed to set access of a memory pool", __pool, __descs.data(), __descs.size());
}

_CCCL_NODISCARD inline bool __mempool_get_access(cudaMemPool_t __pool, device_ref __dev)
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
//! href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html">cudaMemAllocationHandleType</a> docs.
enum class cudaMemAllocationHandleType
{
  cudaMemHandleTypeNone                = 0x0, ///< Does not allow any export mechanism.
  cudaMemHandleTypePosixFileDescriptor = 0x1, ///< Allows a file descriptor to be used for exporting.
  cudaMemHandleTypeWin32               = 0x2, ///< Allows a Win32 NT handle to be used for exporting. (HANDLE)
  cudaMemHandleTypeWin32Kmt            = 0x4, ///< Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)
  cudaMemHandleTypeFabric = 0x8, ///< Allows a fabric handle to be used for exporting. (cudaMemFabricHandle_t)
};

//! @brief \c memory_pool_properties is a wrapper around properties passed to \c device_memory_pool to create a
//! <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">cudaMemPool_t</a>.
struct memory_pool_properties
{
  size_t initial_pool_size                           = 0;
  size_t release_threshold                           = 0;
  cudaMemAllocationHandleType allocation_handle_type = cudaMemAllocationHandleType::cudaMemHandleTypeNone;
};

//! @brief \c device_memory_pool is an owning wrapper around a
//! <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">cudaMemPool_t</a>.
//!
//! It handles creation and destruction of the underlying pool utilizing the provided \c memory_pool_properties.
class device_memory_pool
{
private:
  ::cudaMemPool_t __pool_handle_ = nullptr;

  //! @brief Check whether the specified `cudaMemAllocationHandleType` is supported on the present
  //! CUDA driver/runtime version.
  //! @note This query was introduced in CUDA 11.3 so on CUDA 11.2 this function will only return
  //! true for `cudaMemHandleTypeNone`.
  //! @param __device_id The id of the device to check for support.
  //! @param __handle_type An IPC export handle type to check for support.
  //! @return true if the handle type is supported by cudaDevAttrMemoryPoolSupportedHandleTypes.
  static void __cuda_supports_export_handle_type(const int __device_id, cudaMemAllocationHandleType __handle_type)
  {
    int __supported_handles = static_cast<int>(cudaMemAllocationHandleType::cudaMemHandleTypeNone);
#    if _CCCL_CUDACC_AT_LEAST(11, 3)
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
          ::cudaGetLastError(); // Clear CUDA error state
          ::cuda::__throw_cuda_error(
            ::cudaErrorNotSupported, "Requested IPC memory handle type not supported on given device");
        default:
          ::cudaGetLastError(); // Clear CUDA error state
          ::cuda::__throw_cuda_error(__status, "Failed to call cudaDeviceGetAttribute");
      }
    }
#    endif // _CCCL_CUDACC_BELOW(11, 3)
    if ((static_cast<int>(__handle_type) & __supported_handles) != static_cast<int>(__handle_type))
    {
      ::cuda::__throw_cuda_error(
        ::cudaErrorNotSupported, "Requested IPC memory handle type not supported on given device");
    }
  }

  //! @brief  Creates the CUDA memory pool from the passed in arguments.
  //! @throws cuda_error If the creation of the CUDA memory pool failed.
  //! @returns The created CUDA memory pool.
  _CCCL_NODISCARD static cudaMemPool_t
  __create_cuda_mempool(const int __device_id, memory_pool_properties __properties) noexcept
  {
    ::cuda::experimental::__device_supports_stream_ordered_allocations(__device_id);
    device_memory_pool::__cuda_supports_export_handle_type(__device_id, __properties.allocation_handle_type);

    ::cudaMemPoolProps __pool_properties{};
    __pool_properties.allocType     = ::cudaMemAllocationTypePinned;
    __pool_properties.handleTypes   = ::cudaMemAllocationHandleType(__properties.allocation_handle_type);
    __pool_properties.location.type = ::cudaMemLocationTypeDevice;
    __pool_properties.location.id   = __device_id;
    ::cudaMemPool_t __cuda_pool_handle{};
    _CCCL_TRY_CUDA_API(::cudaMemPoolCreate, "Failed to call cudaMemPoolCreate", &__cuda_pool_handle, &__pool_properties);

    // CUDA drivers before 11.5 have known incompatibilities with the async allocator.
    // We'll disable `cudaMemPoolReuseAllowOpportunistic` if cuda driver < 11.5.
    // See https://github.com/NVIDIA/spark-rapids/issues/4710.
    int __driver_version = 0;
    _CCCL_TRY_CUDA_API(::cudaDriverGetVersion, "Failed to call cudaDriverGetVersion", &__driver_version);

    constexpr int __min_async_version = 11050;
    if (__driver_version < __min_async_version)
    {
      int __disable_reuse = 0;
      _CCCL_TRY_CUDA_API(
        ::cudaMemPoolSetAttribute,
        "Failed to call cudaMemPoolSetAttribute with cudaMemPoolReuseAllowOpportunistic",
        __cuda_pool_handle,
        ::cudaMemPoolReuseAllowOpportunistic,
        &__disable_reuse);
    }

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
      ::cuda::experimental::stream __temp_stream{__device_id};
      void* __ptr{nullptr};
      _CCCL_TRY_CUDA_API(
        ::cudaMallocAsync,
        "device_memory_pool failed to allocate the initial pool size",
        &__ptr,
        __properties.initial_pool_size,
        __temp_stream.get());

      _CCCL_ASSERT_CUDA_API(
        ::cudaFreeAsync, "device_memory_pool failed to free the initial pool allocation", __ptr, __temp_stream.get());
    }
    return __cuda_pool_handle;
  }

  struct __from_handle_t
  {};

  //! @brief Constructs a \c device_memory_pool from a handle taking ownership of the pool
  //! @param __handle The handle to the existing pool
  explicit device_memory_pool(__from_handle_t, ::cudaMemPool_t __handle) noexcept
      : __pool_handle_(__handle)
  {}

public:
  //! @brief Constructs a \c device_memory_pool with the optionally specified initial pool size and release threshold.
  //! If the pool size grows beyond the release threshold, unused memory held by the pool will be released at the next
  //! synchronization event.
  //! @throws cuda_error if the CUDA version does not support ``cudaMallocAsync``.
  //! @param __device_id The device id of the device the stream pool is constructed on.
  //! @param __pool_properties Optional, additional properties of the pool to be created.
  explicit device_memory_pool(const ::cuda::experimental::device_ref __device_id,
                              memory_pool_properties __properties = {})
      : __pool_handle_(__create_cuda_mempool(__device_id.get(), __properties))
  {}

  //! @brief Disables construction from a plain `cudaMemPool_t`. We want to ensure clean ownership semantics.
  device_memory_pool(::cudaMemPool_t) = delete;

  device_memory_pool(device_memory_pool const&)            = delete;
  device_memory_pool(device_memory_pool&&)                 = delete;
  device_memory_pool& operator=(device_memory_pool const&) = delete;
  device_memory_pool& operator=(device_memory_pool&&)      = delete;

  //! @brief Destroys the \c device_memory_pool by releasing the internal ``cudaMemPool_t``.
  ~device_memory_pool() noexcept
  {
    _CCCL_ASSERT_CUDA_API(::cudaMemPoolDestroy, "~device_memory_pool() failed to destroy pool", __pool_handle_);
  }

  //! @brief Tries to release memory.
  //! @param __min_bytes_to_keep the minimal guaranteed size of the pool.
  //! @note If the pool has less than \p __minBytesToKeep reserved, the trim_to operation is a no-op. Otherwise the pool
  //! will be guaranteed to have at least \p __minBytesToKeep bytes reserved after the operation.
  void trim_to(const size_t __min_bytes_to_keep)
  {
    _CCCL_TRY_CUDA_API(::cudaMemPoolTrimTo,
                       "Failed to call cudaMemPoolTrimTo in device_memory_pool::trim_to",
                       __pool_handle_,
                       __min_bytes_to_keep);
  }

  //! @brief Gets the value of an attribute of the pool.
  //! @param __attribute the attribute to be set.
  //! @return The value of the attribute. For boolean attributes any value not equal to 0 equates to true.
  size_t get_attribute(::cudaMemPoolAttr __attr) const
  {
    size_t __value = 0;
    _CCCL_TRY_CUDA_API(
      ::cudaMemPoolGetAttribute,
      "Failed to call cudaMemPoolSetAttribute in device_memory_pool::get_attribute",
      __pool_handle_,
      __attr,
      static_cast<void*>(&__value));
    return __value;
  }

  //! @brief Sets an attribute of the pool to a given value.
  //! @param __attribute the attribute to be set.
  //! @param __value the new value of that attribute.
  //! @note For boolean attributes any non-zero value equates to true.
  void set_attribute(::cudaMemPoolAttr __attr, size_t __value)
  {
    if (__attr == ::cudaMemPoolAttrReservedMemCurrent || __attr == cudaMemPoolAttrUsedMemCurrent)
    {
      _CUDA_VSTD_NOVERSION::__throw_invalid_argument("Invalid attribute passed to device_memory_pool::set_attribute.");
    }
    else if ((__attr == ::cudaMemPoolAttrReservedMemHigh || __attr == cudaMemPoolAttrUsedMemHigh) && __value != 0)
    {
      _CUDA_VSTD_NOVERSION::__throw_invalid_argument(
        "device_memory_pool::set_attribute: It is illegal to set this "
        "attribute to a non-zero value.");
    }

    _CCCL_TRY_CUDA_API(
      ::cudaMemPoolSetAttribute,
      "Failed to call cudaMemPoolSetAttribute in device_memory_pool::set_attribute",
      __pool_handle_,
      __attr,
      static_cast<void*>(&__value));
  }

  //! @brief Enable peer access to this memory pool from the supplied devices
  //!
  //! Device on which this pool resides can be included in the vector.
  //!
  //! @param __devices A vector of `device_ref`s listing devices to enable access for
  void enable_peer_access_from(const ::std::vector<device_ref>& __devices)
  {
    ::cuda::experimental::__mempool_switch_peer_access(
      __pool_handle_, {__devices.data(), __devices.size()}, cudaMemAccessFlagsProtReadWrite);
  }

  //! @brief Enable peer access to this memory pool from the supplied device
  //!
  //! @param __device device_ref indicating for which device the access should be enabled
  void enable_peer_access_from(device_ref __device)
  {
    ::cuda::experimental::__mempool_switch_peer_access(__pool_handle_, {&__device, 1}, cudaMemAccessFlagsProtReadWrite);
  }

  //! @brief Disable peer access to this memory pool from the supplied devices
  //!
  //! Device on which this pool resides can be included in the vector.
  //!
  //! @param __devices A vector of `device_ref`s listing devices to disable access for
  void disable_peer_access_from(const ::std::vector<device_ref>& __devices)
  {
    ::cuda::experimental::__mempool_switch_peer_access(
      __pool_handle_, {__devices.data(), __devices.size()}, cudaMemAccessFlagsProtNone);
  }

  //! @brief Disable peer access to this memory pool from the supplied device
  //!
  //! @param __device device_ref indicating for which device the access should be disable
  void disable_peer_access_from(device_ref __device)
  {
    ::cuda::experimental::__mempool_switch_peer_access(__pool_handle_, {&__device, 1}, cudaMemAccessFlagsProtNone);
  }

  //! @brief Query if memory allocated through this memory resource is accessible by the supplied device
  //!
  //! @param __device device for which the peer access is queried
  _CCCL_NODISCARD bool is_accessible_from(device_ref __device)
  {
    return ::cuda::experimental::__mempool_get_access(__pool_handle_, __device);
  }

  //! @brief Equality comparison with another \c device_memory_pool.
  //! @returns true if the stored ``cudaMemPool_t`` are equal.
  _CCCL_NODISCARD constexpr bool operator==(device_memory_pool const& __rhs) const noexcept
  {
    return __pool_handle_ == __rhs.__pool_handle_;
  }

#    if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison with another \c device_memory_pool.
  //! @returns true if the stored ``cudaMemPool_t`` are not equal.
  _CCCL_NODISCARD constexpr bool operator!=(device_memory_pool const& __rhs) const noexcept
  {
    return __pool_handle_ != __rhs.__pool_handle_;
  }
#    endif // _CCCL_STD_VER <= 2017

  //! @brief Equality comparison with a \c cudaMemPool_t.
  //! @param __rhs A \c cudaMemPool_t.
  //! @returns true if the stored ``cudaMemPool_t`` is equal to \p __rhs.
  _CCCL_NODISCARD_FRIEND constexpr bool operator==(device_memory_pool const& __lhs, ::cudaMemPool_t __rhs) noexcept
  {
    return __lhs.__pool_handle_ == __rhs;
  }

#    if _CCCL_STD_VER <= 2017
  //! @copydoc device_memory_pool::operator==(device_memory_pool const&, ::cudaMemPool_t)
  _CCCL_NODISCARD_FRIEND constexpr bool operator==(::cudaMemPool_t __lhs, device_memory_pool const& __rhs) noexcept
  {
    return __rhs.__pool_handle_ == __lhs;
  }

  //! @copydoc device_memory_pool::operator==(device_memory_pool const&, ::cudaMemPool_t)
  _CCCL_NODISCARD_FRIEND constexpr bool operator!=(device_memory_pool const& __lhs, ::cudaMemPool_t __rhs) noexcept
  {
    return __lhs.__pool_handle_ != __rhs;
  }

  //! @copydoc device_memory_pool::operator==(device_memory_pool const&, ::cudaMemPool_t)
  _CCCL_NODISCARD_FRIEND constexpr bool operator!=(::cudaMemPool_t __lhs, device_memory_pool const& __rhs) noexcept
  {
    return __rhs.__pool_handle_ != __lhs;
  }
#    endif // _CCCL_STD_VER <= 2017

  //! @brief Returns the underlying handle to the CUDA memory pool.
  _CCCL_NODISCARD constexpr cudaMemPool_t get() const noexcept
  {
    return __pool_handle_;
  }

  //! @brief Construct an `device_memory_pool` object from a native `cudaMemPool_t` handle.
  //!
  //! @param __handle The native handle
  //!
  //! @return The constructed `device_memory_pool` object
  //!
  //! @note The constructed `device_memory_pool` object takes ownership of the native handle.
  _CCCL_NODISCARD static device_memory_pool from_native_handle(::cudaMemPool_t __handle) noexcept
  {
    return device_memory_pool(__from_handle_t{}, __handle);
  }

  // Disallow construction from an `int`, e.g., `0`.
  static device_memory_pool from_native_handle(int) = delete;

  // Disallow construction from `nullptr`.
  static device_memory_pool from_native_handle(_CUDA_VSTD::nullptr_t) = delete;
};

} // namespace cuda::experimental

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER(MSVC2017) && _CCCL_CUDACC_AT_LEAST(11, 2)

#endif // _CUDAX__MEMORY_RESOURCE_DEVICE_MEMORY_POOL
