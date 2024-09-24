//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_CUDA_MEMORY_POOL
#define _CUDAX__MEMORY_RESOURCE_CUDA_MEMORY_POOL

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// cudaMallocAsync was introduced in CTK 11.2
#if !defined(_CCCL_COMPILER_MSVC_2017) && !defined(_CCCL_CUDACC_BELOW_11_2)

#  if !defined(_CCCL_CUDA_COMPILER_NVCC) && !defined(_CCCL_CUDA_COMPILER_NVHPC)
#    include <cuda_runtime.h>
#    include <cuda_runtime_api.h>
#  endif // !_CCCL_CUDA_COMPILER_NVCC && !_CCCL_CUDA_COMPILER_NVHPC

#  include <cuda/__memory_resource/get_property.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__memory_resource/resource_ref.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/__new_>
#  include <cuda/stream_ref>

#  include <cuda/experimental/__stream/stream.cuh>

#  if _CCCL_STD_VER >= 2014

//! @file
//! The \c async_memory_pool class provides a wrapper around a `cudaMempool_t`.
namespace cuda::experimental::mr
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

//! @brief \c async_memory_pool_properties is a wrapper around properties passed to \c async_memory_pool to create a
//! <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">cudaMemPool_t</a>.
struct async_memory_pool_properties
{
  size_t initial_pool_size                           = 0;
  size_t release_threshold                           = 0;
  cudaMemAllocationHandleType allocation_handle_type = cudaMemAllocationHandleType::cudaMemHandleTypeNone;
};

//! @brief \c async_memory_pool is an owning wrapper around a
//! <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">cudaMemPool_t</a>.
//!
//! It handles creation and destruction of the underlying pool utilizing the provided \c async_memory_pool_properties.
class async_memory_pool
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
#    if !defined(_CCCL_CUDACC_BELOW_11_3)
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
#    endif //_CCCL_CUDACC_BELOW_11_3
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
  __create_cuda_mempool(const int __device_id, async_memory_pool_properties __properties) noexcept
  {
    ::cuda::experimental::mr::__device_supports_stream_ordered_allocations(__device_id);
    async_memory_pool::__cuda_supports_export_handle_type(__device_id, __properties.allocation_handle_type);

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
        "async_memory_pool failed to allocate the initial pool size",
        &__ptr,
        __properties.initial_pool_size,
        __temp_stream.get());

      _CCCL_ASSERT_CUDA_API(
        ::cudaFreeAsync, "async_memory_pool failed to free the initial pool allocation", __ptr, __temp_stream.get());
    }
    return __cuda_pool_handle;
  }

  struct __from_handle_t
  {};

  //! @brief Constructs a \c async_memory_pool from a handle taking ownership of the pool
  //! @param __handle The handle to the existing pool
  explicit async_memory_pool(__from_handle_t, ::cudaMemPool_t __handle) noexcept
      : __pool_handle_(__handle)
  {}

public:
  //! @brief Constructs a \c async_memory_pool with the optionally specified initial pool size and release threshold.
  //! If the pool size grows beyond the release threshold, unused memory held by the pool will be released at the next
  //! synchronization event.
  //! @throws cuda_error if the CUDA version does not support ``cudaMallocAsync``.
  //! @param __device_id The device id of the device the stream pool is constructed on.
  //! @param __pool_properties Optional, additional properties of the pool to be created.
  explicit async_memory_pool(const ::cuda::experimental::device_ref __device_id,
                             async_memory_pool_properties __properties = {})
      : __pool_handle_(__create_cuda_mempool(__device_id.get(), __properties))
  {}

  //! @brief Disables construction from a plain `cudaMemPool_t`. We want to ensure clean ownership semantics.
  async_memory_pool(::cudaMemPool_t) = delete;

  async_memory_pool(async_memory_pool const&)            = delete;
  async_memory_pool(async_memory_pool&&)                 = delete;
  async_memory_pool& operator=(async_memory_pool const&) = delete;
  async_memory_pool& operator=(async_memory_pool&&)      = delete;

  //! @brief Destroys the \c async_memory_pool by releasing the internal ``cudaMemPool_t``.
  ~async_memory_pool() noexcept
  {
    _CCCL_ASSERT_CUDA_API(::cudaMemPoolDestroy, "~async_memory_pool() failed to destroy pool", __pool_handle_);
  }

  //! @brief Equality comparison with another \c async_memory_pool.
  //! @returns true if the stored ``cudaMemPool_t`` are equal.
  _CCCL_NODISCARD constexpr bool operator==(async_memory_pool const& __rhs) const noexcept
  {
    return __pool_handle_ == __rhs.__pool_handle_;
  }

#    if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison with another \c async_memory_pool.
  //! @returns true if the stored ``cudaMemPool_t`` are not equal.
  _CCCL_NODISCARD constexpr bool operator!=(async_memory_pool const& __rhs) const noexcept
  {
    return __pool_handle_ != __rhs.__pool_handle_;
  }
#    endif // _CCCL_STD_VER <= 2017

  //! @brief Equality comparison with a \c cudaMemPool_t.
  //! @param __rhs A \c cudaMemPool_t.
  //! @returns true if the stored ``cudaMemPool_t`` is equal to \p __rhs.
  _CCCL_NODISCARD_FRIEND constexpr bool operator==(async_memory_pool const& __lhs, ::cudaMemPool_t __rhs) noexcept
  {
    return __lhs.__pool_handle_ == __rhs;
  }

#    if _CCCL_STD_VER <= 2017
  //! @copydoc async_memory_pool::operator==(async_memory_pool const&, ::cudaMemPool_t)
  _CCCL_NODISCARD_FRIEND constexpr bool operator==(::cudaMemPool_t __lhs, async_memory_pool const& __rhs) noexcept
  {
    return __rhs.__pool_handle_ == __lhs;
  }

  //! @copydoc async_memory_pool::operator==(async_memory_pool const&, ::cudaMemPool_t)
  _CCCL_NODISCARD_FRIEND constexpr bool operator!=(async_memory_pool const& __lhs, ::cudaMemPool_t __rhs) noexcept
  {
    return __lhs.__pool_handle_ != __rhs;
  }

  //! @copydoc async_memory_pool::operator==(async_memory_pool const&, ::cudaMemPool_t)
  _CCCL_NODISCARD_FRIEND constexpr bool operator!=(::cudaMemPool_t __lhs, async_memory_pool const& __rhs) noexcept
  {
    return __rhs.__pool_handle_ != __lhs;
  }
#    endif // _CCCL_STD_VER <= 2017

  //! @brief Returns the underlying handle to the CUDA memory pool.
  _CCCL_NODISCARD constexpr cudaMemPool_t get() const noexcept
  {
    return __pool_handle_;
  }

  //! @brief Construct an `async_memory_pool` object from a native `cudaMemPool_t` handle.
  //!
  //! @param __handle The native handle
  //!
  //! @return The constructed `async_memory_pool` object
  //!
  //! @note The constructed `async_memory_pool` object takes ownership of the native handle.
  _CCCL_NODISCARD static async_memory_pool from_native_handle(::cudaMemPool_t __handle) noexcept
  {
    return async_memory_pool(__from_handle_t{}, __handle);
  }

  // Disallow construction from an `int`, e.g., `0`.
  static async_memory_pool from_native_handle(int) = delete;

  // Disallow construction from `nullptr`.
  static async_memory_pool from_native_handle(_CUDA_VSTD::nullptr_t) = delete;
};

} // namespace cuda::experimental::mr

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && !_CCCL_CUDACC_BELOW_11_2

#endif // _CUDAX__MEMORY_RESOURCE_CUDA_MEMORY_POOL
