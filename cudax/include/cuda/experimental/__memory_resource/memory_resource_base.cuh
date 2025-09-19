//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_MEMORY_RESOURCE_BASE_CUH
#define _CUDAX__MEMORY_RESOURCE_MEMORY_RESOURCE_BASE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(CLANG)
#  include <cuda_runtime.h>
#  include <cuda_runtime_api.h>
#endif // _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/__device/device_ref.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/cstddef>
#include <cuda/stream_ref>

#include <cuda/experimental/__memory_resource/any_resource.cuh>
#include <cuda/experimental/__memory_resource/memory_pool_base.cuh>
#include <cuda/experimental/__memory_resource/properties.cuh>
#include <cuda/experimental/__stream/internal_streams.cuh>
#include <cuda/experimental/__stream/stream.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

class __memory_resource_base
{
protected:
  ::cudaMemPool_t __pool_;

  //! @brief Checks whether the passed in alignment is valid.
  //! @param __alignment the alignment to check.
  //! @returns true if \p __alignment is valid.
  [[nodiscard]] static constexpr bool __is_valid_alignment(const size_t __alignment) noexcept
  {
    return __alignment <= ::cuda::mr::default_cuda_malloc_alignment
        && (::cuda::mr::default_cuda_malloc_alignment % __alignment == 0);
  }

public:
  __memory_resource_base(::cuda::std::nullptr_t) = delete;

  //! @brief  Constructs the __memory_resource_base from a \c cudaMemPool_t.
  //! @param __pool The \c cudaMemPool_t used to allocate memory.
  explicit __memory_resource_base(::cudaMemPool_t __pool) noexcept
      : __pool_(__pool)
  {}

  //! @brief Allocate device memory of size at least \p __bytes via cudaMallocFromPoolAsync.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @throws std::invalid_argument In case of invalid alignment.
  //! @throws cuda::cuda_error If an error code was return by the CUDA API call.
  //! @returns Pointer to the newly allocated memory.
  [[nodiscard]] void* allocate_sync(const size_t __bytes,
                                    const size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment)
  {
    if (!__is_valid_alignment(__alignment))
    {
      ::cuda::std::__throw_invalid_argument(
        "Invalid alignment passed to "
        "__memory_resource_base::allocate_sync.");
    }

    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocFromPoolAsync,
      "__memory_resource_base::allocate_sync failed to allocate with cudaMallocFromPoolAsync",
      &__ptr,
      __bytes,
      __pool_,
      __cccl_allocation_stream().get());
    __cccl_allocation_stream().sync();
    return __ptr;
  }

  //! @brief deallocate_sync memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate_sync`.
  //! @param __bytes  The number of bytes that was passed to the allocation call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the allocation call that returned \p __ptr.
  //! @note The pointer passed to `deallocate_sync` must not be in use in a stream. It is the caller's responsibility to
  //! properly synchronize all relevant streams before calling `deallocate_sync`.
  void deallocate_sync(
    void* __ptr, const size_t, [[maybe_unused]] const size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment)
  {
    _CCCL_ASSERT(__is_valid_alignment(__alignment),
                 "Invalid alignment passed to __memory_resource_base::deallocate_sync.");
    _CCCL_ASSERT_CUDA_API(
      ::cudaFreeAsync, "__memory_resource_base::deallocate_sync failed", __ptr, __cccl_allocation_stream().get());
    __cccl_allocation_stream().sync();
  }

  //! @brief Allocate device memory of size at least \p __bytes via `cudaMallocFromPoolAsync`.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @param __stream Stream on which to perform allocation.
  //! @throws std::invalid_argument In case of invalid alignment.
  //! @throws cuda::cuda_error If an error code was return by the cuda api call.
  //! @returns Pointer to the newly allocated memory.
  [[nodiscard]] void* allocate(const ::cuda::stream_ref __stream, const size_t __bytes, const size_t __alignment)
  {
    if (!__is_valid_alignment(__alignment))
    {
      ::cuda::std::__throw_invalid_argument(
        "Invalid alignment passed to "
        "__memory_resource_base::allocate.");
    }

    return allocate(__stream, __bytes);
  }

  //! @brief Allocate device memory of size at least \p __bytes via cudaMallocFromPoolAsync.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __stream Stream on which to perform allocation.
  //! @throws cuda::cuda_error If an error code was return by the cuda api call.
  //! @returns Pointer to the newly allocated memory.
  [[nodiscard]] void* allocate(const ::cuda::stream_ref __stream, const size_t __bytes)
  {
    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocFromPoolAsync,
      "__memory_resource_base::allocate failed to allocate with cudaMallocFromPoolAsync",
      &__ptr,
      __bytes,
      __pool_,
      __stream.get());
    return __ptr;
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`
  //! @param __bytes The number of bytes that was passed to the allocation call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the allocation call that returned \p __ptr.
  //! @param __stream A stream that has a stream ordering relationship with the stream used in the
  //! <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">allocate</a> call
  //! that returned \p __ptr.
  //! @note The pointer passed to `deallocate` must not be in use in a stream other than \p __stream.
  //! It is the caller's responsibility to properly synchronize all relevant streams before calling `deallocate`.
  void deallocate(
    [[maybe_unused]] const ::cuda::stream_ref __stream, void* __ptr, const size_t __bytes, const size_t __alignment)
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _CCCL_ASSERT(__is_valid_alignment(__alignment), "Invalid alignment passed to __memory_resource_base::deallocate.");
    deallocate(__stream, __ptr, __bytes);
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`.
  //! @param __bytes The number of bytes that was passed to the allocation call that returned \p __ptr.
  //! @param __stream A stream that has a stream ordering relationship with the stream used in the
  //! <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">allocate</a> call
  //! that returned \p __ptr.
  //! @note The pointer passed to `deallocate` must not be in use in a stream other than \p __stream.
  //! It is the caller's responsibility to properly synchronize all relevant streams before calling `deallocate`.
  void deallocate(const ::cuda::stream_ref __stream, void* __ptr, size_t)
  {
    _CCCL_ASSERT_CUDA_API(::cudaFreeAsync, "__memory_resource_base::deallocate failed", __ptr, __stream.get());
  }

  //! @brief Enable access to memory allocated through this memory resource by the supplied devices
  //!
  //! Access is controlled through the underlying memory pool, so this
  //! setting is shared between all memory resources created from the same pool.
  //! Device on which this resource allocates memory can be included in the span.
  //!
  //! @param __devices A span of `device_ref`s listing devices to enable access for
  void enable_access_from(::cuda::std::span<const device_ref> __devices)
  {
    ::cuda::experimental::__mempool_set_access(
      __pool_, {__devices.data(), __devices.size()}, cudaMemAccessFlagsProtReadWrite);
  }

  //! @brief Enable access to memory allocated through this memory resource by the supplied device
  //!
  //! Access is controlled through the underlying memory pool, so this
  //! setting is shared between all memory resources created from the same pool.
  //!
  //! @param __device device_ref indicating for which device the access should be enabled
  void enable_access_from(device_ref __device)
  {
    ::cuda::experimental::__mempool_set_access(__pool_, {&__device, 1}, cudaMemAccessFlagsProtReadWrite);
  }

  //! @brief Disable access to memory allocated through this memory resource by the supplied devices
  //!
  //! Access is controlled through the underlying memory pool, so this
  //! setting is shared between all memory resources created from the same pool.
  //! Device on which this resource allocates memory can be included in the span.
  //!
  //! @param __devices A span of `device_ref`s listing devices to disable access for
  void disable_access_from(::cuda::std::span<const device_ref> __devices)
  {
    ::cuda::experimental::__mempool_set_access(
      __pool_, {__devices.data(), __devices.size()}, cudaMemAccessFlagsProtNone);
  }

  //! @brief Disable access to memory allocated through this memory resource by the supplied device
  //!
  //! Access is controlled through the underlying memory pool, so this
  //! setting is shared between all memory resources created from the same pool.
  //!
  //! @param __device device_ref indicating for which device the access should be disabled
  void disable_access_from(device_ref __device)
  {
    ::cuda::experimental::__mempool_set_access(__pool_, {&__device, 1}, cudaMemAccessFlagsProtNone);
  }

  //! @brief Query if memory allocated through this memory resource is accessible by the supplied device
  //!
  //! @param __device device for which the access is queried
  [[nodiscard]] bool is_accessible_from(device_ref __device)
  {
    return ::cuda::experimental::__mempool_get_access(__pool_, __device);
  }

  //! @brief Equality comparison with another __memory_resource_base.
  //! @returns true if underlying \c cudaMemPool_t are equal.
  [[nodiscard]] bool operator==(__memory_resource_base const& __rhs) const noexcept
  {
    return __pool_ == __rhs.__pool_;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison with another __memory_resource_base.
  //! @returns true if underlying \c cudaMemPool_t are not equal.
  [[nodiscard]] bool operator!=(__memory_resource_base const& __rhs) const noexcept
  {
    return __pool_ != __rhs.__pool_;
  }
#endif // _CCCL_STD_VER <= 2017

  [[nodiscard]] constexpr cudaMemPool_t get() const noexcept
  {
    return __pool_;
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__MEMORY_RESOURCE_MEMORY_RESOURCE_BASE_CUH
