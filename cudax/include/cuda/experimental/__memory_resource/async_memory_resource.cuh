//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_CUDA_ASYNC_MEMORY_RESOURCE
#define _CUDAX__MEMORY_RESOURCE_CUDA_ASYNC_MEMORY_RESOURCE

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
#  include <cuda/std/__concepts/__concept_macros.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/__new_>
#  include <cuda/std/cstddef>
#  include <cuda/stream_ref>

#  include <cuda/experimental/__device/device_ref.cuh>
#  include <cuda/experimental/__memory_resource/async_memory_pool.cuh>
#  include <cuda/experimental/__stream/stream.cuh>

#  if _CCCL_STD_VER >= 2014

//! @file
//! The \c async_memory_pool class provides an asynchronous memory resource that allocates device memory in stream
//! order.
namespace cuda::experimental::mr
{

//! @brief global stream to synchronize in the synchronous interface of \c async_memory_resource
inline ::cuda::stream_ref __async_memory_resource_sync_stream()
{
  static ::cuda::experimental::stream __stream{};
  return __stream;
}

//! @rst
//! .. _cudax-memory-resource-async:
//!
//! Stream ordered memory resource
//! ------------------------------
//!
//! ``async_memory_resource`` uses `cudaMallocFromPoolAsync / cudaFreeAsync
//! <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>`__ for allocation/deallocation. A
//! ``async_memory_resource`` is a thin wrapper around a \c cudaMemPool_t.
//!
//! .. warning::
//!
//!    ``async_memory_resource`` does not own the pool and it is the responsibility of the user to ensure that the
//!    lifetime of the pool exceeds the lifetime of the ``async_memory_resource``.
//!
//! @endrst
class async_memory_resource
{
private:
  ::cudaMemPool_t __pool_;

  //! @brief Checks whether the passed in alignment is valid.
  //! @param __alignment the alignment to check.
  //! @returns true if \p __alignment is valid.
  _CCCL_NODISCARD static constexpr bool __is_valid_alignment(const size_t __alignment) noexcept
  {
    return __alignment <= _CUDA_VMR::default_cuda_malloc_alignment
        && (_CUDA_VMR::default_cuda_malloc_alignment % __alignment == 0);
  }

  //! @brief  Returns the default ``cudaMemPool_t`` from the specified device.
  //! @throws cuda_error if retrieving the default ``cudaMemPool_t`` fails.
  //! @returns The default memory pool of the specified device.
  _CCCL_NODISCARD static ::cudaMemPool_t __get_default_mem_pool(const int __device_id)
  {
    ::cuda::experimental::mr::__device_supports_stream_ordered_allocations(__device_id);

    ::cudaMemPool_t __pool;
    _CCCL_TRY_CUDA_API(
      ::cudaDeviceGetDefaultMemPool, "Failed to call cudaDeviceGetDefaultMemPool", &__pool, __device_id);
    return __pool;
  }

public:
  //! @brief Default constructs the async_memory_resource using the default \c cudaMemPool_t of the default device.
  //! @throws cuda_error if retrieving the default \c cudaMemPool_t fails.
  async_memory_resource()
      : __pool_(__get_default_mem_pool(0))
  {}

  //! @brief Constructs a async_memory_resource using the default \c cudaMemPool_t of a given device.
  //! @throws cuda_error if retrieving the default \c cudaMemPool_t fails.
  explicit async_memory_resource(::cuda::experimental::device_ref __device)
      : __pool_(__get_default_mem_pool(__device.get()))
  {}

  async_memory_resource(int)                   = delete;
  async_memory_resource(_CUDA_VSTD::nullptr_t) = delete;

  //! @brief  Constructs the async_memory_resource from a \c cudaMemPool_t.
  //! @param __pool The \c cudaMemPool_t used to allocate memory.
  explicit async_memory_resource(::cudaMemPool_t __pool) noexcept
      : __pool_(__pool)
  {}

  //! @brief  Constructs the async_memory_resource from a \c async_memory_pool by calling get().
  //! @param __pool The \c async_memory_pool used to allocate memory.
  explicit async_memory_resource(async_memory_pool& __pool) noexcept
      : __pool_(__pool.get())
  {}

  //! @brief Allocate device memory of size at least \p __bytes via cudaMallocFromPoolAsync.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @throws std::invalid_argument In case of invalid alignment.
  //! @throws cuda::cuda_error If an error code was return by the CUDA API call.
  //! @returns Pointer to the newly allocated memory.
  _CCCL_NODISCARD void* allocate(const size_t __bytes,
                                 const size_t __alignment = _CUDA_VMR::default_cuda_malloc_alignment)
  {
    if (!__is_valid_alignment(__alignment))
    {
      _CUDA_VSTD_NOVERSION::__throw_invalid_argument(
        "Invalid alignment passed to "
        "async_memory_resource::allocate_async.");
    }

    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocFromPoolAsync,
      "async_memory_resource::allocate failed to allocate with cudaMallocFromPoolAsync",
      &__ptr,
      __bytes,
      __pool_,
      __async_memory_resource_sync_stream().get());
    __async_memory_resource_sync_stream().wait();
    return __ptr;
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`.
  //! @param __bytes  The number of bytes that was passed to the `allocate` call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr.
  //! @note The pointer passed to `deallocate` must not be in use in a stream. It is the caller's responsibility to
  //! properly synchronize all relevant streams before calling `deallocate`.
  void deallocate(void* __ptr, const size_t, const size_t __alignment = _CUDA_VMR::default_cuda_malloc_alignment)
  {
    _LIBCUDACXX_ASSERT(__is_valid_alignment(__alignment),
                       "Invalid alignment passed to async_memory_resource::deallocate.");
    _CCCL_ASSERT_CUDA_API(
      ::cudaFreeAsync, "async_memory_resource::deallocate failed", __ptr, __async_memory_resource_sync_stream().get());
    __async_memory_resource_sync_stream().wait();
    (void) __alignment;
  }

  //! @brief Allocate device memory of size at least \p __bytes via `cudaMallocFromPoolAsync`.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @param __stream Stream on which to perform allocation.
  //! @throws std::invalid_argument In case of invalid alignment.
  //! @throws cuda::cuda_error If an error code was return by the cuda api call.
  //! @returns Pointer to the newly allocated memory.
  _CCCL_NODISCARD void* allocate_async(const size_t __bytes, const size_t __alignment, const ::cuda::stream_ref __stream)
  {
    if (!__is_valid_alignment(__alignment))
    {
      _CUDA_VSTD_NOVERSION::__throw_invalid_argument(
        "Invalid alignment passed to "
        "async_memory_resource::allocate_async.");
    }

    return allocate_async(__bytes, __stream);
  }

  //! @brief Allocate device memory of size at least \p __bytes via cudaMallocFromPoolAsync.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __stream Stream on which to perform allocation.
  //! @throws cuda::cuda_error If an error code was return by the cuda api call.
  //! @returns Pointer to the newly allocated memory.
  _CCCL_NODISCARD void* allocate_async(const size_t __bytes, const ::cuda::stream_ref __stream)
  {
    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocFromPoolAsync,
      "async_memory_resource::allocate_async failed to allocate with cudaMallocFromPoolAsync",
      &__ptr,
      __bytes,
      __pool_,
      __stream.get());
    return __ptr;
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate_async`
  //! @param __bytes The number of bytes that was passed to the `allocate_async` call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the `allocate_async` call that returned \p __ptr.
  //! @param __stream A stream that has a stream ordering relationship with the stream used in the
  //! <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">allocate_async</a> call
  //! that returned \p __ptr.
  //! @note The pointer passed to `deallocate_async` must not be in use in a stream other than \p __stream.
  //! It is the caller's responsibility to properly synchronize all relevant streams before calling `deallocate_async`.
  void deallocate_async(void* __ptr, const size_t __bytes, const size_t __alignment, const ::cuda::stream_ref __stream)
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _LIBCUDACXX_ASSERT(__is_valid_alignment(__alignment),
                       "Invalid alignment passed to async_memory_resource::deallocate.");
    deallocate_async(__ptr, __bytes, __stream);
    (void) __alignment;
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate_async`.
  //! @param __bytes The number of bytes that was passed to the `allocate_async` call that returned \p __ptr.
  //! @param __stream A stream that has a stream ordering relationship with the stream used in the
  //! <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">allocate_async</a> call
  //! that returned \p __ptr.
  //! @note The pointer passed to `deallocate_async` must not be in use in a stream other than \p __stream.
  //! It is the caller's responsibility to properly synchronize all relevant streams before calling `deallocate_async`.
  void deallocate_async(void* __ptr, size_t, const ::cuda::stream_ref __stream)
  {
    _CCCL_ASSERT_CUDA_API(::cudaFreeAsync, "async_memory_resource::deallocate_async failed", __ptr, __stream.get());
  }

  //! @brief Equality comparison with another async_memory_resource.
  //! @returns true if underlying \c cudaMemPool_t are equal.
  _CCCL_NODISCARD constexpr bool operator==(async_memory_resource const& __rhs) const noexcept
  {
    return __pool_ == __rhs.__pool_;
  }
#    if _CCCL_STD_VER <= 2017

  //! @brief Inequality comparison with another \c async_memory_resource.
  //! @returns true if underlying \c cudaMemPool_t are inequal.
  _CCCL_NODISCARD constexpr bool operator!=(async_memory_resource const& __rhs) const noexcept
  {
    return __pool_ != __rhs.__pool_;
  }
#    endif // _CCCL_STD_VER <= 2017

#    if _CCCL_STD_VER >= 2020
  //! @brief Equality comparison between a \c async_memory_resource and another resource.
  //! @param __rhs The resource to compare to.
  //! @returns If the underlying types are equality comparable, returns the result of equality comparison of both
  //! resources. Otherwise, returns false.
  _LIBCUDACXX_TEMPLATE(class _Resource)
  _LIBCUDACXX_REQUIRES((_CUDA_VMR::__different_resource<async_memory_resource, _Resource>) )
  _CCCL_NODISCARD bool operator==(_Resource const& __rhs) const noexcept
  {
    return _CUDA_VMR::resource_ref<>{const_cast<async_memory_resource*>(this)}
        == _CUDA_VMR::resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
#    else // ^^^ C++20 ^^^ / vvv C++17
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(async_memory_resource const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(_CUDA_VMR::__different_resource<async_memory_resource, _Resource>)
  {
    return _CUDA_VMR::resource_ref<>{const_cast<async_memory_resource&>(__lhs)}
        == _CUDA_VMR::resource_ref<>{const_cast<_Resource&>(__rhs)};
  }

  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(_Resource const& __rhs, async_memory_resource const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(_CUDA_VMR::__different_resource<async_memory_resource, _Resource>)
  {
    return _CUDA_VMR::resource_ref<>{const_cast<async_memory_resource&>(__lhs)}
        == _CUDA_VMR::resource_ref<>{const_cast<_Resource&>(__rhs)};
  }

  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(async_memory_resource const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(_CUDA_VMR::__different_resource<async_memory_resource, _Resource>)
  {
    return _CUDA_VMR::resource_ref<>{const_cast<async_memory_resource&>(__lhs)}
        != _CUDA_VMR::resource_ref<>{const_cast<_Resource&>(__rhs)};
  }

  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(_Resource const& __rhs, async_memory_resource const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(_CUDA_VMR::__different_resource<async_memory_resource, _Resource>)
  {
    return _CUDA_VMR::resource_ref<>{const_cast<async_memory_resource&>(__lhs)}
        != _CUDA_VMR::resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
#    endif // _CCCL_STD_VER <= 2017

  //! @brief Returns the underlying handle to the CUDA memory pool.
  _CCCL_NODISCARD constexpr cudaMemPool_t get() const noexcept
  {
    return __pool_;
  }

#    ifndef DOXYGEN_SHOULD_SKIP_THIS // Doxygen cannot handle the friend function
  //! @brief Enables the \c device_accessible property for \c async_memory_resource.
  //! @relates async_memory_resource
  friend constexpr void get_property(async_memory_resource const&, _CUDA_VMR::device_accessible) noexcept {}
#    endif // DOXYGEN_SHOULD_SKIP_THIS
};
static_assert(_CUDA_VMR::resource_with<async_memory_resource, _CUDA_VMR::device_accessible>, "");

} // namespace cuda::experimental::mr

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && !_CCCL_CUDACC_BELOW_11_2

#endif //_CUDAX__MEMORY_RESOURCE_CUDA_ASYNC_MEMORY_RESOURCE
