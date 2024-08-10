/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __CUDAX__MEMORY_RESOURCE_ASYNC_MEMORY_RESOURCE
#define __CUDAX__MEMORY_RESOURCE_ASYNC_MEMORY_RESOURCE

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/cuda_memory_resource.h>
#include <cuda/std/span>
#include <cuda/stream_ref>

#include <cuda/experimental/__device/device_ref.cuh>

namespace cuda::experimental
{
// @brief A memory resource that allocates memory asynchronously.
//
// This memory resource is an extension of `cuda_memory_resource` that provides
// asynchronous allocation and deallocation of memory. The `allocate_async` and
// `deallocate_async` are implemeted in terms of `cudaMallocAsync` and
// `cudaFreeAsync` respectively. The `allocate_async` and `deallocate_async`
// functions takes an additional `stream_ref` parameter that specifies the
// stream on which the (de)allocation should be performed.
class cuda_async_memory_resource : public _CUDA_VMR::cuda_memory_resource
{
public:
  constexpr cuda_async_memory_resource(device_ref __dev) noexcept
      : _CUDA_VMR::cuda_memory_resource(__dev.get())
  {}

  //! @brief Allocate CUDA unified memory of size at least \p __bytes.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @throw cuda::cuda_error of the returned error code
  //! @return Pointer to the newly allocated memory
  _CCCL_NODISCARD void*
  allocate_async(const size_t __bytes, [[maybe_unused]] const size_t __alignment, ::cuda::stream_ref __stream) const
  {
    // // We need to ensure that the provided alignment matches the minimal provided alignment
    // if (!__is_valid_alignment(__alignment))
    // {
    //   _CUDA_VSTD::__throw_bad_alloc();
    // }

    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocAsync, "Failed to allocate memory with cudaMallocAsync.", &__ptr, __bytes, __stream.get());
    return __ptr;
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`
  //! @param __bytes The number of bytes that was passed to the `allocate` call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr.
  void deallocate_async(
    void* __ptr, const size_t, [[maybe_unused]] const size_t __alignment, ::cuda::stream_ref __stream) const
  {
    // // We need to ensure that the provided alignment matches the minimal provided alignment
    // _LIBCUDACXX_ASSERT(__is_valid_alignment(__alignment),
    //                    "Invalid alignment passed to cuda_memory_resource::deallocate.");
    _CCCL_ASSERT_CUDA_API(::cudaFreeAsync, "cudaFreeAsync failed", __ptr, __stream.get());
  }
};
} // namespace cuda::experimental

#endif // __CUDAX__MEMORY_RESOURCE_ASYNC_MEMORY_RESOURCE
