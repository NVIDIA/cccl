/***********************************************************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{

// TODO(gevtushenko/srinivasyadav18): move cudax `device_memory_resource` to `cuda::__device_memory_resource` and remove
// this implementation
struct device_memory_resource
{
  void* allocate(size_t bytes, size_t /* alignment */)
  {
    void* ptr{nullptr};
    _CCCL_TRY_CUDA_API(::cudaMalloc, "allocate failed to allocate with cudaMalloc", &ptr, bytes);
    return ptr;
  }

  void deallocate(void* ptr, size_t /* bytes */)
  {
    _CCCL_ASSERT_CUDA_API(::cudaFree, "deallocate failed", ptr);
  }

  void* allocate(::cuda::stream_ref stream, size_t bytes, size_t /* alignment */)
  {
    return allocate(stream, bytes);
  }

  void* allocate(::cuda::stream_ref stream, size_t bytes)
  {
    void* ptr{nullptr};
    _CCCL_TRY_CUDA_API(::cudaMallocAsync, "allocate failed to allocate with cudaMallocAsync", &ptr, bytes, stream.get());
    return ptr;
  }

  void deallocate(const ::cuda::stream_ref stream, void* ptr, size_t /* bytes */)
  {
    _CCCL_ASSERT_CUDA_API(::cudaFreeAsync, "deallocate failed", ptr, stream.get());
  }
};

} // namespace detail

CUB_NAMESPACE_END
