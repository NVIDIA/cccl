/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <thrust/device_allocator.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/system/cuda/memory_resource.h>
#include <thrust/system/cuda/pointer.h>

#include <cuda_runtime_api.h>

namespace c2h
{
namespace detail
{

// Check available memory prior to calling cudaMalloc.
// This avoids hangups and slowdowns from allocating swap / non-device memory
// on some platforms, namely tegra.
inline cudaError_t checked_cuda_malloc(void** ptr, std::size_t bytes)
{
  std::size_t free_bytes{};
  std::size_t total_bytes{};
  cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess)
  {
    return status;
  }

  // Avoid allocating all available memory:
  constexpr std::size_t padding = 16 * 1024 * 1024; // 16 MiB
  if (free_bytes < (bytes + padding))
  {
    return cudaErrorMemoryAllocation;
  }

  return cudaMalloc(ptr, bytes);
}
} // namespace detail

using checked_cuda_memory_resource =
  thrust::system::cuda::detail::cuda_memory_resource<detail::checked_cuda_malloc, cudaFree, thrust::cuda::pointer<void>>;
template <typename T>
using checked_cuda_allocator =
  thrust::mr::stateless_resource_allocator<T, thrust::device_ptr_memory_resource<checked_cuda_memory_resource>>;

} // namespace c2h
