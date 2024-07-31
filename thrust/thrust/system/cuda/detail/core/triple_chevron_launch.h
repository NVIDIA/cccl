/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/system/cuda/config.h>

#include <cuda/cmath>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{
namespace launcher
{

struct _CCCL_VISIBILITY_HIDDEN triple_chevron
{
  using Size = size_t;
  dim3 const grid;
  dim3 const block;
  Size const shared_mem;
  cudaStream_t const stream;

  THRUST_RUNTIME_FUNCTION triple_chevron(dim3 grid_, dim3 block_, Size shared_mem_ = 0, cudaStream_t stream_ = 0)
      : grid(grid_)
      , block(block_)
      , shared_mem(shared_mem_)
      , stream(stream_)
  {}

  template <class K, class... Args>
  cudaError_t _CCCL_HOST doit_host(K k, Args const&... args) const
  {
    k<<<grid, block, shared_mem, stream>>>(args...);
    return cudaPeekAtLastError();
  }

  template <class T>
  size_t _CCCL_DEVICE align_up(size_t offset) const
  {
    return ::cuda::ceil_div(offset, alignof(T)) * alignof(T);
  }

  size_t _CCCL_DEVICE argument_pack_size(size_t size) const
  {
    return size;
  }

  template <class... Args>
  size_t _CCCL_DEVICE argument_pack_size(size_t size, Args const&...) const
  {
    // TODO(bgruber): replace by fold over comma in C++17 (make sure order of evaluation is left to right!)
    int dummy[] = {(size += align_up<Args>(size) + sizeof(Args), 0)...};
    (void) dummy;
    return size;
  }

  template <class Arg>
  void _CCCL_DEVICE copy_arg(char* buffer, size_t& offset, Arg arg) const
  {
    // TODO(bgruber): we should make sure that we can actually byte-wise copy Arg, but this fails with some tests
    // static_assert(::cuda::std::is_trivially_copyable<Arg>::value, "");

    offset = align_up<Arg>(offset);
    for (int i = 0; i != sizeof(Arg); ++i)
    {
      buffer[offset + i] = reinterpret_cast<const char*>(&arg)[i];
    }
    offset += sizeof(Arg);
  }

  _CCCL_DEVICE void fill_arguments(char*, size_t) const {}

  template <class... Args>
  _CCCL_DEVICE void fill_arguments(char* buffer, size_t offset, Args const&... args) const
  {
    // TODO(bgruber): replace by fold over comma in C++17 (make sure order of evaluation is left to right!)
    int dummy[] = {(copy_arg(buffer, offset, args), 0)...};
    (void) dummy;
  }

#ifdef THRUST_RDC_ENABLED
  template <class K, class... Args>
  cudaError_t _CCCL_DEVICE doit_device(K k, Args const&... args) const
  {
    const size_t size  = argument_pack_size(0, args...);
    void* param_buffer = cudaGetParameterBuffer(64, size);
    fill_arguments((char*) param_buffer, 0, args...);
    return launch_device(k, param_buffer);
  }

  template <class K>
  cudaError_t _CCCL_DEVICE launch_device(K k, void* buffer) const
  {
    return cudaLaunchDevice((void*) k, buffer, dim3(grid), dim3(block), shared_mem, stream);
  }
#else
  template <class K, class... Args>
  cudaError_t _CCCL_DEVICE doit_device(K, Args const&...) const
  {
    return cudaErrorNotSupported;
  }
#endif

  _CCCL_EXEC_CHECK_DISABLE
  template <class K, class... Args>
  THRUST_FUNCTION cudaError_t doit(K k, Args const&... args) const
  {
    NV_IF_TARGET(NV_IS_HOST, (return doit_host(k, args...);), (return doit_device(k, args...);));
  }

}; // struct triple_chevron

} // namespace launcher
} // namespace cuda_cub

THRUST_NAMESPACE_END
