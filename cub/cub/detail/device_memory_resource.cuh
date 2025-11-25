// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_device.cuh>

#include <cuda/__runtime/api_wrapper.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/cstdint>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail
{
// TODO(gevtushenko/srinivasyadav18): move cudax `device_memory_resource` to `cuda::__device_memory_resource` and remove
// this implementation
struct device_memory_resource
{
  CUB_RUNTIME_FUNCTION void* allocate(size_t bytes, size_t /* alignment */)
  {
    void* ptr{nullptr};
    NV_IF_TARGET(
      NV_IS_HOST,
      (_CCCL_TRY_CUDA_API(::cudaMallocAsync, "allocate failed to allocate with cudaMallocAsync", &ptr, bytes, NULL);),
      (_CubLog("%s\n", "cub::detail::device_memory_resource::allocate not supported from device code.");
       ::cuda::std::terminate();));
    _CCCL_ASSERT(ptr != nullptr, "allocate failed to allocate with cudaMallocAsync");
    return ptr;
  }

  CUB_RUNTIME_FUNCTION void deallocate(void* ptr, size_t /* bytes */)
  {
    NV_IF_TARGET( //
      NV_IS_HOST,
      (_CCCL_TRY_CUDA_API(::cudaFree, "deallocate failed", ptr);),
      (_CubLog("%s\n", "cub::detail::device_memory_resource::deallocate not supported from device code.");
       ::cuda::std::terminate();));
  }

  CUB_RUNTIME_FUNCTION void* allocate(::cuda::stream_ref stream, size_t bytes, size_t /* alignment */)
  {
    return allocate(stream, bytes);
  }

  CUB_RUNTIME_FUNCTION void* allocate(::cuda::stream_ref stream, size_t bytes)
  {
    void* ptr{nullptr};
    NV_IF_TARGET( //
      NV_IS_HOST,
      (_CCCL_TRY_CUDA_API(
         ::cudaMallocAsync, "allocate failed to allocate with cudaMallocAsync", &ptr, bytes, stream.get());),
      (_CubLog("%s\n", "cub::detail::device_memory_resource::allocate not supported from device code.");
       ::cuda::std::terminate();));
    return ptr;
  }

  CUB_RUNTIME_FUNCTION void deallocate(::cuda::stream_ref stream, void* ptr, size_t /* bytes */)
  {
    NV_IF_TARGET( //
      NV_IS_HOST,
      (_CCCL_TRY_CUDA_API(::cudaFreeAsync, "deallocate failed", ptr, stream.get());),
      (_CubLog("%s\n", "cub::detail::device_memory_resource::deallocate not supported from device code.");
       ::cuda::std::terminate();));
  }
};
} // namespace detail

CUB_NAMESPACE_END
