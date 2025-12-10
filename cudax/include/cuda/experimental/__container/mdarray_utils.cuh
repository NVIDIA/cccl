//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINER_MDARRAY_UTILS_CUH
#define __CUDAX__CONTAINER_MDARRAY_UTILS_CUH

#include <cuda/std/detail/__config>

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/all_devices.h>
#include <cuda/__device/device_ref.h>
#include <cuda/__mdspan/host_device_mdspan.h>
#include <cuda/__memory_resource/device_memory_pool.h>
#include <cuda/__memory_resource/shared_resource.h>
#include <cuda/std/__utility/delegate_constructors.h>

#include <cuda/experimental/__container/mdarray_base.cuh>

#include <cuda/std/__cccl/prologue.h>

// see also https://github.com/rapidsai/raft/blob/main/cpp/include/raft/core/mdarray.hpp
namespace cuda::experimental
{
template <typename _Alloc>
struct __construct_allocator
{
  [[nodiscard]] _CCCL_HOST_API static _Alloc __do()
  {
    return _Alloc{};
  }

  [[nodiscard]] _CCCL_HOST_API static _Alloc __do(::cuda::device_ref)
  {
    return _Alloc{};
  }
};

template <typename _Resource>
struct __construct_allocator<::cuda::mr::shared_resource<_Resource>>
{
  [[nodiscard]] _CCCL_HOST_API static ::cuda::mr::shared_resource<_Resource> __do(::cuda::device_ref __device)
  {
    return ::cuda::mr::make_shared_resource<_Resource>(__device);
  }
};

template <typename T, typename R>
void __copy(T&& src, R&& dst, ::cudaStream_t stream)
{
  // TODO: implementation
  cub::detail::copy_mdspan::copy(src, dst, stream);
}

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //__CUDAX__CONTAINER_MDARRAY_UTILS_CUH
