//===----------------------------------------------------------------------===//
//
// Part of the CUDA Toolkit, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__UTILITY_SELECT_EXECUTION_SPACE_CUH
#define __CUDAX__UTILITY_SELECT_EXECUTION_SPACE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/properties.h>

#include <cuda/std/__cccl/prologue.h>

#if defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
namespace cuda::experimental
{

enum class _ExecutionSpace
{
  __host,
  __device,
  __host_device,
};

template <class... _Properties>
inline constexpr _ExecutionSpace __select_execution_space =
  ::cuda::mr::__is_host_device_accessible<_Properties...> ? _ExecutionSpace::__host_device
  : ::cuda::mr::__is_device_accessible<_Properties...>
    ? _ExecutionSpace::__device
    : _ExecutionSpace::__host;

} // namespace cuda::experimental

#endif // LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#include <cuda/std/__cccl/epilogue.h>

#endif //__CUDAX__UTILITY_SELECT_EXECUTION_SPACE_CUH
