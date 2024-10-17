//===----------------------------------------------------------------------===//
//
// Part of the CUDA Toolkit, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__UTILITY_SELECT_EXECUTION_SPACE
#define __CUDAX__UTILITY_SELECT_EXECUTION_SPACE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/properties.h>

#if _CCCL_STD_VER >= 2014 && !defined(_CCCL_COMPILER_MSVC_2017) \
  && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
namespace cuda::experimental
{

enum class _ExecutionSpace
{
  __host,
  __device,
  __host_device,
};

template <class... _Properties>
_CCCL_INLINE_VAR constexpr _ExecutionSpace __select_execution_space =
  _CUDA_VMR::__is_host_device_accessible<_Properties...> ? _ExecutionSpace::__host_device
  : _CUDA_VMR::__is_device_accessible<_Properties...>
    ? _ExecutionSpace::__device
    : _ExecutionSpace::__host;

} // namespace cuda::experimental

#endif // _CCCL_STD_VER >= 2014 && !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //__CUDAX__UTILITY_SELECT_EXECUTION_SPACE
