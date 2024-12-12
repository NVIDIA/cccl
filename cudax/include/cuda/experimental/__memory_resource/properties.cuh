//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_PROPERTIES_CUH
#define _CUDAX__MEMORY_RESOURCE_PROPERTIES_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// If the memory resource header was included without the experimental flag,
// tell the user to define the experimental flag.
#if defined(_CUDA_MEMORY_RESOURCE) && !defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
#  error "To use the experimental memory resource, define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"
#endif

// cuda::mr is unavable on MSVC 2017
#if _CCCL_COMPILER(MSVC2017)
#  error "The any_resource header is not supported on MSVC 2017"
#endif // _CCCL_COMPILER(MSVC2017)

#if !defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
#  define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#endif

#include <cuda/__memory_resource/get_property.h>
#include <cuda/__memory_resource/properties.h>

namespace cuda::experimental
{

using ::cuda::mr::device_accessible;
using ::cuda::mr::host_accessible;

//! @brief determines the cudaMemcpyKind needed to transfer memory pointed to by an iterator to a cudax::mdarray
template <bool _IsHostOnly, class _Iter>
_CCCL_INLINE_VAR constexpr cudaMemcpyKind __detect_transfer_kind =
  has_property<_Iter, _CUDA_VMR::device_accessible>
    ? (_IsHostOnly ? cudaMemcpyKind::cudaMemcpyDeviceToHost : cudaMemcpyKind::cudaMemcpyDeviceToDevice)
    : (_IsHostOnly ? cudaMemcpyKind::cudaMemcpyHostToHost : cudaMemcpyKind::cudaMemcpyHostToDevice);

} // namespace cuda::experimental

#endif //_CUDAX__MEMORY_RESOURCE_PROPERTIES_CUH
