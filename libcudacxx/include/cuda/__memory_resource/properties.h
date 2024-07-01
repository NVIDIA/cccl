//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_PROPERTIES_H
#define _CUDA__MEMORY_RESOURCE_PROPERTIES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>

#if !defined(_CCCL_COMPILER_MSVC_2017) && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

#  if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

//! @brief The default alignment by a cudaMalloc{...} call
_LIBCUDACXX_INLINE_VAR constexpr size_t default_cuda_malloc_alignment = 256;

//! @brief The default alignment by a cudaMallocHost{...} call
_LIBCUDACXX_INLINE_VAR constexpr size_t default_cuda_malloc_host_alignment = alignof(_CUDA_VSTD::max_align_t);

//! @brief The device_accessible property signals that the allocated memory is device accessible
struct device_accessible
{};

//! @brief The device_accessible property signals that the allocated memory is host accessible
struct host_accessible
{};

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //_CUDA__MEMORY_RESOURCE_PROPERTIES_H
