//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_MEMORY_RESOURCE___
#define __CUDAX_MEMORY_RESOURCE___

// If the memory resource header was included without the experimental flag,
// tell the user to define the experimental flag.
#ifndef LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#  ifdef _CUDA_MEMORY_RESOURCE
#    error "To use the experimental memory resource, define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"
#  else // ^^^ _CUDA_MEMORY_RESOURCE ^^^ / vvv !_CUDA_MEMORY_RESOURCE vvv
#    warning "To use the experimental memory resource, define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"
#    define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#  endif // _CUDA_MEMORY_RESOURCE
#endif // !LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#include <cuda/experimental/__memory_resource/any_resource.cuh>
#include <cuda/experimental/__memory_resource/device_memory_pool.cuh>
#include <cuda/experimental/__memory_resource/device_memory_resource.cuh>
#include <cuda/experimental/__memory_resource/legacy_managed_memory_resource.cuh>
#include <cuda/experimental/__memory_resource/legacy_pinned_memory_resource.cuh>
#include <cuda/experimental/__memory_resource/pinned_memory_pool.cuh>
#include <cuda/experimental/__memory_resource/pinned_memory_resource.cuh>
#include <cuda/experimental/__memory_resource/properties.cuh>
#include <cuda/experimental/__memory_resource/resource.cuh>
#include <cuda/experimental/__memory_resource/shared_resource.cuh>

#endif // __CUDAX_MEMORY_RESOURCE___
