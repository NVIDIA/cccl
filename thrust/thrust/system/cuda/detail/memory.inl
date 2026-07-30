// SPDX-FileCopyrightText: Copyright (c) 2008-2018, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/system/cuda/detail/malloc_and_free.h>
#include <thrust/system/cuda/memory.h>

#include <cuda/std/limits>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
_CCCL_HOST_DEVICE pointer<void> malloc(std::size_t n)
{
  tag cuda_tag;
  return pointer<void>(thrust::cuda_cub::malloc(cuda_tag, n));
} // end malloc()

template <typename T>
_CCCL_HOST_DEVICE pointer<T> malloc(std::size_t n)
{
  pointer<void> raw_ptr = thrust::cuda_cub::malloc(sizeof(T) * n);
  return pointer<T>(reinterpret_cast<T*>(raw_ptr.get()));
} // end malloc()

_CCCL_HOST_DEVICE void free(pointer<void> ptr)
{
  tag cuda_tag;
  return thrust::cuda_cub::free(cuda_tag, ptr.get());
} // end free()
} // namespace cuda_cub
THRUST_NAMESPACE_END
