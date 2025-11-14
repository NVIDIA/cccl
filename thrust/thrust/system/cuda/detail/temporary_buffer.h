/*
 *  Copyright 2025 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/detail/bad_alloc.h>

#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
// If par_nosync does not have a user provided allocator attached, these
// overloads should be selected.

template <typename T>
_CCCL_HOST ::cuda::std::pair<T*, ::cuda::std::ptrdiff_t> get_temporary_buffer(par_nosync_t&, ::cuda::std::ptrdiff_t n)
{
  void* ptr;
  cudaError_t status = cudaMallocAsync(&ptr, sizeof(T) * n, nullptr);

  if (status != cudaSuccess)
  {
    cudaGetLastError(); // Clear the CUDA global error state.

    // That didn't work. We could be somewhere where async allocation isn't
    // supported like Windows, so try again with cudaMalloc.
    status = cudaMalloc(&ptr, sizeof(T) * n);

    if (status != cudaSuccess)
    {
      throw system::detail::bad_alloc(cuda_category().message(status).c_str());
    }
  }

  return ::cuda::std::make_pair(reinterpret_pointer_cast<T*>(ptr), n);
}

template <typename Pointer>
_CCCL_HOST void return_temporary_buffer(par_nosync_t&, Pointer ptr, ::cuda::std::ptrdiff_t)
{
  void* void_ptr = raw_pointer_cast(ptr);

  cudaError_t status = cudaFreeAsync(void_ptr, nullptr);

  if (status != cudaSuccess)
  {
    cudaGetLastError(); // Clear the CUDA global error state.

    // That didn't work. We could be somewhere where async allocation isn't
    // supported like Windows, so try again with cudaFree.
    status = cudaFree(void_ptr);

    if (status != cudaSuccess)
    {
      throw system::detail::bad_alloc(cuda_category().message(status).c_str());
    }
  }
}

template <typename T>
_CCCL_HOST ::cuda::std::pair<T*, ::cuda::std::ptrdiff_t>
get_temporary_buffer(execute_on_stream_nosync& system, ::cuda::std::ptrdiff_t n)
{
  void* ptr;
  cudaError_t status = cudaMallocAsync(&ptr, sizeof(T) * n, get_stream(system));

  if (status != cudaSuccess)
  {
    cudaGetLastError(); // Clear the CUDA global error state.

    // That didn't work. We could be somewhere where async allocation isn't
    // supported like Windows, so try again with cudaFree.
    status = cudaMalloc(&ptr, sizeof(T) * n);

    if (status != cudaSuccess)
    {
      throw system::detail::bad_alloc(cuda_category().message(status).c_str());
    }
  }

  return ::cuda::std::make_pair(reinterpret_pointer_cast<T*>(ptr), n);
}

template <typename Pointer>
_CCCL_HOST void return_temporary_buffer(execute_on_stream_nosync& system, Pointer ptr, ::cuda::std::ptrdiff_t)
{
  void* void_ptr = raw_pointer_cast(ptr);

  cudaError_t status = cudaFreeAsync(void_ptr, get_stream(system));

  if (status != cudaSuccess)
  {
    cudaGetLastError(); // Clear the CUDA global error state.

    // That didn't work. We could be somewhere where async allocation isn't
    // supported like Windows, so try again with cudaMalloc.
    status = cudaFree(void_ptr);

    if (status != cudaSuccess)
    {
      throw system::detail::bad_alloc(cuda_category().message(status).c_str());
    }
  }
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
