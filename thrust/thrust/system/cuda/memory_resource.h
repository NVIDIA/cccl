/*
 *  Copyright 2018-2020 NVIDIA Corporation
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

/*! \file cuda/memory_resource.h
 *  \brief Memory resources for the CUDA system.
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

#include <thrust/mr/host_memory_resource.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system/cuda/pointer.h>
#include <thrust/system/detail/bad_alloc.h>

THRUST_NAMESPACE_BEGIN

namespace system
{
namespace cuda
{

//! \cond
namespace detail
{

using allocation_fn   = cudaError_t (*)(void**, std::size_t);
using deallocation_fn = cudaError_t (*)(void*);

template <allocation_fn Alloc, deallocation_fn Dealloc, typename Pointer>
class cuda_memory_resource final : public mr::memory_resource<Pointer>
{
public:
  Pointer do_allocate(std::size_t bytes, [[maybe_unused]] std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
  {
    void* ret;
    cudaError_t status = Alloc(&ret, bytes);

    if (status != cudaSuccess)
    {
      cudaGetLastError(); // Clear the CUDA global error state.
      throw thrust::system::detail::bad_alloc(thrust::cuda_category().message(status).c_str());
    }

    return Pointer(ret);
  }

  void do_deallocate(Pointer p, [[maybe_unused]] std::size_t bytes, [[maybe_unused]] std::size_t alignment) override
  {
    cudaError_t status = Dealloc(thrust::detail::pointer_traits<Pointer>::get(p));

    if (status != cudaSuccess)
    {
      thrust::cuda_cub::throw_on_error(status, "CUDA free failed");
    }
  }
};

inline cudaError_t CUDARTAPI cudaMallocManaged(void** ptr, std::size_t bytes)
{
  return ::cudaMallocManaged(ptr, bytes, cudaMemAttachGlobal);
}

using device_memory_resource = detail::cuda_memory_resource<cudaMalloc, cudaFree, thrust::cuda::pointer<void>>;
using managed_memory_resource =
  detail::cuda_memory_resource<detail::cudaMallocManaged, cudaFree, thrust::cuda::universal_pointer<void>>;
using pinned_memory_resource =
  detail::cuda_memory_resource<cudaMallocHost, cudaFreeHost, thrust::cuda::universal_host_pinned_pointer<void>>;

} // namespace detail
//! \endcond

/*! The memory resource for the CUDA system. Uses <tt>cudaMalloc</tt> and wraps
 *  the result with \p cuda::pointer.
 */
using memory_resource = detail::device_memory_resource;
/*! The universal memory resource for the CUDA system. Uses
 *  <tt>cudaMallocManaged</tt> and wraps the result with
 *  \p cuda::universal_pointer.
 */
using universal_memory_resource = detail::managed_memory_resource;
/*! The host pinned memory resource for the CUDA system. Uses
 *  <tt>cudaMallocHost</tt> and wraps the result with \p
 *  cuda::universal_pointer.
 */
using universal_host_pinned_memory_resource = detail::pinned_memory_resource;

} // namespace cuda
} // namespace system

namespace cuda
{
using thrust::system::cuda::memory_resource;
using thrust::system::cuda::universal_host_pinned_memory_resource;
using thrust::system::cuda::universal_memory_resource;
} // namespace cuda

THRUST_NAMESPACE_END
