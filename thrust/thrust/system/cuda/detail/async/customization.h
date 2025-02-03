/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// TODO: Move into system::cuda

#pragma once

#include <thrust/detail/config.h>

#include "cuda/std/__cccl/diagnostic.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/cpp14_required.h>

#if _CCCL_STD_VER >= 2014

#  if _CCCL_HAS_CUDA_COMPILER

#    include <thrust/system/cuda/config.h>

#    include <thrust/detail/execute_with_allocator.h>
#    include <thrust/detail/type_deduction.h>
#    include <thrust/mr/allocator.h>
#    include <thrust/mr/disjoint_sync_pool.h>
#    include <thrust/mr/host_memory_resource.h>
#    include <thrust/mr/sync_pool.h>
#    include <thrust/per_device_resource.h>
#    include <thrust/system/cuda/memory_resource.h>

#    include <cstdint>

THRUST_NAMESPACE_BEGIN

namespace system
{
namespace cuda
{
namespace detail
{

using default_async_host_resource = thrust::mr::synchronized_pool_resource<thrust::host_memory_resource>;

template <typename DerivedPolicy>
auto get_async_host_allocator(thrust::detail::execution_policy_base<DerivedPolicy>&)
  THRUST_RETURNS(thrust::mr::stateless_resource_allocator<std::uint8_t, default_async_host_resource>{})

  ///////////////////////////////////////////////////////////////////////////////

  using default_async_device_resource =
    thrust::mr::disjoint_synchronized_pool_resource<thrust::system::cuda::memory_resource,
                                                    thrust::mr::new_delete_resource>;

template <typename DerivedPolicy>
auto get_async_device_allocator(thrust::detail::execution_policy_base<DerivedPolicy>&)
  THRUST_RETURNS(thrust::per_device_allocator<std::uint8_t, default_async_device_resource, par_t>{})

    template <typename Allocator, template <typename> class BaseSystem>
    auto get_async_device_allocator(thrust::detail::execute_with_allocator<Allocator, BaseSystem>& exec)
      THRUST_RETURNS(exec.get_allocator())

        _CCCL_SUPPRESS_DEPRECATED_PUSH template <typename Allocator, template <typename> class BaseSystem>
        CCCL_DEPRECATED
  auto get_async_device_allocator(thrust::detail::execute_with_allocator_and_dependencies<Allocator, BaseSystem>& exec)
    THRUST_RETURNS(exec.get_allocator()) _CCCL_SUPPRESS_DEPRECATED_POP

  ///////////////////////////////////////////////////////////////////////////////

  using default_async_universal_host_pinned_resource =
    thrust::mr::synchronized_pool_resource<thrust::system::cuda::universal_host_pinned_memory_resource>;

template <typename DerivedPolicy>
auto get_async_universal_host_pinned_allocator(thrust::detail::execution_policy_base<DerivedPolicy>&)
  THRUST_RETURNS(thrust::mr::stateless_resource_allocator<std::uint8_t, default_async_universal_host_pinned_resource>{})

} // namespace detail
} // namespace cuda
} // namespace system

THRUST_NAMESPACE_END

#  endif // _CCCL_CUDA_COMPILER

#endif
