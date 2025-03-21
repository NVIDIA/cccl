/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#if _CCCL_HAS_CUDA_COMPILER()
#  include <thrust/system/cuda/config.h>

#  include <thrust/detail/raw_pointer_cast.h>
#  include <thrust/system/cuda/detail/copy.h>
#  include <thrust/system/cuda/detail/execution_policy.h>

#  include <nv/target>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <typename DerivedPolicy, typename Pointer1, typename Pointer2>
_CCCL_HOST_DEVICE void assign_value(execution_policy<DerivedPolicy>& exec, Pointer1 dst, Pointer2 src)
{
  // Because of https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-arch point 2., if a call from a __host__
  // __device__ function leads to the template instantiation of a __global__ function, then this instantiation needs to
  // happen regardless of whether __CUDA_ARCH__ is defined. Therefore, we make the host path visible outside the
  // NV_IF_TARGET switch. See also NVBug 881631.
  struct HostPath
  {
    _CCCL_HOST auto operator()(execution_policy<DerivedPolicy>& exec, Pointer1 dst, Pointer2 src)
    {
      cuda_cub::copy(exec, src, src + 1, dst);
    }
  };
  NV_IF_TARGET(
    NV_IS_HOST, (HostPath{}(exec, dst, src);), *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src););
}

template <typename System1, typename System2, typename Pointer1, typename Pointer2>
_CCCL_HOST_DEVICE void assign_value(cross_system<System1, System2>& systems, Pointer1 dst, Pointer2 src)
{
  // Because of https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-arch point 2., if a call from a __host__
  // __device__ function leads to the template instantiation of a __global__ function, then this instantiation needs to
  // happen regardless of whether __CUDA_ARCH__ is defined. Therefore, we make the host path visible outside the
  // NV_IF_TARGET switch. See also NVBug 881631.
  struct HostPath
  {
    _CCCL_HOST auto operator()(cross_system<System1, System2>& systems, Pointer1 dst, Pointer2 src)
    {
      // rotate the systems so that they are ordered the same as (src, dst) for the call to thrust::copy
      cross_system<System2, System1> rotated_systems = systems.rotate();
      cuda_cub::copy(rotated_systems, src, src + 1, dst);
    }
  };
  NV_IF_TARGET(NV_IS_HOST,
               (HostPath{}(systems, dst, src);),
               (
                 // XXX forward the true cuda::execution_policy inside systems here instead of materializing a tag
                 cuda::tag cuda_tag; cuda_cub::assign_value(cuda_tag, dst, src);));
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
