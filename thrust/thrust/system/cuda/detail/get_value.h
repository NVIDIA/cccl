// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
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

#if _CCCL_CUDA_COMPILATION()
#  include <thrust/system/cuda/config.h>

#  include <thrust/detail/raw_pointer_cast.h>
#  include <thrust/iterator/iterator_traits.h>
#  include <thrust/system/cuda/detail/cross_system.h>

#  include <nv/target>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <typename DerivedPolicy, typename Pointer>
_CCCL_HOST_DEVICE thrust::detail::it_value_t<Pointer> get_value(execution_policy<DerivedPolicy>& exec, Pointer ptr)
{
  // Because of https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-arch point 2., if a call from a __host__
  // __device__ function leads to the template instantiation of a __global__ function, then this instantiation needs to
  // happen regardless of whether __CUDA_ARCH__ is defined. Therefore, we make the host path visible outside the
  // NV_IF_TARGET switch. See also NVBug 881631.
  struct HostPath
  {
    _CCCL_HOST auto operator()(execution_policy<DerivedPolicy>& exec, Pointer ptr)
    {
      // implemented with assign_value, which requires a type with a default constructor
      thrust::detail::it_value_t<Pointer> result;
      host_system_tag host_tag;
      cross_system<host_system_tag, DerivedPolicy> systems(host_tag, exec);
      assign_value(systems, &result, ptr);
      return result;
    }
  };
  NV_IF_TARGET(NV_IS_DEVICE, return *thrust::raw_pointer_cast(ptr);, (return HostPath{}(exec, ptr);))
}
} // namespace cuda_cub
THRUST_NAMESPACE_END

#endif // _CCCL_CUDA_COMPILATION()
