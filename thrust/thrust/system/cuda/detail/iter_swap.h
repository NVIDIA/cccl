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
#  include <thrust/system/cuda/detail/execution_policy.h>

#  include <cuda/std/__utility/swap.h>

#  include <nv/target>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace detail
{
template <typename Pointer1, typename Pointer2>
CCCL_DETAIL_KERNEL_ATTRIBUTES void iter_swap_kernel(Pointer1 a, Pointer2 b)
{
  using ::cuda::std::swap;
  swap(*raw_pointer_cast(a), *raw_pointer_cast(b));
}
} // namespace detail

template <typename DerivedPolicy, typename Pointer1, typename Pointer2>
inline _CCCL_HOST_DEVICE void iter_swap(thrust::cuda::execution_policy<DerivedPolicy>& exec, Pointer1 a, Pointer2 b)
{
  // Because of https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-arch point 2., if a call from a __host__
  // __device__ function leads to the template instantiation of a __global__ function, then this instantiation needs to
  // happen regardless of whether __CUDA_ARCH__ is defined. Therefore, we make the host path visible outside the
  // NV_IF_TARGET switch. See also NVBug 881631.
  struct host_path
  {
    _CCCL_HOST void operator()(execution_policy<DerivedPolicy>& exec, Pointer1 a, Pointer2 b) const
    {
      const cudaError status =
        detail::triple_chevron(1, 1, 0, stream(exec)).doit(detail::iter_swap_kernel<Pointer1, Pointer2>, a, b);
      throw_on_error(status, "iter_swap: calling iter_swap_kernel failed");
      throw_on_error(synchronize_optional(exec), "iter_swap: sync failed");
    }
  };

  NV_IF_TARGET(NV_IS_HOST,
               (host_path{}(exec, a, b);),
               (using ::cuda::std::swap; //
                swap(*thrust::raw_pointer_cast(a), *thrust::raw_pointer_cast(b));));

} // end iter_swap()

} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
