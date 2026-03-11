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
#  include <thrust/system/cuda/detail/execution_policy.h>

#  include <cuda/std/__utility/swap.h>

#  include <nv/target>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes") // __visibility__ attribute ignored

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

_CCCL_DIAG_POP

#endif // _CCCL_CUDA_COMPILATION()
