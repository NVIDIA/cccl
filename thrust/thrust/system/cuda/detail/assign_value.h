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

#  include <thrust/detail/raw_pointer_cast.h>
#  include <thrust/system/cuda/detail/cross_system.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <nv/target>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes") // __visibility__ attribute ignored

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace detail
{
template <typename T, typename U>
CCCL_DETAIL_KERNEL_ATTRIBUTES void assign_value_kernel(T* dst, const U* src)
{
  *dst = *src;
}
} // namespace detail

template <typename DerivedPolicy, typename Pointer1, typename Pointer2>
_CCCL_HOST_DEVICE void assign_value(execution_policy<DerivedPolicy>& exec, Pointer1 dst, Pointer2 src)
{
  using V1 = thrust::detail::it_value_t<Pointer1>;
  using V2 = thrust::detail::it_value_t<Pointer2>;
  // Because of https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-arch point 2., if a call from a __host__
  // __device__ function leads to the template instantiation of a __global__ function, then this instantiation needs to
  // happen regardless of whether __CUDA_ARCH__ is defined. Therefore, we make the host path visible outside the
  // NV_IF_TARGET switch. See also NVBug 881631.
  struct HostPath
  {
    execution_policy<DerivedPolicy>& exec;
    Pointer1 dst;
    Pointer2 src;

    _CCCL_HOST void operator()() const
    {
      if constexpr (::cuda::std::is_same_v<V1, V2>)
      {
        const cudaError status = trivial_copy_device_to_device(exec, raw_pointer_cast(dst), raw_pointer_cast(src), 1);
        throw_on_error(status, "__copy:: D->D: failed");
      }
      else
      {
        const cudaError status =
          cuda_cub::detail::triple_chevron(1, 1, 0, stream(exec))
            .doit(detail::assign_value_kernel<V1, V2>, raw_pointer_cast(dst), raw_pointer_cast(src));
        throw_on_error(status, "__copy:: D->D with different data types: kernel failed");
        throw_on_error(synchronize_optional(exec), "__copy:: D->D with different data types: sync failed");
      }
    }
  };
  NV_IF_TARGET(NV_IS_HOST,
               // on host, perform device -> device memcpy
               (HostPath{exec, dst, src}();),
               // on device, simply assign
               *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src););
}

namespace detail
{
struct cross_system_assign_host_path
{
  // device -> host copy executed from host
  template <typename System1, typename DerivedPolicy2, typename Pointer1, typename Pointer2>
  _CCCL_HOST void operator()(System1&, execution_policy<DerivedPolicy2>& system2, Pointer1 dst, Pointer2 src)
  {
    thrust::detail::it_value_t<Pointer2> copy_dst;
    const cudaError status = trivial_copy_from_device(&copy_dst, raw_pointer_cast(src), 1, stream(system2));
    *dst                   = copy_dst; // may convert type
    throw_on_error(status, "__copy:: D->H: failed");
  }

  // host -> device copy executed from host
  template <typename DerivedPolicy1, typename System2, typename Pointer1, typename Pointer2>
  _CCCL_HOST void operator()(execution_policy<DerivedPolicy1>& system1, System2&, Pointer1 dst, Pointer2 src)
  {
    thrust::detail::it_value_t<Pointer1> copy_src = *src; // may convert type
    const cudaError status = trivial_copy_to_device(raw_pointer_cast(dst), &copy_src, 1, stream(system1));
    throw_on_error(status, "__copy:: H->D: failed");
  }
};
} // namespace detail

template <typename System1, typename System2, typename Pointer1, typename Pointer2>
_CCCL_HOST_DEVICE void assign_value(cross_system<System1, System2>& systems, Pointer1 dst, Pointer2 src)
{
  NV_IF_TARGET(NV_IS_HOST,
               (detail::cross_system_assign_host_path{}(
                  thrust::detail::derived_cast(systems.sys1), thrust::detail::derived_cast(systems.sys2), dst, src);),
               (*thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src);));
}
} // namespace cuda_cub

THRUST_NAMESPACE_END

_CCCL_DIAG_POP

#endif // _CCCL_CUDA_COMPILATION()
