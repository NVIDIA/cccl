// SPDX-FileCopyrightText: Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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
#  include <thrust/iterator/permutation_iterator.h>
#  include <thrust/system/cuda/detail/transform.h>

#  include <cuda/std/__functional/identity.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class MapIt, class ItemsIt, class ResultIt>
ResultIt _CCCL_HOST_DEVICE
gather(execution_policy<Derived>& policy, MapIt map_first, MapIt map_last, ItemsIt items, ResultIt result)
{
  return cuda_cub::transform(
    policy,
    thrust::make_permutation_iterator(items, map_first),
    thrust::make_permutation_iterator(items, map_last),
    result,
    ::cuda::std::identity{});
}

template <class Derived, class MapIt, class StencilIt, class ItemsIt, class ResultIt, class Predicate>
ResultIt _CCCL_HOST_DEVICE gather_if(
  execution_policy<Derived>& policy,
  MapIt map_first,
  MapIt map_last,
  StencilIt stencil,
  ItemsIt items,
  ResultIt result,
  Predicate predicate)
{
  return cuda_cub::transform_if(
    policy,
    thrust::make_permutation_iterator(items, map_first),
    thrust::make_permutation_iterator(items, map_last),
    stencil,
    result,
    ::cuda::std::identity{},
    predicate);
}

template <class Derived, class MapIt, class StencilIt, class ItemsIt, class ResultIt>
ResultIt _CCCL_HOST_DEVICE gather_if(
  execution_policy<Derived>& policy, MapIt map_first, MapIt map_last, StencilIt stencil, ItemsIt items, ResultIt result)
{
  return cuda_cub::gather_if(policy, map_first, map_last, stencil, items, result, ::cuda::std::identity{});
}
} // namespace cuda_cub
THRUST_NAMESPACE_END

#endif // _CCCL_CUDA_COMPILATION()
