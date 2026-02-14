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

#  include <cuda/__functional/address_stability.h>
#  include <cuda/std/__functional/identity.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class ItemsIt, class MapIt, class ResultIt>
void _CCCL_HOST_DEVICE
scatter(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, MapIt map, ResultIt result)
{
  cuda_cub::transform(
    policy,
    first,
    last,
    thrust::make_permutation_iterator(result, map),
    ::cuda::proclaim_copyable_arguments(::cuda::std::identity{}));
}

template <class Derived, class ItemsIt, class MapIt, class StencilIt, class ResultIt, class Predicate>
void _CCCL_HOST_DEVICE scatter_if(
  execution_policy<Derived>& policy,
  ItemsIt first,
  ItemsIt last,
  MapIt map,
  StencilIt stencil,
  ResultIt result,
  Predicate predicate)
{
  cuda_cub::transform_if(
    policy,
    first,
    last,
    stencil,
    thrust::make_permutation_iterator(result, map),
    ::cuda::proclaim_copyable_arguments(::cuda::std::identity{}),
    predicate);
}

template <class Derived, class ItemsIt, class MapIt, class StencilIt, class ResultIt>
void _CCCL_HOST_DEVICE scatter_if(
  execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, MapIt map, StencilIt stencil, ResultIt result)
{
  cuda_cub::scatter_if(
    policy, first, last, map, stencil, result, ::cuda::proclaim_copyable_arguments(::cuda::std::identity{}));
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
