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
#  include <thrust/detail/internal_functional.h>
#  include <thrust/system/cuda/detail/transform.h>

#  include <cuda/__functional/equal_to_value.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class Iterator, class T>
void _CCCL_HOST_DEVICE
replace(execution_policy<Derived>& policy, Iterator first, Iterator last, T const& old_value, T const& new_value)
{
  cuda_cub::transform_if(
    policy,
    first,
    last,
    first,
    CUB_NS_QUALIFIER::detail::__return_constant<T>{new_value},
    ::cuda::equal_to_value<T>{old_value});
}

template <class Derived, class Iterator, class Predicate, class T>
void _CCCL_HOST_DEVICE
replace_if(execution_policy<Derived>& policy, Iterator first, Iterator last, Predicate pred, T const& new_value)
{
  cuda_cub::transform_if(policy, first, last, first, CUB_NS_QUALIFIER::detail::__return_constant<T>{new_value}, pred);
}

template <class Derived, class Iterator, class StencilIt, class Predicate, class T>
void _CCCL_HOST_DEVICE replace_if(
  execution_policy<Derived>& policy,
  Iterator first,
  Iterator last,
  StencilIt stencil,
  Predicate pred,
  T const& new_value)
{
  cuda_cub::transform_if(
    policy, first, last, stencil, first, CUB_NS_QUALIFIER::detail::__return_constant<T>{new_value}, pred);
}

template <class Derived, class InputIt, class OutputIt, class Predicate, class T>
OutputIt _CCCL_HOST_DEVICE replace_copy_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  Predicate predicate,
  T const& new_value)
{
  using output_type    = thrust::detail::it_value_t<OutputIt>;
  using new_value_if_t = thrust::detail::new_value_if_f<Predicate, T, output_type>;
  return cuda_cub::transform(policy, first, last, result, new_value_if_t{predicate, new_value});
}

template <class Derived, class InputIt, class StencilIt, class OutputIt, class Predicate, class T>
OutputIt _CCCL_HOST_DEVICE replace_copy_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  OutputIt result,
  Predicate predicate,
  T const& new_value)
{
  using output_type    = thrust::detail::it_value_t<OutputIt>;
  using new_value_if_t = thrust::detail::new_value_if_f<Predicate, T, output_type>;
  return cuda_cub::transform(policy, first, last, stencil, result, new_value_if_t{predicate, new_value});
}

template <class Derived, class InputIt, class OutputIt, class T>
OutputIt _CCCL_HOST_DEVICE replace_copy(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  T const& old_value,
  T const& new_value)
{
  return cuda_cub::replace_copy_if(policy, first, last, result, ::cuda::equal_to_value<T>{old_value}, new_value);
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
