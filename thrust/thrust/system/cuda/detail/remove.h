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
#  include <thrust/system/cuda/detail/copy_if.h>

#  include <cuda/__functional/equal_to_value.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
// in-place

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class StencilIt, class Predicate>
InputIt _CCCL_HOST_DEVICE
remove_if(execution_policy<Derived>& policy, InputIt first, InputIt last, StencilIt stencil, Predicate predicate)
{
  THRUST_CDP_DISPATCH((return cuda_cub::detail::copy_if<cub::SelectImpl::SelectPotentiallyInPlace>(
                                policy, first, last, stencil, first, ::cuda::std::not_fn(predicate));),
                      (return thrust::remove_if(cvt_to_seq(derived_cast(policy)), first, last, stencil, predicate);));
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class Predicate>
InputIt _CCCL_HOST_DEVICE remove_if(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate)
{
  THRUST_CDP_DISPATCH(
    (return cuda_cub::detail::copy_if<cub::SelectImpl::SelectPotentiallyInPlace>(
              policy, first, last, static_cast<cub::NullType*>(nullptr), first, ::cuda::std::not_fn(predicate));),
    (return thrust::remove_if(cvt_to_seq(derived_cast(policy)), first, last, predicate);));
}

template <class Derived, class InputIt, class T>
InputIt _CCCL_HOST_DEVICE remove(execution_policy<Derived>& policy, InputIt first, InputIt last, const T& value)
{
  using thrust::placeholders::_1;

  return cuda_cub::remove_if(policy, first, last, _1 == value);
}

// copy

template <class Derived, class InputIt, class StencilIt, class OutputIt, class Predicate>
OutputIt _CCCL_HOST_DEVICE remove_copy_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  OutputIt result,
  Predicate predicate)
{
  return cuda_cub::copy_if(policy, first, last, stencil, result, ::cuda::std::not_fn(predicate));
}

template <class Derived, class InputIt, class OutputIt, class Predicate>
OutputIt _CCCL_HOST_DEVICE
remove_copy_if(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, Predicate predicate)
{
  return cuda_cub::copy_if(policy, first, last, result, ::cuda::std::not_fn(predicate));
}

template <class Derived, class InputIt, class OutputIt, class T>
OutputIt _CCCL_HOST_DEVICE
remove_copy(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, const T& value)
{
  ::cuda::equal_to_value<T> pred{value};
  return cuda_cub::remove_copy_if(policy, first, last, result, pred);
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
