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
#  include <thrust/system/cuda/config.h>

#  include <thrust/iterator/transform_iterator.h>
#  include <thrust/system/cuda/detail/reduce.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/__functional/equal_to_value.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class InputIt, class UnaryPred>
thrust::detail::it_difference_t<InputIt> _CCCL_HOST_DEVICE
count_if(execution_policy<Derived>& policy, InputIt first, InputIt last, UnaryPred unary_pred)
{
  using size_type       = thrust::detail::it_difference_t<InputIt>;
  using flag_iterator_t = transform_iterator<UnaryPred, InputIt, size_type, size_type>;

  return cuda_cub::reduce_n(
    policy,
    flag_iterator_t(first, unary_pred),
    ::cuda::std::distance(first, last),
    size_type(0),
    ::cuda::std::plus<size_type>());
}

template <class Derived, class InputIt, class Value>
thrust::detail::it_difference_t<InputIt> _CCCL_HOST_DEVICE
count(execution_policy<Derived>& policy, InputIt first, InputIt last, Value const& value)
{
  return cuda_cub::count_if(policy, first, last, ::cuda::equal_to_value<Value>{value});
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
