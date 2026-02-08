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
#  include <thrust/detail/type_traits.h>
#  include <thrust/iterator/transform_iterator.h>
#  include <thrust/system/cuda/detail/scan.h>

#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__type_traits/remove_cvref.h>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{
template <class Derived, class InputIt, class OutputIt, class TransformOp, class ScanOp>
OutputIt _CCCL_HOST_DEVICE transform_inclusive_scan(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  TransformOp transform_op,
  ScanOp scan_op)
{
  // Use the transformed input iterator's value type per https://wg21.link/P0571
  using input_type  = thrust::detail::it_value_t<InputIt>;
  using result_type = thrust::detail::invoke_result_t<TransformOp, input_type>;
  using value_type  = ::cuda::std::remove_cvref_t<result_type>;

  using size_type              = thrust::detail::it_difference_t<InputIt>;
  size_type num_items          = static_cast<size_type>(::cuda::std::distance(first, last));
  using transformed_iterator_t = transform_iterator<TransformOp, InputIt, value_type, value_type>;

  return cuda_cub::inclusive_scan_n(policy, transformed_iterator_t(first, transform_op), num_items, result, scan_op);
}

template <class Derived, class InputIt, class OutputIt, class TransformOp, class InitialValueType, class ScanOp>
OutputIt _CCCL_HOST_DEVICE transform_inclusive_scan(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  TransformOp transform_op,
  InitialValueType init,
  ScanOp scan_op)
{
  using input_type  = thrust::detail::it_value_t<InputIt>;
  using result_type = thrust::detail::invoke_result_t<TransformOp, input_type>;
  using value_type  = ::cuda::std::remove_cvref_t<result_type>;

  using size_type              = thrust::detail::it_difference_t<InputIt>;
  size_type num_items          = static_cast<size_type>(::cuda::std::distance(first, last));
  using transformed_iterator_t = transform_iterator<TransformOp, InputIt, value_type, value_type>;

  return cuda_cub::inclusive_scan_n(
    policy, transformed_iterator_t(first, transform_op), num_items, result, init, scan_op);
}

template <class Derived, class InputIt, class OutputIt, class TransformOp, class InitialValueType, class ScanOp>
OutputIt _CCCL_HOST_DEVICE transform_exclusive_scan(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  TransformOp transform_op,
  InitialValueType init,
  ScanOp scan_op)
{
  // Use the initial value type per https://wg21.link/P0571
  using result_type = ::cuda::std::remove_cvref_t<InitialValueType>;

  using size_type              = thrust::detail::it_difference_t<InputIt>;
  size_type num_items          = static_cast<size_type>(::cuda::std::distance(first, last));
  using transformed_iterator_t = transform_iterator<TransformOp, InputIt, result_type, result_type>;

  return cuda_cub::exclusive_scan_n(
    policy, transformed_iterator_t(first, transform_op), num_items, result, init, scan_op);
}
} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
