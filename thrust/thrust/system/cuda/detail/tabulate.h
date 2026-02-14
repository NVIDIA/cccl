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

#  include <thrust/system/cuda/detail/transform.h>
#  include <thrust/system/cuda/execution_policy.h>

#  include <cuda/__functional/address_stability.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/incrementable_traits.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class Iterator, class TabulateOp>
void _CCCL_HOST_DEVICE tabulate(execution_policy<Derived>& policy, Iterator first, Iterator last, TabulateOp tabulate_op)
{
  using size_type  = ::cuda::std::iter_difference_t<Iterator>;
  const auto count = ::cuda::std::distance(first, last);
  cuda_cub::transform_n(
    policy, ::cuda::counting_iterator<size_type>{}, count, first, ::cuda::proclaim_copyable_arguments(tabulate_op));
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
