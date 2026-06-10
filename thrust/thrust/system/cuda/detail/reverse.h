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
#  include <thrust/system/cuda/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class ItemsIt, class ResultIt>
ResultIt _CCCL_HOST_DEVICE reverse_copy(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, ResultIt result);

template <class Derived, class ItemsIt>
void _CCCL_HOST_DEVICE reverse(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last);
} // namespace cuda_cub
THRUST_NAMESPACE_END

#  include <thrust/system/cuda/detail/copy.h>
#  include <thrust/system/cuda/detail/swap_ranges.h>

#  include <cuda/std/__iterator/advance.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/reverse_iterator.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class ItemsIt, class ResultIt>
ResultIt _CCCL_HOST_DEVICE reverse_copy(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, ResultIt result)
{
  return cuda_cub::copy(policy, ::cuda::std::reverse_iterator{last}, ::cuda::std::reverse_iterator{first}, result);
}

template <class Derived, class ItemsIt>
void _CCCL_HOST_DEVICE reverse(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last)
{
  using difference_type = thrust::detail::it_difference_t<ItemsIt>;

  // find the midpoint of [first,last)
  difference_type N = ::cuda::std::distance(first, last);
  ItemsIt mid(first);
  ::cuda::std::advance(mid, N / 2);

  cuda_cub::swap_ranges(policy, first, mid, ::cuda::std::make_reverse_iterator(last));
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
