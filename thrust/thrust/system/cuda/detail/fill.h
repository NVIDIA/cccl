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
#  include <cub/device/device_transform.cuh>

#  include <thrust/fill.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/dispatch.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <typename T>
struct __return_constant
{
  T value;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE auto operator()() const -> T
  {
    return value;
  }
};

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class OutputIterator, class Size, class T>
OutputIterator _CCCL_HOST_DEVICE
fill_n(execution_policy<Derived>& policy, OutputIterator first, Size count, const T& value)
{
  THRUST_CDP_DISPATCH(
    (using Predicate   = CUB_NS_QUALIFIER::detail::transform::always_true_predicate;
     using TransformOp = __return_constant<T>;
     cudaError_t status;
     THRUST_INDEX_TYPE_DISPATCH(
       status,
       (CUB_NS_QUALIFIER::detail::transform::dispatch<CUB_NS_QUALIFIER::detail::transform::requires_stable_address::no>),
       count,
       (::cuda::std::tuple<>{}, first, count_fixed, Predicate{}, TransformOp{value}, cuda_cub::stream(policy)));
     throw_on_error(status, "fill_n: failed inside CUB");
     throw_on_error(synchronize_optional(policy), "fill_n: failed to synchronize");
     return first + count;),
    (return thrust::fill_n(cvt_to_seq(derived_cast(policy)), first, count, value);));
}

template <class Derived, class ForwardIterator, class T>
void _CCCL_HOST_DEVICE
fill(execution_policy<Derived>& policy, ForwardIterator first, ForwardIterator last, const T& value)
{
  cuda_cub::fill_n(policy, first, ::cuda::std::distance(first, last), value);
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
