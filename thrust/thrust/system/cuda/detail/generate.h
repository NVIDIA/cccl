// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/dispatch.h>

#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class OutputIt, class Size, class Generator>
OutputIt _CCCL_HOST_DEVICE
generate_n(execution_policy<Derived>& policy, OutputIt result, Size count, Generator generator)
{
  THRUST_CDP_DISPATCH(
    (using Predicate = CUB_NS_QUALIFIER::detail::transform::always_true_predicate; //
     cudaError_t status;
     THRUST_INDEX_TYPE_DISPATCH(
       status,
       (CUB_NS_QUALIFIER::detail::transform::dispatch_t<
         CUB_NS_QUALIFIER::detail::transform::requires_stable_address::no,
         decltype(count_fixed),
         ::cuda::std::tuple<>,
         OutputIt,
         Predicate,
         Generator>::dispatch),
       count,
       (::cuda::std::tuple<>{}, result, count_fixed, Predicate{}, generator, cuda_cub::stream(policy)));
     throw_on_error(status, "generate_n: failed inside CUB");
     throw_on_error(synchronize_optional(policy), "generate_n: failed to synchronize");
     return result + count;),
    (return thrust::generate_n(cvt_to_seq(derived_cast(policy)), result, count, generator);));
}

template <class Derived, class OutputIt, class Generator>
void _CCCL_HOST_DEVICE generate(execution_policy<Derived>& policy, OutputIt first, OutputIt last, Generator generator)
{
  cuda_cub::generate_n(policy, first, ::cuda::std::distance(first, last), generator);
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
