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

#  include <cub/device/device_for.cuh>

#  include <thrust/detail/function.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/parallel_for.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/std/__iterator/distance.h>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{
// for_each_n
_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class Input, class Size, class UnaryOp>
Input _CCCL_API _CCCL_FORCEINLINE for_each_n(execution_policy<Derived>& policy, Input first, Size count, UnaryOp op)
{
  THRUST_CDP_DISPATCH(
    (cudaStream_t stream = cuda_cub::stream(policy);
     cudaError_t status  = cub::DeviceFor::ForEachN(first, count, op, stream);
     cuda_cub::throw_on_error(status, "parallel_for failed");
     status = cuda_cub::synchronize_optional(policy);
     cuda_cub::throw_on_error(status, "parallel_for: failed to synchronize");),
    (for (Size idx = 0; idx != count; ++idx) { op(raw_reference_cast(*(first + idx))); }));

  return first + count;
}

// for_each
template <class Derived, class Input, class UnaryOp>
Input _CCCL_API _CCCL_FORCEINLINE for_each(execution_policy<Derived>& policy, Input first, Input last, UnaryOp op)
{
  using size_type = thrust::detail::it_difference_t<Input>;
  size_type count = static_cast<size_type>(::cuda::std::distance(first, last));

  return THRUST_NS_QUALIFIER::cuda_cub::for_each_n(policy, first, count, op);
}
} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
