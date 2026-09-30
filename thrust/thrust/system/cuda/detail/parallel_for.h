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

#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/util.h>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{
_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class F, class Size>
void _CCCL_HOST_DEVICE parallel_for(execution_policy<Derived>& policy, F f, Size count)
{
  if (count == 0)
  {
    return;
  }

  // clang-format off
  THRUST_CDP_DISPATCH(
    (cudaStream_t stream = cuda_cub::stream(policy);
     cudaError_t  status = cub::DeviceFor::Bulk(count, f, stream);
     cuda_cub::throw_on_error(status, "parallel_for failed");
     status = cuda_cub::synchronize_optional(policy);
     cuda_cub::throw_on_error(status, "parallel_for: failed to synchronize");),
    // CDP sequential impl:
    (for (Size idx = 0; idx != count; ++idx)
     {
       f(idx);
     }
  ));
  // clang-format on
}
} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
