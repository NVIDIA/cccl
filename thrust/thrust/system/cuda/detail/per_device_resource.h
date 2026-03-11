// SPDX-FileCopyrightText: Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <mutex>
#  include <unordered_map>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{
template <typename MR, typename DerivedPolicy>
_CCCL_HOST MR* get_per_device_resource(execution_policy<DerivedPolicy>&)
{
  static std::mutex map_lock;
  static std::unordered_map<int, MR> device_id_to_resource;

  int device_id;
  thrust::cuda_cub::throw_on_error(cudaGetDevice(&device_id));

  std::lock_guard<std::mutex> lock{map_lock};
  return &device_id_to_resource[device_id];
}
} // namespace cuda_cub

THRUST_NAMESPACE_END

#endif // _CCCL_CUDA_COMPILATION()
