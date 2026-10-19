// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/dispatch/dispatch_rotate.cuh>

CUB_NAMESPACE_BEGIN

using RotateState_t = detail::rotate::RotateState_t;

struct DeviceRotate
{
  template <typename T>
  CUB_RUNTIME_FUNCTION static cudaError_t Rotate(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RotateState_t& state,
    T* d_array,
    size_t num_items,
    size_t rotate_distance,
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceRotate::Rotate");

    return detail::rotate::dispatch(
      d_temp_storage, temp_storage_bytes, state, d_array, num_items, rotate_distance, stream);
  }
};

CUB_NAMESPACE_END
