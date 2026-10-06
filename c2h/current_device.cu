// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/__cccl/assert.h>

#include <cuda_runtime_api.h>

#include <c2h/detail/current_device.cuh>

namespace c2h::detail
{
void assert_current_device(int device) noexcept
{
  int current_device{};
  _CCCL_VERIFY(cudaGetDevice(&current_device) == cudaSuccess, "Failed to get the current CUDA device");
  _CCCL_VERIFY(current_device == device, "A device-bound C2H resource must use the current CUDA device");
}
} // namespace c2h::detail
