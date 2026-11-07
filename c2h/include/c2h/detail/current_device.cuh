// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/__cccl_config>

#include <cuda_runtime_api.h>

namespace c2h::detail
{
//! @brief Verifies that @p device is the current CUDA device.
//!
//! C2H tests run on the device selected by their `--device` argument. Operations using device-bound resources must
//! remain on that device rather than changing the current device implicitly.
_CCCL_HOST_API inline void assert_current_device(int device) noexcept
{
  int current_device{};
  _CCCL_VERIFY(::cudaGetDevice(&current_device) == ::cudaSuccess, "Failed to get the current CUDA device");
  _CCCL_VERIFY(current_device == device, "A device-bound C2H resource must use the current CUDA device");
}
} // namespace c2h::detail
