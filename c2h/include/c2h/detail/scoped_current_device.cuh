// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/std/__exception/cuda_error.h>

#include <cuda_runtime_api.h>

namespace c2h::detail
{
//! @brief Temporarily makes a CUDA device current and restores the previous device when destroyed.
class scoped_current_device
{
public:
  explicit scoped_current_device(int device)
  {
    const ::cudaError_t get_status = ::cudaGetDevice(&m_previous_device);
    if (get_status != ::cudaSuccess)
    {
      throw ::cuda::cuda_error{get_status, "failed to get current device"};
    }

    if (m_previous_device != device)
    {
      const ::cudaError_t set_status = ::cudaSetDevice(device);
      if (set_status != ::cudaSuccess)
      {
        throw ::cuda::cuda_error{set_status, "failed to change current device"};
      }
      m_restore = true;
    }
  }

  scoped_current_device(const scoped_current_device&)            = delete;
  scoped_current_device& operator=(const scoped_current_device&) = delete;

  ~scoped_current_device() noexcept
  {
    if (m_restore)
    {
      // Destructors cannot report device-restoration failures.
      (void) ::cudaSetDevice(m_previous_device);
    }
  }

private:
  int m_previous_device = 0;
  bool m_restore        = false;
};
} // namespace c2h::detail
