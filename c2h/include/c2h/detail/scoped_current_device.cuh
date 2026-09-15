// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/__device/device_ref.h>
#include <cuda/__device/physical_device.h>
#include <cuda/__driver/driver_api.h>

namespace c2h::detail
{
//! @brief Temporarily makes a CUDA device's primary context current and restores the previous context when destroyed.
class scoped_current_device
{
public:
  explicit scoped_current_device(int device)
  {
    const auto context = ::cuda::device_ref{device}.__primary_context();
    ::cuda::__driver::__ctxPush(context);
  }

  scoped_current_device(const scoped_current_device&)            = delete;
  scoped_current_device& operator=(const scoped_current_device&) = delete;

  ~scoped_current_device() noexcept
  {
    try
    {
      (void) ::cuda::__driver::__ctxPop();
    }
    catch (...)
    {
      // Destructors cannot report context-restoration failures.
    }
  }
};
} // namespace c2h::detail
