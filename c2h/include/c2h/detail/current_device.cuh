// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace c2h::detail
{
//! @brief Verifies that @p device is the current CUDA device.
//!
//! C2H tests run on the device selected by their `--device` argument. Operations using device-bound resources must
//! remain on that device rather than changing the current device implicitly.
void assert_current_device(int device) noexcept;
} // namespace c2h::detail
