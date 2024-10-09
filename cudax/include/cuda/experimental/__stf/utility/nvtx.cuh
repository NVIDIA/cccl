//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Wrappers for NVTX
 */

#pragma once

#include <nvtx3/nvToolsExt.h>

namespace cuda::experimental::stf {

/**
 * @brief A RAII style wrapper for NVTX ranges
 */
class nvtx_range {
public:
    nvtx_range(const char* message) { nvtxRangePushA(message); }

    // Noncopyable and nonassignable to avoid multiple pops
    nvtx_range(const nvtx_range&) = delete;
    nvtx_range(nvtx_range&&) = delete;
    nvtx_range& operator=(const nvtx_range&) = delete;

    ~nvtx_range() { nvtxRangePop(); }
};

}  // end namespace cuda::experimental::stf
