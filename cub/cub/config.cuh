// SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * \file
 * Static configuration header for the CUB project.
 */

#pragma once

// For _CCCL_IMPLICIT_SYSTEM_HEADER
#include <cuda/__cccl_config> // IWYU pragma: export

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_arch.cuh> // IWYU pragma: export
#include <cub/util_cpp_dialect.cuh> // IWYU pragma: export
#include <cub/util_macro.cuh> // IWYU pragma: export
#include <cub/util_namespace.cuh> // IWYU pragma: export

#if !_CCCL_COMPILER(NVRTC)
#  include <cuda/__nvtx/nvtx.h>
#endif // !_CCCL_COMPILER(NVRTC)
