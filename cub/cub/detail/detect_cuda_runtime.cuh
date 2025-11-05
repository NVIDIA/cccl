// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * Utilities for CUDA dynamic parallelism.
 */

#pragma once

// We cannot use `cub/config.cuh` here due to circular dependencies
#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// CUDA headers might not be present when using NVRTC, see NVIDIA/cccl#2095 for detail
#if !_CCCL_COMPILER(NVRTC)
#  include <cuda_runtime_api.h>
#endif // !_CCCL_COMPILER(NVRTC)

#ifdef _CCCL_DOXYGEN_INVOKED // Only parse this during doxygen passes:
//! Defined if RDC is enabled and CUB_DISABLE_CDP is not defined.
//! Deprecated [Since 3.2]
#  define CUB_RDC_ENABLED

//! If defined, support for device-side usage of CUB is disabled.
//! Deprecated [Since 3.2]. Use CCCL_DISABLE_CDP instead.
#  define CUB_DISABLE_CDP

//! Execution space for functions that use the CUDA runtime API, e.g. to launch kernels. Such functions are `__host__
//! __device__` when compiling with RDC, otherwise only `__host__`.
//! Deprecated [Since 3.2]
#  define CUB_RUNTIME_FUNCTION
#else // Non-doxygen pass:

#  if _CCCL_HAS_RDC()
#    define CUB_RDC_ENABLED
#  endif // _CCCL_HAS_RDC()

#  ifndef CUB_RUNTIME_FUNCTION
#    define CUB_RUNTIME_FUNCTION _CCCL_CDP_API
#  endif // CUB_RUNTIME_FUNCTION predefined
#endif // Do not document
