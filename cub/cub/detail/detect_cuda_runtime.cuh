/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

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
