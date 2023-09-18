/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <cub/detail/detect_cuda_runtime.cuh>

/**
 * \def THRUST_RUNTIME_FUNCTION
 *
 * Execution space for functions that can use the CUDA runtime API (`__host__`
 * when RDC is off, `__host__ __device__` when RDC is on).
 */
#define THRUST_RUNTIME_FUNCTION CUB_RUNTIME_FUNCTION

/**
 * \def THRUST_RDC_ENABLED
 *
 * Defined if RDC is enabled.
 */
#ifdef CUB_RDC_ENABLED
#define THRUST_RDC_ENABLED
#endif

/**
 * \def __THRUST_HAS_CUDART__
 *
 * Whether or not the active compiler pass is allowed to invoke device kernels
 * or methods from the CUDA runtime API.
 *
 * This macro should not be used in Thrust, as it depends on `__CUDA_ARCH__`
 * and is not compatible with `NV_IF_TARGET`. It is provided for legacy
 * purposes only.
 *
 * Replace any usages with `THRUST_RDC_ENABLED` and `NV_IF_TARGET`.
 */
#ifdef CUB_RUNTIME_ENABLED
#define __THRUST_HAS_CUDART__ 1
#else
#define __THRUST_HAS_CUDART__ 0
#endif

// These definitions were intended for internal use only and are now obsolete.
// If you relied on them, consider porting your code to use the functionality
// in libcu++'s <nv/target> header.
//
// For a temporary workaround, define THRUST_PROVIDE_LEGACY_ARCH_MACROS to make
// them available again. These should be considered deprecated and will be
// fully removed in a future version.
#ifdef THRUST_PROVIDE_LEGACY_ARCH_MACROS
#ifdef __CUDA_ARCH__
#define THRUST_DEVICE_CODE
#endif // __CUDA_ARCH__
#endif // THRUST_PROVIDE_LEGACY_ARCH_MACROS

#ifdef THRUST_AGENT_ENTRY_NOINLINE
#define THRUST_AGENT_ENTRY_INLINE_ATTR __noinline__
#else
#define THRUST_AGENT_ENTRY_INLINE_ATTR __forceinline__
#endif

#define THRUST_DEVICE_FUNCTION __device__ __forceinline__
#define THRUST_HOST_FUNCTION __host__     __forceinline__
#define THRUST_FUNCTION __host__ __device__ __forceinline__
