/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
 * \file
 * Static architectural properties by SM version.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_cpp_dialect.cuh> // IWYU pragma: export
#include <cub/util_macro.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/cmath>
#include <cuda/functional>

// Legacy include; this functionality used to be defined in here.
#include <cub/detail/detect_cuda_runtime.cuh>

CUB_NAMESPACE_BEGIN

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

/// In device code, CUB_PTX_ARCH expands to the PTX version for which we are
/// compiling. In host code, CUB_PTX_ARCH's value is implementation defined.
#  ifndef CUB_PTX_ARCH
#    if defined(_NVHPC_CUDA)
// __NVCOMPILER_CUDA_ARCH__ is the target PTX version, and is defined
// when compiling both host code and device code. Currently, only one
// PTX version can be targeted.
#      define CUB_PTX_ARCH __NVCOMPILER_CUDA_ARCH__
#    elif !defined(__CUDA_ARCH__)
#      define CUB_PTX_ARCH 0
#    else
#      define CUB_PTX_ARCH __CUDA_ARCH__
#    endif
#  endif

/// Maximum number of devices supported.
#  ifndef CUB_MAX_DEVICES
#    define CUB_MAX_DEVICES (128)
#  endif

static_assert(CUB_MAX_DEVICES > 0, "CUB_MAX_DEVICES must be greater than 0.");

/// Number of threads per warp
#  ifndef CUB_LOG_WARP_THREADS
#    define CUB_LOG_WARP_THREADS(unused) (5)
#    define CUB_WARP_THREADS(unused)     (1 << CUB_LOG_WARP_THREADS(0))

#    define CUB_PTX_WARP_THREADS     CUB_WARP_THREADS(0)
#    define CUB_PTX_LOG_WARP_THREADS CUB_LOG_WARP_THREADS(0)
#  endif

/// Number of smem banks
#  ifndef CUB_LOG_SMEM_BANKS
#    define CUB_LOG_SMEM_BANKS(unused) (5)
#    define CUB_SMEM_BANKS(unused)     (1 << CUB_LOG_SMEM_BANKS(0))

#    define CUB_PTX_LOG_SMEM_BANKS CUB_LOG_SMEM_BANKS(0)
#    define CUB_PTX_SMEM_BANKS     CUB_SMEM_BANKS
#  endif

/// Oversubscription factor
#  ifndef CUB_SUBSCRIPTION_FACTOR
#    define CUB_SUBSCRIPTION_FACTOR(unused) (5)
#    define CUB_PTX_SUBSCRIPTION_FACTOR     CUB_SUBSCRIPTION_FACTOR(0)
#  endif

/// Prefer padding overhead vs X-way conflicts greater than this threshold
#  ifndef CUB_PREFER_CONFLICT_OVER_PADDING
#    define CUB_PREFER_CONFLICT_OVER_PADDING(unused) (1)
#    define CUB_PTX_PREFER_CONFLICT_OVER_PADDING     CUB_PREFER_CONFLICT_OVER_PADDING(0)
#  endif

namespace detail
{
// The maximum amount of static shared memory available per thread block
// Note that in contrast to dynamic shared memory, static shared memory is still limited to 48 KB
static constexpr ::cuda::std::size_t max_smem_per_block = 48 * 1024;

template <typename T>
constexpr int bound_scaling_block_threads(uint32_t Nominal4ByteBlockThreads, uint32_t ItemsPerThread)
{
  return ::cuda::minimum<>{}(
    Nominal4ByteBlockThreads, ::cuda::round_up(detail::max_smem_per_block / (sizeof(T) * ItemsPerThread), 32));
}

template <int Nominal4ByteBlockThreads, int Nominal4ByteItemsPerThread, typename T>
struct RegBoundScaling
{
  static constexpr int ITEMS_PER_THREAD =
    ::cuda::maximum<>{}(1u, (Nominal4ByteItemsPerThread * 4) / ::cuda::maximum<>{}(4u, sizeof(T)));

  static constexpr auto BLOCK_THREADS = bound_scaling_block_threads<T>(Nominal4ByteBlockThreads, ITEMS_PER_THREAD);
};

template <int Nominal4ByteBlockThreads, int Nominal4ByteItemsPerThread, typename T>
struct MemBoundScaling
{
  static constexpr int ITEMS_PER_THREAD = ::cuda::maximum<>{}(
    1u, ::cuda::minimum<>{}((Nominal4ByteItemsPerThread * 4) / sizeof(T), Nominal4ByteItemsPerThread * 2u));

  static constexpr auto BLOCK_THREADS = bound_scaling_block_threads<T>(Nominal4ByteBlockThreads, ITEMS_PER_THREAD);
};

} // namespace detail

template <int Nominal4ByteBlockThreads, int Nominal4ByteItemsPerThread, typename T>
struct CCCL_DEPRECATED_BECAUSE("Internal-only implementation details") RegBoundScaling
    : detail::MemBoundScaling<Nominal4ByteBlockThreads, Nominal4ByteItemsPerThread, T>
{};

template <int Nominal4ByteBlockThreads, int Nominal4ByteItemsPerThread, typename T>
struct CCCL_DEPRECATED_BECAUSE("Internal-only implementation details") MemBoundScaling
    : detail::MemBoundScaling<Nominal4ByteBlockThreads, Nominal4ByteItemsPerThread, T>
{};

#endif // Do not document

CUB_NAMESPACE_END
