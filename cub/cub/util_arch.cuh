// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

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

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/round_up.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>

// Legacy include; this functionality used to be defined in here.
#include <cub/detail/detect_cuda_runtime.cuh>

CUB_NAMESPACE_BEGIN

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

/// In device code, CUB_PTX_ARCH expands to the PTX version for which we are
/// compiling. In host code, CUB_PTX_ARCH's value is implementation defined.
#  ifndef CUB_PTX_ARCH
// deprecated in 3.1
#    if _CCCL_CUDA_COMPILER(NVHPC)
// __NVCOMPILER_CUDA_ARCH__ is the target PTX version, and is defined
// when compiling both host code and device code. Currently, only one
// PTX version can be targeted.
#      define CUB_PTX_ARCH __NVCOMPILER_CUDA_ARCH__
#    else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVHPC) vvv
#      define CUB_PTX_ARCH _CCCL_PTX_ARCH()
#    endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) ^^^
#  endif

/// Maximum number of devices supported.
#  ifndef CUB_MAX_DEVICES
//! Deprecated [Since 3.0]
#    define CUB_MAX_DEVICES (128)
#  endif
static_assert(CUB_MAX_DEVICES > 0, "CUB_MAX_DEVICES must be greater than 0.");

/// Number of threads per warp
#  ifndef CUB_LOG_WARP_THREADS
//! Deprecated [Since 3.0]
#    define CUB_LOG_WARP_THREADS(unused) (5)
//! Deprecated [Since 3.0]
#    define CUB_WARP_THREADS(unused) (1 << CUB_LOG_WARP_THREADS(0))

//! Deprecated [Since 3.0]
#    define CUB_PTX_WARP_THREADS CUB_WARP_THREADS(0)
//! Deprecated [Since 3.0]
#    define CUB_PTX_LOG_WARP_THREADS CUB_LOG_WARP_THREADS(0)
#  endif

/// Number of smem banks
#  ifndef CUB_LOG_SMEM_BANKS
//! Deprecated [Since 3.0]
#    define CUB_LOG_SMEM_BANKS(unused) (5)
//! Deprecated [Since 3.0]
#    define CUB_SMEM_BANKS(unused) (1 << CUB_LOG_SMEM_BANKS(0))

//! Deprecated [Since 3.0]
#    define CUB_PTX_LOG_SMEM_BANKS CUB_LOG_SMEM_BANKS(0)
//! Deprecated [Since 3.0]
#    define CUB_PTX_SMEM_BANKS CUB_SMEM_BANKS
#  endif

/// Oversubscription factor
#  ifndef CUB_SUBSCRIPTION_FACTOR
//! Deprecated [Since 3.0]
#    define CUB_SUBSCRIPTION_FACTOR(unused) (5)
//! Deprecated [Since 3.0]
#    define CUB_PTX_SUBSCRIPTION_FACTOR CUB_SUBSCRIPTION_FACTOR(0)
#  endif

/// Prefer padding overhead vs X-way conflicts greater than this threshold
#  ifndef CUB_PREFER_CONFLICT_OVER_PADDING
//! Deprecated [Since 3.0]
#    define CUB_PREFER_CONFLICT_OVER_PADDING(unused) (1)
//! Deprecated [Since 3.0]
#    define CUB_PTX_PREFER_CONFLICT_OVER_PADDING CUB_PREFER_CONFLICT_OVER_PADDING(0)
#  endif

namespace detail
{
inline constexpr int max_devices       = CUB_MAX_DEVICES;
inline constexpr int warp_threads      = CUB_PTX_WARP_THREADS;
inline constexpr int log2_warp_threads = CUB_PTX_LOG_WARP_THREADS;
inline constexpr int smem_banks        = CUB_SMEM_BANKS(0);
inline constexpr int log2_smem_banks   = CUB_PTX_LOG_SMEM_BANKS;

inline constexpr int subscription_factor           = CUB_PTX_SUBSCRIPTION_FACTOR;
inline constexpr bool prefer_conflict_over_padding = CUB_PTX_PREFER_CONFLICT_OVER_PADDING;

// The maximum amount of shared memory available per thread block for eternity. Every current and future CUDA
// architecture has and will have at least this amount of shared memory. This is also the maximum size of total static
// shared memory in a kernel. Note that dynamic shared memory may be larger than this amount.
static constexpr ::cuda::std::size_t max_smem_per_block = 48 * 1024;

struct scaling_result
{
  int items_per_thread;
  int block_threads;
};

[[nodiscard]] _CCCL_API inline constexpr auto
scale_reg_bound(int nominal_4B_block_threads, int nominal_4B_items_per_thread, int target_type_size) -> scaling_result
{
  const int items_per_thread =
    (::cuda::std::max) (1, nominal_4B_items_per_thread * 4 / (::cuda::std::max) (4, target_type_size));
  const int block_threads =
    (::cuda::std::min) (nominal_4B_block_threads,
                        ::cuda::ceil_div(int{max_smem_per_block} / (target_type_size * items_per_thread), 32) * 32);
  return {items_per_thread, block_threads};
}

template <int Nominal4ByteBlockThreads, int Nominal4ByteItemsPerThread, typename T>
struct RegBoundScaling
{
private:
  static constexpr auto result = scale_reg_bound(Nominal4ByteBlockThreads, Nominal4ByteItemsPerThread, int{sizeof(T)});

public:
  static constexpr int ITEMS_PER_THREAD = result.items_per_thread;
  static constexpr int BLOCK_THREADS    = result.block_threads;
};

[[nodiscard]] _CCCL_API inline constexpr auto
scale_mem_bound(int nominal_4B_block_threads, int nominal_4B_items_per_thread, int target_type_size) -> scaling_result
{
  const int items_per_thread =
    ::cuda::std::clamp(nominal_4B_items_per_thread * 4 / target_type_size, 1, nominal_4B_items_per_thread * 2);
  const int block_threads =
    (::cuda::std::min) (nominal_4B_block_threads,
                        ::cuda::round_up(int{max_smem_per_block} / (target_type_size * items_per_thread), 32));
  return {items_per_thread, block_threads};
}

template <int Nominal4ByteBlockThreads, int Nominal4ByteItemsPerThread, typename T>
struct MemBoundScaling
{
private:
  static constexpr auto result = scale_mem_bound(Nominal4ByteBlockThreads, Nominal4ByteItemsPerThread, int{sizeof(T)});

public:
  static constexpr int ITEMS_PER_THREAD = result.items_per_thread;
  static constexpr int BLOCK_THREADS    = result.block_threads;
};

template <int Nominal4ByteBlockThreads, int Nominal4ByteItemsPerThread, typename = void>
struct NoScaling
{
  static constexpr int ITEMS_PER_THREAD = Nominal4ByteItemsPerThread;
  static constexpr int BLOCK_THREADS    = Nominal4ByteBlockThreads;
};
} // namespace detail
#endif // Do not document

CUB_NAMESPACE_END
