// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/atomic>
#include <cuda/cmath>

#include <algorithm>

CUB_NAMESPACE_BEGIN
namespace detail
{
namespace rotate
{
constexpr int WS               = 32; // warp size
constexpr int BYTES_PER_SECTOR = 32; // cache sector size
constexpr int REGS_PER_SM      = 65536;

using device_flag_t = cuda::atomic<int, cuda::thread_scope_device>;

#ifdef __CUDA_ARCH__
#  if (__CUDA_ARCH__ > 800 && __CUDA_ARCH__ < 900) || __CUDA_ARCH__ >= 1100
constexpr size_t MAX_TPSM = 1536;
#  elif __CUDA_ARCH__ == 750
constexpr size_t MAX_TPSM = 1024;
#  else
constexpr size_t MAX_TPSM = 2048;
#  endif

#  if __CUDA_ARCH__ == 750
constexpr int SHMEM_PER_SM = 64 * 1024;
#  elif __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 870
constexpr int SHMEM_PER_SM = 164 * 1024;
#  elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ >= 1200
constexpr int SHMEM_PER_SM = 100 * 1024;
#  elif __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ <= 1100
constexpr int SHMEM_PER_SM = 228 * 1024;
#  else
#    error "Unknown device architecture, please define SHMEM_PER_SM for __CUDA_ARCH__"
#  endif

#  define CUB_ROTATE_LB(x, y) __launch_bounds__(x, y)
#else
constexpr int SHMEM_PER_SM = 1;
constexpr int MAX_TPSM     = 1;
#  define CUB_ROTATE_LB(x, y)
#endif // __CUDA_ARCH__

__device__ constexpr bool hasTMA()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  return true;
#else
  return false;
#endif
}

constexpr int get_pipeline_depth(const int tile_bytes, const int shmem_per_sm)
{
  const int depth = shmem_per_sm / (tile_bytes + 1'000); // extra bytes to ensure no spilling
  return depth > 0 ? depth : 1;
}

constexpr std::pair<int, int> get_launch_bounds(const int tile_bytes, const int shmem_per_sm, const int max_tpsm)
{
  const int BLOCKS_PER_SM = shmem_per_sm / (tile_bytes + 1'000); // extra bytes to ensure no spilling
  if (BLOCKS_PER_SM <= 0)
  {
    return {1, 1}; // shmem too small (e.g. host-side fallback); return safe defaults
  }

  const int BLOCK_SIZE = cuda::prev_power_of_two(max_tpsm / BLOCKS_PER_SM);
  return {BLOCK_SIZE, BLOCKS_PER_SM};
}

constexpr bool USE_SHORT_PIPELINE = false;

namespace rotate_short
{
constexpr int TILE_BYTES                  = 32 * 1024;
inline constexpr int NUM_CONSUMER_THREADS = 512;
inline constexpr int BLOCK_SIZE           = NUM_CONSUMER_THREADS + WS;
inline constexpr int BLOCKS_PER_SM        = 1;
} // namespace rotate_short

namespace rotate_short_no_pipeline
{
constexpr int TILE_BYTES            = 32 * 1024;
constexpr auto LAUNCH_BOUNDS        = get_launch_bounds(TILE_BYTES, SHMEM_PER_SM, MAX_TPSM);
inline constexpr auto BLOCK_SIZE    = LAUNCH_BOUNDS.first;
inline constexpr auto BLOCKS_PER_SM = LAUNCH_BOUNDS.second;
} // namespace rotate_short_no_pipeline

namespace rotate_long
{
constexpr int TILE_BYTES            = 32 * 1024;
constexpr auto LAUNCH_BOUNDS        = get_launch_bounds(TILE_BYTES, SHMEM_PER_SM, MAX_TPSM);
inline constexpr auto BLOCK_SIZE    = LAUNCH_BOUNDS.first;
inline constexpr auto BLOCKS_PER_SM = LAUNCH_BOUNDS.second;
} // namespace rotate_long
} // namespace rotate
} // namespace detail
CUB_NAMESPACE_END
