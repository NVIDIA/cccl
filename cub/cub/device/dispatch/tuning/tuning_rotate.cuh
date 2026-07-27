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

struct short_algorithm
{
  static constexpr int tile_bytes = 18 * 1024;

  // How many contiguous tiles a CTA grabs at once.
  static constexpr int tiles_per_grab = 6;

  // Number of shared-memory tile buffers per block.
  static constexpr int pipeline_stages = 2;

  static constexpr auto launch_bounds = get_launch_bounds(tile_bytes * pipeline_stages, SHMEM_PER_SM, MAX_TPSM);
  static constexpr int block_size     = launch_bounds.first;
  static constexpr int blocks_per_sm  = launch_bounds.second;

  // Limit the number of registers per thread when copying from shared to global.
  static constexpr int max_regs_per_thread_override = 4;
};

struct long_algorithm
{
  static constexpr int tile_bytes = 32 * 1024;

  // Keep dependency chains below the cooperative grid size by this many CTAs.
  static constexpr int cooperative_grid_safety_margin = 50;

  // Use a closed-form processing order while either side fits within this many tiles.
  static constexpr int max_direct_dependency_distance = 256;

  static constexpr size_t naive_fallback_min_size     = 1'000'000'000;
  static constexpr double naive_fallback_max_fraction = 0.3;
  static constexpr size_t naive_fallback_min_distance_bytes = 500'000'000;

  static constexpr int pipeline_stages = 2;

  static constexpr auto launch_bounds = get_launch_bounds(tile_bytes * pipeline_stages, SHMEM_PER_SM, MAX_TPSM);
  static constexpr int block_size     = launch_bounds.first;
  static constexpr int blocks_per_sm  = launch_bounds.second;

  // Cap the shared->global store register buffer at the BUFFERED_ITERS=1 floor (= min(4, REGS_PER_T)/4)
  // so the store loop interleaves each per-uint4 shmem load with its write-through gmem store (higher
  // store-side memory-level parallelism / DRAM utilization) instead of buffering the whole tile in
  // registers before storing. Swept 4/8/16 under the pipeline -- 4 is the tuned winner.
  static constexpr int max_regs_per_thread_override = 4;
};
} // namespace rotate
} // namespace detail
CUB_NAMESPACE_END
