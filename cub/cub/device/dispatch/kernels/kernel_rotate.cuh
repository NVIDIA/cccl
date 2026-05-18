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

#include <cub/cub.cuh>
#include <cub/device/dispatch/tuning/tuning_rotate.cuh>

#include <cuda/atomic>
#include <cuda/barrier>
#include <cuda/cmath>
#include <cuda/std/utility>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <cooperative_groups.h>

CUB_NAMESPACE_BEGIN
namespace detail
{
namespace rotate
{
// ============================================================================
// Device-side utility functions
// ============================================================================

__device__ __forceinline__ void bar_sync(int barrier_id, int num_threads)
{
  asm volatile("bar.sync %0, %1;" ::"r"(barrier_id), "r"(num_threads));
}

struct named_barrier_sync
{
  int barrier_id_;
  int num_threads_;
  uint32_t tid_;
  __device__ uint32_t thread_rank() const
  {
    return tid_;
  }
  __device__ void sync()
  {
    bar_sync(barrier_id_, num_threads_);
  }
};

template <typename T, int BLOCK_SIZE, class CG>
__device__ inline void overcopy_memcpy_async(
  T* shmem_dst,
  T* src,
  const int size,
  const int num_elems_past_alignment,
  CG& group,
  cuda::barrier<cuda::thread_scope_block>& bar)
{
  constexpr auto ELEMS_PER_SECTOR = BYTES_PER_SECTOR / sizeof(T);
  assert(reinterpret_cast<uintptr_t>(shmem_dst) % BYTES_PER_SECTOR == 0);
  // Round up to multiple of sector size
  const uint32_t overcopy_tail_elems = cuda::round_up((size + num_elems_past_alignment), ELEMS_PER_SECTOR) - size;

  T* aligned_src = src - num_elems_past_alignment;

  assert(reinterpret_cast<uintptr_t>(aligned_src) % BYTES_PER_SECTOR == 0
         && (size + overcopy_tail_elems) % ELEMS_PER_SECTOR == 0);
  cuda::memcpy_async(
    group,
    shmem_dst,
    aligned_src,
    cuda::aligned_size_t<BYTES_PER_SECTOR>((size + overcopy_tail_elems) * sizeof(T)),
    bar);
}

// Copies a tile from shared memory to global memory through registers.
// Buffers up to MAX_REGS_PER_THREAD uint32_t registers worth of data from shmem
// BEFORE calling sync_op.sync(), then writes to gmem AFTER.  Any remaining
// iterations that do not fit in the register budget are processed post-sync.
// sync_op must provide sync() and thread_rank().
//
// MAX_REGS_PER_THREAD should be set to REGS_PER_SM / (BLOCKS_PER_SM * BLOCK_SIZE)
// so that the register buffering does not reduce occupancy.
template <typename T, int NUM_THREADS, int TILE_BYTES, int MAX_REGS_PER_THREAD, typename SyncOp>
__device__ inline void shared_to_global_through_regs(T* dst, T* src, uint32_t const bytes_to_load, SyncOp& sync_op)
{
  static_assert(TILE_BYTES % (NUM_THREADS * static_cast<int>(sizeof(uint32_t))) == 0);
  constexpr int REGS_PER_T     = TILE_BYTES / (NUM_THREADS * static_cast<int>(sizeof(uint32_t)));
  constexpr int ITERS          = REGS_PER_T / 4; // each uint4 = 4 regs
  constexpr int CHUNK_REGS     = cuda::std::min(MAX_REGS_PER_THREAD, REGS_PER_T);
  constexpr int BUFFERED_ITERS = CHUNK_REGS / 4;
  static_assert(BUFFERED_ITERS >= 1 && BUFFERED_ITERS <= ITERS);

  auto const tid = sync_op.thread_rank();

  if (bytes_to_load < TILE_BYTES)
  {
    uint32_t const elems_to_load = bytes_to_load / sizeof(T);
    sync_op.sync();
    for (uint32_t i = tid; i < elems_to_load; i += NUM_THREADS)
    {
      dst[i] = src[i];
    }
  }
  else
  {
    assert((reinterpret_cast<uintptr_t>(dst) % BYTES_PER_SECTOR) == 0);
    if ((reinterpret_cast<uintptr_t>(src) % sizeof(uint4)) == 0)
    {
      uint4* new_src = reinterpret_cast<uint4*>(src);
      uint4* new_dst = reinterpret_cast<uint4*>(dst);

      uint4 regs[BUFFERED_ITERS];
#pragma unroll
      for (int k = 0; k < BUFFERED_ITERS; ++k)
      {
        regs[k] = new_src[tid + k * NUM_THREADS];
      }
      sync_op.sync();
#pragma unroll
      for (int k = 0; k < BUFFERED_ITERS; ++k)
      {
        new_dst[tid + k * NUM_THREADS] = regs[k];
      }
#pragma unroll
      for (int k = BUFFERED_ITERS; k < ITERS; ++k)
      {
        new_dst[tid + k * NUM_THREADS] = new_src[tid + k * NUM_THREADS];
      }
    }
    else if constexpr (sizeof(T) >= 4)
    {
      uint32_t* new_src = reinterpret_cast<uint32_t*>(src);
      uint4* new_dst    = reinterpret_cast<uint4*>(dst);

      uint32_t const group = (tid >> 3) & 3;

      auto load_swizzled = [&](int k) {
        int const i         = tid + k * NUM_THREADS;
        uint32_t const base = i * 4;
        uint32_t w[4];
#pragma unroll
        for (int r = 0; r < 4; ++r)
        {
          w[r] = new_src[base + ((r + group) & 3)];
        }
        switch (group)
        {
          case 0:
            return make_uint4(w[0], w[1], w[2], w[3]);
          case 1:
            return make_uint4(w[3], w[0], w[1], w[2]);
          case 2:
            return make_uint4(w[2], w[3], w[0], w[1]);
          default:
            return make_uint4(w[1], w[2], w[3], w[0]);
        }
      };

      uint4 regs[BUFFERED_ITERS];
#pragma unroll
      for (int k = 0; k < BUFFERED_ITERS; ++k)
      {
        regs[k] = load_swizzled(k);
      }
      sync_op.sync();
#pragma unroll
      for (int k = 0; k < BUFFERED_ITERS; ++k)
      {
        new_dst[tid + k * NUM_THREADS] = regs[k];
      }
#pragma unroll
      for (int k = BUFFERED_ITERS; k < ITERS; ++k)
      {
        new_dst[tid + k * NUM_THREADS] = load_swizzled(k);
      }
    }
    else
    {
      int const bytes_to_word_end = sizeof(uint32_t) - (reinterpret_cast<uintptr_t>(src) % sizeof(uint32_t));
      uint32_t* new_src =
        reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(src) - (sizeof(uint32_t) - bytes_to_word_end));
      assert(reinterpret_cast<uintptr_t>(new_src) % sizeof(uint32_t) == 0);

      const int shift_len = 8 * (sizeof(uint32_t) - bytes_to_word_end);

      uint4* new_dst = reinterpret_cast<uint4*>(dst);

      uint32_t const group = (tid >> 3) & 3;

      auto load_funnel_shifted = [&](int k) {
        int const i         = tid + k * NUM_THREADS;
        uint32_t const base = i * 4;
        uint32_t w[5];
#pragma unroll
        for (int r = 0; r < 5; ++r)
        {
          int col = r + group;
          w[r]    = new_src[base + (col >= 5 ? col - 5 : col)];
        }
#define FS(a, b) __funnelshift_rc(w[a], w[b], shift_len)
        uint4 val;
        switch (group)
        {
          case 0:
            val = make_uint4(FS(0, 1), FS(1, 2), FS(2, 3), FS(3, 4));
            break;
          case 1:
            val = make_uint4(FS(4, 0), FS(0, 1), FS(1, 2), FS(2, 3));
            break;
          case 2:
            val = make_uint4(FS(3, 4), FS(4, 0), FS(0, 1), FS(1, 2));
            break;
          default:
            val = make_uint4(FS(2, 3), FS(3, 4), FS(4, 0), FS(0, 1));
            break;
        }
#undef FS
        return val;
      };

      uint4 regs[BUFFERED_ITERS];
#pragma unroll
      for (int k = 0; k < BUFFERED_ITERS; ++k)
      {
        regs[k] = load_funnel_shifted(k);
      }
      sync_op.sync();
#pragma unroll
      for (int k = 0; k < BUFFERED_ITERS; ++k)
      {
        new_dst[tid + k * NUM_THREADS] = regs[k];
      }
#pragma unroll
      for (int k = BUFFERED_ITERS; k < ITERS; ++k)
      {
        new_dst[tid + k * NUM_THREADS] = load_funnel_shifted(k);
      }
    }
  }
}

// ============================================================================
// Tile coordinate helpers (host + device)
// ============================================================================

struct Dependencies
{
  int32_t deps_[3];
  uint8_t num_dependencies_;
};

namespace tile_detail
{
template <typename T>
__host__ __device__ uint32_t get_neg_head_size(size_t const arr_size, size_t const rot_dist, uint32_t const head_size)
{
  constexpr auto ELEMS_PER_SECTOR = BYTES_PER_SECTOR / sizeof(T);
  // Sector offset of the negative region's start (= arr + arr_size - rot_dist).
  uint32_t const arr_offset = head_size == 0u ? 0u : (ELEMS_PER_SECTOR - head_size);
  uint32_t const dst_offset = static_cast<uint32_t>((arr_offset + (arr_size - rot_dist)) % ELEMS_PER_SECTOR);
  // 0 when the region's start is already sector-aligned; otherwise the slack to the
  // next sector boundary, in [1, ELEMS_PER_SECTOR - 1].
  return dst_offset == 0u ? 0u : (ELEMS_PER_SECTOR - dst_offset);
}

__host__ __device__ size_t get_overwrite_start(
  size_t const arr_size,
  size_t const rot_dist,
  uint32_t const nominal_tile_size,
  int32_t const tile_ix,
  uint32_t const head_size,
  uint32_t const neg_head_size)
{
  if (tile_ix >= 0)
  {
    return static_cast<size_t>(head_size) + static_cast<size_t>(tile_ix) * nominal_tile_size;
  }
  else
  {
    return (arr_size - rot_dist) + neg_head_size + static_cast<size_t>(-tile_ix - 1) * nominal_tile_size;
  }
}

__host__ __device__ uint32_t get_tile_size(
  size_t const arr_size,
  size_t const rot_dist,
  uint32_t const nominal_tile_size,
  int32_t const tile_ix,
  size_t const start,
  uint32_t const head_size,
  uint32_t const neg_head_size)
{
  if (tile_ix >= 0)
  {
    if (start + nominal_tile_size >= arr_size)
    {
      size_t const main_size   = arr_size - rot_dist - head_size;
      uint32_t const remainder = static_cast<uint32_t>(main_size % nominal_tile_size);
      return remainder == 0u ? nominal_tile_size : remainder;
    }
    else
    {
      return nominal_tile_size;
    }
  }
  else if (start + nominal_tile_size >= rot_dist)
  {
    size_t const main_size   = rot_dist - neg_head_size;
    uint32_t const remainder = static_cast<uint32_t>(main_size % nominal_tile_size);
    return remainder == 0u ? nominal_tile_size : remainder;
  }
  else
  {
    return nominal_tile_size;
  }
}

__host__ __device__ size_t get_tile_start(
  size_t const rot_dist,
  uint32_t const nominal_tile_size,
  int32_t const tile_ix,
  uint32_t const head_size,
  uint32_t const neg_head_size)
{
  if (tile_ix >= 0)
  {
    return rot_dist + static_cast<size_t>(head_size) + static_cast<size_t>(tile_ix) * nominal_tile_size;
  }
  else
  {
    return neg_head_size + static_cast<size_t>(-tile_ix - 1) * nominal_tile_size;
  }
}

__host__ __device__ uint32_t
get_num_negative_tiles(size_t const rot_dist, uint32_t const nominal_tile_size, uint32_t const neg_head_size)
{
  return cuda::ceil_div(rot_dist - neg_head_size, nominal_tile_size);
}

__host__ __device__ uint32_t get_num_positive_tiles(
  size_t const arr_size, size_t const rot_dist, uint32_t const nominal_tile_size, uint32_t const head_size)
{
  return cuda::ceil_div(arr_size - rot_dist - head_size, nominal_tile_size);
}

template <typename T>
__host__ __device__ Dependencies get_dependencies(
  size_t const arr_size,
  size_t const rot_dist,
  uint32_t const nominal_tile_size,
  int32_t const tile_ix,
  uint32_t const /*num_tiles*/,
  uint32_t const head_size)
{
  Dependencies item_deps{{0, 0, 0}, 0};

  uint32_t const neg_head_size = get_neg_head_size<T>(arr_size, rot_dist, head_size);
  uint32_t const num_neg       = get_num_negative_tiles(rot_dist, nominal_tile_size, neg_head_size);

  size_t overwrite_start =
    get_overwrite_start(arr_size, rot_dist, nominal_tile_size, tile_ix, head_size, neg_head_size);
  uint32_t const tile_size = get_tile_size(
    arr_size,
    rot_dist,
    nominal_tile_size,
    tile_ix,
    get_tile_start(rot_dist, nominal_tile_size, tile_ix, head_size, neg_head_size),
    head_size,
    neg_head_size);
  size_t const overwrite_end = overwrite_start + tile_size - 1;
  // Extend the overwrite start to cover the head destination owned by this tile so that
  // any tile whose source overlaps the head dst is picked up as a dependency.
  if (tile_ix == 0 && head_size > 0u)
  {
    overwrite_start = 0;
  }
  else if (tile_ix == -1 && neg_head_size > 0u)
  {
    overwrite_start = arr_size - rot_dist;
  }

  auto snap = [&](size_t pos) -> int32_t {
    if (pos < neg_head_size)
    {
      return -1; // pos head src -> tile -1
    }
    if (pos < rot_dist)
    {
      return -static_cast<int32_t>((pos - neg_head_size) / nominal_tile_size + 1);
    }
    if (pos < rot_dist + head_size)
    {
      return 0; // pos head src -> tile 0
    }
    return static_cast<int32_t>((pos - rot_dist - head_size) / nominal_tile_size);
  };

  int32_t cur        = snap(overwrite_start);
  int32_t const last = snap(overwrite_end);

  auto advance = [&]() {
    if (cur < 0)
    {
      cur--;
      if (static_cast<uint32_t>(cuda::std::abs(cur)) > num_neg)
      {
        cur = 0; // jump from -num_neg past pos head to BFS pos 0
      }
    }
    else
    {
      cur++;
    }
  };
  auto should_continue = [&]() {
    return ((cuda::std::abs(cur) <= cuda::std::abs(last)) || ((cur < 0) && (last >= 0))) && !(last < 0 && cur >= 0);
  };

  item_deps.deps_[0]          = cur;
  item_deps.num_dependencies_ = 1;
  advance();
  if (should_continue())
  {
    item_deps.deps_[1]          = cur;
    item_deps.num_dependencies_ = 2;
    advance();
    if (should_continue())
    {
      item_deps.deps_[2]          = cur;
      item_deps.num_dependencies_ = 3;
    }
  }
  return item_deps;
}

__host__ __device__ int32_t arr_ix_to_tile_ix(size_t const arr_ix, uint32_t const num_negative_tiles)
{
  return arr_ix < num_negative_tiles
         ? -static_cast<int32_t>(arr_ix + 1)
         : (static_cast<int32_t>(arr_ix - num_negative_tiles));
};

__host__ __device__ size_t tile_ix_to_arr_ix(int32_t const tile_ix, uint32_t const num_negative_tiles)
{
  return tile_ix < 0 ? cuda::std::abs(tile_ix + 1) : static_cast<size_t>(tile_ix + num_negative_tiles);
};
} // namespace tile_detail

// ============================================================================
// Short-distance rotate kernel
// ============================================================================

namespace rotate_short
{
template <typename T>
int get_shmem_usage(cudaStream_t stream)
{
  int device;
  cudaStreamGetDevice(stream, &device);

  int shmem_per_sm;
  cudaDeviceGetAttribute(&shmem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);

  const auto pipeline_depth = get_pipeline_depth(TILE_BYTES, shmem_per_sm);
  return pipeline_depth * (TILE_BYTES + BYTES_PER_SECTOR);
}

__global__ void setup_kernel(void* d_temp_storage, size_t const temp_bytes, size_t const num_sms)
{
  auto* flags       = reinterpret_cast<device_flag_t*>(d_temp_storage);
  auto const tid    = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  for (int i = tid; i < num_sms; i += stride)
  {
    // Initialize as blocked
    flags[i].store(0, cuda::memory_order_relaxed);
  }
}

template <typename T>
CUB_ROTATE_LB(BLOCK_SIZE, BLOCKS_PER_SM)
__global__ void rotate_short_kernel(
  T* arr,
  size_t const size,
  void* d_temp_storage,
  size_t const temp_size,
  size_t const rotate_distance,
  size_t const num_tiles,
  uint32_t const head_size)
{
  assert(blockDim.x == BLOCK_SIZE);
  constexpr int TILE_SIZE         = TILE_BYTES / sizeof(T);
  constexpr auto ELEMS_PER_SECTOR = BYTES_PER_SECTOR / sizeof(T);

  assert(rotate_distance <= TILE_SIZE);
  assert(head_size < ELEMS_PER_SECTOR);

  constexpr int P = get_pipeline_depth(TILE_BYTES, SHMEM_PER_SM);

  alignas(BYTES_PER_SECTOR) extern __shared__ unsigned char smem_raw[];
  T(*cache)[TILE_SIZE + ELEMS_PER_SECTOR] = reinterpret_cast<T(*)[TILE_SIZE + ELEMS_PER_SECTOR]>(smem_raw);

  // Space in shared memory for the first chunk's head tile used for alignment of the others
  __shared__ T head_tile[ELEMS_PER_SECTOR];

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> bars[P];

  __shared__ cuda::atomic<uint32_t, cuda::thread_scope_block> num_loaded;
  __shared__ cuda::atomic<uint32_t, cuda::thread_scope_block> num_consumed;

  auto* flags = reinterpret_cast<device_flag_t*>(d_temp_storage);

  auto const tid         = threadIdx.x;
  bool const is_producer = tid < WS;

  if (tid < P)
  {
    init(&bars[tid], NUM_CONSUMER_THREADS);
  }
  if (tid == 0)
  {
    num_loaded   = 0;
    num_consumed = 0;
  }
  __syncthreads();

  assert(2 * rotate_distance <= size);
  assert(size > TILE_SIZE);

  const auto nominal_tiles_per_chunk = num_tiles / gridDim.x;
  const uint32_t tiles_to_process =
    blockIdx.x == gridDim.x - 1 ? nominal_tiles_per_chunk + num_tiles % gridDim.x : nominal_tiles_per_chunk;
  assert(tiles_to_process > 0 || nominal_tiles_per_chunk == 0);

  // Blocks with no tiles to process must still set their flag
  // to prevent deadlocks in subsequent blocks waiting on them
  if (tiles_to_process == 0)
  {
    if (tid == 0)
    {
      flags[blockIdx.x].store(1, cuda::memory_order_release);
      flags[blockIdx.x].notify_all();
    }
    return;
  }

  const auto chunk_start_ix = rotate_distance + head_size + blockIdx.x * nominal_tiles_per_chunk * TILE_SIZE;
  uint32_t const overcopy_extra_head_elems = static_cast<uint32_t>(rotate_distance % ELEMS_PER_SECTOR);
  bool const is_first_chunk                = chunk_start_ix == rotate_distance + head_size;

  auto get_tile_load_index = [&](uint32_t tile_num) -> size_t {
    // We load from the back forwards
    return chunk_start_ix + (tiles_to_process - 1 - tile_num) * TILE_SIZE;
  };

  auto get_bytes_to_load = [&](uint32_t tile_num) -> uint32_t {
    bool const is_last_chunks_first_tile = blockIdx.x == gridDim.x - 1 && tile_num == 0;
    if (!is_last_chunks_first_tile)
    {
      return TILE_SIZE * sizeof(T);
    }
    uint32_t const remainder = (size - rotate_distance - head_size) % TILE_SIZE;
    return (remainder == 0u ? TILE_SIZE : remainder) * sizeof(T);
  };

  // ========================= PRODUCER (warp 0) =========================
  if (is_producer)
  {
    auto warp = cooperative_groups::tiled_partition<WS>(cooperative_groups::this_thread_block());

    auto producer_load = [&](uint32_t tile_num) {
      int const slot       = tile_num % P;
      T* src               = arr + get_tile_load_index(tile_num);
      uint32_t const bytes = get_bytes_to_load(tile_num);
      while (tile_num >= num_consumed.load(cuda::memory_order_acquire) + P)
      {
        __nanosleep(0);
      }
      if constexpr (hasTMA())
      {
        cooperative_groups::invoke_one(warp, [&] {
          overcopy_memcpy_async<T, WS>(cache[slot], src, bytes / sizeof(T), overcopy_extra_head_elems, warp, bars[slot]);
        });
      }
      else
      {
        overcopy_memcpy_async<T, WS>(cache[slot], src, bytes / sizeof(T), overcopy_extra_head_elems, warp, bars[slot]);
      }
      if (tid == 0)
      {
        num_loaded++;
      }
    };

    // Load head tile
    if (is_first_chunk)
    {
      for (uint32_t i = tid; i < head_size; i += WS)
      {
        head_tile[i] = arr[rotate_distance + i];
      }
    }

    // Main loop
    for (int i = 0; i < static_cast<int>(tiles_to_process); i++)
    {
      producer_load(i);
    }
  }
  // ======================== CONSUMERS (warps 1+) ========================
  else
  {
    uint32_t const consumer_tid = tid - WS;

    // wait on the first tile and set global flag
    if (consumer_tid == 0)
    {
      while (num_loaded.load(cuda::memory_order_acquire) == 0)
      {
        __nanosleep(0);
      }
    }
    bars[0].arrive_and_wait();
    if (consumer_tid == 0)
    {
      flags[blockIdx.x].store(1, cuda::memory_order_release);
      flags[blockIdx.x].notify_all();
    }

    for (int i = 0; i < static_cast<int>(tiles_to_process); i++)
    {
      int const slot = i % P;

      // Handle inter-block dependencies on the last tile
      if (i == static_cast<int>(tiles_to_process) - 1)
      {
        if (is_first_chunk)
        {
          assert(blockIdx.x == 0 || blockIdx.x == gridDim.x - 1);
          if (consumer_tid == 0)
          {
            flags[gridDim.x - 1].wait(0, cuda::memory_order_acquire);
            flags[gridDim.x - 2].wait(0, cuda::memory_order_acquire);
          }
          bar_sync(1, NUM_CONSUMER_THREADS);
          for (uint32_t j = consumer_tid; j < rotate_distance; j += NUM_CONSUMER_THREADS)
          {
            arr[size - rotate_distance + j] = arr[j];
          }
          bar_sync(1, NUM_CONSUMER_THREADS);
        }
        else
        {
          if (consumer_tid == 0)
          {
            flags[blockIdx.x - 1].wait(0, cuda::memory_order_acquire);
          }
          bar_sync(1, NUM_CONSUMER_THREADS);
        }
      }

      // Store tile from shmem to gmem
      auto const load_index = get_tile_load_index(i);
      T* src                = cache[slot] + overcopy_extra_head_elems;
      T* dst                = arr + load_index - rotate_distance;
      assert(load_index > rotate_distance || head_size == 0);

      // wait on next tile to finish loading (all consumers participate)
      if (i < static_cast<int>(tiles_to_process) - 1)
      {
        if (consumer_tid == 0)
        {
          while (num_loaded.load(cuda::memory_order_acquire) <= static_cast<uint32_t>(i + 1))
          {
            __nanosleep(0);
          }
        }
        bars[(slot + 1) % P].arrive_and_wait();
      }
      named_barrier_sync nbs{1, NUM_CONSUMER_THREADS, consumer_tid};
      constexpr int MAX_REGS = REGS_PER_SM / (BLOCKS_PER_SM * BLOCK_SIZE);
      shared_to_global_through_regs<T, NUM_CONSUMER_THREADS, TILE_BYTES, MAX_REGS>(dst, src, get_bytes_to_load(i), nbs);
      bar_sync(1, NUM_CONSUMER_THREADS);
      if (consumer_tid == 0)
      {
        num_consumed++;
      }
    }
  }

  __syncthreads();

  if (is_first_chunk && tid < WS)
  {
    for (uint32_t i = tid; i < head_size; i += WS)
    {
      arr[i] = head_tile[i];
    }
  }
}
} // namespace rotate_short

// ============================================================================
// Short-distance rotate kernel (no pipelining)
// ============================================================================

namespace rotate_short_no_pipeline
{
__global__ void setup_kernel(void* d_temp_storage, size_t const temp_bytes, size_t const num_tiles)
{
  auto* counter     = reinterpret_cast<int*>(d_temp_storage);
  auto const tid    = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  if (tid == 0)
  {
    counter[0] = num_tiles - 1;
  }
  auto* flags = reinterpret_cast<device_flag_t*>(counter + 1);
  for (int i = tid; i < num_tiles; i += stride)
  {
    flags[i].store(0, cuda::memory_order_relaxed);
  }
}

template <typename T>
__global__ void rotate_tiny_kernel(T* arr, size_t const size, size_t const rotate_distance)
{
  extern __shared__ unsigned char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);

  assert(2 * rotate_distance > size || size <= TILE_BYTES / sizeof(T));
  assert(rotate_distance > 0 && rotate_distance < size);

  if (blockIdx.x == 0)
  {
    for (size_t i = threadIdx.x; i < size; i += blockDim.x)
    {
      smem[i] = arr[i];
    }
    __syncthreads();

    size_t const tail = size - rotate_distance;
    for (size_t i = threadIdx.x; i < tail; i += blockDim.x)
    {
      arr[i] = smem[i + rotate_distance];
    }
    for (size_t i = threadIdx.x; i < rotate_distance; i += blockDim.x)
    {
      arr[tail + i] = smem[i];
    }
  }
}

template <typename T>
CUB_ROTATE_LB(BLOCK_SIZE, BLOCKS_PER_SM)
__global__ void rotate_short_kernel_no_pipeline(
  T* arr,
  size_t const size,
  void* d_temp_storage,
  size_t const temp_size,
  size_t const rotate_distance,
  size_t const num_tiles,
  uint32_t const head_size)
{
  assert(blockDim.x == BLOCK_SIZE);
  constexpr int TILE_SIZE         = TILE_BYTES / sizeof(T);
  constexpr auto ELEMS_PER_SECTOR = BYTES_PER_SECTOR / sizeof(T);

  assert(rotate_distance <= TILE_SIZE);
  assert(head_size < ELEMS_PER_SECTOR);
  assert(2 * rotate_distance <= size);
  assert(size > TILE_SIZE);

  alignas(BYTES_PER_SECTOR) __shared__ T cache[TILE_SIZE + ELEMS_PER_SECTOR];
  __shared__ int tile_ix;
  __shared__ T head_tile_cache[ELEMS_PER_SECTOR];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;

  auto* tile_counter = reinterpret_cast<cuda::atomic<int, cuda::thread_scope_device>*>(d_temp_storage);
  auto* flags        = reinterpret_cast<device_flag_t*>(reinterpret_cast<int*>(d_temp_storage) + 1);

  auto const tid = threadIdx.x;
  auto cta       = cooperative_groups::this_thread_block();

  if (tid == 0)
  {
    init(&bar, BLOCK_SIZE);
  }

  uint32_t const overcopy_extra_head_elems = static_cast<uint32_t>(rotate_distance % ELEMS_PER_SECTOR);

  bool first_iteration = true;
  while (true)
  {
    if (tid == 0)
    {
      tile_ix = tile_counter->fetch_sub(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    int const curr_tile = tile_ix;
    if (curr_tile < 0)
    {
      if (!first_iteration)
      {
        bar.arrive_and_wait();
      }
      return;
    }

    bool const is_first_tile = static_cast<size_t>(curr_tile) == num_tiles - 1;
    uint32_t bytes_to_load   = TILE_SIZE * sizeof(T);
    if (is_first_tile)
    {
      uint32_t const remainder = (size - rotate_distance - head_size) % TILE_SIZE;
      bytes_to_load            = (remainder == 0u ? TILE_SIZE : remainder) * sizeof(T);
    }

    size_t const load_index = rotate_distance + head_size + static_cast<size_t>(curr_tile) * TILE_SIZE;
    T* src                  = arr + load_index;

    if (!first_iteration)
    {
      bar.arrive_and_wait();
    }
    else
    {
      first_iteration = false;
    }
    overcopy_memcpy_async<T, BLOCK_SIZE>(cache, src, bytes_to_load / sizeof(T), overcopy_extra_head_elems, cta, bar);
    bar.arrive_and_wait();

    bool const is_last_tile = curr_tile == 0;
    if (tid == 0)
    {
      flags[curr_tile].store(1, cuda::memory_order_release);
      flags[curr_tile].notify_all();
      auto const ix = is_last_tile ? num_tiles - 1 : curr_tile - 1;
      flags[ix].wait(0, cuda::memory_order_acquire);
    }

    if (is_last_tile)
    {
      // Save head data before it gets overwritten by our tile store
      if (head_size > 0u)
      {
        for (uint32_t i = tid; i < head_size; i += BLOCK_SIZE)
        {
          head_tile_cache[i] = arr[rotate_distance + i];
        }
      }

      // Copy first rotate_distance elements to end of array
      flags[num_tiles - 1].wait(0, cuda::memory_order_acquire);
      __syncthreads();
      for (size_t i = tid; i < rotate_distance; i += BLOCK_SIZE)
      {
        arr[size - rotate_distance + i] = arr[i];
      }
      __syncthreads();
    }

    T* dst                 = arr + load_index - rotate_distance;
    constexpr int MAX_REGS = REGS_PER_SM / (BLOCKS_PER_SM * BLOCK_SIZE);
    shared_to_global_through_regs<T, BLOCK_SIZE, TILE_BYTES, MAX_REGS>(
      dst, cache + overcopy_extra_head_elems, bytes_to_load, cta);

    // Tile 0 owns the head: write it after storing the main tile
    if (is_last_tile && head_size > 0u)
    {
      __syncthreads();
      for (uint32_t i = tid; i < head_size; i += BLOCK_SIZE)
      {
        arr[i] = head_tile_cache[i];
      }
    }
  }
}
} // namespace rotate_short_no_pipeline

// ============================================================================
// Long-distance rotate kernel
// ============================================================================

namespace rotate_long
{
__global__ void setup_kernel(void* d_temp_storage, size_t const temp_bytes, size_t const num_tiles)
{
  auto* counter     = reinterpret_cast<uint32_t*>(d_temp_storage);
  auto const tid    = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  if (tid == 0)
  {
    counter[0] = 0;
  }
  auto* flags = reinterpret_cast<device_flag_t*>(counter + 1 + num_tiles); // skip ordering array
  for (int i = tid; i < num_tiles; i += stride)
  {
    // Initialize as blocked
    flags[i].store(0, cuda::memory_order_relaxed);
  }
}

template <typename T>
CUB_ROTATE_LB(BLOCK_SIZE, BLOCKS_PER_SM)
__global__ void rotate_long_kernel(
  T* arr,
  size_t const size,
  void* d_temp_storage,
  size_t const temp_size,
  size_t const rotate_distance,
  size_t const num_tiles,
  uint32_t const head_size)
{
  assert(blockDim.x == BLOCK_SIZE);
  constexpr int TILE_SIZE         = TILE_BYTES / sizeof(T);
  constexpr auto ELEMS_PER_SECTOR = BYTES_PER_SECTOR / sizeof(T);
  alignas(BYTES_PER_SECTOR) __shared__ T cache[TILE_SIZE + ELEMS_PER_SECTOR]; // +ELEMS_PER_SECTOR to avoid shmem bank
                                                                              // conflicts when writing back to global
                                                                              // memory
  __shared__ int tile_ordering_ix;
  // Tile -1 owns the negative head used for sector alignment of the negative tiles destinations.
  // Tile 0 owns the positive head used for sector alignment of the positive tiles destinations.
  __shared__ T head_tile_cache[ELEMS_PER_SECTOR];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;

  auto* tiles_processed  = reinterpret_cast<cuda::atomic<uint32_t, cuda::thread_scope_device>*>(d_temp_storage);
  auto* processing_order = reinterpret_cast<int32_t*>(tiles_processed + 1); // first uint32_t is the counter of
                                                                            // processed items
  auto* flags = reinterpret_cast<device_flag_t*>(processing_order + num_tiles);

  uint32_t const neg_head_size = tile_detail::get_neg_head_size<T>(size, rotate_distance, head_size);

  auto const tid = threadIdx.x;
  auto cta       = cooperative_groups::this_thread_block();

  if (tid == 0)
  {
    init(&bar, BLOCK_SIZE);
  }
  while (true)
  {
    if (tid == 0)
    {
      tile_ordering_ix = tiles_processed->fetch_add(1, cuda::memory_order_seq_cst);
    }
    __syncthreads();
    auto const curr_tile_ordering_ix = tile_ordering_ix;
    if (curr_tile_ordering_ix >= num_tiles)
    {
      return;
    }
    auto const curr_tile = processing_order[curr_tile_ordering_ix];
    auto const curr_tile_start =
      tile_detail::get_tile_start(rotate_distance, TILE_SIZE, curr_tile, head_size, neg_head_size);
    auto const curr_tile_size = tile_detail::get_tile_size(
      size, rotate_distance, TILE_SIZE, curr_tile, curr_tile_start, head_size, neg_head_size);
    assert(curr_tile_size <= TILE_SIZE);

    bool const owns_pos_head      = (curr_tile == 0) && (head_size > 0u);
    bool const owns_neg_head      = (curr_tile == -1) && (neg_head_size > 0u);
    uint32_t const head_load_size = owns_pos_head ? head_size : (owns_neg_head ? neg_head_size : 0u);
    size_t const head_src_off     = owns_pos_head ? rotate_distance : 0;
    size_t const head_dst_off     = owns_pos_head ? 0 : (size - rotate_distance);

    if (head_load_size > 0u && tid < WS)
    {
      for (uint32_t i = tid; i < head_load_size; i += WS)
      {
        head_tile_cache[i] = arr[head_src_off + i];
      }
    }

    // Regular tile load via the standard memcpy_async pipeline.
    T* src                         = arr + curr_tile_start;
    uint32_t const unaligned_elems = (reinterpret_cast<uintptr_t>(src) % BYTES_PER_SECTOR) / sizeof(T);

    overcopy_memcpy_async<T, BLOCK_SIZE>(cache, src, curr_tile_size, unaligned_elems, cta, bar);
    bar.arrive_and_wait();

    if (tid == 0)
    {
      auto const num_negative_tiles = tile_detail::get_num_negative_tiles(rotate_distance, TILE_SIZE, neg_head_size);
      flags[tile_detail::tile_ix_to_arr_ix(curr_tile, num_negative_tiles)].store(1, cuda::memory_order_release);
      flags[tile_detail::tile_ix_to_arr_ix(curr_tile, num_negative_tiles)].notify_all();
      // Poll dependencies
      auto const deps =
        tile_detail::get_dependencies<T>(size, rotate_distance, TILE_SIZE, curr_tile, num_tiles, head_size);
      // static loop to keep deps in registers
      for (int i = 0; i < 3; ++i)
      {
        if (i < deps.num_dependencies_)
        {
          flags[tile_detail::tile_ix_to_arr_ix(deps.deps_[i], num_negative_tiles)].wait(0, cuda::memory_order_acquire);
        }
      }
    }

    // Copy cached tile to destination
    T* dst = arr
           + tile_detail::get_overwrite_start(size, rotate_distance, TILE_SIZE, curr_tile, head_size, neg_head_size);
    src                    = cache + unaligned_elems;
    constexpr int MAX_REGS = REGS_PER_SM / (BLOCKS_PER_SM * BLOCK_SIZE);
    shared_to_global_through_regs<T, BLOCK_SIZE, TILE_BYTES, MAX_REGS>(dst, src, curr_tile_size * sizeof(T), cta);

    if (head_load_size > 0u && tid < WS)
    {
      for (uint32_t i = tid; i < head_load_size; i += WS)
      {
        arr[head_dst_off + i] = head_tile_cache[i];
      }
    }
  }
}
} // namespace rotate_long
} // namespace rotate
} // namespace detail
CUB_NAMESPACE_END
