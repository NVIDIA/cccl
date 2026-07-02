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
// BEFORE calling sync_op.sync(), to maximally overlap the atomic polling with other work, then writes to gmem AFTER.  Any remaining
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

  constexpr int VEC_TILE_BYTES = ITERS * 4 * NUM_THREADS * static_cast<int>(sizeof(uint32_t));

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
        // Write-through store to not pollute L2
        cub::ThreadStore<cub::STORE_WT>(new_dst + tid + k * NUM_THREADS, regs[k]);
      }
#pragma unroll
      for (int k = BUFFERED_ITERS; k < ITERS; ++k)
      {
        cub::ThreadStore<cub::STORE_WT>(new_dst + tid + k * NUM_THREADS, new_src[tid + k * NUM_THREADS]);
      }
    }
    else if ((reinterpret_cast<uintptr_t>(src) % sizeof(uint32_t)) == 0)
    {
      uint32_t* new_src = reinterpret_cast<uint32_t*>(src);
      uint4* new_dst    = reinterpret_cast<uint4*>(dst);

      // s = words `src` is above the 16B boundary; in {1,2,3} in this branch.
      uint32_t const s         = (static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src)) % sizeof(uint4)) / sizeof(uint32_t);
      uint4 const* aligned_src = reinterpret_cast<uint4 const*>(new_src - s);
      bool const is_top_lane      = (tid % WS) == (WS - 1);

      auto load_funnel = [&](int k) -> uint4 {
        int const j   = tid + k * NUM_THREADS;
        uint4 const A = aligned_src[j];
        // Boundary words = the first `s` words of B = aligned_src[j+1] = the neighbor lane's A.
        // Shuffle them from lane+1; the top lane has no in-warp neighbor, so load B directly.
        uint32_t bx = __shfl_down_sync(0xffffffffu, A.x, 1);
        uint32_t by = (s >= 2u) ? __shfl_down_sync(0xffffffffu, A.y, 1) : 0u;
        uint32_t bz = (s >= 3u) ? __shfl_down_sync(0xffffffffu, A.z, 1) : 0u;
        if (is_top_lane)
        {
          uint4 const B = aligned_src[j + 1];
          bx            = B.x;
          by            = B.y;
          bz            = B.z;
        }
        switch (s)
        {
          case 1:
            return make_uint4(A.y, A.z, A.w, bx);
          case 2:
            return make_uint4(A.z, A.w, bx, by);
          default: // s == 3
            return make_uint4(A.w, bx, by, bz);
        }
      };

      uint4 regs[BUFFERED_ITERS];
#pragma unroll
      for (int k = 0; k < BUFFERED_ITERS; ++k)
      {
        regs[k] = load_funnel(k);
      }
      sync_op.sync();
#pragma unroll
      for (int k = 0; k < BUFFERED_ITERS; ++k)
      {
        cub::ThreadStore<cub::STORE_WT>(new_dst + tid + k * NUM_THREADS, regs[k]);
      }
#pragma unroll
      for (int k = BUFFERED_ITERS; k < ITERS; ++k)
      {
        cub::ThreadStore<cub::STORE_WT>(new_dst + tid + k * NUM_THREADS, load_funnel(k));
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
        cub::ThreadStore<cub::STORE_WT>(new_dst + tid + k * NUM_THREADS, regs[k]);
      }
#pragma unroll
      for (int k = BUFFERED_ITERS; k < ITERS; ++k)
      {
        cub::ThreadStore<cub::STORE_WT>(new_dst + tid + k * NUM_THREADS, load_funnel_shifted(k));
      }
    }

    // TODO: see if adapting the tile size to remove the tail improves perf
    if constexpr (VEC_TILE_BYTES < TILE_BYTES)
    {
      // The uncovered tail [VEC_TILE_BYTES, TILE_BYTES) is a contiguous src->dst block copy: the
      // rotation lives entirely in dst's gmem offset, so a plain element copy is byte-identical for
      // all three realignment branches above.
      uint32_t const elems_to_load  = bytes_to_load / sizeof(T);
      constexpr uint32_t tail_begin = VEC_TILE_BYTES / static_cast<int>(sizeof(T));
      for (uint32_t i = tail_begin + tid; i < elems_to_load; i += NUM_THREADS)
      {
        dst[i] = src[i];
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
// Short-distance rotate kernel (no pipelining)
// ============================================================================

namespace rotate_short
{
// Dynamic shared-memory bytes for the multi-stage short kernel: PIPELINE_STAGES tile
// buffers, each TILE_BYTES + BYTES_PER_SECTOR (overcopy / bank-conflict slack).
template <typename T>
constexpr int get_shmem_usage()
{
  constexpr auto ELEMS_PER_SECTOR = BYTES_PER_SECTOR / static_cast<int>(sizeof(T));
  constexpr int TILE_SIZE         = TILE_BYTES / static_cast<int>(sizeof(T));
  // Must match SLOT_BYTES inside the kernel (each slot rounded up to 128B for TMA alignment).
  constexpr int SLOT_BYTES = cuda::round_up((TILE_SIZE + ELEMS_PER_SECTOR) * static_cast<int>(sizeof(T)), 128);
  return PIPELINE_STAGES * SLOT_BYTES;
}

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
  constexpr int P                 = PIPELINE_STAGES;

  assert(rotate_distance <= TILE_SIZE);
  assert(head_size < ELEMS_PER_SECTOR);
  assert(2 * rotate_distance <= size);
  assert(size > TILE_SIZE);

  constexpr int SLOT_BYTES = cuda::round_up((TILE_SIZE + ELEMS_PER_SECTOR) * static_cast<int>(sizeof(T)), BYTES_PER_SECTOR);
  constexpr int SLOT_ELEMS = SLOT_BYTES / static_cast<int>(sizeof(T));
  alignas(BYTES_PER_SECTOR) extern __shared__ unsigned char smem_raw[];
  T(*cache)[SLOT_ELEMS] = reinterpret_cast<T(*)[SLOT_ELEMS]>(smem_raw);

  constexpr int B = TILES_PER_GRAB;

  __shared__ int tile_ix_buf[P];
  __shared__ bool is_top_buf[P]; // tile is the highest of its contiguous grab-run
  __shared__ bool is_bot_buf[P]; // tile is the lowest of its contiguous grab-run
  __shared__ int batch_next; // next (descending) tile to issue within the current run
  __shared__ int batch_lo; // lowest tile index in the current run
  __shared__ T head_tile_cache[ELEMS_PER_SECTOR];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> bars[P];

  auto* tile_counter = reinterpret_cast<cuda::atomic<int, cuda::thread_scope_device>*>(d_temp_storage);
  auto* flags        = reinterpret_cast<device_flag_t*>(reinterpret_cast<int*>(d_temp_storage) + 1);

  auto cta       = cooperative_groups::this_thread_block();
  auto const tid = cta.thread_rank();

  if (tid < P)
  {
    init(&bars[tid], BLOCK_SIZE);
  }
  if (tid == 0)
  {
    batch_next = -1; // no run claimed yet -> force a fetch on the first grab
    batch_lo   = 0;
  }
  __syncthreads();

  uint32_t const overcopy_extra_head_elems = static_cast<uint32_t>(rotate_distance % ELEMS_PER_SECTOR);
  constexpr int MAX_REGS_OCCUPANCY         = REGS_PER_SM / (BLOCKS_PER_SM * BLOCK_SIZE);
  constexpr int MAX_REGS = MAX_REGS_PER_THREAD_OVERRIDE > 0 ? MAX_REGS_PER_THREAD_OVERRIDE : MAX_REGS_OCCUPANCY;

  auto bytes_to_load_for = [&](int curr_tile) -> uint32_t {
    bool const is_first_tile = static_cast<size_t>(curr_tile) == num_tiles - 1;
    if (!is_first_tile)
    {
      return TILE_SIZE * sizeof(T);
    }
    uint32_t const remainder = (size - rotate_distance - head_size) % TILE_SIZE;
    return (remainder == 0u ? TILE_SIZE : remainder) * sizeof(T);
  };

  // Claim the next tile (descending) from this CTA's current contiguous run, refilling the run
  // from the global counter via a single fetch_sub(B) when it is exhausted, issue the tile's
  // async load into slot `slot`, and record the tile index plus whether it is the top/bottom
  // of its run.  Returns the grabbed tile index (negative => no more work).
  auto grab_and_load = [&](int slot) -> int {
    if (tid == 0)
    {
      bool new_run = false;
      if (batch_next < batch_lo)
      {
        // Claim a new contiguous run, top tile first (descending), exactly as the original
        // descending counter did -- but the global counter counts UP from 0 (fetch_add) so the
        // temp buffer can be zero-initialized with a plain cudaMemsetAsync (no separate
        // setup_kernel launch, a large fixed-cost fraction for the small 256MiB arrays). The
        // ascending claim value is mapped to a descending run top, reproducing the proven
        // high->low tile claim order and run partition; only the counter *storage* changed.
        int const claim = tile_counter->fetch_add(B, cuda::memory_order_relaxed);
        int const hi    = static_cast<int>(num_tiles) - 1 - claim; // descending run top
        batch_next      = hi; // negative once past the end
        batch_lo        = (hi - (B - 1) < 0) ? 0 : hi - (B - 1);
        new_run         = true;
      }
      int const t       = batch_next; // negative once the counter is past the end
      tile_ix_buf[slot] = t;
      // The first tile issued of a run is its top; the tile equal to batch_lo is its bottom.
      is_top_buf[slot] = new_run && (t >= 0);
      is_bot_buf[slot] = (t == batch_lo);
      --batch_next;
    }
    __syncthreads();
    int const curr_tile = tile_ix_buf[slot];
    if (curr_tile < 0)
    {
      return curr_tile;
    }
    size_t const load_index = rotate_distance + head_size + static_cast<size_t>(curr_tile) * TILE_SIZE;
    overcopy_memcpy_async<T, BLOCK_SIZE>(
      cache[slot],
      arr + load_index,
      bytes_to_load_for(curr_tile) / sizeof(T),
      overcopy_extra_head_elems,
      cta,
      bars[slot]);
    return curr_tile;
  };

  // Wait until slot's load has landed in shmem, then publish this tile's load-complete flag.
  // Splitting flag publication from the store is what keeps the per-CTA pipeline deadlock-free:
  // tiles are grabbed in descending order, but a tile's store waits on its *predecessor's* flag,
  // so every in-flight tile must publish its flag (when its data is in shmem) before any
  // dependent store can proceed.
  auto await_and_publish = [&](int slot) {
    int const curr_tile = tile_ix_buf[slot];
    bars[slot].arrive_and_wait();
    // Only the top tile of a run is ever waited on by another CTA (whose run-bottom predecessor
    // is exactly this tile), so only run tops publish a device-scope flag.
    if (tid == 0 && is_top_buf[slot])
    {
      flags[curr_tile].store(1, cuda::memory_order_release);
      flags[curr_tile].notify_all();
    }
  };

  // Store the tile held in slot back to gmem, shifted left by rotate_distance.  Waits on the
  // predecessor tile's load-complete flag first (the in-place RAW ordering).
  auto store_tile = [&](int slot) {
    int const curr_tile          = tile_ix_buf[slot];
    bool const is_last_tile      = curr_tile == 0;
    bool const is_bot            = is_bot_buf[slot];
    uint32_t const bytes_to_load = bytes_to_load_for(curr_tile);
    size_t const load_index      = rotate_distance + head_size + static_cast<size_t>(curr_tile) * TILE_SIZE;

    // Only a run-bottom tile's predecessor lives in another CTA; interior tiles rely on the
    // in-CTA pipeline (predecessor loaded before this tile is stored), so they skip the flag.
    if (tid == 0 && is_bot && !is_last_tile)
    {
      flags[curr_tile - 1].wait(0, cuda::memory_order_acquire);
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

    T* dst       = arr + load_index - rotate_distance;
    T* store_src = cache[slot] + overcopy_extra_head_elems;
    shared_to_global_through_regs<T, BLOCK_SIZE, TILE_BYTES, MAX_REGS>(dst, store_src, bytes_to_load, cta);

    // Tile 0 owns the head: write it after storing the main tile
    if (is_last_tile && head_size > 0u)
    {
      __syncthreads();
      for (uint32_t i = tid; i < head_size; i += BLOCK_SIZE)
      {
        arr[i] = head_tile_cache[i];
      }
    }
  };

  // Software pipeline keeping up to P cp.async tile loads in flight, tracked by three monotone
  // grab-order cursors (grab-order index g maps to ring slot g % P):
  //   issued    : loads issued (cp.async in flight)
  //   published : loads awaited + flags[tile] published (load-complete signalled to other CTAs)
  //   stored    : tiles written back
  // with the ring invariant   stored <= published <= issued <= stored + P.
  int issued     = 0;
  int published  = 0;
  int stored     = 0;
  bool exhausted = false;

  auto try_issue = [&]() {
    if (!exhausted && (issued - stored) < P)
    {
      if (grab_and_load(issued % P) < 0)
      {
        exhausted = true;
      }
      else
      {
        ++issued;
      }
    }
  };

  // Prime the pipeline: issue up to P loads.
  while (!exhausted && (issued - stored) < P)
  {
    try_issue();
  }
  if (issued == 0)
  {
    return; // no tiles for this CTA
  }

  while (stored < issued)
  {
    // Eagerly await + publish every issued-but-unpublished load.  The awaited load for the
    // oldest slot was issued up to P iterations ago, so this rarely stalls; publishing it now
    // (before the dependent store) keeps every CTA's flags visible and the chain unblocked.
    while (published < issued)
    {
      await_and_publish(published % P);
      ++published;
    }

    // Store the oldest tile; the next load (issued below) overlaps this store.
    store_tile(stored % P);
    ++stored;

    // Refill the freed slot, keeping a load in flight to overlap the next store.
    try_issue();
  }

}
} // namespace rotate_short

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
