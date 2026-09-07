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
#include <cub/util_arch.cuh>

#include <cuda/atomic>
#include <cuda/barrier>
#include <cuda/cmath>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cassert>
#include <cstdint>

#include <cooperative_groups.h>

CUB_NAMESPACE_BEGIN
namespace detail
{
namespace rotate
{
inline constexpr int BYTES_PER_SECTOR = 32;
using device_flag_t                   = cuda::atomic<int, cuda::thread_scope_device>;

enum class RotDir
{
  Left,
  Right
};

// Which visit-order the long kernel uses to claim tiles.  Circulant is host-staged in
// processing_order[]; the two small-side orders are trivial closed forms of the claim ticket that
// the kernel evaluates inline, so they need no host->device ordering copy.
enum class OrderMode : uint32_t
{
  Circulant = 0, // host-staged processing_order[]
  Negative  = 1, // small-side negative closed form: pos == 0 ? 0 : num_tiles - pos
  Positive  = 2, // small-side positive closed form: pos
};

struct short_algorithm
{};

struct long_algorithm
{};

template <typename Algorithm>
[[nodiscard]] _CCCL_API constexpr auto get_algorithm_policy(const rotate_policy& policy)
{
  if constexpr (::cuda::std::is_same_v<Algorithm, short_algorithm>)
  {
    return policy.short_algorithm;
  }
  else
  {
    static_assert(::cuda::std::is_same_v<Algorithm, long_algorithm>);
    return policy.long_algorithm;
  }
}

_CCCL_HOST_DEVICE _CCCL_FORCEINLINE device_flag_t* get_long_flags(void* temp_storage)
{
  return reinterpret_cast<device_flag_t*>(reinterpret_cast<uint32_t*>(temp_storage) + 1);
}

_CCCL_HOST_DEVICE _CCCL_FORCEINLINE uint32_t* get_long_processing_order(void* temp_storage, size_t num_tiles)
{
  return reinterpret_cast<uint32_t*>(get_long_flags(temp_storage) + num_tiles);
}

// ============================================================================
// Device-side utility functions
// ============================================================================

template <typename T, class CG>
_CCCL_DEVICE void overcopy_memcpy_async(
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
// BEFORE calling sync_op.sync(), to maximally overlap the atomic polling with other work, then writes to gmem AFTER.
// Any remaining iterations that do not fit in the register budget are processed post-sync. sync_op must provide sync().
//
// MAX_REGS_PER_THREAD is a tuning parameter and must leave room for at least one uint4.
template <typename T, int NUM_THREADS, int TILE_BYTES, int MAX_REGS_PER_THREAD, typename SyncOp>
_CCCL_DEVICE void shared_to_global_through_regs(T* dst, T* src, uint32_t const bytes_to_load, SyncOp& sync_op)
{
  constexpr int REGS_PER_T     = TILE_BYTES / (NUM_THREADS * sizeof(uint32_t));
  constexpr int ITERS          = REGS_PER_T / 4; // each uint4 = 4 regs
  constexpr int CHUNK_REGS     = cuda::std::min(MAX_REGS_PER_THREAD, REGS_PER_T);
  constexpr int BUFFERED_ITERS = CHUNK_REGS / 4;
  static_assert(BUFFERED_ITERS >= 1 && BUFFERED_ITERS <= ITERS);

  constexpr int VEC_TILE_BYTES = ITERS * 4 * NUM_THREADS * sizeof(uint32_t);

  auto const tid = threadIdx.x;

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
      uint32_t const s         = (reinterpret_cast<uintptr_t>(src) % sizeof(uint4)) / sizeof(uint32_t);
      uint4 const* aligned_src = reinterpret_cast<uint4 const*>(new_src - s);

      auto load_funnel = [&](int k) -> uint4 {
        int const j   = tid + k * NUM_THREADS;
        uint4 const A = aligned_src[j];
        uint4 const B = aligned_src[j + 1];
        switch (s)
        {
          case 1:
            return make_uint4(A.y, A.z, A.w, B.x);
          case 2:
            return make_uint4(A.z, A.w, B.x, B.y);
          default: // s == 3
            return make_uint4(A.w, B.x, B.y, B.z);
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
      uintptr_t const src_addr = reinterpret_cast<uintptr_t>(src);
      uint4 const* aligned16   = reinterpret_cast<uint4 const*>(src_addr & ~uintptr_t{15});
      uint32_t const boff      = src_addr & uintptr_t{15}; // in [1,15], boff % 4 != 0
      uint32_t const bw        = boff >> 2; // whole 32-bit words src is above the 16B boundary: 0..3
      uint32_t const shift_len = (boff & 3u) * 8u; // sub-word bit shift: 8, 16 or 24

      uint4* new_dst = reinterpret_cast<uint4*>(dst);

      auto load_funnel_shifted = [&](int k) {
        int const i    = tid + k * NUM_THREADS;
        uint4 const Lo = aligned16[i];
        uint4 const Hi = aligned16[i + 1];
#define FS(a, b) __funnelshift_rc(a, b, shift_len)
        uint4 val;
        switch (bw)
        {
          case 0:
            val = make_uint4(FS(Lo.x, Lo.y), FS(Lo.y, Lo.z), FS(Lo.z, Lo.w), FS(Lo.w, Hi.x));
            break;
          case 1:
            val = make_uint4(FS(Lo.y, Lo.z), FS(Lo.z, Lo.w), FS(Lo.w, Hi.x), FS(Hi.x, Hi.y));
            break;
          case 2:
            val = make_uint4(FS(Lo.z, Lo.w), FS(Lo.w, Hi.x), FS(Hi.x, Hi.y), FS(Hi.y, Hi.z));
            break;
          default: // bw == 3
            val = make_uint4(FS(Lo.w, Hi.x), FS(Hi.x, Hi.y), FS(Hi.y, Hi.z), FS(Hi.z, Hi.w));
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

    if constexpr (VEC_TILE_BYTES < TILE_BYTES)
    {
      // The uncovered tail [VEC_TILE_BYTES, TILE_BYTES) is a contiguous src->dst block copy: the
      // rotation lives entirely in dst's gmem offset, so a plain element copy is byte-identical for
      // all three realignment branches above.
      uint32_t const elems_to_load  = bytes_to_load / sizeof(T);
      constexpr uint32_t tail_begin = VEC_TILE_BYTES / sizeof(T);
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

struct DependencyRange
{
  uint32_t begin_;
  uint32_t end_;
};

namespace tile_detail
{
template <RotDir Dir>
_CCCL_DEVICE size_t
physical_interval_start(size_t const array_size, size_t const logical_start, size_t const interval_size)
{
  if constexpr (Dir == RotDir::Left)
  {
    return logical_start;
  }
  else
  {
    return array_size - logical_start - interval_size;
  }
}

template <typename T>
_CCCL_HOST_DEVICE uint32_t get_neg_head_size(size_t const arr_size, size_t const rot_dist, uint32_t const head_size)
{
  constexpr auto ELEMS_PER_SECTOR = BYTES_PER_SECTOR / sizeof(T);
  // Sector offset of the negative region's start (= arr + arr_size - rot_dist).
  uint32_t const arr_offset = head_size == 0u ? 0u : (ELEMS_PER_SECTOR - head_size);
  uint32_t const dst_offset = (arr_offset + (arr_size - rot_dist)) % ELEMS_PER_SECTOR;
  // 0 when the region's start is already sector-aligned; otherwise the slack to the
  // next sector boundary, in [1, ELEMS_PER_SECTOR - 1].
  return dst_offset == 0u ? 0u : (ELEMS_PER_SECTOR - dst_offset);
}

_CCCL_HOST_DEVICE size_t get_overwrite_start(
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

_CCCL_HOST_DEVICE uint32_t get_tile_size(
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
      uint32_t const remainder = main_size % nominal_tile_size;
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
    uint32_t const remainder = main_size % nominal_tile_size;
    return remainder == 0u ? nominal_tile_size : remainder;
  }
  else
  {
    return nominal_tile_size;
  }
}

_CCCL_HOST_DEVICE size_t get_tile_start(
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

_CCCL_HOST_DEVICE uint32_t
get_num_negative_tiles(size_t const rot_dist, uint32_t const nominal_tile_size, uint32_t const neg_head_size)
{
  return cuda::ceil_div(rot_dist - neg_head_size, nominal_tile_size);
}

uint32_t get_num_positive_tiles(
  size_t const arr_size, size_t const rot_dist, uint32_t const nominal_tile_size, uint32_t const head_size)
{
  return cuda::ceil_div(arr_size - rot_dist - head_size, nominal_tile_size);
}

_CCCL_HOST_DEVICE _CCCL_FORCEINLINE DependencyRange get_dependencies_from(
  size_t const arr_size,
  size_t const rot_dist,
  uint32_t const nominal_tile_size,
  int32_t const tile_ix,
  uint32_t const head_size,
  uint32_t const neg_head_size,
  uint32_t const num_negative_tiles,
  size_t overwrite_start,
  uint32_t const tile_size)
{
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

  auto snap_to_order = [&](size_t pos) -> uint32_t {
    if (pos < neg_head_size)
    {
      return 0;
    }
    if (pos < rot_dist)
    {
      return static_cast<uint32_t>((pos - neg_head_size) / nominal_tile_size);
    }
    if (pos < rot_dist + head_size)
    {
      return num_negative_tiles;
    }
    return num_negative_tiles + static_cast<uint32_t>((pos - rot_dist - head_size) / nominal_tile_size);
  };

  uint32_t const first_order = snap_to_order(overwrite_start);
  uint32_t const last_order  = snap_to_order(overwrite_end);
  assert(first_order <= last_order && last_order - first_order < 3);
  return {first_order, last_order + 1};
}

DependencyRange get_dependencies(
  size_t const arr_size,
  size_t const rot_dist,
  uint32_t const nominal_tile_size,
  int32_t const tile_ix,
  uint32_t const head_size,
  uint32_t const neg_head_size,
  uint32_t const num_negative_tiles)
{
  size_t const overwrite_start =
    get_overwrite_start(arr_size, rot_dist, nominal_tile_size, tile_ix, head_size, neg_head_size);
  uint32_t const tile_size = get_tile_size(
    arr_size,
    rot_dist,
    nominal_tile_size,
    tile_ix,
    get_tile_start(rot_dist, nominal_tile_size, tile_ix, head_size, neg_head_size),
    head_size,
    neg_head_size);
  return get_dependencies_from(
    arr_size,
    rot_dist,
    nominal_tile_size,
    tile_ix,
    head_size,
    neg_head_size,
    num_negative_tiles,
    overwrite_start,
    tile_size);
}

_CCCL_HOST_DEVICE int32_t arr_ix_to_tile_ix(uint32_t const arr_ix, uint32_t const num_negative_tiles)
{
  return arr_ix < num_negative_tiles
         ? -static_cast<int32_t>(arr_ix + 1)
         : (static_cast<int32_t>(arr_ix - num_negative_tiles));
};
} // namespace tile_detail

template <typename T>
constexpr int get_shmem_usage(const kernel_policy& policy)
{
  const int tile_size            = policy.tile_bytes / sizeof(T);
  constexpr int elems_per_sector = BYTES_PER_SECTOR / sizeof(T);
  const int slot_bytes           = cuda::round_up((tile_size + elems_per_sector) * sizeof(T), BYTES_PER_SECTOR);
  return policy.pipeline_stages * slot_bytes;
}

template <typename Algorithm, typename T, typename PolicySelector = policy_selector>
struct pipeline_shared_storage;

template <typename T, typename PolicySelector>
struct pipeline_shared_storage<short_algorithm, T, PolicySelector>
{
  static constexpr int pipeline_stages  = current_policy<PolicySelector>().short_algorithm.kernel.pipeline_stages;
  static constexpr int elems_per_sector = BYTES_PER_SECTOR / sizeof(T);

  struct tile
  {
    int index;
    bool is_run_first;
    bool is_run_last;

    _CCCL_DEVICE bool should_publish() const
    {
      return is_run_first;
    }
  };

  tile tiles[pipeline_stages];
  int next_tile;
  int run_last;
  T head_cache[elems_per_sector];
  cuda::barrier<cuda::thread_scope_block> bars[pipeline_stages];
};

template <typename T, typename PolicySelector>
struct pipeline_shared_storage<long_algorithm, T, PolicySelector>
{
  static constexpr int pipeline_stages  = current_policy<PolicySelector>().long_algorithm.kernel.pipeline_stages;
  static constexpr int elems_per_sector = BYTES_PER_SECTOR / sizeof(T);

  struct tile
  {
    uint32_t work_index;
    uint32_t index;
    uint32_t size;
    uint32_t unaligned_elems;
    size_t src_offset;
    size_t dst_offset;
    DependencyRange dependencies;
    uint32_t head_size;
    size_t head_src_offset;
    size_t head_dst_offset;

    _CCCL_DEVICE bool should_publish() const
    {
      return true;
    }
  };

  tile tiles[pipeline_stages];
  // The positive and negative head tiles can be in flight in different stages at the same time.
  T head_cache[pipeline_stages][elems_per_sector];
  cuda::barrier<cuda::thread_scope_block> bars[pipeline_stages];
};

// TODO: this may cause unnecessary register usage
template <typename Algorithm, RotDir Dir, typename T, typename PolicySelector = policy_selector>
struct pipeline_context
{
  using shared_storage = pipeline_shared_storage<Algorithm, T, PolicySelector>;

  T* array;
  size_t size;
  void* temp_storage;
  size_t rotate_distance;
  size_t num_tiles;
  uint32_t head_size;
  OrderMode order_mode;
  T* cache;
  shared_storage& shared;

  _CCCL_DEVICE T* tile_cache(int slot) const
  {
    constexpr auto kernel          = get_algorithm_policy<Algorithm>(current_policy<PolicySelector>()).kernel;
    constexpr int tile_size        = kernel.tile_bytes / sizeof(T);
    constexpr int elems_per_sector = BYTES_PER_SECTOR / sizeof(T);
    constexpr int slot_bytes       = cuda::round_up((tile_size + elems_per_sector) * sizeof(T), BYTES_PER_SECTOR);
    constexpr int slot_elems       = slot_bytes / sizeof(T);
    return cache + slot * slot_elems;
  }
};

template <RotDir Dir, typename T, typename PolicySelector>
_CCCL_DEVICE device_flag_t* get_flags(pipeline_context<short_algorithm, Dir, T, PolicySelector> const& context)
{
  return reinterpret_cast<device_flag_t*>(reinterpret_cast<int*>(context.temp_storage) + 1);
}

template <RotDir Dir, typename T, typename PolicySelector>
_CCCL_DEVICE device_flag_t* get_flags(pipeline_context<long_algorithm, Dir, T, PolicySelector> const& context)
{
  return get_long_flags(context.temp_storage);
}

template <RotDir Dir, typename T, typename PolicySelector>
_CCCL_DEVICE uint32_t* get_processing_order(pipeline_context<long_algorithm, Dir, T, PolicySelector> const& context)
{
  return get_long_processing_order(context.temp_storage, context.num_tiles);
}

// ============================================================================
// Short-distance rotate operations
// ============================================================================

namespace rotate_tiny
{
template <RotDir Dir, typename T, typename PolicySelector = policy_selector>
_CCCL_KERNEL_ATTRIBUTES void rotate_tiny_kernel(T* arr, size_t const size, size_t const rotate_distance)
{
  constexpr auto policy = current_policy<PolicySelector>().short_algorithm;
  extern __shared__ unsigned char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);

  assert(size <= policy.kernel.tile_bytes / sizeof(T));
  assert(rotate_distance > 0 && rotate_distance <= size / 2);

  if (blockIdx.x == 0)
  {
    for (size_t i = threadIdx.x; i < size; i += blockDim.x)
    {
      smem[i] = arr[i];
    }
    __syncthreads();

    size_t const main_size = size - rotate_distance;
    for (size_t i = threadIdx.x; i < rotate_distance; i += blockDim.x)
    {
      arr[(Dir == RotDir::Left ? main_size : 0) + i] = smem[(Dir == RotDir::Left ? 0 : main_size) + i];
    }
    for (size_t i = threadIdx.x; i < main_size; i += blockDim.x)
    {
      arr[(Dir == RotDir::Left ? 0 : rotate_distance) + i] = smem[(Dir == RotDir::Left ? rotate_distance : 0) + i];
    }
  }
}
} // namespace rotate_tiny

template <typename Algorithm, RotDir Dir, typename T, typename PolicySelector>
_CCCL_DEVICE void initialize_pipeline(pipeline_context<Algorithm, Dir, T, PolicySelector>& context)
{
  constexpr auto kernel          = get_algorithm_policy<Algorithm>(current_policy<PolicySelector>()).kernel;
  constexpr int tile_size        = kernel.tile_bytes / sizeof(T);
  constexpr int elems_per_sector = BYTES_PER_SECTOR / sizeof(T);
  auto const tid                 = threadIdx.x;

  if (tid < kernel.pipeline_stages)
  {
    init(&context.shared.bars[tid], kernel.threads_per_block);
  }

  if constexpr (cuda::std::is_same_v<Algorithm, short_algorithm>)
  {
    assert(context.rotate_distance <= tile_size);
    assert(context.head_size < elems_per_sector);
    assert(2 * context.rotate_distance <= context.size);
    assert(context.size > tile_size);

    if (tid == 0)
    {
      context.shared.next_tile = -1;
      context.shared.run_last  = 0;
    }
  }
  __syncthreads();
}

template <RotDir Dir, typename T, typename PolicySelector>
_CCCL_DEVICE uint32_t get_tile_size(pipeline_context<short_algorithm, Dir, T, PolicySelector> const& context,
                                    uint32_t tile_index)
{
  constexpr auto kernel   = current_policy<PolicySelector>().short_algorithm.kernel;
  constexpr int tile_size = kernel.tile_bytes / sizeof(T);
  if (tile_index != context.num_tiles - 1)
  {
    return tile_size;
  }

  uint32_t const remainder = (context.size - context.rotate_distance - context.head_size) % tile_size;
  return remainder == 0u ? tile_size : remainder;
}

// Claim the next tile from this CTA's descending run and issue its asynchronous load.
template <RotDir Dir, typename T, typename PolicySelector>
_CCCL_DEVICE bool claim_and_load(pipeline_context<short_algorithm, Dir, T, PolicySelector>& context, int slot)
{
  constexpr auto policy          = current_policy<PolicySelector>().short_algorithm;
  constexpr int full_tile_size   = policy.kernel.tile_bytes / sizeof(T);
  constexpr int elems_per_sector = BYTES_PER_SECTOR / sizeof(T);
  constexpr int tiles_per_grab   = policy.tiles_per_grab;
  auto* tile_counter = reinterpret_cast<cuda::atomic<int, cuda::thread_scope_device>*>(context.temp_storage);
  auto cta           = cooperative_groups::this_thread_block();
  auto const tid     = threadIdx.x;
  auto& tile         = context.shared.tiles[slot];

  if (tid == 0)
  {
    bool is_run_first = false;
    if (context.shared.next_tile < context.shared.run_last)
    {
      int const claim          = tile_counter->fetch_add(tiles_per_grab, cuda::memory_order_relaxed);
      int const first          = static_cast<int>(context.num_tiles) - 1 - claim;
      context.shared.next_tile = first;
      context.shared.run_last  = (first < tiles_per_grab - 1) ? 0 : first - (tiles_per_grab - 1);
      is_run_first             = first >= 0;
    }

    tile.index        = context.shared.next_tile;
    tile.is_run_first = is_run_first;
    tile.is_run_last  = tile.index == context.shared.run_last;
    --context.shared.next_tile;
  }
  __syncthreads();

  if (tile.index < 0)
  {
    return false;
  }

  uint32_t const tile_size = get_tile_size(context, tile.index);
  size_t const logical_src =
    context.rotate_distance + context.head_size + static_cast<size_t>(tile.index) * full_tile_size;
  size_t const src_offset  = tile_detail::physical_interval_start<Dir>(context.size, logical_src, tile_size);
  uint32_t unaligned_elems = context.rotate_distance % elems_per_sector;
  if constexpr (Dir == RotDir::Right)
  {
    unaligned_elems = (reinterpret_cast<uintptr_t>(context.array + src_offset) % BYTES_PER_SECTOR) / sizeof(T);
  }

  overcopy_memcpy_async<T>(
    context.tile_cache(slot), context.array + src_offset, tile_size, unaligned_elems, cta, context.shared.bars[slot]);
  return true;
}

template <typename Algorithm, RotDir Dir, typename T, typename PolicySelector>
_CCCL_DEVICE void publish_tile(pipeline_context<Algorithm, Dir, T, PolicySelector>& context, int slot)
{
  auto* flags      = get_flags(context);
  auto const& tile = context.shared.tiles[slot];
  context.shared.bars[slot].arrive_and_wait();

  if (threadIdx.x == 0 && tile.should_publish())
  {
    flags[tile.index].store(1, cuda::memory_order_release);
    flags[tile.index].notify_all();
  }
}

template <RotDir Dir, typename T, typename PolicySelector>
_CCCL_DEVICE void wait_for_dependencies(pipeline_context<short_algorithm, Dir, T, PolicySelector>& context, int slot)
{
  auto* flags             = get_flags(context);
  auto const& tile        = context.shared.tiles[slot];
  bool const is_last_tile = tile.index == 0;

  if (threadIdx.x == 0 && tile.is_run_last && !is_last_tile)
  {
    flags[tile.index - 1].wait(0, cuda::memory_order_acquire);
  }

  if (is_last_tile)
  {
    flags[context.num_tiles - 1].wait(0, cuda::memory_order_acquire);
    __syncthreads();
  }
}

template <RotDir Dir, typename T, typename PolicySelector>
_CCCL_DEVICE void store_tile(pipeline_context<short_algorithm, Dir, T, PolicySelector>& context, int slot)
{
  constexpr auto kernel          = current_policy<PolicySelector>().short_algorithm.kernel;
  constexpr int full_tile_size   = kernel.tile_bytes / sizeof(T);
  constexpr int elems_per_sector = BYTES_PER_SECTOR / sizeof(T);
  auto cta                       = cooperative_groups::this_thread_block();
  auto const tid                 = threadIdx.x;
  int const tile_index           = context.shared.tiles[slot].index;
  bool const is_last_tile        = tile_index == 0;
  uint32_t const tile_size       = get_tile_size(context, tile_index);
  size_t const logical_src =
    context.rotate_distance + context.head_size + static_cast<size_t>(tile_index) * full_tile_size;
  size_t const logical_dst = logical_src - context.rotate_distance;
  size_t const dst_offset  = tile_detail::physical_interval_start<Dir>(context.size, logical_dst, tile_size);

  if (is_last_tile)
  {
    if (context.head_size > 0u)
    {
      size_t const head_src =
        tile_detail::physical_interval_start<Dir>(context.size, context.rotate_distance, context.head_size);
      for (uint32_t i = tid; i < context.head_size; i += kernel.threads_per_block)
      {
        context.shared.head_cache[i] = context.array[head_src + i];
      }
    }

    size_t const wrap_src = tile_detail::physical_interval_start<Dir>(context.size, 0, context.rotate_distance);
    size_t const wrap_dst = tile_detail::physical_interval_start<Dir>(
      context.size, context.size - context.rotate_distance, context.rotate_distance);
    for (size_t i = tid; i < context.rotate_distance; i += kernel.threads_per_block)
    {
      context.array[wrap_dst + i] = context.array[wrap_src + i];
    }
    __syncthreads();

    if (context.head_size > 0u)
    {
      size_t const head_dst = tile_detail::physical_interval_start<Dir>(context.size, 0, context.head_size);
      for (uint32_t i = tid; i < context.head_size; i += kernel.threads_per_block)
      {
        context.array[head_dst + i] = context.shared.head_cache[i];
      }
    }
  }

  size_t const src_offset  = tile_detail::physical_interval_start<Dir>(context.size, logical_src, tile_size);
  uint32_t unaligned_elems = context.rotate_distance % elems_per_sector;
  if constexpr (Dir == RotDir::Right)
  {
    unaligned_elems = (reinterpret_cast<uintptr_t>(context.array + src_offset) % BYTES_PER_SECTOR) / sizeof(T);
  }
  shared_to_global_through_regs<T, kernel.threads_per_block, kernel.tile_bytes, kernel.max_regs_per_thread>(
    context.array + dst_offset, context.tile_cache(slot) + unaligned_elems, tile_size * sizeof(T), cta);
}

// ============================================================================
// Long-distance rotate operations
// ============================================================================

template <RotDir Dir, typename T, typename PolicySelector>
_CCCL_DEVICE bool claim_and_load(pipeline_context<long_algorithm, Dir, T, PolicySelector>& context, int slot)
{
  constexpr auto kernel   = current_policy<PolicySelector>().long_algorithm.kernel;
  constexpr int tile_size = kernel.tile_bytes / sizeof(T);
  auto* counter           = reinterpret_cast<cuda::atomic<uint32_t, cuda::thread_scope_device>*>(context.temp_storage);
  auto* processing_order  = get_processing_order(context);
  auto cta                = cooperative_groups::this_thread_block();
  auto const tid          = threadIdx.x;
  auto& tile              = context.shared.tiles[slot];
  uint32_t const neg_head_size =
    tile_detail::get_neg_head_size<T>(context.size, context.rotate_distance, context.head_size);
  uint32_t const num_negative_tiles =
    tile_detail::get_num_negative_tiles(context.rotate_distance, tile_size, neg_head_size);

  if (tid == 0)
  {
    tile.work_index = counter->fetch_add(1, cuda::memory_order_relaxed);
    if (tile.work_index < context.num_tiles)
    {
      tile.index =
        context.order_mode == OrderMode::Circulant
          ? processing_order[tile.work_index]
          : (context.order_mode == OrderMode::Negative
               ? (tile.work_index == 0u ? 0u : static_cast<uint32_t>(context.num_tiles) - tile.work_index)
               : tile.work_index);

      int32_t const tile_index = tile_detail::arr_ix_to_tile_ix(tile.index, num_negative_tiles);
      size_t const logical_src =
        tile_detail::get_tile_start(context.rotate_distance, tile_size, tile_index, context.head_size, neg_head_size);
      tile.size = tile_detail::get_tile_size(
        context.size, context.rotate_distance, tile_size, tile_index, logical_src, context.head_size, neg_head_size);
      assert(tile.size <= tile_size);
      tile.src_offset = tile_detail::physical_interval_start<Dir>(context.size, logical_src, tile.size);
      tile.unaligned_elems =
        (reinterpret_cast<uintptr_t>(context.array + tile.src_offset) % BYTES_PER_SECTOR) / sizeof(T);

      bool const owns_pos_head      = tile_index == 0 && context.head_size > 0u;
      bool const owns_neg_head      = tile_index == -1 && neg_head_size > 0u;
      tile.head_size                = owns_pos_head ? context.head_size : (owns_neg_head ? neg_head_size : 0u);
      size_t const logical_head_src = owns_pos_head ? context.rotate_distance : 0;
      tile.head_src_offset = tile_detail::physical_interval_start<Dir>(context.size, logical_head_src, tile.head_size);
    }
  }
  __syncthreads();

  if (tile.work_index >= context.num_tiles)
  {
    return false;
  }

  overcopy_memcpy_async<T>(
    context.tile_cache(slot),
    context.array + tile.src_offset,
    tile.size,
    tile.unaligned_elems,
    cta,
    context.shared.bars[slot]);

  if (tile.head_size > 0u && tid < warp_threads)
  {
    for (uint32_t i = tid; i < tile.head_size; i += warp_threads)
    {
      context.shared.head_cache[slot][i] = context.array[tile.head_src_offset + i];
    }
  }

  // Compute store-side state while the asynchronous tile copy is in flight.
  if (tid == 0)
  {
    int32_t const tile_index = tile_detail::arr_ix_to_tile_ix(tile.index, num_negative_tiles);
    size_t const logical_dst = tile_detail::get_overwrite_start(
      context.size, context.rotate_distance, tile_size, tile_index, context.head_size, neg_head_size);
    tile.dependencies = tile_detail::get_dependencies_from(
      context.size,
      context.rotate_distance,
      tile_size,
      tile_index,
      context.head_size,
      neg_head_size,
      num_negative_tiles,
      logical_dst,
      tile.size);
    tile.dst_offset = tile_detail::physical_interval_start<Dir>(context.size, logical_dst, tile.size);

    bool const owns_pos_head      = tile_index == 0 && context.head_size > 0u;
    size_t const logical_head_dst = owns_pos_head ? 0 : (context.size - context.rotate_distance);
    tile.head_dst_offset = tile_detail::physical_interval_start<Dir>(context.size, logical_head_dst, tile.head_size);
  }
  return true;
}

template <RotDir Dir, typename T, typename PolicySelector>
_CCCL_DEVICE void wait_for_dependencies(pipeline_context<long_algorithm, Dir, T, PolicySelector>& context, int slot)
{
  auto* flags = get_flags(context);

  if (threadIdx.x == 0)
  {
    auto const dependencies = context.shared.tiles[slot].dependencies;
    for (uint32_t dependency = dependencies.begin_; dependency < dependencies.end_; ++dependency)
    {
      flags[dependency].wait(0, cuda::memory_order_acquire);
    }
  }
}

template <RotDir Dir, typename T, typename PolicySelector>
_CCCL_DEVICE void store_tile(pipeline_context<long_algorithm, Dir, T, PolicySelector>& context, int slot)
{
  constexpr auto kernel = current_policy<PolicySelector>().long_algorithm.kernel;
  auto cta              = cooperative_groups::this_thread_block();
  auto const tid        = threadIdx.x;
  auto const& tile      = context.shared.tiles[slot];

  shared_to_global_through_regs<T, kernel.threads_per_block, kernel.tile_bytes, kernel.max_regs_per_thread>(
    context.array + tile.dst_offset, context.tile_cache(slot) + tile.unaligned_elems, tile.size * sizeof(T), cta);

  if (tile.head_size > 0u && tid < warp_threads)
  {
    for (uint32_t i = tid; i < tile.head_size; i += warp_threads)
    {
      context.array[tile.head_dst_offset + i] = context.shared.head_cache[slot][i];
    }
  }
}

// ============================================================================
// Pipeline execution
// ============================================================================
template <typename Algorithm, RotDir Dir, typename T, typename PolicySelector>
_CCCL_DEVICE void run_tile_pipeline(pipeline_context<Algorithm, Dir, T, PolicySelector>& context)
{
  constexpr auto kernel         = get_algorithm_policy<Algorithm>(current_policy<PolicySelector>()).kernel;
  constexpr int pipeline_stages = kernel.pipeline_stages;
  int issued                    = 0;
  int published                 = 0;
  int stored                    = 0;
  bool exhausted                = false;

  // Fill the pipeline. A claim that reports exhaustion does not occupy a slot.
  while (!exhausted && issued - stored < pipeline_stages)
  {
    exhausted = !claim_and_load(context, issued % pipeline_stages);
    if (!exhausted)
    {
      ++issued;
    }
  }

  while (stored < issued)
  {
    // Publish every in-flight load before waiting: a tile may depend on another tile
    // currently owned by this CTA.
    while (published < issued)
    {
      publish_tile(context, published % pipeline_stages);
      ++published;
    }

    wait_for_dependencies(context, stored % pipeline_stages);
    store_tile(context, stored % pipeline_stages);
    ++stored;

    // The store freed one slot, so issue at most one replacement load.
    if (!exhausted)
    {
      exhausted = !claim_and_load(context, issued % pipeline_stages);
      if (!exhausted)
      {
        ++issued;
      }
    }
  }
}

template <typename Algorithm, RotDir Dir, typename T, typename PolicySelector = policy_selector>
__launch_bounds__(get_algorithm_policy<Algorithm>(current_policy<PolicySelector>()).kernel.threads_per_block,
                  get_algorithm_policy<Algorithm>(current_policy<PolicySelector>()).kernel.blocks_per_sm)
  _CCCL_KERNEL_ATTRIBUTES void rotate_kernel(
    T* arr,
    size_t const size,
    void* temp_storage,
    size_t const rotate_distance,
    size_t const num_tiles,
    uint32_t const head_size,
    OrderMode const order_mode)
{
  constexpr auto policy = current_policy<PolicySelector>();
  constexpr auto kernel = get_algorithm_policy<Algorithm>(policy).kernel;
  assert(blockDim.x == kernel.threads_per_block);

  alignas(BYTES_PER_SECTOR) extern __shared__ unsigned char smem_raw[];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ pipeline_shared_storage<Algorithm, T, PolicySelector> shared;

  pipeline_context<Algorithm, Dir, T, PolicySelector> context{
    arr, size, temp_storage, rotate_distance, num_tiles, head_size, order_mode, reinterpret_cast<T*>(smem_raw), shared};
  initialize_pipeline(context);
  run_tile_pipeline(context);
}
} // namespace rotate
} // namespace detail
CUB_NAMESPACE_END
