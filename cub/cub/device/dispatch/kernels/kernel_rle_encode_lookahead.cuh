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

#include <cub/detail/warpspeed/squad/squad.cuh>
#include <cub/device/dispatch/tuning/tuning_rle_encode.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_macro.cuh>

#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail::rle::encode
{
namespace ptx = ::cuda::ptx;

_CCCL_HOST_DEVICE_API constexpr int num_total_threads(const RleLookaheadPolicy& policy)
{
  const int num_total_warps = 1 /*load*/ + policy.compute_warps + 1 /*poll*/ + policy.store_warps + 1 /*bookkeeper*/;
  return num_total_warps * 32;
}

// This is important for position staging on dense cases (16 way bank conflicts).
__device__ __forceinline__ int swizzle_xor_stride32(int x)
{
  return x ^ (x >> 5);
}

constexpr unsigned full_mask = 0xffffffffu;

__device__ __forceinline__ void wait_parity(cuda::std::uint64_t* bar, unsigned parity)
{
  while (!ptx::mbarrier_try_wait_parity(bar, parity))
  {
  }
}

// tile_partial_states: one dword per tile, layout: u64 [published_tag:32][open_len:16][run_count:16]
// states are cleared by rle_init_states every launch, since we do not own temp storage!
// an aligned 64-bit access is already non-tearing, but atomic_ref doesn't hurt and has clear semantics
// Nan: we could use u32 layouts [ready_bit:1][open_len:15][run_count:16], but we choose to use u64 to
// 1. w << 32 is free (u64 is already split into 2 registers), so we save a bit of time (theoretically)
// 2. to use the same layout as warpspeed scan
constexpr unsigned tile_published = 1u;

struct TilePartialStateT
{
  cuda::std::uint64_t dword;

  __device__ __forceinline__ unsigned published_tag() const
  {
    return (unsigned) (dword >> 32);
  }

  __device__ __forceinline__ int run_count() const
  {
    return (int) (dword & 0xffffu);
  }

  __device__ __forceinline__ int open_len() const
  {
    return (int) ((dword >> 16) & 0xffffu);
  }

  static __device__ __forceinline__ TilePartialStateT pack(int run_count, int open_len)
  {
    return {((cuda::std::uint64_t) tile_published << 32) | ((cuda::std::uint64_t) (unsigned) open_len << 16)
            | (cuda::std::uint64_t) (unsigned) run_count};
  }
};

__device__ __forceinline__ void
publish_state(TilePartialStateT* tile_state_arr, int tile_idx, int run_count, int open_len)
{
  cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_device> a(tile_state_arr[tile_idx].dword);
  a.store(TilePartialStateT::pack(run_count, open_len).dword, cuda::memory_order_relaxed);
}

// return the state (even if not yet publish for this launch, caller checks it)
// we do not want to spin here
__device__ __forceinline__ TilePartialStateT load_state(TilePartialStateT* tile_state_arr, int tile_idx)
{
  cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_device> a(tile_state_arr[tile_idx].dword);
  return {a.load(cuda::memory_order_relaxed)};
}

// CRITICAL: from choose_signed_offset, it is guaranteed that OffT covers the whole index space.
// Therefore, in the kernel, the type of the prefix (run_count, open_len) should always be OffT.
template <class OffT, bool = (sizeof(OffT) > 4)>
struct PrefixT;

template <class OffT>
struct PrefixT<OffT, false>
{
  cuda::std::uint64_t dword;

  static __device__ __forceinline__ PrefixT pack(OffT run_count, OffT open_len)
  {
    return {((cuda::std::uint64_t) (unsigned) open_len << 32) | (unsigned) run_count};
  }

  __device__ __forceinline__ OffT run_count() const
  {
    return (OffT) (unsigned) (dword & 0xffffffffull);
  }

  __device__ __forceinline__ OffT open_len() const
  {
    return (OffT) (unsigned) (dword >> 32);
  }
};

template <class OffT>
struct alignas(16) PrefixT<OffT, true>
{
  cuda::std::uint64_t packed_run_count;
  cuda::std::uint64_t packed_open_len;

  static __device__ __forceinline__ PrefixT pack(OffT run_count, OffT open_len)
  {
    return {(cuda::std::uint64_t) run_count, (cuda::std::uint64_t) open_len};
  }

  __device__ __forceinline__ OffT run_count() const
  {
    return (OffT) packed_run_count;
  }

  __device__ __forceinline__ OffT open_len() const
  {
    return (OffT) packed_open_len;
  }
};

// position of the n-th set bit of flag_mask, requires popc(flag_mask) > rank. Implementation is binary search.
// __fns(flag_mask, 0, rank+1) computes the same thing but has NO hardware op on sm_100a and is slower
// TODO (Nan): as per discussion with Federico, this could be in libcudacxx
__device__ __forceinline__ int nth_set_bit(unsigned flag_mask, int rank)
{
  // each step: if the wanted bit is not among the low half's set bits, skip that half entirely
  int bit_position         = 0;
  int set_bits_in_low_half = __popc(flag_mask & 0xffffu);
  if (rank >= set_bits_in_low_half)
  {
    rank -= set_bits_in_low_half;
    bit_position += 16;
    flag_mask >>= 16;
  }
  set_bits_in_low_half = __popc(flag_mask & 0xffu);
  if (rank >= set_bits_in_low_half)
  {
    rank -= set_bits_in_low_half;
    bit_position += 8;
    flag_mask >>= 8;
  }
  set_bits_in_low_half = __popc(flag_mask & 0xfu);
  if (rank >= set_bits_in_low_half)
  {
    rank -= set_bits_in_low_half;
    bit_position += 4;
    flag_mask >>= 4;
  }
  set_bits_in_low_half = __popc(flag_mask & 0x3u);
  if (rank >= set_bits_in_low_half)
  {
    rank -= set_bits_in_low_half;
    bit_position += 2;
    flag_mask >>= 2;
  }
  if (rank >= (int) (flag_mask & 1u))
  {
    bit_position += 1;
  }
  return bit_position;
}

// width = how many low lanes participate, i.e. lanes [0, width): lane i returns lane_value(0) + ... + lane_value(i).
// lanes in [width, 32) must still call this, but their return values are unspecified.
// (TODO) Nan: swapping this with cub::WarpScan caused perf regression (-3% in worst cases). This needs investigation.
// The primary suspect is cub::WarpScan uses asm VOLATILE and it could change codegen
template <int width>
__device__ __forceinline__ int warp_inclusive_scan_add(int lane_value, int lane_id)
{
  static_assert(1 <= width && width <= 32, "the scan operates within a single warp");
#pragma unroll
  for (int offset = 1; offset < width; offset <<= 1)
  {
    const int predecessor_partial = __shfl_up_sync(full_mask, lane_value, offset);
    if (lane_id >= offset)
    {
      lane_value += predecessor_partial;
    }
  }
  return lane_value;
}

struct WarpTileRunScanT
{
  int lane_run_count;
  int lane_runs_before;
};

// we need this because STORE and BOOKKEEPER both recalculate from slot_warp_run_counts
template <int compute_warps>
__device__ __forceinline__ WarpTileRunScanT scan_warp_tile_run_counts(const int* slot_warp_run_counts, int lane_id)
{
  const int lane_run_count = (lane_id < compute_warps) ? slot_warp_run_counts[lane_id] : 0;
  const int lane_scan      = warp_inclusive_scan_add<compute_warps>(lane_run_count, lane_id);
  return {lane_run_count, lane_scan - lane_run_count};
}

template <int tile_size, int slot_pad, class KeyT>
__device__ __forceinline__ void load_tile_keys(
  KeyT* slot,
  const KeyT* d_keys,
  int tile_id,
  int tile_len,
  bool first_tile,
  bool last_tile,
  unsigned base_skip,
  cuda::std::uint64_t* full_bar,
  int lane_id)
{
  if (lane_id == 0)
  {
    // if it is not first tile, we overcopy 16B to the left to get last key from last tile
    const unsigned nbytes     = (unsigned) (((size_t) tile_len + (first_tile ? 0 : slot_pad)) * sizeof(KeyT));
    const unsigned span_bytes = (nbytes + base_skip + 15u) & ~15u;
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, full_bar, span_bytes);
    ptx::cp_async_bulk_ignore_oob(
      ptx::space_shared,
      ptx::space_global,
      slot + (first_tile ? slot_pad : 0),
      (const KeyT*) ((const char*) (d_keys + (size_t) tile_id * tile_size - (first_tile ? 0 : slot_pad)) - base_skip),
      span_bytes,
      first_tile ? base_skip : 0u,
      last_tile ? (span_bytes - base_skip - nbytes) : 0u,
      full_bar);
  }
  __syncwarp();
}

__device__ __forceinline__ int
clc_next_tile_id(uint4& clc_resp, cuda::std::uint64_t& clc_bar, int pipeline_gen, int num_tiles, int lane_id)
{
  int nxt = num_tiles; // if no more work was cancellable
  if (lane_id == 0)
  {
    wait_parity(&clc_bar, (unsigned) (pipeline_gen & 1));
    // try_cancel wrote clc_resp via the async proxy
    ptx::fence_proxy_async(ptx::space_shared);
    const uint4 resp_snapshot = clc_resp;
    ptx::fence_proxy_async(ptx::space_shared);
    const bool canceled = ptx::clusterlaunchcontrol_query_cancel_is_canceled(resp_snapshot);
    if (canceled)
    {
      nxt = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(resp_snapshot);
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &clc_bar, 16);
      ptx::clusterlaunchcontrol_try_cancel(&clc_resp, &clc_bar);
    }
  }
  return __shfl_sync(full_mask, nxt, 0);
}

// calculate head_flags: each iter is 32 consecutive elements (lane L owns loc = warp_tile_offset + iter*32 + L)
// head = (key != predecessor)
template <int items_per_thread, class KeyT>
__device__ __forceinline__ unsigned
compute_head_flags(const KeyT* key_buf, int warp_tile_offset, int tile_len, int tile_id, int lane_id, int skip_elems)
{
  static_assert(items_per_thread <= 32, "one lane per iter requires items_per_thread<=32");
  unsigned my_flags = 0;
#pragma unroll
  for (int iter = 0; iter < items_per_thread; ++iter)
  {
    const int loc             = warp_tile_offset + iter * 32 + lane_id;
    const KeyT key            = (loc < tile_len) ? key_buf[loc + skip_elems] : KeyT{};
    const KeyT pred           = key_buf[loc + skip_elems - 1]; // loc==0 reads the over fetched slot[slot_pad-1]
    const int is_global_first = (tile_id == 0 && loc == 0);
    const int head            = (loc < tile_len) ? (is_global_first ? 1 : (key != pred)) : 0;
    const unsigned flags      = __ballot_sync(full_mask, head);
    if (lane_id == iter)
    {
      my_flags = flags;
    }
  }
  return my_flags;
}

template <int compute_warps>
__device__ __forceinline__ void reduce_and_publish_tile_state(
  TilePartialStateT* tile_partial_states,
  int tile_id,
  int tile_len,
  const int* slot_warp_run_counts,
  const int* slot_warp_last_heads,
  int lane_id)
{
  // compute_warps<=32 so one lane/warp fits (in practice we will never have anything close to 32)
  static_assert(compute_warps <= 32, "compute_warps must be less than 32!");
  const bool active        = lane_id < compute_warps;
  const int warp_run_count = active ? slot_warp_run_counts[lane_id] : 0;
  const int run_count      = __reduce_add_sync(full_mask, warp_run_count);
  // last head = the highest-index warp that has any run (its last_head is the tile's last head)
  const unsigned warps_with_runs = __ballot_sync(full_mask, active && warp_run_count > 0);
  int last_head_idx              = -1;
  // if we have any heads, get last head index
  if (warps_with_runs)
  {
    const int last_warp_with_runs = 31 - __clz(warps_with_runs);
    // broadcast the index from last lane to all
    last_head_idx = __shfl_sync(full_mask, active ? slot_warp_last_heads[lane_id] : -1, last_warp_with_runs);
  }
  if (lane_id == 0)
  {
    const int open_len = (run_count > 0) ? (tile_len - last_head_idx) : tile_len;
    // CRITICAL: publish as soon as possible, this is why we calculate head_flags first
    publish_state(tile_partial_states, tile_id, run_count, open_len);
  }
}

template <int items_per_thread>
__device__ __forceinline__ void
stage_head_positions(unsigned my_flags, short* pos_dst, int warp_tile_offset, int lane_id)
{
  // we store run R at warp_tile_offset + (R ^ (R>>5)) to avoid bank conflicts for dense cases
  // (CRITICAL for MaxSeg=1,2,4)
  int head_scan = __popc(my_flags); // start: this word's head count
  head_scan     = warp_inclusive_scan_add<32>(head_scan, lane_id);
  // head_scan is a running sum of run_count, so each lane know each chunk's base
  const int runs_before_word = head_scan - __popc(my_flags);
  if (lane_id < items_per_thread)
  {
    const int word_pos     = warp_tile_offset + lane_id * 32; // element position of bit 0 of this word
    unsigned pending_heads = my_flags; // this word's head mask; we need to "peel" it headbit by headbit
    int run_index          = runs_before_word; // run-order slot for this word's next head
    while (pending_heads)
    {
      const int head_offset = __ffs(pending_heads) - 1; // offset (0..31) of the next head within the word
      pos_dst[warp_tile_offset + swizzle_xor_stride32(run_index)] = (short) (word_pos + head_offset);
      ++run_index;
      pending_heads &= (pending_heads - 1); // clear the lowest set bit
    }
  }
}

struct RunSpanT
{
  int head_pos_in_warp_tile;
  int next_head_pos;
};

// the compute warp may deem this warp tile too sparse to be worth the position-staging, and in that case it will write
// only the 32 head-flag words. Then, it is up to the store warps to "decode" the positions from the headflags.
// one warp tile is 32 chunks x 32 elements, so lane i owns word i.
// This buys 2.5% BWUtil in the MaxSegSize{2^4, 2^6, 2^8}
struct HeadFlagDecodeT
{
  unsigned lane_head_flag_word;
  int lane_runs_before_word;
  int lane_first_head_from_word;

  __device__ __forceinline__ HeadFlagDecodeT(const unsigned* slot_head_flags, int warp_tile_id, int lane_id)
  {
    lane_head_flag_word                = slot_head_flags[warp_tile_id * 32 + lane_id];
    const int lane_word_run_count      = __popc(lane_head_flag_word);
    const int lane_word_run_count_scan = warp_inclusive_scan_add<32>(lane_word_run_count, lane_id);
    // lane i: # of runs starting in head_flag words [0, i), i.e. in elements [0, i*32)
    lane_runs_before_word = lane_word_run_count_scan - lane_word_run_count;
    // lane i -> first head position in head flag words [i, 32)
    // if our own run_count is >0, the head is here!
    // empty should be +infinity, since we use min
    lane_first_head_from_word = lane_word_run_count ? (lane_id * 32 + __ffs(lane_head_flag_word) - 1) : 0x7fffffff;
    // if not, we loop to find the next head in flag word [i, 32). this is just a fold with min
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
    {
      const int shuffled_first_head = __shfl_down_sync(full_mask, lane_first_head_from_word, offset);
      lane_first_head_from_word =
        min(lane_first_head_from_word, (lane_id + offset < 32) ? shuffled_first_head : 0x7fffffff);
    }
    // now, lane i holds the next head in [i, 32). we precalculate this in parallel
  }

  __device__ __forceinline__ RunSpanT decode_run(int run_idx) const
  {
    // first question: which head_flag word contains my run's (run_idx) head?
    // lane_runs_before_word's row i = number of heads in words [0, i)
    // the word containing run_dex is then the largest i with runs_before(i) that is <= j
    // we do binary search over the distributed lane_runs_before_word table held across the warp
    int flag_word_idx = 0;
#pragma unroll
    for (int step = 16; step; step >>= 1)
    {
      // propose candidate
      const int candidate_word_idx = flag_word_idx + step;
      // read the i'th row
      const int candidate_runs_before = __shfl_sync(full_mask, lane_runs_before_word, candidate_word_idx & 31);
      if (candidate_word_idx < 32 && candidate_runs_before <= run_idx)
      {
        flag_word_idx = candidate_word_idx;
      }
    }
    // the lane now knows the index of the word containing its head
    // we need to convert it to the element position
    // where is my head in the word?
    const int run_rank_in_word = run_idx - __shfl_sync(full_mask, lane_runs_before_word, flag_word_idx);
    // get the actual word
    const unsigned flag_word = __shfl_sync(full_mask, lane_head_flag_word, flag_word_idx);
    // where's the first head in any word after mine?
    const int first_head_after_word = __shfl_sync(full_mask, lane_first_head_from_word, (flag_word_idx + 1) & 31);
    // how many heads my word has?
    const int flag_word_run_count = __popc(flag_word);
    // position of my head inside the word
    const int head_bit_in_word =
      nth_set_bit(flag_word, (run_rank_in_word < flag_word_run_count) ? run_rank_in_word : 0);
    const int head_pos_in_warp_tile = flag_word_idx * 32 + head_bit_in_word;
    // where does my run end? try find the position of next head in word
    const int next_head_in_word = flag_word_idx * 32 + __ffs(flag_word & (~1u << head_bit_in_word)) - 1;
    // does my word contain a head after mine? if not, next_head_in_word is garbage, and we use first_head_after_word
    const int next_head_pos = (run_rank_in_word + 1 < flag_word_run_count) ? next_head_in_word : first_head_after_word;
    return {head_pos_in_warp_tile, next_head_pos};
  }
};

template <int window_size_cap, class PolicySelector, class OffT>
__device__ __forceinline__ void poll_fold_windows(
  TilePartialStateT* tile_partial_states,
  int tile_id,
  int& last_seen_tile_id,
  OffT& last_seen_prefix_run_count,
  OffT& last_seen_prefix_open_length,
  int lane_id,
  int& dense_mode)
{
  constexpr int poll_loads_per_lane = current_policy<PolicySelector>().lookahead.poll_loads_per_lane;
  static_assert(window_size_cap >= 1 && window_size_cap <= 32 * poll_loads_per_lane,
                "the fold window must be covered by the lanes");
  while (last_seen_tile_id < tile_id)
  {
    const int remain = tile_id - last_seen_tile_id;
    // # of tiles to fold this iteration
    const int window_size                               = remain < window_size_cap ? remain : window_size_cap;
    const int lane_first_tile_id                        = last_seen_tile_id + lane_id;
    const int lane_tile_count                           = (window_size - lane_id + 31) >> 5;
    TilePartialStateT packed_words[poll_loads_per_lane] = {}; // must zero initialize
    bool ready;
    // first, all tile state in window must be ready
    do
    {
      ready = true;
#pragma unroll
      for (int i = 0; i < poll_loads_per_lane; ++i)
      {
        // we only try if that state is not published
        if (i < lane_tile_count && packed_words[i].published_tag() != tile_published)
        {
          packed_words[i] = load_state(tile_partial_states, lane_first_tile_id + i * 32);
          if (packed_words[i].published_tag() != tile_published)
          {
            ready = false;
          }
        }
      }
    } while (__ballot_sync(full_mask, !ready) != 0u);
    int lane_run_count = 0, lane_last_tile_with_runs_in_window = -1;
    // now, we fold the window
#pragma unroll
    for (int i = 0; i < poll_loads_per_lane; ++i)
    {
      if (i < lane_tile_count)
      {
        // aggregate run_count per lane, this is fine since run_count is commutative
        lane_run_count += packed_words[i].run_count();
        // norminate the highest tile id with runs
        lane_last_tile_with_runs_in_window =
          (packed_words[i].run_count() > 0) ? (i * 32 + lane_id) : lane_last_tile_with_runs_in_window;
      }
    }
    // vote for the highest tile id with runs
    const int last_tile_with_runs_in_window = __reduce_max_sync(full_mask, lane_last_tile_with_runs_in_window);
    int lane_open_length                    = 0;
#pragma unroll
    // how long is the window_size's unfinished run?
    for (int i = 0; i < poll_loads_per_lane; ++i)
    {
      // if this tile id >= the highest tile id with runs
      if (i < lane_tile_count && i * 32 + lane_id >= last_tile_with_runs_in_window)
      {
        lane_open_length += packed_words[i].open_len();
      }
    }
    const int window_run_count   = __reduce_add_sync(full_mask, lane_run_count);
    const int window_open_length = __reduce_add_sync(full_mask, lane_open_length);
    // dense_mode is true if window_run_count > 128
    dense_mode = window_run_count > (window_size << 7);
    // combine last_seen_prefix with the window_size aggregate
    const OffT new_run_count = last_seen_prefix_run_count + window_run_count;
    const OffT new_open_length =
      (window_run_count > 0) ? (OffT) window_open_length : (last_seen_prefix_open_length + window_open_length);
    last_seen_prefix_run_count   = new_run_count;
    last_seen_prefix_open_length = new_open_length;
    last_seen_tile_id += window_size;
  }
}

template <class PolicySelector, class OffT>
__device__ __forceinline__ void poll_and_fold(
  TilePartialStateT* tile_partial_states,
  int tile_id,
  int& last_seen_tile_id,
  OffT& last_seen_prefix_run_count,
  OffT& last_seen_prefix_open_length,
  int lane_id,
  int& dense_mode,
  OffT& curr_prefix_run_count,
  OffT& curr_prefix_open_length)
{
  // adaptive poll: we decide the window size based on the density of the runs. this buys ~5% BWUtil
  // the 2 window sizes: 96 and 160 = 32 * 5 are decided by the # of SM on blackwell
  // i.e. since we know the residency is 1 CTA per SM, each generation is 148 tiles ahead
  // therefore, with window_size=96, we split it in 2. with window_size=160 we do it in one pass.
  if (dense_mode)
  // when it is dense, compute has a slower rate of publishing tile states. so we wait for a smaller window first and
  // fold it. as we fold the small window, more tiles in the next window are becoming ready, so we get some overlapping
  {
    poll_fold_windows<96, PolicySelector>(
      tile_partial_states,
      tile_id,
      last_seen_tile_id,
      last_seen_prefix_run_count,
      last_seen_prefix_open_length,
      lane_id,
      dense_mode);
  }
  else
  // when it is sparse, compute has a high rate of publishing tile states. so we just poll the big window at once
  {
    poll_fold_windows<32 * current_policy<PolicySelector>().lookahead.poll_loads_per_lane, PolicySelector>(
      tile_partial_states,
      tile_id,
      last_seen_tile_id,
      last_seen_prefix_run_count,
      last_seen_prefix_open_length,
      lane_id,
      dense_mode);
  }
  curr_prefix_run_count   = last_seen_prefix_run_count;
  curr_prefix_open_length = last_seen_prefix_open_length;
}

// we aim for 1 block/SM since it is easier to manage resources: we do not need to worry about occupancy anymore
template <typename PolicySelector, class KeyT, class LenT, class NumRunsT, class OffT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void device_rle_encode_lookahead_body(
  const KeyT* __restrict__ d_keys,
  KeyT* __restrict__ d_unique,
  LenT* __restrict__ d_counts,
  NumRunsT* __restrict__ d_num_runs,
  TilePartialStateT* __restrict__ tile_partial_states,
  OffT num_items,
  int num_tiles)
{
  static constexpr RleLookaheadPolicy policy = current_policy<PolicySelector>().lookahead;
  CUB_DETAIL_STATIC_ISH_ASSERT(16 % sizeof(KeyT) == 0, "KeyT size must be a power of two <= 16");
  CUB_DETAIL_STATIC_ISH_ASSERT(alignof(KeyT) <= 16, "Alignment <= 16");
  CUB_DETAIL_STATIC_ISH_ASSERT(
    policy.items_per_thread >= 1 && policy.items_per_thread <= 32, "items_per_thread must be in [1, 32]");
  CUB_DETAIL_STATIC_ISH_ASSERT(
    policy.compute_warps >= 1 && policy.compute_warps <= 31, "compute_warps must be in [1, 31]");
  CUB_DETAIL_STATIC_ISH_ASSERT(policy.store_warps == policy.compute_warps, "store_warps must equal compute_warps");
  CUB_DETAIL_STATIC_ISH_ASSERT(policy.key_ring_stages >= 1, "at least one pipeline stage");
  CUB_DETAIL_STATIC_ISH_ASSERT(policy.pos_ring_stages >= 1 && 2 * policy.pos_ring_stages >= policy.key_ring_stages,
                               "pos ring parity wait aliases unless 2*pos_ring_stages >= key_ring_stages");
  CUB_DETAIL_STATIC_ISH_ASSERT(policy.tile_size() <= 0xffff && policy.tile_size() <= 32768,
                               "tile_size must fit the 16-bit state words and signed 16-bit staged positions");
  CUB_DETAIL_STATIC_ISH_ASSERT(num_total_threads(policy) <= 1024, "a CTA is capped at 1024 threads");
  CUB_DETAIL_STATIC_ISH_ASSERT(
    policy.buf_per_lane() * ((int) sizeof(KeyT) + 4) <= 64, "reg-buf rounds must fit the 64B/lane register budget");
  CUB_DETAIL_STATIC_ISH_ASSERT(
    cuda::std::is_integral_v<OffT> && policy.tile_size() <= cuda::std::numeric_limits<OffT>::max(),
    "OffT must be an integer type wide enough for one tile");
  constexpr int items_per_thread       = policy.items_per_thread;
  constexpr int compute_warps          = policy.compute_warps;
  constexpr int store_warps            = policy.store_warps;
  constexpr int key_ring_stages        = policy.key_ring_stages;
  constexpr int pos_ring_stages        = policy.pos_ring_stages;
  constexpr int flag_staging_threshold = policy.flag_staging_threshold;
  constexpr int warp_tile_size         = policy.warp_tile_size();
  constexpr int tile_size              = policy.tile_size();
  constexpr int slot_pad               = policy.slot_pad((int) sizeof(KeyT));
  constexpr int slot_stride            = policy.slot_stride((int) sizeof(KeyT), (int) alignof(KeyT));
  using PrefixT                        = rle::encode::PrefixT<OffT>;
  // [key_ring_stages][tile_size] input keys
  // [key_ring_stages][tile_size] int16 staged head positions
  extern __shared__ char smem_raw[];
  KeyT* const tile_buf = (KeyT*) smem_raw;
  short* const pos_buf = (short*) (tile_buf + (size_t) key_ring_stages * slot_stride);
  __shared__ int tile_id_buf[key_ring_stages]; // which global tile each ring slot holds (LOAD gets it with try_cancel)
  __shared__ int warp_run_counts[key_ring_stages][compute_warps]; // per compute warp run counts
  __shared__ unsigned head_flag_buf[key_ring_stages][compute_warps * 32]; // staged head-flag words
  __shared__ int warp_first_heads[key_ring_stages][compute_warps]; // per compute warp first head idx (-1 if none)
  __shared__ int warp_last_heads[key_ring_stages][compute_warps]; // per compute warp last head idx (-1 if none)

  // for POLL to pass STORE packed [open_len_prefix:32][run_count_prefix:32]
  __shared__ PrefixT prefix_packed[key_ring_stages];

  // STORE --pos_buf_free--> COMPUTE staging (this is because we have the case where pos_ring_stages < key_ring_stages);
  // if it is mapped 1:1, then this would have been protected by empty / fall as well, but here we need an extra barrier
  __shared__ cuda::std::uint64_t pos_buf_free[pos_ring_stages];
  // LOAD --full--> COMPUTE & POLL
  // COMPUTE(all warps) --computed--> COMPUTE w0, then cw0 calculates & publishes this tile's aggregate to the global
  // POLL --prefixed--> STORE
  // STORE --empty--> LOAD & POLL
  __shared__ cuda::std::uint64_t full[key_ring_stages];
  __shared__ cuda::std::uint64_t computed[key_ring_stages], prefixed[key_ring_stages], empty[key_ring_stages];
  // COMPUTE warp w --staged_warp_tile[w]--> STORE: we arrive per warp tile handoff
  // i.e. store warps start working to drain a warp-tile as soon as ITS positions are staged
  __shared__ cuda::std::uint64_t staged_warp_tile[key_ring_stages][compute_warps];

  // try_cancel writes a 16-byte response into clc_resp + completes clc_bar's tx.
  __shared__ __align__(16) uint4 clc_resp;
  __shared__ cuda::std::uint64_t clc_bar;

  const int thr_id         = threadIdx.x;
  const int lane_id        = thr_id & 31;
  const int blk_id         = blockIdx.x;
  const unsigned base_skip = (alignof(KeyT) < 16) ? ((unsigned) (size_t) d_keys & 15u) : 0u;
  const int skip_elems     = (int) (base_skip / sizeof(KeyT));
  if (thr_id == 0)
  {
    for (int slot_id = 0; slot_id < key_ring_stages; ++slot_id)
    {
      ptx::mbarrier_init(&full[slot_id], 1);
      ptx::mbarrier_init(&computed[slot_id], compute_warps); // every compute warp arrives
      ptx::mbarrier_init(&prefixed[slot_id], 1);
      ptx::mbarrier_init(&empty[slot_id], store_warps + 1); // store warps + the bookkeeper
      for (int cw = 0; cw < compute_warps; ++cw)
      {
        ptx::mbarrier_init(&staged_warp_tile[slot_id][cw], 1); // that compute warp's lane0
      }
    }
    for (int p = 0; p < pos_ring_stages; ++p)
    {
      ptx::mbarrier_init(&pos_buf_free[p], store_warps);
    }

    ptx::mbarrier_init(&clc_bar, 1); // 1 arrival
  }
  // normal smem writes (e.g. mbarrier_init) go through the generic proxy
  // the TMA operations access shared memory through the async proxy. these are separate visibility domains,
  // so the init writes are not automatically visible to TMA.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();

  constexpr warpspeed::SquadDesc squadLoad{0, 1};
  constexpr warpspeed::SquadDesc squadCompute{1, compute_warps};
  constexpr warpspeed::SquadDesc squadPoll{2, 1};
  constexpr warpspeed::SquadDesc squadStore{3, store_warps};
  constexpr warpspeed::SquadDesc squadBookkeeper{4, 1};
  constexpr warpspeed::SquadDesc squads[] = {squadLoad, squadCompute, squadPoll, squadStore, squadBookkeeper};

  warpspeed::squadDispatch(
    warpspeed::getSpecialRegisters(), squads, [&](warpspeed::Squad squad) _CCCL_FORCEINLINE_LAMBDA {
      // if you are load
      if (squad == squadLoad)
      {
        // CLC tile assignment: gen0 tile = this CTA's launch id (blockIdx.x)
        int tile_id = blk_id;
        if (lane_id == 0)
        {
          // 16 is the try_cancel byte tx
          ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &clc_bar, 16);
          ptx::clusterlaunchcontrol_try_cancel(&clc_resp, &clc_bar);
        }
        for (int pipeline_gen = 0;; ++pipeline_gen)
        {
          const int slot_id  = pipeline_gen % key_ring_stages; // which slot is this?
          const int slot_gen = pipeline_gen / key_ring_stages; // how many times is this slot used?
          if (pipeline_gen >= key_ring_stages)
          {
            // need to wait for slot to be free
            wait_parity(&empty[slot_id], (unsigned) ((slot_gen - 1) & 1));
          }
          if (lane_id == 0)
          {
            tile_id_buf[slot_id] = tile_id;
          }
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&full[slot_id]);
            }
            __syncwarp();
            break;
          }
          // over-fetch one 16B chunk to the left, so that we get last tiles last key
          // tile 0 has no predecessor and skips the over-fetch
          const bool first_tile = (tile_id == 0);
          const int tile_len    = (int) min((OffT) tile_size, num_items - (OffT) tile_id * tile_size);
          load_tile_keys<tile_size, slot_pad>(
            tile_buf + (size_t) slot_id * slot_stride,
            d_keys,
            tile_id,
            tile_len,
            first_tile,
            tile_id == num_tiles - 1,
            base_skip,
            &full[slot_id],
            lane_id);
          // consume the prefetched cancel, this is ok since it should be fast to get next cancelled id
          tile_id = clc_next_tile_id(clc_resp, clc_bar, pipeline_gen, num_tiles, lane_id);
        }
      }
      // if you are compute
      else if (squad == squadCompute)
      {
        const int compute_warp_id  = squad.warpRank();
        const int warp_tile_offset = compute_warp_id * warp_tile_size;
        for (int pipeline_gen = 0;; ++pipeline_gen)
        {
          const int slot_id  = pipeline_gen % key_ring_stages;
          const int slot_gen = pipeline_gen / key_ring_stages;
          wait_parity(&full[slot_id], (unsigned) (slot_gen & 1));
          const int tile_id = tile_id_buf[slot_id];
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              // STORE waits computed + its warp-tile's staged_warp_tile, so arrive both
              ptx::mbarrier_arrive(&computed[slot_id]);
              ptx::mbarrier_arrive(&staged_warp_tile[slot_id][compute_warp_id]);
            }
            break;
          }
          // slot is ready!
          const KeyT* key_buf = tile_buf + (size_t) slot_id * slot_stride + slot_pad;
          const int tile_len  = (int) min((OffT) tile_size, num_items - (OffT) tile_id * tile_size);
          int local_run_count = 0, warp_first_head = -1, warp_last_head = -1;
          short* const pos_dst = pos_buf + (size_t) (pipeline_gen % pos_ring_stages) * tile_size;
          const unsigned my_flags =
            compute_head_flags<items_per_thread>(key_buf, warp_tile_offset, tile_len, tile_id, lane_id, skip_elems);
          local_run_count = __reduce_add_sync(full_mask, __popc(my_flags));
          // each lane in a warp now has a mask that tells which chunk is non empty
          const unsigned nonempty_chunk_mask = __ballot_sync(full_mask, my_flags != 0u);
          // if warptile is non empty (has heads), we get the location of warps first head and last head
          if (nonempty_chunk_mask)
          {
            const int first_chunk           = __ffs(nonempty_chunk_mask) - 1;
            const int last_chunk            = 31 - __clz(nonempty_chunk_mask);
            const unsigned first_chunk_mask = __shfl_sync(full_mask, my_flags, first_chunk);
            const unsigned last_chunk_mask  = __shfl_sync(full_mask, my_flags, last_chunk);
            warp_first_head                 = warp_tile_offset + first_chunk * 32 + (__ffs(first_chunk_mask) - 1);
            warp_last_head                  = warp_tile_offset + last_chunk * 32 + 31 - __clz(last_chunk_mask);
          }
          // now, we calculate warptile aggregates
          if (lane_id == 0)
          {
            warp_run_counts[slot_id][compute_warp_id]  = local_run_count;
            warp_first_heads[slot_id][compute_warp_id] = warp_first_head;
            warp_last_heads[slot_id][compute_warp_id]  = warp_last_head;
            ptx::mbarrier_arrive(&computed[slot_id]); // each compute warp arrives
          }
          // warp 0 waits all compute warp arrivals so that every warp's results are visible
          // then collect results from all warptiles and publish the tile run count and tile open len
          if (compute_warp_id == 0)
          {
            wait_parity(&computed[slot_id], (unsigned) (slot_gen & 1));
            reduce_and_publish_tile_state<compute_warps>(
              tile_partial_states, tile_id, tile_len, warp_run_counts[slot_id], warp_last_heads[slot_id], lane_id);
          }
          // now we start to stage head positions per warp tile, if a warptile has enough runs
          // (it is only worth it when we have more runs by a certain threshold per warp tile)
          // (otherwise, it is cheaper to recalculate positions from head_flags directly)
          const bool stage_flags = (local_run_count < flag_staging_threshold);
          if (stage_flags)
          {
            head_flag_buf[slot_id][compute_warp_id * 32 + lane_id] = my_flags;
          }
          // CRITICAL: When stage_flags is true, we skip waiting on the pos_buf barriers too. This buys 3% BWUtil in
          // some cells. Generally, skipping a wait like this would cause a race (phasebit can only encode parity).
          // But we can prove, when 2 * pos_ring_stages >= key_ring_stages, this race would not happen.
          // Proof. let's say P = pos_ring_stages and S = key_ring_stages, and we are at pipeline_generation g with
          // g / P = n. Let's say for g, stage_flags is false, so g is waiting for phase (n - 1) to complete, i.e.
          // the phase bit of the barrier is no longer (n - 1) % 2. If g - P waits and arrives properly, this is
          // sound. However, if g - P skipped the wait, when the barrier is no longer (n - 1) % 2, it could also mean
          // it just flipped from (n - 3) % 2, i.e. g - 2P could also be using the slot! This is a race/hazard:
          // For this slot, we have 3 dependence types:
          //   1. RAW: ST warps of gen g must read the slot after COMPUTE warps wrote them, i.e. write(g) must be
          //      before reads(g). This is protected by staged_warp_tile.
          //   2. WAR: COMPUTE warps of gen g must write after the ST warps finish reading g - P / g - 2P's, i.e.
          //      reads(g- P, g - 2P, ...) must happen before write(g). This is now unprotected after the wait is
          //      skipped. We are going to prove that with 2P >= S, this is guaranteed.
          //   3. WAW: write(g - 2P) must happen before write(g). This is guaranteed because each CW write to
          //      pos_dst[warp_tile_offset + swizzle_xor_stride32(run_idx)], i.e. each segment only has 1 writer.
          //      WAW is guaranteed by the progression of each CW.
          // On WAR: notice that when g is in flight, load's wait on empty(g - S) must have passed. This means for all
          // store warps STw, they must have arrived empty(h) for all h <= g - S. Given 2P >= S, they all must have
          // arrived empty(g - 2P). Since we always arrive pos_buf_free before empty, this means they all must have
          // arrived pos_buf_free(g - 2P) too. So there is no race.
          else
          {
            if constexpr (pos_ring_stages < key_ring_stages)
            {
              // the pos slot is shared by pipeline_gens g, g+pos_ring_stages, ...
              // need to wait for it to be cleared by STORE
              if (pipeline_gen >= pos_ring_stages)
              {
                wait_parity(&pos_buf_free[pipeline_gen % pos_ring_stages],
                            (unsigned) ((pipeline_gen / pos_ring_stages - 1) & 1));
              }
            }
            stage_head_positions<items_per_thread>(my_flags, pos_dst, warp_tile_offset, lane_id);
          } // stage flags
          __syncwarp();
          if (lane_id == 0)
          {
            ptx::mbarrier_arrive(&staged_warp_tile[slot_id][compute_warp_id]); // this warp-tile's positions ready
          }
        }
      }
      // if you are poll
      else if (squad == squadPoll)
      {
        int last_seen_tile_id             = 0;
        OffT last_seen_prefix_run_count   = 0;
        OffT last_seen_prefix_open_length = 0;
        int poll_dense_mode               = 1;
        for (int pipeline_gen = 0;; ++pipeline_gen)
        {
          const int slot_id  = pipeline_gen % key_ring_stages;
          const int slot_gen = pipeline_gen / key_ring_stages;
          wait_parity(&full[slot_id], (unsigned) (slot_gen & 1));
          const int tile_id = tile_id_buf[slot_id];
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&prefixed[slot_id]);
            }
            break;
          }
          OffT curr_prefix_run_count, curr_prefix_open_length;
          poll_and_fold<PolicySelector>(
            tile_partial_states,
            tile_id,
            last_seen_tile_id,
            last_seen_prefix_run_count,
            last_seen_prefix_open_length,
            lane_id,
            poll_dense_mode,
            curr_prefix_run_count,
            curr_prefix_open_length);
          __syncwarp();
          if (lane_id == 0)
          {
            prefix_packed[slot_id] = PrefixT::pack(curr_prefix_run_count, curr_prefix_open_length);
            ptx::mbarrier_arrive(&prefixed[slot_id]); // prefix ready, store may proceed
          }
        }
      }
      // if you are store
      else if (squad == squadStore)
      {
        const int store_warp_idx = squad.warpRank();
        for (int pipeline_gen = 0;; ++pipeline_gen)
        {
          const int slot_id = pipeline_gen % key_ring_stages;
          // wait for computed (1/3): all per-warp-tile metadata (run counts, first/last heads)
          wait_parity(&computed[slot_id], (unsigned) ((pipeline_gen / key_ring_stages) & 1));
          const int tile_id = tile_id_buf[slot_id];
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&empty[slot_id]);
            }
            break;
          }
          // store warps and compute warps are decoupled
          // fewer store warps than compute warps has no winning regime since warp slots are not scarce at 1 block/SM
          static_assert(store_warps >= compute_warps && store_warps % compute_warps == 0,
                        "store warps: a whole multiple of compute warps");
          // per-warp-tile run bases (lane i owns warp-tile i's count/base) and done BEFORE the wait on prefixed so they
          // overlap
          // lane i: run-count sum over warp-tiles [0, i) = where warp-tile i's runs begin within the tile
          const auto [lane_warp_tile_run_count, lane_runs_before_warp_tile] =
            scan_warp_tile_run_counts<compute_warps>(warp_run_counts[slot_id], lane_id);
          const KeyT* tile_keys = tile_buf + (size_t) slot_id * slot_stride + slot_pad;
          // staged positions
          const short* run_positions      = pos_buf + (size_t) (pipeline_gen % pos_ring_stages) * tile_size;
          const int warp_tile_id          = store_warp_idx;
          const int warp_tile_run_count   = __shfl_sync(full_mask, lane_warp_tile_run_count, warp_tile_id);
          const int runs_before_warp_tile = __shfl_sync(full_mask, lane_runs_before_warp_tile, warp_tile_id);
          // if our register budget allows it and it is worth it, we can buffer intermediate results in register
          // and arrive empty early. this buys 2.5% BWUtil at the worst segments
          if (warp_tile_run_count >= 1 && warp_tile_run_count < flag_staging_threshold)
          {
            // wait for staged_warp_tile (2/3)
            wait_parity(&staged_warp_tile[slot_id][warp_tile_id], (unsigned) ((pipeline_gen / key_ring_stages) & 1));
            constexpr int buf_per_lane = policy.buf_per_lane();
            KeyT buf_key[buf_per_lane];
            int buf_run_length[buf_per_lane];
            const int warp_tile_offset = warp_tile_id * warp_tile_size;
            const int num_rounds       = (warp_tile_run_count + 31) >> 5;
            const HeadFlagDecodeT dec(head_flag_buf[slot_id], warp_tile_id, lane_id);
#pragma unroll
            for (int it = 0; it < buf_per_lane; ++it)
            {
              if (it >= num_rounds)
              {
                break;
              }
              const int run_idx  = it * 32 + lane_id;
              const RunSpanT run = dec.decode_run(run_idx);
              buf_key[it]        = (run_idx < warp_tile_run_count)
                                   ? tile_keys[warp_tile_offset + run.head_pos_in_warp_tile + skip_elems]
                                   : KeyT{};
              buf_run_length[it] = run.next_head_pos - run.head_pos_in_warp_tile;
            }
            __syncwarp();
            if (lane_id == 0)
            {
              if constexpr (pos_ring_stages < key_ring_stages)
              {
                ptx::mbarrier_arrive(&pos_buf_free[pipeline_gen % pos_ring_stages]);
              }
            }
            // wait for prefixed (3/3)
            wait_parity(&prefixed[slot_id], (unsigned) ((pipeline_gen / key_ring_stages) & 1));
            const OffT global_runs_before_warp_tile = prefix_packed[slot_id].run_count() + runs_before_warp_tile;
#pragma unroll
            for (int it = 0; it < buf_per_lane; ++it)
            {
              if (it >= num_rounds)
              {
                break;
              }
              const int run_idx = it * 32 + lane_id;
              if (run_idx < warp_tile_run_count)
              {
                const OffT global_run_idx = global_runs_before_warp_tile + run_idx;
                d_unique[global_run_idx]  = buf_key[it];
                if (run_idx + 1 < warp_tile_run_count)
                {
                  d_counts[global_run_idx] = buf_run_length[it];
                }
              }
            }
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&empty[slot_id]);
            }
            continue;
          } // reg buf
          // if not reg buffed, we do the normal things, i.e. prefixed wait, then staged_warp_tile, then drain
          // wait for prefixed (2/3)
          wait_parity(&prefixed[slot_id], (unsigned) ((pipeline_gen / key_ring_stages) & 1));
          const OffT curr_prefix_run_count = prefix_packed[slot_id].run_count();
          // wait for staged_warp_tile (3/3)
          wait_parity(&staged_warp_tile[slot_id][warp_tile_id], (unsigned) ((pipeline_gen / key_ring_stages) & 1));
          // drain writes warp tile (warp_tile_id)'s staged output into the global arrays.
          // Per run: gather its key from the run's head position -> d_unique,
          // and write its length -> d_counts (= next run's head pos - this run's head pos).
          // The warp tile's last run spans into the next warp-tile, so its length is fixed up separately.
          const OffT global_runs_before_warp_tile = curr_prefix_run_count + runs_before_warp_tile;
          const int warp_tile_offset              = warp_tile_id * warp_tile_size;
#pragma unroll 2
          for (int run_idx = lane_id; run_idx < warp_tile_run_count; run_idx += 32)
          {
            const OffT global_run_idx = global_runs_before_warp_tile + run_idx;
            const int head_pos        = (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx)];
            d_unique[global_run_idx]  = tile_keys[head_pos + skip_elems]; // gather the run's key at its head position
            if (run_idx + 1 < warp_tile_run_count)
            {
              // within-warp delta (next head - this head); the last run is fixed separately
              const int run_length =
                (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx + 1)] - head_pos;
              d_counts[global_run_idx] = run_length;
            }
          }
          __syncwarp();
          if (lane_id == 0)
          {
            if constexpr (pos_ring_stages < key_ring_stages)
            {
              ptx::mbarrier_arrive(&pos_buf_free[pipeline_gen % pos_ring_stages]);
            }
            // store done, load may proceed!
            ptx::mbarrier_arrive(&empty[slot_id]);
          }
        }
      }
      // if you are the bookkeeper
      else
      {
        for (int pipeline_gen = 0;; ++pipeline_gen)
        {
          const int slot_id = pipeline_gen % key_ring_stages;
          wait_parity(&computed[slot_id], (unsigned) ((pipeline_gen / key_ring_stages) & 1));
          const int tile_id = tile_id_buf[slot_id];
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&empty[slot_id]);
            }
            break;
          }
          const int tile_len = (int) min((OffT) tile_size, num_items - (OffT) tile_id * tile_size);
          const bool is_last = (tile_id == num_tiles - 1);
          // same scan as the store warps (lane i = warp-tile i)
          const auto [lane_warp_tile_run_count, lane_runs_before_warp_tile] =
            scan_warp_tile_run_counts<compute_warps>(warp_run_counts[slot_id], lane_id);
          const int tile_total_runs =
            __shfl_sync(full_mask, lane_runs_before_warp_tile + lane_warp_tile_run_count, compute_warps - 1);
          const unsigned nonempty_warp_tiles_mask = __ballot_sync(full_mask, lane_warp_tile_run_count > 0);
          wait_parity(&prefixed[slot_id], (unsigned) ((pipeline_gen / key_ring_stages) & 1));
          const PrefixT packed_prefix        = prefix_packed[slot_id];
          const OffT curr_prefix_run_count   = packed_prefix.run_count();
          const OffT curr_prefix_open_length = packed_prefix.open_len();
          // per-warp-tile boundary: a warp-tile's last run is closed by the next nonempty warp-tile's
          // first head. lane L handles warp-tile L.
          if (lane_id < compute_warps && lane_warp_tile_run_count > 0)
          {
            const unsigned later_wts = nonempty_warp_tiles_mask >> (lane_id + 1); // nonempty warp-tiles after L
            const OffT last_run_global_idx =
              curr_prefix_run_count + lane_runs_before_warp_tile + lane_warp_tile_run_count - 1;
            if (later_wts)
            {
              const int next_wt             = lane_id + 1 + __ffs(later_wts) - 1;
              d_counts[last_run_global_idx] = warp_first_heads[slot_id][next_wt] - warp_last_heads[slot_id][lane_id];
            }
            else if (is_last)
            {
              // if we are the last warptile of the whole input, we end here
              d_counts[last_run_global_idx] = tile_len - warp_last_heads[slot_id][lane_id];
            }
            // else: this run is open in this tile, now this became a job for the next tile (see below)
          }
          __syncwarp();
          // now we need to finish last tile's open run
          if (lane_id == 0)
          {
            const bool any_head  = (nonempty_warp_tiles_mask != 0);
            const int first_head = any_head ? warp_first_heads[slot_id][__ffs(nonempty_warp_tiles_mask) - 1] : -1;
            // if our tile has a head, i.e. it stops here
            if (any_head && curr_prefix_run_count > 0)
            {
              d_counts[curr_prefix_run_count - 1] = curr_prefix_open_length + first_head;
            }
            // if we are last tile with no head: we have to close it here
            if (is_last && !any_head && curr_prefix_run_count > 0)
            {
              d_counts[curr_prefix_run_count - 1] = curr_prefix_open_length + tile_len;
            }
            // otherwise, next tile's problem
            if (is_last)
            {
              *d_num_runs = (NumRunsT) (curr_prefix_run_count + tile_total_runs);
            }
            ptx::mbarrier_arrive(&empty[slot_id]); // bookkeeping done, slot may recycle
          }
        }
      }
    });
}

// CUB temp storage is caller scratch with no contents contract between calls, so the states are
// cleared on EVERY launch (same as stock CUB's init kernels)
template <class StateT>
_CCCL_KERNEL_ATTRIBUTES void DeviceRleEncodeLookaheadInitKernel(StateT* states, long long n_states)
{
  const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_states)
  {
    states[i] = StateT{}; // tag 0 never matches the published tag (1)
  }
}

template <typename PolicySelector>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL int get_device_rle_encode_lookahead_launch_bounds() noexcept
{
  return num_total_threads(current_policy<PolicySelector>().lookahead);
}

// need a variable template for clang in CUDA mode to avoid:
// error: 'launch_bounds' attribute requires parameter 0 to be an integer constant
template <typename PolicySelector>
inline constexpr int device_rle_encode_lookahead_launch_bounds =
  get_device_rle_encode_lookahead_launch_bounds<PolicySelector>();

template <typename PolicySelector, class KeyT, class LenT, class NumRunsT, class OffT>
__launch_bounds__(device_rle_encode_lookahead_launch_bounds<PolicySelector>, 1)
  _CCCL_KERNEL_ATTRIBUTES void DeviceRleEncodeLookaheadKernel(
    const KeyT* __restrict__ d_keys,
    KeyT* __restrict__ d_unique,
    LenT* __restrict__ d_counts,
    NumRunsT* __restrict__ d_num_runs,
    TilePartialStateT* tile_partial_states,
    OffT num_items,
    int num_tiles)
{
  static constexpr RleEncodePolicy active_policy = current_policy<PolicySelector>();
  if constexpr (active_policy.algorithm == RleAlgorithm::lookahead)
  {
#if _CCCL_CUDACC_AT_LEAST(12, 8) && __cccl_ptx_isa >= 860
    NV_IF_TARGET(NV_PROVIDES_SM_100,
                 (device_rle_encode_lookahead_body<PolicySelector>(
                    d_keys, d_unique, d_counts, d_num_runs, tile_partial_states, num_items, num_tiles);))
#else // _CCCL_CUDACC_AT_LEAST(12, 8) && __cccl_ptx_isa >= 860
    static_assert(sizeof(KeyT) == 0,
                  "Implementation bug: Tuning policy selected lookahead, but CUDA compiler does not support it");
#endif // _CCCL_CUDACC_AT_LEAST(12, 8) && __cccl_ptx_isa >= 860
  }
  // for a lookback policy this kernel compiles to an empty stub: the fatbin carries this symbol for every
  // target architecture, and targets whose policy resolves to lookback must still compile (the host dispatch
  // never launches the kernel on such devices)
}
} // namespace detail::rle::encode

CUB_NAMESPACE_END
