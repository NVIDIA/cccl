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
#include <cub/warp/warp_scan.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__memory/ptr_rebind.h>
#include <cuda/ptx>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#if !_CCCL_HAS_NV_ATOMIC_BUILTINS()
#  include <cuda/atomic>
#endif // !_CCCL_HAS_NV_ATOMIC_BUILTINS()

CUB_NAMESPACE_BEGIN

namespace detail::rle::encode
{
// the kernel (and everything it needs) only exists from PTX ISA 9.2 (CUDA 13.2): the load warp requires the
// cp.async.bulk .ignore_oob qualifier. Below that, the dispatch layer compiles the lookahead path out entirely.
#if __cccl_ptx_isa >= 920
namespace ptx = ::cuda::ptx;

_CCCL_HOST_DEVICE_API constexpr int num_total_threads(const RleLookaheadPolicy& policy)
{
  const int num_total_warps =
    1 /*load*/ + policy.compute_warps + 1 /*poll*/ + policy.compute_warps /*store*/ + 1 /*bookkeeper*/;
  return num_total_warps * detail::warp_threads;
}

// This is important for position staging on dense cases (16 way bank conflicts).
_CCCL_DEVICE_API _CCCL_FORCEINLINE int swizzle_xor_stride32(int x)
{
  return x ^ (x >> detail::log2_warp_threads);
}

constexpr unsigned full_mask = 0xffffffffu;

_CCCL_DEVICE_API _CCCL_FORCEINLINE void wait_parity(::cuda::std::uint64_t* bar, unsigned parity)
{
  while (!ptx::mbarrier_try_wait_parity(bar, parity))
  {
  }
}

// stages is a runtime value now (we pick the ring depths at launch). now a runtime divide is super expensive
// and costs ~3-6% BWUtil, so now we have to maintain a "cursor" that is far more cheaper.
struct ring_cursor_t
{
  int slot        = 0;
  unsigned parity = 0;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void advance(int stages)
  {
    if (++slot == stages)
    {
      slot = 0;
      parity ^= 1u;
    }
  }
};

// tile_partial_states: one dword per tile, layout: u64 [published_tag:32][open_len:16][run_count:16]
// states are cleared by rle_init_states every launch, since we do not own temp storage!
// an aligned 64-bit access is already non-tearing, but atomic_ref doesn't hurt and has clear semantics
// Nan: we could use u32 layouts [ready_bit:1][open_len:15][run_count:16], but we choose to use u64 to
// 1. w << 32 is free (u64 is already split into 2 registers), so we save a bit of time (theoretically)
// 2. to use the same layout as warpspeed scan
inline constexpr ::cuda::std::uint32_t tile_published = 1u;

struct tile_partial_state_t
{
  ::cuda::std::uint64_t dword;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned published_tag() const
  {
    return static_cast<unsigned>(dword >> 32);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE int run_count() const
  {
    return static_cast<int>(dword & 0xffffu);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE int open_len() const
  {
    return static_cast<int>((dword >> 16) & 0xffffu);
  }

  static _CCCL_DEVICE_API _CCCL_FORCEINLINE tile_partial_state_t pack(int run_count, int open_len)
  {
    return {(static_cast<::cuda::std::uint64_t>(tile_published) << 32)
            | (static_cast<::cuda::std::uint64_t>(static_cast<unsigned>(open_len)) << 16)
            | static_cast<::cuda::std::uint64_t>(static_cast<unsigned>(run_count))};
  }
};

_CCCL_DEVICE_API _CCCL_FORCEINLINE void
publish_state(tile_partial_state_t* tile_state_arr, int tile_idx, int run_count, int open_len)
{
  ::cuda::std::uint64_t packed = tile_partial_state_t::pack(run_count, open_len).dword;
#  if _CCCL_HAS_NV_ATOMIC_BUILTINS()
  __nv_atomic_store(&tile_state_arr[tile_idx].dword, &packed, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
#  else // ^^^ _CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^ / vvv !_CCCL_HAS_NV_ATOMIC_BUILTINS() vvv
  ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device> a(tile_state_arr[tile_idx].dword);
  a.store(packed, ::cuda::memory_order_relaxed);
#  endif // !_CCCL_HAS_NV_ATOMIC_BUILTINS()
}

// return the state (even if not yet publish for this launch, caller checks it)
// we do not want to spin here
_CCCL_DEVICE_API _CCCL_FORCEINLINE tile_partial_state_t load_state(tile_partial_state_t* tile_state_arr, int tile_idx)
{
#  if _CCCL_HAS_NV_ATOMIC_BUILTINS()
  ::cuda::std::uint64_t dword;
  __nv_atomic_load(&tile_state_arr[tile_idx].dword, &dword, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
  return {dword};
#  else // ^^^ _CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^ / vvv !_CCCL_HAS_NV_ATOMIC_BUILTINS() vvv
  ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device> a(tile_state_arr[tile_idx].dword);
  return {a.load(::cuda::memory_order_relaxed)};
#  endif // !_CCCL_HAS_NV_ATOMIC_BUILTINS()
}

// CRITICAL: from choose_signed_offset, it is guaranteed that OffT covers the whole index space.
// Therefore, in the kernel, the type of the prefix (run_count, open_len) should always be OffT.
template <class OffT, bool = (sizeof(OffT) > 4)>
struct prefix_t;

template <class OffT>
struct prefix_t<OffT, false>
{
  ::cuda::std::uint64_t dword;

  static _CCCL_DEVICE_API _CCCL_FORCEINLINE prefix_t pack(OffT run_count, OffT open_len)
  {
    return {(static_cast<::cuda::std::uint64_t>(static_cast<unsigned>(open_len)) << 32)
            | static_cast<unsigned>(run_count)};
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE OffT run_count() const
  {
    return static_cast<OffT>(static_cast<unsigned>(dword & 0xffffffffull));
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE OffT open_len() const
  {
    return static_cast<OffT>(static_cast<unsigned>(dword >> 32));
  }
};

template <class OffT>
struct alignas(16) prefix_t<OffT, true>
{
  ::cuda::std::uint64_t packed_run_count;
  ::cuda::std::uint64_t packed_open_len;

  static _CCCL_DEVICE_API _CCCL_FORCEINLINE prefix_t pack(OffT run_count, OffT open_len)
  {
    return {static_cast<::cuda::std::uint64_t>(run_count), static_cast<::cuda::std::uint64_t>(open_len)};
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE OffT run_count() const
  {
    return static_cast<OffT>(packed_run_count);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE OffT open_len() const
  {
    return static_cast<OffT>(packed_open_len);
  }
};

// position of the n-th set bit of flag_mask, requires popc(flag_mask) > rank. Implementation is binary search.
// __fns(flag_mask, 0, rank+1) computes the same thing but has NO hardware op on sm_100a and is slower
// TODO (Nan): as per discussion with Federico, this could be in libcudacxx
_CCCL_DEVICE_API _CCCL_FORCEINLINE int nth_set_bit(unsigned flag_mask, int rank)
{
  // each step: if the wanted bit is not among the low half's set bits, skip that half entirely
  int bit_position = 0;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int half_width = 16; half_width >= 1; half_width /= 2)
  {
    const int set_bits_in_low_half = __popc(flag_mask & ((1u << half_width) - 1u));
    if (rank >= set_bits_in_low_half)
    {
      rank -= set_bits_in_low_half;
      bit_position += half_width;
      flag_mask >>= half_width;
    }
  }
  return bit_position;
}

struct warp_tile_run_scan_t
{
  int lane_run_count;
  int lane_runs_before;
};

// we need this because STORE and BOOKKEEPER both recalculate from slot_warp_run_counts
template <int ComputeWarps>
_CCCL_DEVICE_API _CCCL_FORCEINLINE warp_tile_run_scan_t
scan_warp_tile_run_counts(const int* slot_warp_run_counts, int lane_id)
{
  const int lane_run_count = (lane_id < ComputeWarps) ? slot_warp_run_counts[lane_id] : 0;
  typename WarpScan<int>::TempStorage warp_scan_storage;
  int lane_scan;
  WarpScan<int>(warp_scan_storage).ExclusiveSum(lane_run_count, lane_scan);
  return {lane_run_count, lane_scan};
}

template <int tile_size, int slot_pad, class KeyT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void load_tile_keys(
  KeyT* slot,
  const KeyT* d_keys,
  int tile_id,
  int tile_len,
  bool first_tile,
  bool last_tile,
  unsigned base_skip,
  ::cuda::std::uint64_t* full_bar,
  int lane_id,
  bool keys_staged)
{
  if (lane_id == 0)
  {
    if (!keys_staged)
    {
      // vvv regressed case: no TMA; full now only means "tile_id_buf[slot] is valid" vvv
      ptx::mbarrier_arrive(full_bar);
      // ^^^ regressed case ^^^
    }
    else
    {
      // if it is not first tile, we overcopy 16B to the left to get last key from last tile
      const unsigned nbytes = static_cast<unsigned>((tile_len + (first_tile ? 0 : slot_pad)) * int{sizeof(KeyT)});
      const unsigned span_bytes =
        (nbytes + base_skip + (detail::bulk_copy_min_align - 1)) & ~unsigned{detail::bulk_copy_min_align - 1};
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, full_bar, span_bytes);
      ptx::cp_async_bulk_ignore_oob(
        ptx::space_shared,
        ptx::space_global,
        slot + (first_tile ? slot_pad : 0),
        ::cuda::ptr_rebind<KeyT>(
          ::cuda::ptr_rebind<char>(d_keys + static_cast<size_t>(tile_id) * tile_size - (first_tile ? 0 : slot_pad))
          - base_skip),
        span_bytes,
        first_tile ? base_skip : 0u,
        last_tile ? (span_bytes - base_skip - nbytes) : 0u,
        full_bar);
    }
  }
  __syncwarp();
}

// CLC is fast, so folding it into LOAD won't cost performance, and it saves us 32 threads
_CCCL_DEVICE_API _CCCL_FORCEINLINE int
clc_next_tile_id(uint4& clc_resp, ::cuda::std::uint64_t& clc_bar, int pipeline_gen, int num_tiles, int lane_id)
{
  int next = num_tiles; // if no more work was cancellable
  if (lane_id == 0)
  {
    wait_parity(&clc_bar, static_cast<unsigned>(pipeline_gen & 1));
    // try_cancel wrote clc_resp via the async proxy
    // TODO(nan): possibly unnecessary; the mbarrier try_wait visibility guarantee is documented for cp.async.bulk
    // but not for CLC (doc gap) -- keep the defensive fence until the PTX docs or gonzalobg confirm
    ptx::fence_proxy_async(ptx::space_shared);
    const uint4 resp_snapshot = clc_resp;
    ptx::fence_proxy_async(ptx::space_shared);
    const bool canceled = ptx::clusterlaunchcontrol_query_cancel_is_canceled(resp_snapshot);
    if (canceled)
    {
      next = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(resp_snapshot);
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &clc_bar, 16);
      ptx::clusterlaunchcontrol_try_cancel(&clc_resp, &clc_bar);
    }
  }
  return __shfl_sync(full_mask, next, 0);
}

// calculate head_flags: each iter is 32 consecutive elements (lane L owns loc = warp_tile_offset + iter*32 + L)
// head = (key != predecessor)
template <int ItemsPerThread, bool KeysStaged, class KeyT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned compute_head_flags(
  const KeyT* key_buf, int warp_tile_offset, int tile_len, int tile_id, int lane_id, [[maybe_unused]] int skip_elems)
{
  static_assert(ItemsPerThread <= 32, "one lane per iter requires ItemsPerThread<=32");
  unsigned my_flags = 0;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int iter = 0; iter < ItemsPerThread; ++iter)
  {
    // each iteration handles one 32-element chunk of the warp tile; lane i compares element i of the chunk
    const int loc = warp_tile_offset + iter * detail::warp_threads + lane_id;
    int key_idx;
    int pred_idx;
    if constexpr (KeysStaged)
    {
      key_idx  = loc + skip_elems;
      pred_idx = loc + skip_elems - 1; // loc==0 reads the over fetched slot[slot_pad-1]
    }
    else
    {
      // vvv regressed case: plain global loads have no ignore_oob, so clamp the tail reads into the input.
      // the clamped values are garbage, but (loc < tile_len) below already zeroes those heads vvv
      key_idx  = (::cuda::std::min) (loc, tile_len - 1);
      pred_idx = (tile_id == 0) ? (::cuda::std::max) (key_idx - 1, 0) : key_idx - 1;
      // ^^^ regressed case ^^^
    }

    // a head = key differs from its predecessor; the global first element is always a head
    const KeyT key            = key_buf[key_idx];
    const KeyT pred           = key_buf[pred_idx];
    const int is_global_first = (tile_id == 0 && loc == 0);
    const int head            = (loc < tile_len) ? (is_global_first ? 1 : !(key == pred)) : 0;

    // gather the chunk's 32 head bits into one flag word; lane i keeps chunk i's word
    const unsigned flags = __ballot_sync(full_mask, head);
    if (lane_id == iter)
    {
      my_flags = flags;
    }
  }
  return my_flags;
}

template <int compute_warps>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void reduce_and_publish_tile_state(
  tile_partial_state_t* tile_partial_states,
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
  const unsigned warps_with_runs = __ballot_sync(full_mask, warp_run_count > 0);
  // CRITICAL: publish as soon as possible, this is why we calculate head_flags first
  // headless tile deserves a special branch because it is very common when problem is sparse
  if (warps_with_runs)
  {
    const int last_warp_with_runs = 31 - ::cuda::std::countl_zero(warps_with_runs);
    if (lane_id == last_warp_with_runs)
    {
      const int open_len = tile_len - slot_warp_last_heads[lane_id];
      publish_state(tile_partial_states, tile_id, run_count, open_len);
    }
  }
  else if (lane_id == 0)
  {
    publish_state(tile_partial_states, tile_id, run_count, tile_len);
  }
}

template <int ItemsPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
stage_head_positions(unsigned my_flags, position_t* pos_dst, int warp_tile_offset, int lane_id)
{
  // we store run R at warp_tile_offset + (R ^ (R>>5)) to avoid bank conflicts for dense cases
  // (CRITICAL for MaxSeg=1,2,4)
  const int word_run_count = __popc(my_flags); // this word's head count
  typename WarpScan<int>::TempStorage warp_scan_storage;
  int runs_before_word;
  // runs_before_word is a running sum of run_count, so each lane know each chunk's base
  WarpScan<int>(warp_scan_storage).ExclusiveSum(word_run_count, runs_before_word);
  if (lane_id < ItemsPerThread)
  {
    const int word_pos = warp_tile_offset + lane_id * detail::warp_threads; // element position of bit 0 of this word
    unsigned pending_heads = my_flags; // this word's head mask; we need to "peel" it headbit by headbit
    for (int run_index = runs_before_word; pending_heads; ++run_index)
    {
      const int head_offset = __ffs(static_cast<int>(pending_heads)) - 1; // offset (0..31) of the next head within the
                                                                          // word
      pos_dst[warp_tile_offset + swizzle_xor_stride32(run_index)] = static_cast<position_t>(word_pos + head_offset);
      pending_heads &= (pending_heads - 1); // clear the lowest set bit
    }
  }
}

struct run_span_t
{
  int head_pos_in_warp_tile;
  int next_head_pos;
};

// the compute warp may deem this warp tile too sparse to be worth the position-staging, and in that case it will write
// only the 32 head-flag words. Then, it is up to the store warps to "decode" the positions from the headflags.
// one warp tile is 32 chunks x 32 elements, so lane i owns word i.
// This buys 2.5% BWUtil in the MaxSegSize{2^4, 2^6, 2^8}
template <int ItemsPerThread>
struct head_flag_decode_t
{
  unsigned lane_head_flag_word;
  int lane_runs_before_word;
  int lane_first_head_from_word;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE head_flag_decode_t(const unsigned* slot_head_flags, int warp_tile_id, int lane_id)
  {
    lane_head_flag_word           = slot_head_flags[warp_tile_id * detail::warp_threads + lane_id];
    const int lane_word_run_count = __popc(lane_head_flag_word);
    typename WarpScan<int>::TempStorage warp_scan_storage;
    WarpScan<int>(warp_scan_storage).ExclusiveSum(lane_word_run_count, lane_runs_before_word);
    // lane i: # of runs starting in head_flag words [0, i), i.e. in elements [0, i*32)
    // lane i -> first head position in head flag words [i, 32)
    // if our own run_count is >0, the head is here!
    // empty should be +infinity, since we use min
    lane_first_head_from_word = lane_word_run_count ? (lane_id * 32 + __ffs(lane_head_flag_word) - 1) : 0x7fffffff;
    // if not, we loop to find the next head in flag word [i, 32). this is just a fold with min
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int offset = 1; offset < 32; offset <<= 1)
    {
      const int shuffled_first_head = __shfl_down_sync(full_mask, lane_first_head_from_word, offset);
      lane_first_head_from_word =
        (::cuda::std::min) (lane_first_head_from_word, (lane_id + offset < 32) ? shuffled_first_head : 0x7fffffff);
    }
    // now, lane i holds the next head in [i, 32). we precalculate this in parallel
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE run_span_t decode_run(int run_idx) const
  {
    // first question: which head_flag word contains my run's (run_idx) head?
    // lane_runs_before_word's row i = number of heads in words [0, i)
    // the word containing run_dex is then the largest i with runs_before(i) that is <= j
    // we do binary search over the distributed lane_runs_before_word table held across the warp
    int flag_word_idx = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int step = 16; step; step >>= 1)
    {
      // propose candidate
      const int candidate_word_idx = flag_word_idx + step;
      // read the i'th row
      const int candidate_runs_before = __shfl_sync(full_mask, lane_runs_before_word, candidate_word_idx);
      if (candidate_word_idx < ItemsPerThread && candidate_runs_before <= run_idx)
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
    // NOTE: for the last run head in warp tile, next_head_pos is garbage
    return {head_pos_in_warp_tile, next_head_pos};
  }
};

template <int window_size_cap, class PolicySelector, class OffT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void poll_fold_windows(
  tile_partial_state_t* tile_partial_states,
  int tile_id,
  int& first_unseen_tile_id,
  OffT& last_seen_prefix_run_count,
  OffT& last_seen_prefix_open_length,
  int lane_id,
  int& dense_mode)
{
  constexpr int poll_items_per_thread = current_policy<PolicySelector>().lookahead.poll_items_per_thread;
  static_assert(window_size_cap >= 1 && window_size_cap <= detail::warp_threads * poll_items_per_thread,
                "the fold window must be covered by the lanes");
  while (first_unseen_tile_id < tile_id)
  {
    const int remain = tile_id - first_unseen_tile_id;
    // # of tiles to fold this iteration
    const int window_size     = (::cuda::std::min) (remain, window_size_cap);
    const int lane_tile_count = (window_size - lane_id + (detail::warp_threads - 1)) >> detail::log2_warp_threads;
    tile_partial_state_t packed_words[poll_items_per_thread] = {}; // must zero initialize

    bool ready;
    // first, all tile state in window must be ready
    do
    {
      ready = true;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < poll_items_per_thread; ++i)
      {
        // we only try if that state is not published
        if (i < lane_tile_count && packed_words[i].published_tag() != tile_published)
        {
          packed_words[i] =
            load_state(tile_partial_states, first_unseen_tile_id + (i * detail::warp_threads + lane_id));
          if (packed_words[i].published_tag() != tile_published)
          {
            ready = false;
          }
        }
      }
    } while (__any_sync(full_mask, !ready));

    constexpr int tile_size = current_policy<PolicySelector>().lookahead.tile_size();
    int lane_run_count = 0, lane_last_packed = -1;
    // now, we fold the window
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < poll_items_per_thread; ++i)
    {
      if (i < lane_tile_count)
      {
        // aggregate run_count per lane, this is fine since run_count is commutative
        lane_run_count += packed_words[i].run_count();
        // nominate the highest tile id with runs, carrying its open_len in the low bits
        const int packed = ((i * detail::warp_threads + lane_id) << 16) | packed_words[i].open_len();
        lane_last_packed = (packed_words[i].run_count() > 0) ? packed : lane_last_packed;
      }
    }
    // vote for the highest tile id with runs
    const int last_packed = __reduce_max_sync(full_mask, lane_last_packed);

    // reduce across the warp, then roll this window into the running prefix
    const int window_run_count = __reduce_add_sync(full_mask, lane_run_count);
    // every folded tile is full, and run-less tiles publish open_len == tile_size, so the open length is the
    // last run-carrying tile's open_len plus tile_size for each tile after it
    const int window_open_length =
      (last_packed < 0)
        ? window_size * tile_size
        : (last_packed & 0xffff) + (window_size - 1 - (last_packed >> 16)) * tile_size;
    // dense_mode is true if the window has more than dense_mode_runs_per_tile runs per tile
    dense_mode = window_run_count > window_size * current_policy<PolicySelector>().lookahead.dense_mode_runs_per_tile;
    // combine last_seen_prefix with the window_size aggregate
    const OffT new_run_count     = last_seen_prefix_run_count + window_run_count;
    const OffT new_open_length   = (window_run_count > 0) ? static_cast<OffT>(window_open_length)
                                                          : (last_seen_prefix_open_length + window_open_length);
    last_seen_prefix_run_count   = new_run_count;
    last_seen_prefix_open_length = new_open_length;
    first_unseen_tile_id += window_size;
  }
}

template <class PolicySelector, class OffT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void poll_and_fold(
  tile_partial_state_t* tile_partial_states,
  int tile_id,
  int& first_unseen_tile_id,
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
  {
    // when it is dense, compute has a slower rate of publishing tile states. so we wait for a smaller window first and
    // fold it. as we fold the small window, more tiles in the next window are becoming ready, so we get some
    // overlapping
    poll_fold_windows<detail::warp_threads * current_policy<PolicySelector>().lookahead.dense_poll_items_per_thread,
                      PolicySelector>(
      tile_partial_states,
      tile_id,
      first_unseen_tile_id,
      last_seen_prefix_run_count,
      last_seen_prefix_open_length,
      lane_id,
      dense_mode);
  }
  else
  {
    // when it is sparse, compute has a high rate of publishing tile states. so we just poll the big window at once
    poll_fold_windows<detail::warp_threads * current_policy<PolicySelector>().lookahead.poll_items_per_thread,
                      PolicySelector>(
      tile_partial_states,
      tile_id,
      first_unseen_tile_id,
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
  tile_partial_state_t* __restrict__ tile_partial_states,
  OffT num_items,
  int num_tiles,
  int key_ring_stages,
  int pos_ring_stages,
  bool keys_staged)
{
  static constexpr RleLookaheadPolicy policy = current_policy<PolicySelector>().lookahead;
  static_assert(16 % sizeof(KeyT) == 0, "KeyT size must be a power of two <= 16");
  static_assert(alignof(KeyT) <= 16, "Alignment <= 16");

  static_assert(policy.items_per_thread >= 1 && policy.items_per_thread <= 32, "items_per_thread must be in [1, 32]");
  static_assert(policy.compute_warps >= 1 && policy.compute_warps <= 31, "compute_warps must be in [1, 31]");
  static_assert(policy.key_ring_stages >= 1, "at least one pipeline stage");

  static_assert(
    policy.pos_ring_stages >= 1
      && RleLookaheadPolicy::max_key_stages_per_pos_stage * policy.pos_ring_stages >= policy.key_ring_stages,
    "pos ring parity wait aliases unless 2*pos_ring_stages >= key_ring_stages");
  static_assert(policy.floor_pos_ring_stages() <= policy.pos_ring_stages
                  && RleLookaheadPolicy::max_key_stages_per_pos_stage * policy.floor_pos_ring_stages()
                       >= policy.floor_key_ring_stages(),
                "the unstaged floor configuration must satisfy the pos ring parity bound");
  static_assert(policy.floor_dyn_smem_bytes() + RleLookaheadPolicy::static_smem_budget <= detail::max_smem_per_block,
                "the unstaged floor configuration must launch within the default shared memory limit on every device");

  static_assert(
    policy.tile_size() <= 0xffff && policy.tile_size() - 1 <= ::cuda::std::numeric_limits<position_t>::max(),
    "tile_size must fit the 16-bit state words and the staged position type");
  static_assert(policy.poll_items_per_thread >= 3 && policy.poll_items_per_thread <= 32,
                "poll_items_per_thread must be in [3, 32] so the fold windows cover the dense cap and the int "
                "open-length accumulator cannot overflow");

  static_assert(num_total_threads(policy) <= 1024, "a CTA is capped at 1024 threads");
  static_assert(policy.decode_items_per_thread() * int{sizeof(KeyT) + sizeof(int)} <= 64,
                "reg-buf rounds must fit the 64B/lane register budget");
  static_assert(
    (::cuda::std::is_same_v<OffT, ::cuda::std::int32_t> || ::cuda::std::is_same_v<OffT, ::cuda::std::int64_t>)
      && policy.tile_size() <= ::cuda::std::numeric_limits<OffT>::max(),
    "OffT must be the internal signed 32/64-bit offset type, wide enough for one tile");
  constexpr int items_per_thread    = policy.items_per_thread;
  constexpr int compute_warps       = policy.compute_warps;
  constexpr int store_warps         = policy.compute_warps; // one store warp drains each compute warp's tile
  constexpr int max_key_ring_stages = policy.key_ring_stages;
  constexpr int max_pos_ring_stages = policy.pos_ring_stages;
  _CCCL_ASSERT(key_ring_stages >= 1 && key_ring_stages <= max_key_ring_stages, "invalid key_ring_stages");
  _CCCL_ASSERT(pos_ring_stages >= 1 && pos_ring_stages <= max_pos_ring_stages, "invalid pos_ring_stages");
  _CCCL_ASSERT(RleLookaheadPolicy::max_key_stages_per_pos_stage * pos_ring_stages >= key_ring_stages,
               "pos ring parity wait aliases");
  constexpr int flag_staging_threshold = policy.flag_staging_threshold;
  // in the regressed case: always stage positions, so the store warps never run the flag-decode drain vvv
  const int staging_threshold  = keys_staged ? flag_staging_threshold : 0;
  constexpr int warp_tile_size = policy.warp_tile_size();
  constexpr int tile_size      = policy.tile_size();
  constexpr int slot_pad       = policy.slot_pad(static_cast<int>(sizeof(KeyT)));
  constexpr int slot_stride    = policy.slot_stride(static_cast<int>(sizeof(KeyT)), static_cast<int>(alignof(KeyT)));
  using prefix_t               = rle::encode::prefix_t<OffT>;
  // [key_ring_stages][tile_size] input keys
  // [key_ring_stages][tile_size] int16 staged head positions
  extern __shared__ char smem_raw[];
  KeyT* const tile_buf = reinterpret_cast<KeyT*>(smem_raw);
  // when keys are not staged, the positions ring sits at the base
  position_t* const pos_buf =
    reinterpret_cast<position_t*>(tile_buf + (keys_staged ? static_cast<size_t>(key_ring_stages) * slot_stride : 0));
  __shared__ int tile_id_buf[max_key_ring_stages]; // which global tile each ring slot holds (LOAD gets it with
                                                   // try_cancel)
  __shared__ int warp_run_counts[max_key_ring_stages][compute_warps]; // per compute warp run counts
  __shared__ unsigned head_flag_buf[max_key_ring_stages][compute_warps * detail::warp_threads]; // staged head-flag
                                                                                                // words
  __shared__ int warp_first_heads[max_key_ring_stages][compute_warps]; // per compute warp first head idx (-1 if none)
  __shared__ int warp_last_heads[max_key_ring_stages][compute_warps]; // per compute warp last head idx (-1 if none)

  // for POLL to pass STORE packed [open_len_prefix:32][run_count_prefix:32]
  __shared__ prefix_t prefix_packed[max_key_ring_stages];

  // STORE --pos_buf_free--> COMPUTE staging (this is because we have the case where pos_ring_stages < key_ring_stages);
  // if it is mapped 1:1, then this would have been protected by empty / fall as well, but here we need an extra barrier
  __shared__ ::cuda::std::uint64_t pos_buf_free[max_pos_ring_stages];
  // barrier dependency graph, per key-ring slot; A = arrives, W = waits (arrival counts in the init loop below)
  //
  // spine:  LOAD --> {COMPUTE, POLL} --> {STORE, BOOKKEEPER} --> LOAD  (slot recycles)
  //
  //                       LOAD   COMPUTE(all)   POLL   STORE(all)   BOOKKEEPER
  // clc_bar               A,W                                                    next stolen tile id, 1 in flight
  // full                   A          W          W
  // computed                         A,W(w0)             W             W        w0 then publishes the tile aggregate
  // staged_warp_tile[w]               A                  W                       per-warp-tile handoff to its drainer
  // prefixed                                     A       W             W
  // pos_buf_free                      W                  A                       staging gate, pos ring is shallower
  // empty                  W                              A             A        POLL never waits on empty: it is
  //                                                                              transitively gated by LOAD's next full
  __shared__ ::cuda::std::uint64_t full[max_key_ring_stages];
  __shared__ ::cuda::std::uint64_t computed[max_key_ring_stages], prefixed[max_key_ring_stages],
    empty[max_key_ring_stages];
  // COMPUTE warp w --staged_warp_tile[w]--> STORE: we arrive per warp tile handoff
  // i.e. store warps start working to drain a warp-tile as soon as ITS positions are staged
  __shared__ ::cuda::std::uint64_t staged_warp_tile[max_key_ring_stages][compute_warps];

  // try_cancel writes a 16-byte response into clc_resp + completes clc_bar's tx.
  __shared__ __align__(16) uint4 clc_resp;
  __shared__ ::cuda::std::uint64_t clc_bar;
  static_assert(
    sizeof(tile_id_buf) + sizeof(warp_run_counts) + sizeof(head_flag_buf) + sizeof(warp_first_heads)
        + sizeof(warp_last_heads) + sizeof(prefix_packed) + sizeof(pos_buf_free) + sizeof(full) + sizeof(computed)
        + sizeof(prefixed) + sizeof(empty) + sizeof(staged_warp_tile) + sizeof(clc_resp) + sizeof(clc_bar)
      <= RleLookaheadPolicy::static_smem_budget,
    "static shared memory exceeds the budget assumed by the floor launch guarantee");

  const int tid     = threadIdx.x;
  const int lane_id = tid & 31;
  const int bid     = blockIdx.x;
  const unsigned base_skip =
    (int{alignof(KeyT)} < detail::bulk_copy_min_align)
      ? (static_cast<unsigned>(::cuda::std::bit_cast<::cuda::std::uintptr_t>(d_keys))
         & unsigned{detail::bulk_copy_min_align - 1})
      : 0u;
  const int skip_elems = static_cast<int>(base_skip / sizeof(KeyT));
  if (tid == 0)
  {
    for (int slot_id = 0; slot_id < max_key_ring_stages; ++slot_id)
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
    for (int p = 0; p < max_pos_ring_stages; ++p)
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
        int tile_id = bid;
        if (lane_id == 0)
        {
          // 16 is the try_cancel byte tx
          ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &clc_bar, 16);
          ptx::clusterlaunchcontrol_try_cancel(&clc_resp, &clc_bar);
        }
        ring_cursor_t key_ring;
        for (int pipeline_gen = 0;; ++pipeline_gen, key_ring.advance(key_ring_stages))
        {
          const int slot_id = key_ring.slot; // which slot is this?
          if (pipeline_gen >= key_ring_stages)
          {
            // need to wait for slot to be free
            wait_parity(&empty[slot_id], key_ring.parity ^ 1u);
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
          const int tile_len    = static_cast<int>(
            (::cuda::std::min) (static_cast<OffT>(tile_size), num_items - static_cast<OffT>(tile_id) * tile_size));
          load_tile_keys<tile_size, slot_pad>(
            tile_buf + static_cast<size_t>(slot_id) * slot_stride,
            d_keys,
            tile_id,
            tile_len,
            first_tile,
            tile_id == num_tiles - 1,
            base_skip,
            &full[slot_id],
            lane_id,
            keys_staged);
          // consume the prefetched cancel, this is ok since it should be fast to get next cancelled id
          tile_id = clc_next_tile_id(clc_resp, clc_bar, pipeline_gen, num_tiles, lane_id);
        }
      }
      // if you are compute
      else if (squad == squadCompute)
      {
        const int compute_warp_id  = squad.warpRank();
        const int warp_tile_offset = compute_warp_id * warp_tile_size;
        ring_cursor_t key_ring;
        ring_cursor_t pos_ring;
        for (int pipeline_gen = 0;;
             ++pipeline_gen, key_ring.advance(key_ring_stages), pos_ring.advance(pos_ring_stages))
        {
          const int slot_id = key_ring.slot;
          wait_parity(&full[slot_id], key_ring.parity);
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
          const int tile_len = static_cast<int>(
            (::cuda::std::min) (static_cast<OffT>(tile_size), num_items - static_cast<OffT>(tile_id) * tile_size));
          int local_run_count = 0, warp_first_head = -1, warp_last_head = -1;
          position_t* const pos_dst = pos_buf + static_cast<size_t>(pos_ring.slot) * tile_size;
          unsigned my_flags;
          if (keys_staged)
          {
            const KeyT* key_buf = tile_buf + static_cast<size_t>(slot_id) * slot_stride + slot_pad;
            my_flags            = compute_head_flags<items_per_thread, true>(
              key_buf, warp_tile_offset, tile_len, tile_id, lane_id, skip_elems);
          }
          else
          {
            // vvv regressed case: we load compute flags straight from global vvv
            const KeyT* key_buf = d_keys + static_cast<size_t>(tile_id) * tile_size;
            my_flags =
              compute_head_flags<items_per_thread, false>(key_buf, warp_tile_offset, tile_len, tile_id, lane_id, 0);
            // ^^^ regressed case ^^^
          }
          local_run_count = __reduce_add_sync(full_mask, __popc(my_flags));
          // each lane in a warp now has a mask that tells which chunk is non empty
          const unsigned nonempty_chunk_mask = __ballot_sync(full_mask, my_flags != 0u);
          // if warptile is non empty (has heads), we get the location of warps first head and last head
          if (nonempty_chunk_mask)
          {
            const int first_chunk = __ffs(nonempty_chunk_mask) - 1;
            const int last_chunk  = 31 - ::cuda::std::countl_zero(nonempty_chunk_mask);
            // measured alternative: letting lanes first_chunk/last_chunk store the heads directly (skipping both
            // shuffles) regresses ~1%: the conditional stores plus the extra __syncwarp needed before the computed
            // arrive cost more than the two shuffles
            const unsigned first_chunk_mask = __shfl_sync(full_mask, my_flags, first_chunk);
            const unsigned last_chunk_mask  = __shfl_sync(full_mask, my_flags, last_chunk);
            _CCCL_ASSERT((first_chunk_mask != 0u) && (last_chunk_mask != 0u), "nonempty chunk with an empty mask");
            warp_first_head = warp_tile_offset + first_chunk * 32 + (__ffs(first_chunk_mask) - 1);
            warp_last_head  = warp_tile_offset + last_chunk * 32 + 31 - ::cuda::std::countl_zero(last_chunk_mask);
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
            wait_parity(&computed[slot_id], key_ring.parity);
            reduce_and_publish_tile_state<compute_warps>(
              tile_partial_states, tile_id, tile_len, warp_run_counts[slot_id], warp_last_heads[slot_id], lane_id);
          }
          // now we start to stage head positions per warp tile, if a warptile has enough runs
          // (it is only worth it when we have more runs by a certain threshold per warp tile)
          // (otherwise, it is cheaper to recalculate positions from head_flags directly)
          // Paired with STORE's predicate; they intentionally disagree at run_count == 0 (see there).
          const bool stage_flags = (local_run_count < staging_threshold);
          if (pos_ring_stages < key_ring_stages)
          {
            // the pos slot is shared by pipeline_gens g, g+pos_ring_stages, ...
            // need to wait for it to be cleared by STORE
            if (pipeline_gen >= pos_ring_stages)
            {
              wait_parity(&pos_buf_free[pos_ring.slot], pos_ring.parity ^ 1u);
            }
          }
          if (stage_flags)
          {
            head_flag_buf[slot_id][compute_warp_id * detail::warp_threads + lane_id] = my_flags;
          }
          else
          {
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
        // running prefix state: everything folded so far, i.e. tiles [0, first_unseen_tile_id)
        int first_unseen_tile_id          = 0;
        OffT last_seen_prefix_run_count   = 0;
        OffT last_seen_prefix_open_length = 0;
        int poll_dense_mode               = 1;
        ring_cursor_t key_ring;
        for (int pipeline_gen = 0;; ++pipeline_gen, key_ring.advance(key_ring_stages))
        {
          const int slot_id = key_ring.slot;
          wait_parity(&full[slot_id], key_ring.parity);
          const int tile_id = tile_id_buf[slot_id];
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&prefixed[slot_id]);
            }
            break;
          }

          // fold every predecessor tile's published state into this tile's exclusive prefix
          OffT curr_prefix_run_count, curr_prefix_open_length;
          poll_and_fold<PolicySelector>(
            tile_partial_states,
            tile_id,
            first_unseen_tile_id,
            last_seen_prefix_run_count,
            last_seen_prefix_open_length,
            lane_id,
            poll_dense_mode,
            curr_prefix_run_count,
            curr_prefix_open_length);
          __syncwarp();

          // hand the prefix to the STORE warps and the bookkeeper
          if (lane_id == 0)
          {
            prefix_packed[slot_id] = prefix_t::pack(curr_prefix_run_count, curr_prefix_open_length);
            // CRITICAL: POLL doesn't participate in empty. This arrive gates STORE -> empty -> LOAD's
            // rewrite of the slot, so all reads of slot_id must precede it. Same chain bounds full's phase.
            ptx::mbarrier_arrive(&prefixed[slot_id]); // prefix ready, store may proceed
          }
        }
      }
      // if you are store
      else if (squad == squadStore)
      {
        const int store_warp_idx = squad.warpRank();
        ring_cursor_t key_ring;
        ring_cursor_t pos_ring;
        for (int pipeline_gen = 0;;
             ++pipeline_gen, key_ring.advance(key_ring_stages), pos_ring.advance(pos_ring_stages))
        {
          const int slot_id = key_ring.slot;
          // wait for computed (1/3): all per-warp-tile metadata (run counts, first/last heads)
          wait_parity(&computed[slot_id], key_ring.parity);
          const int tile_id = tile_id_buf[slot_id];
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&empty[slot_id]);
            }
            break;
          }
          // lane i: run-count sum over warp-tiles [0, i) = where warp-tile i's runs begin within the tile
          // we do this BEFORE the wait on prefixed so they overlap
          const auto [lane_warp_tile_run_count, lane_runs_before_warp_tile] =
            scan_warp_tile_run_counts<compute_warps>(warp_run_counts[slot_id], lane_id);
          // staged positions
          const position_t* run_positions = pos_buf + static_cast<size_t>(pos_ring.slot) * tile_size;
          const int warp_tile_id          = store_warp_idx;
          const int warp_tile_run_count   = warp_run_counts[slot_id][warp_tile_id];
          const int runs_before_warp_tile = __shfl_sync(full_mask, lane_runs_before_warp_tile, warp_tile_id);
          // if the compute warp decided to skip staging for this warp tile, the positions were never staged:
          // decode them from the head flags and buffer intermediate results in register
          // At run_count == 0 COMPUTE stages nothing but we fall through to the staged drain; safe only
          // because the loop bound run_idx < warp_tile_run_count == 0 never reads the pos slot.
          if (warp_tile_run_count >= 1 && warp_tile_run_count < staging_threshold)
          {
            // wait for staged_warp_tile (2/3)
            wait_parity(&staged_warp_tile[slot_id][warp_tile_id], key_ring.parity);
            constexpr int decode_items_per_thread = policy.decode_items_per_thread();
            KeyT buf_key[decode_items_per_thread];
            int buf_run_length[decode_items_per_thread];
            const int warp_tile_offset = warp_tile_id * warp_tile_size;
            const int num_rounds       = ::cuda::ceil_div(warp_tile_run_count, detail::warp_threads);
            _CCCL_ASSERT(num_rounds <= decode_items_per_thread, "register buffer must cover all decoding rounds");
            const head_flag_decode_t<items_per_thread> dec(head_flag_buf[slot_id], warp_tile_id, lane_id);
            const KeyT* tile_keys = tile_buf + static_cast<size_t>(slot_id) * slot_stride + slot_pad;
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int it = 0; it < decode_items_per_thread; ++it)
            {
              if (it >= num_rounds)
              {
                break;
              }
              const int run_idx    = it * detail::warp_threads + lane_id;
              const run_span_t run = dec.decode_run(run_idx);
              // the run's key is its LAST element (the one before the next run's head). The clamp keeps the
              // gather in bounds when next_head_pos is garbage (the warp tile's last run); that run's key and
              // count are both dead here and written by the bookkeeper instead.
              const int last_pos = (::cuda::std::min) (run.next_head_pos - 1, warp_tile_size - 1);
              buf_key[it]        = tile_keys[warp_tile_offset + last_pos + skip_elems];
              buf_run_length[it] = run.next_head_pos - run.head_pos_in_warp_tile;
            }

            __syncwarp();
            if (lane_id == 0)
            {
              if (pos_ring_stages < key_ring_stages)
              {
                ptx::mbarrier_arrive(&pos_buf_free[pos_ring.slot]);
              }
            }

            // wait for prefixed (3/3)
            wait_parity(&prefixed[slot_id], key_ring.parity);
            const OffT global_runs_before_warp_tile = prefix_packed[slot_id].run_count() + runs_before_warp_tile;
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int it = 0; it < decode_items_per_thread; ++it)
            {
              if (it >= num_rounds)
              {
                break;
              }
              const int run_idx         = it * detail::warp_threads + lane_id;
              const OffT global_run_idx = global_runs_before_warp_tile + run_idx;
              if (run_idx + 1 < warp_tile_run_count)
              {
                // the warp tile's last run ends outside this warp tile: its key and count are the
                // bookkeeper's job
                d_unique[global_run_idx] = buf_key[it];
                d_counts[global_run_idx] = buf_run_length[it];
              }
            }
            __syncwarp();
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&empty[slot_id]);
            }
            continue;
          } // reg buf
          // if not reg buffed, we do the normal things, i.e. prefixed wait, then staged_warp_tile, then drain
          // wait for prefixed (2/3)
          wait_parity(&prefixed[slot_id], key_ring.parity);
          const OffT curr_prefix_run_count = prefix_packed[slot_id].run_count();
          // wait for staged_warp_tile (3/3)
          wait_parity(&staged_warp_tile[slot_id][warp_tile_id], key_ring.parity);
          // writes warp tile (warp_tile_id)'s staged output into the global arrays.
          // Per run: gather its key from the run's head position -> d_unique,
          // and write its length -> d_counts (= next run's head pos - this run's head pos).
          // The warp tile's last run spans into the next warp-tile, so its length is fixed up separately.
          const OffT global_runs_before_warp_tile = curr_prefix_run_count + runs_before_warp_tile;
          const int warp_tile_offset              = warp_tile_id * warp_tile_size;
          // the run's key is its LAST element, the one before the next run's head;
          // the warp tile's last run (key and count) is fixed up by the bookkeeper
          const int full_runs = warp_tile_run_count - 1;
          if (keys_staged)
          {
            const KeyT* tile_keys = tile_buf + static_cast<size_t>(slot_id) * slot_stride + slot_pad;
            int chunk_base        = 0;
            _CCCL_PRAGMA_UNROLL(2)
            for (; chunk_base + detail::warp_threads <= full_runs; chunk_base += detail::warp_threads)
            {
              const int run_idx         = chunk_base + lane_id;
              const OffT global_run_idx = global_runs_before_warp_tile + run_idx;
              const int head_pos = static_cast<int>(run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx)]);
              const int next_head_pos =
                static_cast<int>(run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx + 1)]);
              d_unique[global_run_idx] = tile_keys[next_head_pos - 1 + skip_elems];
              d_counts[global_run_idx] = next_head_pos - head_pos;
            }
            const int run_idx = chunk_base + lane_id;
            if (run_idx < full_runs)
            {
              const OffT global_run_idx = global_runs_before_warp_tile + run_idx;
              const int head_pos = static_cast<int>(run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx)]);
              const int next_head_pos =
                static_cast<int>(run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx + 1)]);
              d_unique[global_run_idx] = tile_keys[next_head_pos - 1 + skip_elems];
              d_counts[global_run_idx] = next_head_pos - head_pos;
            }
          }
          else
          {
            // vvv regressed case vvv
            const KeyT* tile_keys = d_keys + static_cast<size_t>(tile_id) * tile_size;
            int chunk_base        = 0;
            _CCCL_PRAGMA_UNROLL(2)
            for (; chunk_base + detail::warp_threads <= full_runs; chunk_base += detail::warp_threads)
            {
              const int run_idx         = chunk_base + lane_id;
              const OffT global_run_idx = global_runs_before_warp_tile + run_idx;
              const int head_pos = static_cast<int>(run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx)]);
              const int next_head_pos =
                static_cast<int>(run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx + 1)]);
              d_unique[global_run_idx] = tile_keys[next_head_pos - 1];
              d_counts[global_run_idx] = next_head_pos - head_pos;
            }
            const int run_idx = chunk_base + lane_id;
            if (run_idx < full_runs)
            {
              const OffT global_run_idx = global_runs_before_warp_tile + run_idx;
              const int head_pos = static_cast<int>(run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx)]);
              const int next_head_pos =
                static_cast<int>(run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx + 1)]);
              d_unique[global_run_idx] = tile_keys[next_head_pos - 1];
              d_counts[global_run_idx] = next_head_pos - head_pos;
            }
            // ^^^ regressed case ^^^
          }
          __syncwarp();
          if (lane_id == 0)
          {
            if (pos_ring_stages < key_ring_stages)
            {
              ptx::mbarrier_arrive(&pos_buf_free[pos_ring.slot]);
            }
            // store done, load may proceed!
            ptx::mbarrier_arrive(&empty[slot_id]);
          }
        }
      }
      // if you are the bookkeeper
      else
      {
        ring_cursor_t key_ring;
        for (int pipeline_gen = 0;; ++pipeline_gen, key_ring.advance(key_ring_stages))
        {
          const int slot_id = key_ring.slot;
          wait_parity(&computed[slot_id], key_ring.parity);
          const int tile_id = tile_id_buf[slot_id];
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&empty[slot_id]);
            }
            break;
          }
          const int tile_len = static_cast<int>(
            (::cuda::std::min) (static_cast<OffT>(tile_size), num_items - static_cast<OffT>(tile_id) * tile_size));
          const bool is_last_tile = (tile_id == num_tiles - 1);
          // same scan as the store warps (lane i = warp-tile i)
          const auto [lane_warp_tile_run_count, lane_runs_before_warp_tile] =
            scan_warp_tile_run_counts<compute_warps>(warp_run_counts[slot_id], lane_id);
          const int tile_total_runs =
            __shfl_sync(full_mask, lane_runs_before_warp_tile + lane_warp_tile_run_count, compute_warps - 1);
          const unsigned nonempty_warp_tiles_mask = __ballot_sync(full_mask, lane_warp_tile_run_count > 0);
          // every count this warp closes gets its key written too: a run's key is its LAST element,
          // which lives in the closing tile's keys (position -1 reads the over-fetched boundary element)
          const KeyT* bk_tile_keys = keys_staged ? tile_buf + static_cast<size_t>(slot_id) * slot_stride + slot_pad
                                                 : d_keys + static_cast<size_t>(tile_id) * tile_size;
          const int bk_key_skip    = keys_staged ? skip_elems : 0;
          // wait for prefixed
          wait_parity(&prefixed[slot_id], key_ring.parity);
          const prefix_t packed_prefix       = prefix_packed[slot_id];
          const OffT curr_prefix_run_count   = packed_prefix.run_count();
          const OffT curr_prefix_open_length = packed_prefix.open_len();
          // per-warp-tile boundary: a warp-tile's last run is closed by the next nonempty warp-tile's irst head.
          // lane L handles warp-tile L.
          if (lane_id < compute_warps && lane_warp_tile_run_count > 0)
          {
            // nonempty warp-tiles after L (this is a mask)
            const unsigned later_nonempty_warp_tiles = nonempty_warp_tiles_mask >> (lane_id + 1);
            const OffT last_run_global_idx =
              curr_prefix_run_count + lane_runs_before_warp_tile + lane_warp_tile_run_count - 1;
            if (later_nonempty_warp_tiles)
            {
              const int next_nonempty_warp_tile = lane_id + 1 + __ffs(later_nonempty_warp_tiles) - 1;
              const int close_pos               = warp_first_heads[slot_id][next_nonempty_warp_tile];
              d_unique[last_run_global_idx]     = bk_tile_keys[close_pos - 1 + bk_key_skip];
              d_counts[last_run_global_idx]     = close_pos - warp_last_heads[slot_id][lane_id];
            }
            else if (is_last_tile)
            {
              // if we are the last warptile of the whole input, we end here
              d_unique[last_run_global_idx] = bk_tile_keys[tile_len - 1 + bk_key_skip];
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
              // first_head == 0 reads the over-fetched boundary element (the previous tile's last key)
              d_unique[curr_prefix_run_count - 1] = bk_tile_keys[first_head - 1 + bk_key_skip];
              d_counts[curr_prefix_run_count - 1] = curr_prefix_open_length + first_head;
            }
            // if we are last tile with no head: we have to close it here
            if (is_last_tile && !any_head && curr_prefix_run_count > 0)
            {
              d_unique[curr_prefix_run_count - 1] = bk_tile_keys[tile_len - 1 + bk_key_skip];
              d_counts[curr_prefix_run_count - 1] = curr_prefix_open_length + tile_len;
            }
            // otherwise, next tile's problem
            if (is_last_tile)
            {
              *d_num_runs = (NumRunsT) (curr_prefix_run_count + tile_total_runs);
            }
            ptx::mbarrier_arrive(&empty[slot_id]); // bookkeeping done, slot may recycle
          }
        }
      }
    });
}

template <class PolicySelector, class StateT>
_CCCL_KERNEL_ATTRIBUTES void DeviceRleEncodeLookaheadInitKernel(StateT* states, ::cuda::std::int64_t n_states)
{
  const ::cuda::std::int64_t i = (::cuda::std::int64_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_states)
  {
    states[i] = StateT{};
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
    tile_partial_state_t* tile_partial_states,
    OffT num_items,
    int num_tiles,
    int key_ring_stages,
    int pos_ring_stages,
    bool keys_staged)
{
  static constexpr RleEncodePolicy active_policy = current_policy<PolicySelector>();
  if constexpr (active_policy.algorithm == RleAlgorithm::lookahead)
  {
    NV_IF_TARGET(
      NV_PROVIDES_SM_100,
      (device_rle_encode_lookahead_body<PolicySelector>(
         d_keys,
         d_unique,
         d_counts,
         d_num_runs,
         tile_partial_states,
         num_items,
         num_tiles,
         key_ring_stages,
         pos_ring_stages,
         keys_staged);))
  }
}
#endif // __cccl_ptx_isa >= 920
} // namespace detail::rle::encode

CUB_NAMESPACE_END
