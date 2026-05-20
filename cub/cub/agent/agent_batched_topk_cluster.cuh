// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Cluster-based per-segment top-k agent.
//!
//! Prototype that exercises CUDA thread block clusters as a replacement for
//! the multi-kernel + global histogram pipeline used by cub::DeviceTopK. Each
//! cluster processes exactly one segment.
//!
//! Histogram strategy (Pattern C):
//!   1. Every block lays out `hist[num_buckets]` at the same offset in its own
//!      shared memory. Each block accumulates a block-private histogram using
//!      block-scope atomicAdd_block (cheap, SMEM-local).
//!   2. After a cluster-wide barrier, every non-leader block walks its
//!      histogram and folds its bucket counts into the leader block's `hist`
//!      via cluster-scope DSMEM atomics. The leader's `hist` therefore plays
//!      a dual role: its own block-private histogram in step 1, then the
//!      cluster-merged histogram after the second cluster sync.
//!   3. The leader's thread 0 prefix-scans `hist`, identifies the bucket of
//!      the k-th key, and updates the cluster-shared `state`. Every block
//!      reads `state.kth_key_bits` from the leader via DSMEM at the start of
//!      the next pass.
//!
//! Output cursors live in the same cluster-shared `state` and are reached the
//! same way (cluster-scope DSMEM atomics).

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_topk.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/dispatch_topk.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cooperative_groups.h>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk_cluster
{
// -----------------------------------------------------------------------------
// Tuning policy
// -----------------------------------------------------------------------------
struct cluster_topk_policy
{
  int cluster_size;
  int threads_per_block;
  int items_per_thread;
  int bits_per_pass;
};

// -----------------------------------------------------------------------------
// Cluster-shared state. Lives in the leader block's shared memory and is
// reached from every block of the cluster through DSMEM.
// -----------------------------------------------------------------------------
template <typename KeyT, typename OffsetT, typename OutOffsetT>
struct alignas(16) cluster_topk_state
{
  using key_prefix_t = detail::topk::key_prefix_storage_t<KeyT>;

  OffsetT len;
  OutOffsetT k;
  key_prefix_t kth_key_bits;
  OutOffsetT out_cnt;
  OutOffsetT out_back_cnt;
  // Set by the leader after `leader_identify_kth_bucket` whenever the
  // identified bucket holds exactly `k` items (every candidate is part of
  // the top-k). Read by every block of the cluster at the top of the next
  // radix pass through DSMEM. Carried in the cluster-shared state so the
  // value survives the cluster sync that ends the current pass.
  ::cuda::std::uint32_t early_stop;
};

// -----------------------------------------------------------------------------
// Cluster top-k agent
// -----------------------------------------------------------------------------
template <int ClusterSize,
          int ThreadsPerBlock,
          int ItemsPerThread,
          int BitsPerPass,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
struct agent_batched_topk_cluster
{
  // ---------------------------------------------------------------------------
  // Types / constants
  // ---------------------------------------------------------------------------
  using key_it_t = it_value_t<KeyInputItItT>;
  using key_t    = it_value_t<key_it_t>;

  using segment_size_val_t = typename SegmentSizeParameterT::value_type;
  using num_segments_val_t = typename NumSegmentsParameterT::value_type;

  using offset_t     = ::cuda::std::uint32_t;
  using out_offset_t = ::cuda::std::uint32_t;
  using state_t      = cluster_topk_state<key_t, offset_t, out_offset_t>;
  using key_prefix_t = typename state_t::key_prefix_t;

  static constexpr int cluster_size      = ClusterSize;
  static constexpr int threads_per_block = ThreadsPerBlock;
  static constexpr int items_per_thread  = ItemsPerThread;
  static constexpr int bits_per_pass     = BitsPerPass;
  static constexpr int tile_items        = threads_per_block * items_per_thread;
  static constexpr int cluster_tile      = cluster_size * tile_items;
  static constexpr int num_buckets       = 1 << bits_per_pass;

  using decomposer_t = detail::identity_decomposer_t;

  // ---------------------------------------------------------------------------
  // Block-scan used by the leader block to prefix-sum its merged histogram
  // ---------------------------------------------------------------------------
  // Constraint: every leader thread owns at most one bucket. Threads beyond
  // num_buckets contribute zero to the scan, which keeps the prefix-sum loop
  // body trivial (1 item per thread, no inner loops).
  static_assert(num_buckets <= threads_per_block,
                "Cluster top-k requires num_buckets <= threads_per_block (1 bin per leader thread or fewer)");
  using block_scan_t = BlockScan<offset_t, threads_per_block, BLOCK_SCAN_WARP_SCANS>;

  // ---------------------------------------------------------------------------
  // Shared memory storage
  // ---------------------------------------------------------------------------
  // The same layout is allocated by every block of the cluster so that any
  // block can reach the leader's fields at a known offset via DSMEM. Each
  // block populates its own `hist` block-locally; after the first cluster
  // sync the non-leader blocks fold their bucket counts into the leader's
  // `hist` through DSMEM atomics. `state` is meaningful only in the leader
  // block; the other blocks reach it exclusively through the DSMEM mapping.
  // `broadcast_kth` / `broadcast_early_stop` are per-block fan-out slots used
  // to share the leader-computed values across threads of each block.
  struct _TempStorage
  {
    offset_t hist[num_buckets];
    state_t state;
    key_prefix_t broadcast_kth;
    ::cuda::std::uint32_t broadcast_early_stop;
    typename block_scan_t::TempStorage scan_storage;
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  // ---------------------------------------------------------------------------
  // Per-thread members
  // ---------------------------------------------------------------------------
  _TempStorage& temp_storage;
  KeyInputItItT d_key_segments_it;
  KeyOutputItItT d_key_segments_out_it;
  SegmentSizeParameterT segment_sizes;
  KParameterT k_param;
  SelectDirectionParameterT select_directions;
  NumSegmentsParameterT num_segments;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE agent_batched_topk_cluster(
    TempStorage& temp_storage_,
    KeyInputItItT d_key_segments_it_,
    KeyOutputItItT d_key_segments_out_it_,
    SegmentSizeParameterT segment_sizes_,
    KParameterT k_param_,
    SelectDirectionParameterT select_directions_,
    NumSegmentsParameterT num_segments_)
      : temp_storage(temp_storage_.Alias())
      , d_key_segments_it(d_key_segments_it_)
      , d_key_segments_out_it(d_key_segments_out_it_)
      , segment_sizes(segment_sizes_)
      , k_param(k_param_)
      , select_directions(select_directions_)
      , num_segments(num_segments_)
  {}

  // ---------------------------------------------------------------------------
  // Main entry point
  // ---------------------------------------------------------------------------
  // Prototype targets SM 9.0+ only; older architectures are unsupported by the
  // dispatch and never reach this kernel.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Process()
  {
    process_impl();
  }

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE void reset_hist()
  {
    for (int i = static_cast<int>(threadIdx.x); i < num_buckets; i += threads_per_block)
    {
      temp_storage.hist[i] = 0;
    }
  }

  // Parallel prefix sum (cub::BlockScan) over the leader's merged histogram
  // plus identification of the bucket holding the k-th item. Every thread of
  // the leader block participates with one bucket; threads with rank beyond
  // num_buckets contribute zero. The single thread that owns the k-th bucket
  // writes the per-pass state. The caller must guarantee the leader block
  // has finished its DSMEM merge before invoking this.
  _CCCL_DEVICE _CCCL_FORCEINLINE void leader_identify_kth_bucket(int pass)
  {
    // Capture `state.k` before the scan: this is the only legal window where
    // every thread is guaranteed to read the previous pass's value. The
    // owning thread overwrites `state.k` in the if-block below, so any read
    // after that point would race with that write.
    const out_offset_t target_k = temp_storage.state.k;

    const int bucket        = static_cast<int>(threadIdx.x);
    const bool owns_bucket  = bucket < num_buckets;
    const offset_t hist_val = owns_bucket ? temp_storage.hist[bucket] : offset_t{0};
    offset_t prefix         = 0;

    block_scan_t(temp_storage.scan_storage).ExclusiveSum(hist_val, prefix);

    // Exactly one thread satisfies `prefix < target_k <= prefix + hist_val`.
    if (owns_bucket && prefix < target_k && prefix + hist_val >= target_k)
    {
      const out_offset_t new_k = target_k - static_cast<out_offset_t>(prefix);
      const offset_t new_len   = hist_val;
      temp_storage.state.len   = new_len;
      temp_storage.state.k     = new_k;
      // Early-stop opportunity: the bucket holding the k-th key contains
      // exactly the remaining `k` items. Every candidate is therefore part
      // of the top-k, so subsequent radix passes only redistribute the same
      // items across finer buckets without changing the final result.
      temp_storage.state.early_stop =
        (static_cast<out_offset_t>(new_len) == new_k) ? ::cuda::std::uint32_t{1} : ::cuda::std::uint32_t{0};
      detail::topk::set_kth_key_bits<key_t, bits_per_pass>(temp_storage.state.kth_key_bits, pass, bucket);
    }
  }

  // -------------------------------------------------------------------------
  // Per-direction implementation
  // -------------------------------------------------------------------------
  template <detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void run(
    ::cooperative_groups::cluster_group& cluster,
    num_segments_val_t segment_id,
    unsigned int cluster_rank,
    segment_size_val_t segment_size,
    out_offset_t k)
  {
    using extract_bin_op_t         = detail::topk::extract_bin_op_t<key_t, SelectDirection, bits_per_pass, decomposer_t>;
    using identify_candidates_op_t = detail::topk::identify_candidates_op_t<key_t, SelectDirection, bits_per_pass, decomposer_t>;

    constexpr int total_bits = int{sizeof(key_t)} * 8;
    constexpr int num_passes = detail::topk::calc_num_passes<key_t>(bits_per_pass);

    // Sentinel key used to pad partial tiles. Worst-possible value for the
    // requested direction so the sentinel can't land in the selected set.
    const key_t pad_key = (SelectDirection == detail::topk::select::max)
                          ? ::cuda::std::numeric_limits<key_t>::lowest()
                          : ::cuda::std::numeric_limits<key_t>::max();

    auto block_keys_in = d_key_segments_it[segment_id];
    const auto block_offset_in_cluster =
      static_cast<segment_size_val_t>(static_cast<int>(cluster_rank) * tile_items);

    // Striped load into per-thread registers. Each thread holds
    // `items_per_thread` keys spanning the block's tile within the cluster
    // tile. Out-of-segment slots receive sentinel padding.
    key_t thread_keys[items_per_thread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < items_per_thread; ++j)
    {
      const segment_size_val_t local_idx  = static_cast<segment_size_val_t>(j * threads_per_block + threadIdx.x);
      const segment_size_val_t global_idx = block_offset_in_cluster + local_idx;
      thread_keys[j] = (global_idx < segment_size) ? block_keys_in[global_idx] : pad_key;
    }

    // DSMEM pointers into the leader block's shared memory.
    offset_t* leader_hist = cluster.map_shared_rank(temp_storage.hist, 0);
    state_t* leader_state = cluster.map_shared_rank(&temp_storage.state, 0);

    // Per-block local copy of `kth_key_bits` so each key check hits the
    // block's own SMEM rather than DSMEM during the histogram loop.
    key_prefix_t kth_key_bits_local = {};

    // Tracks the highest pass count that actually executed. Without early
    // stop this stays at `num_passes`; with early stop it captures the pass
    // at which we broke out so the final filter can construct its identify
    // operator at the matching radix level.
    int last_pass = num_passes;

    for (int pass = 0; pass < num_passes; ++pass)
    {
      const bool is_first_pass = (pass == 0);

      // Refresh per-block local kth_key_bits from the leader's state. For
      // pass 0 the bits are all zero (no filtering yet) so we skip the read.
      // We also pull the leader's `early_stop` flag here so every block has
      // an opportunity to bail out before doing any more histogram work.
      if (!is_first_pass)
      {
        if (threadIdx.x == 0)
        {
          temp_storage.broadcast_kth = leader_state->kth_key_bits;
#ifndef CUB_DISABLE_CLUSTER_TOPK_EARLY_STOP
          temp_storage.broadcast_early_stop = leader_state->early_stop;
#endif
        }
        __syncthreads();
        kth_key_bits_local = temp_storage.broadcast_kth;
#ifndef CUB_DISABLE_CLUSTER_TOPK_EARLY_STOP
        if (temp_storage.broadcast_early_stop != ::cuda::std::uint32_t{0})
        {
          last_pass = pass;
          break;
        }
#endif
      }

      // Every block (including the leader) starts each pass with a fresh,
      // empty `hist`. For pass 0 the leader's initial reset in process_impl()
      // covered the leader's slot, but every non-leader block must also
      // reset its own. We just always reset here for symmetry.
      reset_hist();
      __syncthreads();

      identify_candidates_op_t identify_op(&kth_key_bits_local, pass, total_bits, decomposer_t{});
      extract_bin_op_t extract_op(pass, total_bits, decomposer_t{});

      // Step 1: block-private histogram, cheap block-scope atomic adds.
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < items_per_thread; ++j)
      {
        const key_t key = thread_keys[j];
        const bool keep = is_first_pass || (identify_op(key) == detail::topk::candidate_class::candidate);
        if (keep)
        {
          const int bucket = extract_op(key);
          atomicAdd_block(&temp_storage.hist[bucket], offset_t{1});
        }
      }
      // Cluster-wide barrier: every block must finish its block-scope atomic
      // adds before any non-leader block starts DSMEM atomic adds into the
      // leader's `hist`. A plain __syncthreads() would only order the local
      // block's writes; the leader's `atomicAdd_block`s would still race with
      // remote blocks' cluster-scope `atomicAdd`s targeting the same memory.
      cluster.sync();

      // Step 2: non-leader blocks fold their bucket counts into the leader's
      // `hist` via cluster-scope DSMEM atomics. The leader's data is already
      // present in `hist`; if the leader also merged from itself, its
      // contribution would be counted twice.
      if (cluster_rank != 0)
      {
        for (int i = static_cast<int>(threadIdx.x); i < num_buckets; i += threads_per_block)
        {
          const offset_t v = temp_storage.hist[i];
          if (v != 0)
          {
            atomicAdd(leader_hist + i, v);
          }
        }
      }
      cluster.sync();

      // Step 3: the leader prefix-scans its (now merged) `hist` and updates
      // the cluster-shared `state`. Subsequent reads (next-pass refresh, last
      // filter) all observe these writes after the next cluster sync.
      if (cluster_rank == 0)
      {
        leader_identify_kth_bucket(pass);
      }
      cluster.sync();
    }

    // -----------------------------------------------------------------------
    // Final filter pass: write strictly-selected items to the front of the
    // output; back-fill candidates that share the k-th key's prefix bits.
    // -----------------------------------------------------------------------
    if (threadIdx.x == 0)
    {
      temp_storage.broadcast_kth = leader_state->kth_key_bits;
    }
    __syncthreads();
    kth_key_bits_local = temp_storage.broadcast_kth;

    auto block_keys_out        = d_key_segments_out_it[segment_id];
    const out_offset_t num_kth = leader_state->k; // remaining k after the radix passes

    // The pass argument controls how many radix levels of `kth_key_bits` are
    // considered significant. After an early-stop break at the start of pass
    // `last_pass`, only the first `last_pass` digits of the splitter have
    // been set; comparing all bits would treat the (still-zero) trailing
    // digits as smaller and erroneously reject candidates that share the
    // identified prefix.
    identify_candidates_op_t identify_op(&kth_key_bits_local, last_pass, total_bits, decomposer_t{});

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < items_per_thread; ++j)
    {
      const segment_size_val_t local_idx  = static_cast<segment_size_val_t>(j * threads_per_block + threadIdx.x);
      const segment_size_val_t global_idx = block_offset_in_cluster + local_idx;
      if (global_idx >= segment_size)
      {
        continue;
      }
      const key_t key = thread_keys[j];
      const auto res  = identify_op(key);
      if (res == detail::topk::candidate_class::selected)
      {
        const out_offset_t pos = atomicAdd(&leader_state->out_cnt, out_offset_t{1});
        block_keys_out[pos]    = key;
      }
      else if (res == detail::topk::candidate_class::candidate)
      {
        const out_offset_t back_pos = atomicAdd(&leader_state->out_back_cnt, out_offset_t{1});
        if (back_pos < num_kth)
        {
          const out_offset_t pos = k - 1 - back_pos;
          block_keys_out[pos]    = key;
        }
      }
    }

    // Final cluster barrier: hold every block in the cluster until all DSMEM
    // atomics into the leader's state are complete. Without this, a fast
    // block (e.g. one whose tile is entirely padding) can return while another
    // block is still writing to leader-resident memory through DSMEM, which
    // surfaces as a "cluster target block not present" exception.
    cluster.sync();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void process_impl()
  {
    ::cooperative_groups::cluster_group cluster = ::cooperative_groups::this_cluster();
    const unsigned int cluster_rank             = cluster.block_rank();
    const auto segment_id                       = static_cast<num_segments_val_t>(blockIdx.x / cluster_size);

    if (segment_id >= num_segments.get_param(0))
    {
      return;
    }

    const auto segment_size = static_cast<segment_size_val_t>(segment_sizes.get_param(segment_id));
    const auto k_requested  = static_cast<out_offset_t>(k_param.get_param(segment_id));
    const auto k =
      static_cast<out_offset_t>((::cuda::std::min) (static_cast<segment_size_val_t>(k_requested), segment_size));

    if (k == 0)
    {
      return;
    }

    // The leader block initializes the cluster-shared state. Every block
    // (including the leader) will reset its own `hist` at the top of the
    // per-pass loop.
    if (cluster_rank == 0 && threadIdx.x == 0)
    {
      temp_storage.state.len          = static_cast<offset_t>(segment_size);
      temp_storage.state.k            = k;
      temp_storage.state.kth_key_bits = {};
      temp_storage.state.out_cnt      = 0;
      temp_storage.state.out_back_cnt = 0;
      temp_storage.state.early_stop   = 0;
    }
    cluster.sync();

    const bool ok = detail::params::dispatch_discrete(
      select_directions,
      segment_id,
      [this, &cluster, segment_id, cluster_rank, segment_size, k](auto direction_tag) {
        constexpr detail::topk::select Direction = decltype(direction_tag)::value;
        this->template run<Direction>(cluster, segment_id, cluster_rank, segment_size, k);
      });
    _CCCL_ASSERT(ok, "Unsupported select direction for cluster top-k");
    (void) ok;
  }
};

} // namespace detail::batched_topk_cluster

CUB_NAMESPACE_END
