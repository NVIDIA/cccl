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
//! same way (cluster-scope DSMEM atomics). Defining
//! `CUB_ENABLE_CLUSTER_TOPK_BLOCK_AGG_OUTPUT` switches the final filter pass
//! to a block-aggregated variant that first counts selected / candidate keys
//! in block-local shared-memory counters and then issues a single DSMEM
//! atomic per block per counter to reserve a contiguous output range. This
//! lets us A/B the two strategies without touching the rest of the kernel.
//!
//! Defining `CUB_ENABLE_CLUSTER_TOPK_SCAN_THEN_REDUCE` flips the histogram
//! pipeline from reduce-then-scan to scan-then-reduce: every block does a
//! block-local `InclusiveSum` over its own histogram, then folds the scans
//! (not the raw counts) into the leader. The leader then identifies the
//! k-th bucket directly from `hist[bucket]` / `hist[bucket - 1]`.

#pragma once

#define CUB_ENABLE_CLUSTER_TOPK_SCAN_THEN_REDUCE

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_topk.cuh>
#include <cub/block/block_load_to_shared.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/dispatch_topk.cuh>
#include <cub/device/dispatch/kernels/kernel_transform.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/round_up.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstdint>
#include <cuda/std/inplace_vector>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/utility>

#include <nv/target>

#include <cooperative_groups.h>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk_cluster
{
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

// Dynamic-SMEM layout shared by dispatch and the agent. `block_tile_capacity` is the physical per-CTA
// resident capacity passed to the kernel; `cluster_tile_capacity` reserves one chunk of logical coverage
// for a possible unaligned head chunk.
template <typename KeyT, int ChunkBytes, int LoadAlignBytes>
struct smem_block_tile_layout
{
  static constexpr int chunk_items = ChunkBytes / int{sizeof(KeyT)};
  static constexpr int slot_alignment =
    (::cuda::std::max) (LoadAlignBytes, detail::LoadToSharedBufferAlignBytes<KeyT>());
  static constexpr int slot_stride_bytes =
    ::cuda::round_up(detail::LoadToSharedBufferSizeBytes<KeyT>(chunk_items) + slot_alignment, slot_alignment);
  static constexpr int base_padding_bytes = (alignof(KeyT) > 16) ? slot_alignment : 0;

  [[nodiscard]] _CCCL_HOST_DEVICE static constexpr ::cuda::std::uint32_t
  block_tile_capacity(int dynamic_smem_bytes) noexcept
  {
    const int usable_bytes = (::cuda::std::max) (0, dynamic_smem_bytes - base_padding_bytes);
    const int slots        = usable_bytes / slot_stride_bytes;
    return static_cast<::cuda::std::uint32_t>(slots * chunk_items);
  }

  template <typename SizeT>
  [[nodiscard]] _CCCL_HOST_DEVICE static constexpr SizeT
  cluster_tile_capacity(int cluster_blocks, ::cuda::std::uint32_t physical_block_tile_capacity) noexcept
  {
    const auto physical_cluster_tile_items =
      static_cast<SizeT>(cluster_blocks) * static_cast<SizeT>(physical_block_tile_capacity);
    const auto head_chunk_reserve = static_cast<SizeT>(chunk_items);
    return (physical_cluster_tile_items > head_chunk_reserve)
           ? physical_cluster_tile_items - head_chunk_reserve
           : SizeT{0};
  }
};

// -----------------------------------------------------------------------------
// Cluster top-k agent
// -----------------------------------------------------------------------------
// Cluster width is a runtime value (see `process_impl` for the readback), so
// it is not a template parameter; per-block block_tile layout is still controlled
// by the template parameters below.
template <int ThreadsPerBlock,
          int UnrollFactor,
          int PipelineStages,
          int ChunkBytes,
          int LoadAlignBytes,
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

  static constexpr int threads_per_block = ThreadsPerBlock;
  static constexpr int load_align_bytes  = LoadAlignBytes;
  static constexpr int bits_per_pass     = BitsPerPass;
  static constexpr int num_buckets       = 1 << bits_per_pass;
  using smem_layout_t                    = smem_block_tile_layout<key_t, ChunkBytes, LoadAlignBytes>;
  static constexpr int chunk_items       = smem_layout_t::chunk_items;
  static constexpr int slot_alignment    = smem_layout_t::slot_alignment;
  static constexpr int slot_stride_bytes = smem_layout_t::slot_stride_bytes;

  static_assert(PipelineStages > 0);
  static_assert(ChunkBytes > 0);
  static_assert(LoadAlignBytes > 0);
  static_assert(ChunkBytes % LoadAlignBytes == 0);
  static_assert(LoadAlignBytes % int{sizeof(key_t)} == 0);
  static_assert(chunk_items > 0);

  using decomposer_t = detail::identity_decomposer_t;
  using block_load_t = BlockLoadToShared<threads_per_block>;

  static constexpr bool use_block_load_to_shared =
    THRUST_NS_QUALIFIER::is_trivially_relocatable_v<key_t> && THRUST_NS_QUALIFIER::is_contiguous_iterator_v<key_it_t>;

  // ---------------------------------------------------------------------------
  // Block-scan used by the leader block to prefix-sum its merged histogram
  // ---------------------------------------------------------------------------
  // The leader-block prefix sum spans `num_buckets` entries. Each thread owns
  // `buckets_per_thread` consecutive buckets in a blocked arrangement; entries
  // past the last valid bucket contribute zero so the histogram size is no
  // longer constrained to be `<= threads_per_block`.
  static constexpr int buckets_per_thread = ::cuda::ceil_div(num_buckets, threads_per_block);
  using block_scan_t                      = BlockScan<offset_t, threads_per_block, BLOCK_SCAN_WARP_SCANS>;

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
    typename block_load_t::TempStorage load_storage[PipelineStages];
#ifdef CUB_ENABLE_CLUSTER_TOPK_BLOCK_AGG_OUTPUT
    // Final-filter block-aggregation slots. First populated by per-thread
    // `atomicAdd_block` calls that count the block's selected / candidate
    // keys, then overwritten by thread 0 with the leader-side base position
    // returned from a single DSMEM atomic per counter, so the same slots
    // double as the broadcast channel for the base to the rest of the block.
    out_offset_t block_out_cnt;
    out_offset_t block_out_back_cnt;
#endif
  };

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::span<char> block_tile_buffer() const
  {
    const int slots = static_cast<int>(block_tile_capacity / chunk_items);
    return {key_slots, static_cast<::cuda::std::size_t>(slots * slot_stride_bytes)};
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE key_t* slot_keys_unpadded(int slot) const
  {
    return reinterpret_cast<key_t*>(key_slots + slot * slot_stride_bytes);
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::span<char>
  available_block_tile_buffer(char* buffer_begin) const
  {
    const auto buffer = block_tile_buffer();
    char* const end   = ::cuda::std::data(buffer) + static_cast<int>(::cuda::std::size(buffer));
    _CCCL_ASSERT(buffer_begin >= ::cuda::std::data(buffer) && buffer_begin <= end, "Invalid block_tile buffer cursor");
    _CCCL_ASSERT(::cuda::is_aligned(buffer_begin, detail::LoadToSharedBufferAlignBytes<key_t>()),
                 "block_tile buffer cursor must satisfy BlockLoadToShared's shared-memory alignment");
    return {buffer_begin, static_cast<::cuda::std::size_t>(end - buffer_begin)};
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  append_contiguous_span(::cuda::std::span<key_t>& merged, ::cuda::std::span<key_t> next) const
  {
    const int next_count = static_cast<int>(::cuda::std::size(next));
    _CCCL_ASSERT(static_cast<::cuda::std::size_t>(next_count) == ::cuda::std::size(next),
                 "Resident key span length must fit in int");
    if (next_count == 0)
    {
      return;
    }

    const int merged_count = static_cast<int>(::cuda::std::size(merged));
    _CCCL_ASSERT(static_cast<::cuda::std::size_t>(merged_count) == ::cuda::std::size(merged),
                 "Resident key span length must fit in int");
    if (merged_count == 0)
    {
      merged = next;
      return;
    }

    _CCCL_ASSERT(::cuda::std::data(merged) + merged_count == ::cuda::std::data(next),
                 "BlockLoadToShared returned non-contiguous resident key spans");
    merged = {::cuda::std::data(merged), static_cast<::cuda::std::size_t>(merged_count + next_count)};
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE int span_size(::cuda::std::span<key_t> keys) const
  {
    const int count = static_cast<int>(::cuda::std::size(keys));
    _CCCL_ASSERT(static_cast<::cuda::std::size_t>(count) == ::cuda::std::size(keys),
                 "Resident key span length must fit in int");
    return count;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE char* span_end(::cuda::std::span<key_t> keys) const
  {
    return reinterpret_cast<char*>(::cuda::std::data(keys) + span_size(keys));
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE offset_t
  aligned_head_items(const key_t* base, offset_t segment_size) const
  {
    const auto base_addr = reinterpret_cast<::cuda::std::uintptr_t>(base);
    const auto rem       = base_addr % static_cast<::cuda::std::uintptr_t>(load_align_bytes);
    const auto bytes =
      (rem == 0) ? ::cuda::std::uintptr_t{0} : static_cast<::cuda::std::uintptr_t>(load_align_bytes) - rem;
    const auto items = static_cast<offset_t>(bytes / static_cast<::cuda::std::uintptr_t>(sizeof(key_t)));
    return (::cuda::std::min) (items, segment_size);
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE offset_t num_chunks(offset_t segment_size, offset_t head_items) const
  {
    const offset_t remaining = segment_size - head_items;
    return ((head_items != 0) ? offset_t{1} : offset_t{0})
         + static_cast<offset_t>(::cuda::ceil_div(remaining, offset_t{chunk_items}));
  }

  struct chunk_desc
  {
    offset_t offset;
    int count;
  };

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE chunk_desc
  get_chunk(offset_t chunk_idx, offset_t segment_size, offset_t head_items) const
  {
    offset_t offset = chunk_idx * offset_t{chunk_items};
    if (head_items != 0)
    {
      if (chunk_idx == 0)
      {
        return {offset_t{0}, static_cast<int>(head_items)};
      }
      offset = head_items + (chunk_idx - 1) * offset_t{chunk_items};
    }
    const offset_t remaining = segment_size - offset;
    return {offset, static_cast<int>((::cuda::std::min) (remaining, offset_t{chunk_items}))};
  }

  template <typename PtrT>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE bool is_aligned_chunk(PtrT base, const chunk_desc chunk) const
  {
    const auto begin = reinterpret_cast<::cuda::std::uintptr_t>(base + chunk.offset);
    const auto end   = begin + static_cast<::cuda::std::uintptr_t>(chunk.count) * sizeof(key_t);
    return (begin % static_cast<::cuda::std::uintptr_t>(load_align_bytes) == 0)
        && (end % static_cast<::cuda::std::uintptr_t>(load_align_bytes) == 0);
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE offset_t
  num_rank_chunks(offset_t chunks, unsigned int cluster_rank, unsigned int cluster_blocks) const
  {
    return (cluster_rank < chunks)
           ? static_cast<offset_t>((chunks - 1 - cluster_rank) / cluster_blocks + 1)
           : offset_t{0};
  }

  template <typename F>
  _CCCL_DEVICE _CCCL_FORCEINLINE void for_each_chunk_key(::cuda::std::span<key_t> chunk_keys, F&& f) const
  {
    const int chunk_count = span_size(chunk_keys);
    const int iterations  = ::cuda::ceil_div(chunk_count, threads_per_block);
    detail::transform::unrolled_for<UnrollFactor>(iterations, [&](int j) {
      const int local = j * threads_per_block + static_cast<int>(threadIdx.x);
      if (local < chunk_count)
      {
        f(::cuda::std::data(chunk_keys)[local]);
      }
    });
  }

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
  char* key_slots;
  offset_t block_tile_capacity;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE agent_batched_topk_cluster(
    TempStorage& temp_storage_,
    KeyInputItItT d_key_segments_it_,
    KeyOutputItItT d_key_segments_out_it_,
    SegmentSizeParameterT segment_sizes_,
    KParameterT k_param_,
    SelectDirectionParameterT select_directions_,
    NumSegmentsParameterT num_segments_,
    char* key_slots_,
    offset_t block_tile_capacity_)
      : temp_storage(temp_storage_.Alias())
      , d_key_segments_it(d_key_segments_it_)
      , d_key_segments_out_it(d_key_segments_out_it_)
      , segment_sizes(segment_sizes_)
      , k_param(k_param_)
      , select_directions(select_directions_)
      , num_segments(num_segments_)
      , key_slots(key_slots_)
      , block_tile_capacity(block_tile_capacity_)
  {}

  // ---------------------------------------------------------------------------
  // Main entry point
  // ---------------------------------------------------------------------------
  // SM 9.0+ only. `_CG_HAS_CLUSTER_GROUP` keeps the body and the
  // `process_impl` definition consistent across NVCC and clang-cuda/clangd;
  // `NV_IF_TARGET` strips the call from NVCC's sub-SM-9.0 device passes.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Process()
  {
#if defined(_CG_HAS_CLUSTER_GROUP)
    NV_IF_TARGET(NV_PROVIDES_SM_90, (process_impl();));
#endif
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
  // plus identification of the bucket holding the k-th item. Each thread of
  // the leader block contributes `buckets_per_thread` consecutive buckets in
  // a blocked arrangement; entries past `num_buckets` contribute zero. The
  // single (thread, slot) pair that owns the k-th bucket writes the per-pass
  // state. The caller must guarantee the leader block has finished its DSMEM
  // merge before invoking this.
  _CCCL_DEVICE _CCCL_FORCEINLINE void leader_identify_kth_bucket(int pass)
  {
    // Capture `state.k` before the scan: this is the only legal window where
    // every thread is guaranteed to read the previous pass's value. The
    // owning thread overwrites `state.k` in the if-block below, so any read
    // after that point would race with that write.
    const out_offset_t target_k = temp_storage.state.k;

    offset_t hist_vals[buckets_per_thread];
    offset_t prefixes[buckets_per_thread];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < buckets_per_thread; ++j)
    {
      const int bucket = static_cast<int>(threadIdx.x) * buckets_per_thread + j;
      hist_vals[j]     = (bucket < num_buckets) ? temp_storage.hist[bucket] : offset_t{0};
    }

    block_scan_t(temp_storage.scan_storage).ExclusiveSum(hist_vals, prefixes);

    // Exactly one (thread, slot) pair satisfies
    // `prefix < target_k <= prefix + hist_val`.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < buckets_per_thread; ++j)
    {
      const int bucket = static_cast<int>(threadIdx.x) * buckets_per_thread + j;
      if (bucket < num_buckets && prefixes[j] < target_k && prefixes[j] + hist_vals[j] >= target_k)
      {
        const out_offset_t new_k = target_k - static_cast<out_offset_t>(prefixes[j]);
        const offset_t new_len   = hist_vals[j];
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
  }

#ifdef CUB_ENABLE_CLUSTER_TOPK_SCAN_THEN_REDUCE
  // Scan-then-reduce counterpart of `leader_identify_kth_bucket`. Assumes
  // `hist[i]` already holds the cluster-wide inclusive prefix sum. The
  // per-bucket count is recovered as `inclusive - exclusive`, where
  // `exclusive` is `hist[bucket - 1]` (or `0` for `bucket == 0`).
  //
  // `target_k` is `state.k` read by the caller before the `cluster.sync()`
  // that precedes this call, so that barrier orders the read against the
  // write this function may issue to `state.k`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void leader_identify_kth_bucket_from_inclusive(int pass, out_offset_t target_k)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < buckets_per_thread; ++j)
    {
      const int bucket = static_cast<int>(threadIdx.x) * buckets_per_thread + j;
      if (bucket >= num_buckets)
      {
        continue;
      }
      const offset_t inclusive = temp_storage.hist[bucket];
      const offset_t exclusive = (bucket == 0) ? offset_t{0} : temp_storage.hist[bucket - 1];
      if (exclusive < target_k && inclusive >= target_k)
      {
        const out_offset_t new_k = target_k - static_cast<out_offset_t>(exclusive);
        const offset_t new_len   = inclusive - exclusive;
        temp_storage.state.len   = new_len;
        temp_storage.state.k     = new_k;
        temp_storage.state.early_stop =
          (static_cast<out_offset_t>(new_len) == new_k) ? ::cuda::std::uint32_t{1} : ::cuda::std::uint32_t{0};
        detail::topk::set_kth_key_bits<key_t, bits_per_pass>(temp_storage.state.kth_key_bits, pass, bucket);
      }
    }
  }
#endif

  // -------------------------------------------------------------------------
  // Per-direction implementation
  // -------------------------------------------------------------------------
  // Stripped on sub-SM-9.0 device passes; uses `cluster_group`, which is only
  // declared when `_CG_HAS_CLUSTER_GROUP` is set.
#if defined(_CG_HAS_CLUSTER_GROUP)
  template <detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  run(::cooperative_groups::cluster_group& cluster,
      num_segments_val_t segment_id,
      unsigned int cluster_rank,
      segment_size_val_t segment_size,
      out_offset_t k)
  {
    using extract_bin_op_t = detail::topk::extract_bin_op_t<key_t, SelectDirection, bits_per_pass, decomposer_t>;
    using identify_candidates_op_t =
      detail::topk::identify_candidates_op_t<key_t, SelectDirection, bits_per_pass, decomposer_t>;

    constexpr int total_bits = int{sizeof(key_t)} * 8;
    constexpr int num_passes = detail::topk::calc_num_passes<key_t>(bits_per_pass);

    auto block_keys_in              = d_key_segments_it[segment_id];
    const auto segment_size_u32     = static_cast<offset_t>(segment_size);
    const unsigned int cluster_size = cluster.num_blocks();

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

    offset_t head_items = 0;
    if constexpr (use_block_load_to_shared)
    {
      auto* block_keys_base = THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(block_keys_in);
      head_items            = aligned_head_items(block_keys_base, segment_size_u32);
    }
    // The generic fallback does not use BlockLoadToShared's alignment hint or peeling path, so it can keep a simple
    // uniform chunking (`head_items == 0`). The two chunkings may assign keys to CTAs differently, but top-k only
    // depends on the multiset of keys covered by the cluster.
    const offset_t chunks    = num_chunks(segment_size_u32, head_items);
    const offset_t my_chunks = num_rank_chunks(chunks, cluster_rank, cluster_size);
    // The launch coverage check reserves one extra chunk for the possible unaligned head, so every local chunk remains
    // resident for later radix passes while only the BlockLoadToShared instances are reused round-robin.
    _CCCL_ASSERT(my_chunks * offset_t{chunk_items} <= block_tile_capacity,
                 "Dynamic shared memory block_tile is too small");

    ::cuda::std::span<key_t> resident_keys;

    reset_hist();
    __syncthreads();

    {
      extract_bin_op_t extract_op(0, total_bits, decomposer_t{});
      auto add_first_pass = [&](const key_t& key) {
        const int bucket = extract_op(key);
        if (cluster_rank == 0)
        {
          atomicAdd(&temp_storage.hist[bucket], offset_t{1});
        }
        else
        {
          atomicAdd_block(&temp_storage.hist[bucket], offset_t{1});
        }
      };

      if constexpr (use_block_load_to_shared)
      {
        auto* block_keys_base = THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(block_keys_in);
        // BlockLoadToShared is non-copyable and non-movable; keep the active pipeline local and emplace-only.
        ::cuda::std::inplace_vector<block_load_t, PipelineStages> loaders;
        ::cuda::std::inplace_vector<typename block_load_t::CommitToken, PipelineStages> tokens;
        ::cuda::std::inplace_vector<::cuda::std::span<key_t>, PipelineStages> pending_spans;
        const int prologue = (::cuda::std::min) (PipelineStages, static_cast<int>(my_chunks));
        char* next_dst     = key_slots;

        for (int stage = 0; stage < prologue; ++stage)
        {
          const offset_t chunk_idx = static_cast<offset_t>(cluster_rank) + static_cast<offset_t>(stage * cluster_size);
          const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
          const bool aligned_chunk = is_aligned_chunk(block_keys_base, chunk);
          auto& loader             = loaders.emplace_back(temp_storage.load_storage[stage]);
          const ::cuda::std::span<const key_t> src{
            block_keys_base + chunk.offset, static_cast<::cuda::std::size_t>(chunk.count)};
          if (aligned_chunk)
          {
            pending_spans.emplace_back(
              loader.template CopyAsync<key_t, load_align_bytes>(available_block_tile_buffer(next_dst), src));
          }
          else
          {
            pending_spans.emplace_back(loader.template CopyAsync<key_t>(available_block_tile_buffer(next_dst), src));
          }
          next_dst = span_end(pending_spans[stage]);
          tokens.emplace_back(loader.Commit());
        }

        for (offset_t p = 0; p < my_chunks; ++p)
        {
          const int stage          = static_cast<int>(p % static_cast<offset_t>(prologue));
          const offset_t chunk_idx = static_cast<offset_t>(cluster_rank) + p * static_cast<offset_t>(cluster_size);
          const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
          loaders[stage].Wait(::cuda::std::move(tokens[stage]));
          append_contiguous_span(resident_keys, pending_spans[stage]);
          for_each_chunk_key(pending_spans[stage], add_first_pass);

          if (p + static_cast<offset_t>(prologue) < my_chunks)
          {
            const offset_t next_chunk_idx = static_cast<offset_t>(cluster_rank)
                                          + (p + static_cast<offset_t>(prologue)) * static_cast<offset_t>(cluster_size);
            const auto next_chunk         = get_chunk(next_chunk_idx, segment_size_u32, head_items);
            const bool next_aligned_chunk = is_aligned_chunk(block_keys_base, next_chunk);
            const ::cuda::std::span<const key_t> src{
              block_keys_base + next_chunk.offset, static_cast<::cuda::std::size_t>(next_chunk.count)};
            if (next_aligned_chunk)
            {
              pending_spans[stage] =
                loaders[stage].template CopyAsync<key_t, load_align_bytes>(available_block_tile_buffer(next_dst), src);
            }
            else
            {
              pending_spans[stage] =
                loaders[stage].template CopyAsync<key_t>(available_block_tile_buffer(next_dst), src);
            }
            next_dst      = span_end(pending_spans[stage]);
            tokens[stage] = loaders[stage].Commit();
          }
        }
      }
      else
      {
        for (offset_t p = 0; p < my_chunks; ++p)
        {
          const offset_t chunk_idx = static_cast<offset_t>(cluster_rank) + p * static_cast<offset_t>(cluster_size);
          const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
          key_t* const chunk_keys  = slot_keys_unpadded(static_cast<int>(p));
          const int iterations     = ::cuda::ceil_div(chunk.count, threads_per_block);
          detail::transform::unrolled_for<UnrollFactor>(iterations, [&](int j) {
            const int local = j * threads_per_block + static_cast<int>(threadIdx.x);
            if (local < chunk.count)
            {
              const key_t key =
                block_keys_in[static_cast<segment_size_val_t>(chunk.offset + static_cast<offset_t>(local))];
              chunk_keys[local] = key;
              add_first_pass(key);
            }
          });
        }
      }
      const int resident_count = span_size(resident_keys);
      _CCCL_ASSERT(resident_count == 0 || static_cast<offset_t>(resident_count) <= block_tile_capacity,
                   "Dynamic shared memory block_tile is too small");
      __syncthreads();
    }

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
#  ifndef CUB_DISABLE_CLUSTER_TOPK_EARLY_STOP
          temp_storage.broadcast_early_stop = leader_state->early_stop;
#  endif
        }
        __syncthreads();
        kth_key_bits_local = temp_storage.broadcast_kth;
#  ifndef CUB_DISABLE_CLUSTER_TOPK_EARLY_STOP
        if (temp_storage.broadcast_early_stop != ::cuda::std::uint32_t{0})
        {
          last_pass = pass;
          // Order the `broadcast_kth` load above against thread 0's
          // overwrite of `broadcast_kth` in the final filter pass below.
          // The normal loop exit gets this from the last iteration's
          // `cluster.sync()`; the early-stop path skips that sync.
          __syncthreads();
          break;
        }
#  endif
      }

      if (!is_first_pass)
      {
        // Every block (including the leader) starts each non-first pass with
        // a fresh, empty `hist`. Pass 0 was fused with the load pipeline above.
        reset_hist();
        __syncthreads();

        identify_candidates_op_t identify_op(&kth_key_bits_local, pass, total_bits, decomposer_t{});
        extract_bin_op_t extract_op(pass, total_bits, decomposer_t{});

        // Step 1: block-private histogram. The leader uses device-scope
        // `atomicAdd` so it is mutually atomic per the PTX ISA with the
        // remote device-scope `atomicAdd`s that non-leaders issue against
        // the same SMEM through DSMEM in Step 2. Non-leaders only write to
        // their own `hist[]` and keep the cheaper `atomicAdd_block`.
        // TODO(https://github.com/NVIDIA/cccl/issues/73): collapse both
        // branches onto cluster-scope atomics once
        // `cuda::thread_scope_cluster` is exposed in libcudacxx.
        auto add_hist = [&](const key_t& key) {
          if (identify_op(key) == detail::topk::candidate_class::candidate)
          {
            const int bucket = extract_op(key);
            if (cluster_rank == 0)
            {
              atomicAdd(&temp_storage.hist[bucket], offset_t{1});
            }
            else
            {
              atomicAdd_block(&temp_storage.hist[bucket], offset_t{1});
            }
          }
        };

        if constexpr (use_block_load_to_shared)
        {
          for_each_chunk_key(resident_keys, add_hist);
        }
        else
        {
          for (offset_t p = 0; p < my_chunks; ++p)
          {
            const offset_t chunk_idx = static_cast<offset_t>(cluster_rank) + p * static_cast<offset_t>(cluster_size);
            const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
            key_t* const chunk_keys  = slot_keys_unpadded(static_cast<int>(p));
            for_each_chunk_key({chunk_keys, static_cast<::cuda::std::size_t>(chunk.count)}, add_hist);
          }
        }
      }

#  ifdef CUB_ENABLE_CLUSTER_TOPK_SCAN_THEN_REDUCE
      // Step 1b (scan-then-reduce): in-place block-local `InclusiveSum`
      // over `hist[]`. The barrier publishes Step 1's atomic writes to
      // the scan's reads.
      __syncthreads();
      {
        offset_t bucket_vals[buckets_per_thread];
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < buckets_per_thread; ++j)
        {
          const int bucket = static_cast<int>(threadIdx.x) * buckets_per_thread + j;
          bucket_vals[j]   = (bucket < num_buckets) ? temp_storage.hist[bucket] : offset_t{0};
        }
        block_scan_t(temp_storage.scan_storage).InclusiveSum(bucket_vals, bucket_vals);
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < buckets_per_thread; ++j)
        {
          const int bucket = static_cast<int>(threadIdx.x) * buckets_per_thread + j;
          if (bucket < num_buckets)
          {
            temp_storage.hist[bucket] = bucket_vals[j];
          }
        }
      }
      // Cluster-wide barrier: the leader's scan write-back to `hist[]`
      // is non-atomic and would be lost if a remote Step 2 RMW
      // interleaved with it, so every block must finish its write-back
      // before anyone starts the fold.
      cluster.sync();
#  else
      // Local barrier is enough: all Step 1 / Step 2 writes to `hist[]`
      // are atomic at compatible scopes (see Step 1 dispatch). The
      // cluster-wide ordering before Step 3's leader read of `hist[]`
      // is supplied by the `cluster.sync()` further below.
      __syncthreads();
#  endif

      // Step 2: non-leader blocks fold their per-bucket values
      // (raw counts in the reduce-then-scan path, block-local inclusive
      // scans in the scan-then-reduce path) into the leader's `hist`
      // via DSMEM atomics. The leader skips this to avoid double-counting
      // its own contribution. `atomicAdd` matches the leader's
      // device-scope Step 1 atomic (see comment there).
      // TODO(https://github.com/NVIDIA/cccl/issues/73): use a
      // cluster-scope atomic once `cuda::thread_scope_cluster` is
      // exposed in libcudacxx.
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

#  ifdef CUB_ENABLE_CLUSTER_TOPK_SCAN_THEN_REDUCE
      // Hoist the leader's read of `state.k` above the upcoming
      // `cluster.sync()` so that barrier separates it from the write
      // `leader_identify_kth_bucket_from_inclusive` may issue to
      // `state.k`. Unconditional in every block to avoid predication;
      // safe because `process_impl` initializes `state` everywhere.
      const out_offset_t leader_target_k = temp_storage.state.k;
#  endif

      cluster.sync();

      // Step 3: the leader walks the merged `hist` (raw counts in the
      // reduce-then-scan path, cluster-wide inclusive scan in the
      // scan-then-reduce path) and updates the cluster-shared `state`.
      // Subsequent reads (next-pass refresh, last filter) all observe
      // these writes after the next cluster sync.
      if (cluster_rank == 0)
      {
#  ifdef CUB_ENABLE_CLUSTER_TOPK_SCAN_THEN_REDUCE
        leader_identify_kth_bucket_from_inclusive(pass, leader_target_k);
#  else
        leader_identify_kth_bucket(pass);
#  endif
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
#  ifdef CUB_ENABLE_CLUSTER_TOPK_BLOCK_AGG_OUTPUT
      // Piggyback the block-local output-counter init on the broadcast_kth
      // publish: the `__syncthreads()` below already orders these writes
      // against the per-thread `atomicAdd_block` calls in the first phase
      // of the block-aggregated path, so no dedicated init+sync is needed.
      temp_storage.block_out_cnt      = 0;
      temp_storage.block_out_back_cnt = 0;
#  endif
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

#  ifdef CUB_ENABLE_CLUSTER_TOPK_BLOCK_AGG_OUTPUT
    // Block-aggregated variant: each thread first reserves its slot in the
    // block's local counters via cheap `atomicAdd_block`, then thread 0 of
    // every block claims the block's contiguous range in the leader's
    // counters with a single DSMEM `atomicAdd` per counter. This collapses
    // up to the block's resident key count DSMEM atomics per block
    // down to two, at the cost of one extra block-wide barrier (the local
    // counters are zeroed inside the pre-existing broadcast_kth publish)
    // and one additional pass over the per-thread keys to materialize the
    // writes.

    auto count_selected = [&](const key_t& key) {
      const auto res = identify_op(key);
      if (res == detail::topk::candidate_class::selected)
      {
        atomicAdd_block(&temp_storage.block_out_cnt, out_offset_t{1});
      }
      else if (res == detail::topk::candidate_class::candidate)
      {
        atomicAdd_block(&temp_storage.block_out_back_cnt, out_offset_t{1});
      }
    };
    if constexpr (use_block_load_to_shared)
    {
      for_each_chunk_key(resident_keys, count_selected);
    }
    else
    {
      for (offset_t p = 0; p < my_chunks; ++p)
      {
        const offset_t chunk_idx = static_cast<offset_t>(cluster_rank) + p * static_cast<offset_t>(cluster_size);
        const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
        key_t* const chunk_keys  = slot_keys_unpadded(static_cast<int>(p));
        for_each_chunk_key({chunk_keys, static_cast<::cuda::std::size_t>(chunk.count)}, count_selected);
      }
    }
    __syncthreads();

    // Thread 0 of every block (leader block included) reserves the block's
    // contiguous range in the leader's counters. The returned old value is
    // the block's base; we stash it back into the same shared-memory slot
    // so the rest of the block can pick it up after the barrier without a
    // separate broadcast field. For the leader block these atomics target
    // its own shared memory; for non-leaders they go through DSMEM.
    if (threadIdx.x == 0)
    {
      const out_offset_t fwd_count    = temp_storage.block_out_cnt;
      const out_offset_t back_count   = temp_storage.block_out_back_cnt;
      temp_storage.block_out_cnt      = atomicAdd(&leader_state->out_cnt, fwd_count);
      temp_storage.block_out_back_cnt = atomicAdd(&leader_state->out_back_cnt, back_count);
    }
    __syncthreads();

    const out_offset_t fwd_base  = temp_storage.block_out_cnt;
    const out_offset_t back_base = temp_storage.block_out_back_cnt;

    if (threadIdx.x == 0)
    {
      temp_storage.block_out_cnt      = 0;
      temp_storage.block_out_back_cnt = 0;
    }
    __syncthreads();

    auto write_selected = [&](const key_t& key) {
      const auto res = identify_op(key);
      if (res == detail::topk::candidate_class::selected)
      {
        const out_offset_t pos = fwd_base + atomicAdd_block(&temp_storage.block_out_cnt, out_offset_t{1});
        block_keys_out[pos]    = key;
      }
      else if (res == detail::topk::candidate_class::candidate)
      {
        const out_offset_t back_pos = back_base + atomicAdd_block(&temp_storage.block_out_back_cnt, out_offset_t{1});
        if (back_pos < num_kth)
        {
          const out_offset_t pos = k - 1 - back_pos;
          block_keys_out[pos]    = key;
        }
      }
    };
    if constexpr (use_block_load_to_shared)
    {
      for_each_chunk_key(resident_keys, write_selected);
    }
    else
    {
      for (offset_t p = 0; p < my_chunks; ++p)
      {
        const offset_t chunk_idx = static_cast<offset_t>(cluster_rank) + p * static_cast<offset_t>(cluster_size);
        const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
        key_t* const chunk_keys  = slot_keys_unpadded(static_cast<int>(p));
        for_each_chunk_key({chunk_keys, static_cast<::cuda::std::size_t>(chunk.count)}, write_selected);
      }
    }
#  else
    auto write_selected = [&](const key_t& key) {
      const auto res = identify_op(key);
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
    };
    if constexpr (use_block_load_to_shared)
    {
      for_each_chunk_key(resident_keys, write_selected);
    }
    else
    {
      for (offset_t p = 0; p < my_chunks; ++p)
      {
        const offset_t chunk_idx = static_cast<offset_t>(cluster_rank) + p * static_cast<offset_t>(cluster_size);
        const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
        key_t* const chunk_keys  = slot_keys_unpadded(static_cast<int>(p));
        for_each_chunk_key({chunk_keys, static_cast<::cuda::std::size_t>(chunk.count)}, write_selected);
      }
    }
#  endif

    // Final cluster barrier: hold every block in the cluster until all DSMEM
    // atomics into the leader's state are complete. Without this, a fast
    // block (e.g. one whose block_tile is entirely padding) can return while another
    // block is still writing to leader-resident memory through DSMEM, which
    // surfaces as a "cluster target block not present" exception.
    cluster.sync();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void process_impl()
  {
    ::cooperative_groups::cluster_group cluster = ::cooperative_groups::this_cluster();
    const unsigned int cluster_rank             = cluster.block_rank();
    // Runtime cluster width matches the launch attribute the dispatch passed
    // to `cudaLaunchKernelExC` (or the kernel's `__cluster_dims__` on CDP).
    const unsigned int cluster_blocks = cluster.num_blocks();
    const auto segment_id             = static_cast<num_segments_val_t>(blockIdx.x / cluster_blocks);

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

    bool segment_fits_offset = true;
    if constexpr (sizeof(segment_size_val_t) > sizeof(offset_t))
    {
      segment_fits_offset =
        segment_size <= static_cast<segment_size_val_t>(::cuda::std::numeric_limits<offset_t>::max());
    }
    const auto max_cluster_tile_capacity = smem_layout_t::template cluster_tile_capacity<segment_size_val_t>(
      static_cast<int>(cluster_blocks), block_tile_capacity);
    if (!segment_fits_offset || segment_size > max_cluster_tile_capacity)
    {
      _CCCL_ASSERT(false, "Segment exceeds the selected cluster top-k cluster_tile capacity");
      return;
    }

    // Every block's thread 0 initializes its local `state`. Only the
    // leader's copy is semantically read (non-leaders reach the cluster
    // state through `leader_state`), but mirroring the writes everywhere
    // keeps the scan-then-reduce path's unconditional `state.k` load
    // safe under compute-sanitizer. Every block will reset its own
    // `hist` at the top of the per-pass loop.
    if (threadIdx.x == 0)
    {
      temp_storage.state.len          = static_cast<offset_t>(segment_size);
      temp_storage.state.k            = k;
      temp_storage.state.kth_key_bits = {};
      temp_storage.state.out_cnt      = 0;
      temp_storage.state.out_back_cnt = 0;
      temp_storage.state.early_stop   = 0;
    }
    cluster.sync();

    [[maybe_unused]] const bool ok = detail::params::dispatch_discrete(
      select_directions, segment_id, [this, &cluster, segment_id, cluster_rank, segment_size, k](auto direction_tag) {
        constexpr detail::topk::select Direction = decltype(direction_tag)::value;
        this->template run<Direction>(cluster, segment_id, cluster_rank, segment_size, k);
      });
    _CCCL_ASSERT(ok, "Unsupported select direction for cluster top-k");
  }
#endif // _CG_HAS_CLUSTER_GROUP
};
} // namespace detail::batched_topk_cluster

CUB_NAMESPACE_END
