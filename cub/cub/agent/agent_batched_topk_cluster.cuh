// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Cluster-based per-segment top-k agent.
//!
//! Prototype that exercises CUDA thread block clusters as a replacement for
//! the multi-kernel + global histogram pipeline used by cub::DeviceTopK. Each
//! cluster processes exactly one segment.
//!
//! Histogram strategy (block-private accumulation, then DSMEM merge into the leader):
//!   1. Every block lays out `hist[num_buckets]` at the same offset in its own
//!      shared memory. Each block accumulates a block-private histogram using
//!      cluster-scope shared-space `red` reductions (cheap, SMEM-local), so the
//!      leader's adds stay mutually atomic with the DSMEM folds below (see
//!      `hist_inc`).
//!   2. After a cluster-wide barrier, every non-leader block walks its
//!      histogram and folds its bucket counts into the leader block's `hist`
//!      via cluster-scope DSMEM atomics. The leader's `hist` therefore plays
//!      a dual role: its own block-private histogram in step 1, then the
//!      cluster-merged histogram after the second cluster sync.
//!   3. The leader's thread 0 prefix-scans `hist`, identifies the bucket of
//!      the k-th key, and updates the cluster-shared `state`. Every block
//!      reads the leader's packed `state.result_pair` (splitter bucket plus
//!      early-stop flag) from the leader via DSMEM at the end of each pass and
//!      folds the bucket into its own local splitter key.
//!
//! The final filter places each block's output through per-CTA shared-memory
//! atomics, seeded by a cross-CTA prefix scan of two independent 32-bit DSMEM
//! reductions (each block's selected-front and candidate-back base offsets)
//! fused directly into the placement counters; no cluster-wide output cursor is
//! kept in `state`.

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
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/__cmath/round_down.h>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__memcpy_async/elect_one.h>
#include <cuda/__memory/align_up.h>
#include <cuda/__memory/ptr_rebind.h>
#include <cuda/__ptx/instructions/cp_async_bulk.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__ptx/instructions/mbarrier_arrive.h>
#include <cuda/__ptx/instructions/mbarrier_init.h>
#include <cuda/__ptx/instructions/mbarrier_wait.h>
#include <cuda/argument>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/utility>

#include <nv/target>

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
  // Per-pass leader result, packed so every block pulls it from the leader cluster-wide in one naturally-aligned
  // 64-bit DSMEM load (instead of two separate reads). Low 32 bits: the splitter `kth_bucket`, which every block folds
  // into its own `kth_key_bits_local` (so the full splitter key is never broadcast). High 32 bits: the `early_stop`
  // flag, set when the splitter bucket holds exactly the remaining `k` (every candidate wins, so further passes cannot
  // change it).
  ::cuda::std::uint64_t result_pair;

  // Decode/encode an already-loaded `result_pair`, keeping the bit layout (see above) defined in one place.
  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static ::cuda::std::uint32_t
  kth_bucket_of(::cuda::std::uint64_t packed)
  {
    return static_cast<::cuda::std::uint32_t>(packed);
  }
  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static bool is_early_stop(::cuda::std::uint64_t packed)
  {
    return (packed >> 32) != ::cuda::std::uint64_t{0};
  }
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void set_result(::cuda::std::uint32_t bucket, bool is_stop)
  {
    result_pair = static_cast<::cuda::std::uint64_t>(bucket) | (static_cast<::cuda::std::uint64_t>(is_stop) << 32);
  }
};

// Dynamic-SMEM layout shared by dispatch and the agent. `block_tile_capacity` is the physical per-CTA resident
// capacity; `cluster_tile_capacity` is the cluster's total resident coverage. The unaligned head is staged as an edge
// in static SMEM (`edge_keys`), not a chunk slot, so the full physical capacity is usable (no head reservation).
template <typename KeyT, int ChunkBytes, int LoadAlignBytes>
struct smem_block_tile_layout
{
  static constexpr int chunk_bytes      = ChunkBytes; // one chunk == one slot; this is the slot stride
  static constexpr int chunk_items      = ChunkBytes / int{sizeof(KeyT)};
  static constexpr int load_align_items = LoadAlignBytes / int{sizeof(KeyT)};
  // Base/slot alignment: at least the load alignment, bumped up for over-aligned key types. The slot stride is exactly
  // one chunk (`ChunkBytes`), so `ChunkBytes` must be a multiple of this for consecutive slots to stay aligned, which
  // also rejects a key alignment larger than a whole chunk at compile time (it makes no sense).
  static constexpr int slot_alignment = (::cuda::std::max) (LoadAlignBytes, int{alignof(KeyT)});
  // Reserve one alignment quantum so the agent can round the dynamic-SMEM base up to `slot_alignment` (>= LoadAlign),
  // giving every bulk-copy destination the same `load_align` alignment the gmem sources already have.
  static constexpr int base_padding_bytes = slot_alignment;
  static_assert(ChunkBytes % slot_alignment == 0, "ChunkBytes must be a multiple of the load and key alignment");

  [[nodiscard]] _CCCL_HOST_DEVICE static constexpr ::cuda::std::uint32_t
  block_tile_capacity(int dynamic_smem_bytes) noexcept
  {
    const int usable_bytes = (::cuda::std::max) (0, dynamic_smem_bytes - base_padding_bytes);
    const int slots        = usable_bytes / ChunkBytes;
    return static_cast<::cuda::std::uint32_t>(slots * chunk_items);
  }

  template <typename SizeT>
  [[nodiscard]] _CCCL_HOST_DEVICE static constexpr SizeT
  cluster_tile_capacity(int cluster_blocks, ::cuda::std::uint32_t physical_block_tile_capacity) noexcept
  {
    return static_cast<SizeT>(cluster_blocks) * static_cast<SizeT>(physical_block_tile_capacity);
  }
};

// -----------------------------------------------------------------------------
// Occupancy-free cluster-blocks arithmetic (shared by host dispatch and device agent)
// -----------------------------------------------------------------------------
// Whether a segment takes the barrier-free single-CTA path: resident in one CTA (`<= block_tile_capacity`) and at/below
// the single-CTA tuning threshold. Occupancy- and head-alignment-independent, so the host fast path and the device
// collapse decision agree exactly. 64-bit math for wide segment-size types / loose bounds.
[[nodiscard]] _CCCL_HOST_DEVICE constexpr bool is_single_cta_eligible(
  ::cuda::std::uint64_t segment_size, ::cuda::std::uint64_t block_tile_capacity, int single_block_max_seg_size) noexcept
{
  return segment_size <= block_tile_capacity
      && segment_size <= static_cast<::cuda::std::uint64_t>(single_block_max_seg_size);
}

// Effective cluster blocks implied by a chunk count: a CTA joins the effective cluster iff it would own at least
// `min_chunks_per_block` chunks, clamped to `[1, cluster_blocks_cap]`. Identical arithmetic on both sides; only the cap
// differs (the live cluster size on the device, max launchable blocks on the host). `min_chunks_per_block` is
// `static_assert`ed positive, so the divide is well-defined. 64-bit math.
[[nodiscard]] _CCCL_HOST_DEVICE constexpr unsigned int effective_cluster_blocks_from_chunks(
  ::cuda::std::uint64_t chunks, int min_chunks_per_block, unsigned int cluster_blocks_cap) noexcept
{
  const auto blocks = chunks / static_cast<::cuda::std::uint64_t>(min_chunks_per_block);
  return static_cast<unsigned int>(
    ::cuda::std::clamp(blocks, ::cuda::std::uint64_t{1}, static_cast<::cuda::std::uint64_t>(cluster_blocks_cap)));
}

// Cluster top-k agent
// -----------------------------------------------------------------------------
// Cluster blocks is a runtime value (see `process_impl` for the readback), so
// it is not a template parameter; per-block block_tile layout is still controlled
// by the template parameters below.
//
// `Determinism` / `TieBreak` carry the requested `cuda::execution` requirements down from the dispatch layer. Any
// deterministic guarantee selects the cluster-wide, index-ordered tie-break scan (and the blocked chunk partition it
// depends on) over the nondeterministic racing atomics; `not_guaranteed` keeps the atomics. On the deterministic path
// `prefer_larger_index` reverses the scan order so the largest-index ties win (else the smallest).
template <int ThreadsPerBlock,
          int HistogramItemsPerThread,
          int PipelineStages,
          int ChunkBytes,
          int LoadAlignBytes,
          int BitsPerPass,
          int TieBreakItemsPerThread,
          int SingleBlockMaxSegSize,
          int MinChunksPerBlock,
          int CopyItemsPerThread,
          ::cuda::execution::determinism::__determinism_t Determinism,
          ::cuda::execution::tie_break::__tie_break_t TieBreak,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
struct agent_batched_topk_cluster
{
  // ---------------------------------------------------------------------------
  // Types / constants
  // ---------------------------------------------------------------------------
  using key_it_t   = it_value_t<KeyInputItItT>;
  using key_t      = it_value_t<key_it_t>;
  using value_it_t = it_value_t<ValueInputItItT>;
  using value_t    = it_value_t<value_it_t>;

  // Keys-only when the value payload type is `cub::NullType` (mirrors the baseline batched top-k agent). The value
  // iterators are then never dereferenced and the final filter's value writes are compiled out.
  static constexpr bool is_keys_only = ::cuda::std::is_same_v<value_t, cub::NullType>;

  using segment_size_val_t = typename ::cuda::args::__traits<SegmentSizeParameterT>::element_type;
  using num_segments_val_t = typename ::cuda::args::__traits<NumSegmentsParameterT>::element_type;

  // 32-bit covers every supported segment: the public entry caps the statically-known maximum segment size at 2^21, so
  // a runtime value exceeding its declared bound is a caller precondition violation (undefined behavior). Unsigned
  // because all offsets, ranks, and block counts are non-negative (segment sizes are clamped to >= 0 upstream).
  // `prime_placement_counters` also returns its two prefix lanes packed into one `uint64_t`, which needs 32-bit lanes.
  using offset_t     = ::cuda::std::uint32_t;
  using out_offset_t = ::cuda::std::uint32_t;
  using state_t      = cluster_topk_state<key_t, offset_t, out_offset_t>;
  using key_prefix_t = typename state_t::key_prefix_t;

  // See the class comment: deterministic guarantees select the index-ordered scan; `is_tie_reversed` flips its order.
  static constexpr bool need_determinism =
    Determinism != ::cuda::execution::determinism::__determinism_t::__not_guaranteed;
  static constexpr bool is_tie_reversed =
    TieBreak == ::cuda::execution::tie_break::__tie_break_t::__prefer_larger_index;

  // Push direction of the cross-CTA prefix scan (`prime_placement_counters`). The leader must be *last* in scan
  // order so it derives its own (merged-away) counts from the predecessor sum: deterministic-prefer-smaller puts the
  // leader at the last effective rank and scans ascending; every other config (deterministic-prefer-larger and the
  // whole non-deterministic path) keeps the leader at rank 0 and scans descending, which makes rank 0 last. Matches the
  // `leader_rank` computed in `run`.
  static constexpr bool is_scan_descending = !(need_determinism && !is_tie_reversed);

  // The deterministic final scan visits chunks in global-index order and bails early (`final_filter_should_stop`), so
  // keeping the *first-visited* chunks resident lets it skip re-reading the streamed overflow. Ascending visits low
  // indices first (the default low-resident split); descending (prefer-larger) visits high indices first, so we flip
  // the split to keep the high-index chunks resident (see `run`), restoring symmetry. Never set on the
  // non-deterministic path.
  static constexpr bool is_residency_reversed = need_determinism && is_tie_reversed;

  // Segments at or below this size that also fit resident in one CTA take the single-CTA fast path (see
  // `process_impl`).
  static constexpr int single_block_max_seg_size = SingleBlockMaxSegSize;

  // A CTA joins a segment's effective cluster only if it would own at least this many chunks (see `run`). At 1 the
  // effective cluster is just the CTAs that receive any chunk. Must be positive: it is the divisor in
  // `effective_cluster_blocks_from_chunks` and a zero-chunk CTA has no work.
  static constexpr int min_chunks_per_block = MinChunksPerBlock;
  static_assert(min_chunks_per_block >= 1, "min_chunks_per_block must be positive");

  // Enable the per-segment effective-single-CTA runtime path only when the host could not size the launch to each
  // segment's exact size: any per-segment sequence (`!is_single_value`) or a `deferred` single value. For host-exact
  // `immediate`/`constant` singles the dispatch already picks the matching cluster blocks, so the logic is compiled
  // out.
  static constexpr bool enable_runtime_single_cta =
    !::cuda::args::__traits<SegmentSizeParameterT>::is_single_value
    || ::cuda::args::__traits<SegmentSizeParameterT>::is_deferred;

  static constexpr int threads_per_block          = ThreadsPerBlock;
  static constexpr int histogram_items_per_thread = HistogramItemsPerThread;
  static constexpr int load_align_bytes           = LoadAlignBytes;
  static constexpr int bits_per_pass              = BitsPerPass;
  static constexpr int tie_break_items_per_thread = TieBreakItemsPerThread;
  static constexpr int copy_items_per_thread      = CopyItemsPerThread; // select-all copy fast path

  // Static upper bound on segment size: exact for constant/immediate sizes, the type maximum for runtime sizes.
  static constexpr ::cuda::std::int64_t static_max_segment_size =
    ::cuda::args::__traits<SegmentSizeParameterT>::highest;

  // Segments small enough to always be single-CTA resident (one contiguous SMEM span, see `run`) need at most
  // `ceil(static_max_segment_size / threads_per_block)` sweep rounds. We clamp each per-thread unroll down to that
  // bound to trim predication/registers on sub-tile segments. Larger/unbounded types keep the full unroll (codegen
  // unchanged); the guard also keeps the rounds arithmetic in `int` range.
  static constexpr bool should_clamp_items_to_segment =
    static_max_segment_size > 0 && static_max_segment_size < single_block_max_seg_size;
  static constexpr int segment_rounds_ceil =
    should_clamp_items_to_segment
      ? static_cast<int>(
          ::cuda::ceil_div(static_max_segment_size, static_cast<::cuda::std::int64_t>(threads_per_block)))
      : 0;
  static constexpr int segment_rounds_floor =
    should_clamp_items_to_segment ? static_cast<int>(static_max_segment_size / threads_per_block) : 0;

  // Clamp a per-thread unroll down to the segment's bounded round count (only when `should_clamp_items_to_segment`);
  // larger/unbounded segments keep the full tuning width, and the guard keeps the rounds arithmetic in `int`.
  [[nodiscard]] static constexpr int clamp_unroll(int rounds, int tuning)
  {
    return should_clamp_items_to_segment ? ::cuda::std::clamp(rounds, 1, tuning) : tuning;
  }

  // Two clamp flavors. The `floor` clamp pairs with an unpredicated main loop over full tiles plus a single
  // non-unrolled remainder loop (the chunk helper `for_each_chunk_key_impl` and the copy fast path); the `ceil` clamp
  // keeps the whole resident segment inside one tile for fully-predicated loops (the final filters' `process_tiles`).
  static constexpr int histogram_items_per_thread_clamped =
    clamp_unroll(segment_rounds_floor, histogram_items_per_thread);
  static constexpr int tie_break_items_per_thread_clamped =
    clamp_unroll(segment_rounds_ceil, tie_break_items_per_thread);
  static constexpr int copy_items_per_thread_clamped = clamp_unroll(segment_rounds_floor, copy_items_per_thread);
  static_assert(histogram_items_per_thread_clamped >= 1, "histogram_items_per_thread_clamped must be positive");
  static_assert(tie_break_items_per_thread_clamped >= 1, "tie_break_items_per_thread_clamped must be positive");
  static_assert(copy_items_per_thread_clamped >= 1, "copy_items_per_thread_clamped must be positive");

  static constexpr int num_buckets      = 1 << bits_per_pass;
  using smem_layout_t                   = smem_block_tile_layout<key_t, ChunkBytes, LoadAlignBytes>;
  static constexpr int chunk_items      = smem_layout_t::chunk_items;
  static constexpr int load_align_items = smem_layout_t::load_align_items;
  static constexpr int slot_alignment   = smem_layout_t::slot_alignment;

  // (pointer, count) carrier for a contiguous run of SMEM-staged keys, used only where the length must travel with the
  // base across call boundaries (the resident window and the overflow stream's per-stage view); elsewhere: `key_t*` +
  // count.
  using smem_keys_t = ::cuda::std::span<key_t>;

  // Tie-break unroll for either filter's streamed overflow, which feeds `process_tiles` one chunk slot at a time: clamp
  // items so the tile (`threads_per_block * items`) stays within a chunk, bounding the per-tile early-exit to <= chunk
  // granularity. Resident/edge regions keep the full `tie_break_items_per_thread_clamped`. Floors at 1.
  static constexpr int tie_break_items_streamed =
    ::cuda::std::clamp(chunk_items / threads_per_block, 1, tie_break_items_per_thread_clamped);
  static_assert(tie_break_items_streamed >= 1, "tie_break_items_streamed must be positive");

  static_assert(PipelineStages > 0);
  // `load_phase` tracks one parity bit per stage in a 32-bit mask.
  static_assert(PipelineStages <= 32, "PipelineStages must fit in the 32-bit per-stage phase mask");
  // `__block_elect_one` elects via a full-warp `__shfl_sync` + `elect.sync`, so the block must be whole warps.
  static_assert(ThreadsPerBlock % detail::warp_threads == 0, "ThreadsPerBlock must be a multiple of the warp size");
  static_assert(CopyItemsPerThread > 0, "copy_items_per_thread must be positive");
  static_assert(HistogramItemsPerThread > 0, "histogram_items_per_thread must be positive");
  static_assert(TieBreakItemsPerThread > 0, "tie_break_items_per_thread must be positive");
  static_assert(ChunkBytes > 0);
  static_assert(LoadAlignBytes > 0);
  static_assert(ChunkBytes % LoadAlignBytes == 0, "ChunkBytes must be a multiple of LoadAlignBytes");
  // The hybrid load relies on the aligned bulk-copy path being exact (no scalar guard), which requires the load
  // alignment to be a power of two and at least the bulk-copy minimum alignment. Mirrors `is_valid_cluster_policy`
  // (checked by the dispatch on the whole policy); repeated here to also guard direct agent instantiations.
  static_assert(LoadAlignBytes >= detail::bulk_copy_min_align, "LoadAlignBytes must be >= bulk_copy_min_align");
  static_assert(::cuda::is_power_of_two(LoadAlignBytes), "LoadAlignBytes must be a power of two");
  static_assert(chunk_items > 0);

  using decomposer_t = detail::identity_decomposer_t;
  // Resident/streamed chunks are pulled into the block_tile with elected-thread `cp.async.bulk`/TMA copies against
  // raw per-stage mbarriers; see the async bulk-copy pipeline helpers below.

  // The aligned bulk (TMA) path tiles gmem into `load_align`-aligned units and dense-packs them by bytes into smem.
  // That is only sound when a `load_align` unit is a whole number of keys and a key has no internal padding; otherwise
  // (e.g. `float3`: 12 bytes, 4-byte aligned) a `load_align` boundary would slice a key or leave bubbles in the
  // resident range, so such types fall back to plain per-element loads/stores.
  static constexpr bool key_is_bulk_tileable =
    int{sizeof(key_t)} == int{alignof(key_t)} && LoadAlignBytes % int{sizeof(key_t)} == 0;
  static constexpr bool use_block_load_to_shared =
    THRUST_NS_QUALIFIER::is_trivially_relocatable_v<key_t> && THRUST_NS_QUALIFIER::is_contiguous_iterator_v<key_it_t>
    && key_is_bulk_tileable;

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
  // `front_local_cnt`/`back_local_cnt` are the final-filter output-slot counters, but they first serve as the cross-CTA
  // scan accumulators: peers add their selected/candidate counts into them through DSMEM (`add_remote_prefix`) while
  // this block seeds its own back base locally, leaving each counter primed with this block's absolute region base
  // (front = `sel_prefix`, back = `num_selected + cand_prefix`). Because peers reach them over DSMEM they must sit at
  // an identical offset in every block's storage. `num_strictly_selected` accumulates this block's strictly-selected
  // count across passes, and `my_candidates` holds the last pass's splitter-bucket count.
  struct _TempStorage
  {
    offset_t hist[num_buckets];
    state_t state;
    offset_t front_local_cnt;
    offset_t back_local_cnt;
    offset_t num_strictly_selected;
    offset_t my_candidates;
    typename block_scan_t::TempStorage scan_storage;
    // One mbarrier handle per pipeline stage, shared by the resident load and the overflow stream and reused
    // (ping-ponged) across radix passes; all are initialized once up front by `init_load_barriers`.
    ::cuda::std::uint64_t load_mbar[PipelineStages];
    // Persistent unaligned boundary edges (block-load path only): the head prefix (`[0, head_edge_cap_items)`, on rank
    // 0) and the peeled tail suffix (`[head_edge_cap_items, 2 * head_edge_cap_items)`, on the tail owner whenever it is
    // unaligned), each strictly `< load_align_items` keys. Loaded once in the first pass and folded into every pass +
    // the final filter. Block-local (never reached through DSMEM).
    key_t edge_keys[2 * load_align_items];
  };
  // Split point of `edge_keys`: head edge in `[0, head_edge_cap_items)`, tail edge in `[head_edge_cap_items, 2 *
  // head_edge_cap_items)`.
  static constexpr int head_edge_cap_items = load_align_items;

  struct chunk_desc
  {
    offset_t offset;
    int count;
  };

  // Chunk `chunk_idx` of the aligned region, beginning on a `load_align` boundary (`head_items + chunk_idx *
  // chunk_items`, with `head_items` aligning the base and `chunk_items` a multiple of `load_align_items`): so every
  // chunk has a zero prefix and only the last chunk can carry an unaligned suffix (its trailing `< load_align_items`
  // items). A chunk's aligned bulk (what the guard-free TMA path copies) is `round_down(count, load_align_items)`
  // (`== count` for interior chunks); the tail's `count - bulk` remainder is peeled into `edge_keys`.
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE chunk_desc
  get_chunk(offset_t chunk_idx, offset_t segment_size, offset_t head_items) const
  {
    const offset_t offset = head_items + chunk_idx * offset_t{chunk_items};
    // Callers pass a chunk index `< chunks`, so the chunk is non-empty and unsigned `remaining` below can't underflow.
    _CCCL_ASSERT(offset < segment_size, "get_chunk: chunk index lies past the segment end");
    const offset_t remaining = segment_size - offset;
    return {offset, static_cast<int>((::cuda::std::min) (remaining, offset_t{chunk_items}))};
  }

  // Assignment of the cluster's global chunk indices `[0, chunks)` to its CTAs. A rank owns `count` chunks; its i-th
  // owned chunk has global index `global_index(i) = first + i * stride`. The single mapping point lets the rest of the
  // agent (resident load, overflow stream, per-pass scans) stay agnostic to the layout chosen by
  // `make_chunk_partition`.
  struct chunk_partition
  {
    offset_t first; // global index of this rank's first owned chunk
    offset_t stride; // distance between consecutive owned chunks (`cluster_blocks` strided, `1` blocked)
    offset_t count; // number of chunks owned by this rank

    [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE offset_t global_index(offset_t local) const
    {
      _CCCL_ASSERT(local < count, "global_index: rank-local chunk index out of range");
      return first + local * stride;
    }
  };

  // Decides which global chunks a cluster rank owns. Both layouts keep chunk 0 on rank 0 (which also stages the head
  // edge) and the tail (chunk `chunks-1`) on a single rank, and leave the per-chunk alignment, the resident/streaming
  // split, and the streaming ping-pong untouched, because all of those depend only on the global chunk index, not on
  // which rank owns it.
  //
  //   * Strided (default): chunk `i` goes to rank `i % cluster_blocks`, so each CTA walks `first, first+S, first+2S,
  //   ...`.
  //   * Blocked (deterministic path): each CTA owns a contiguous run of `ceil_div(chunks, S)` chunks (the last
  //     non-empty rank gets the short remainder).
  //
  // The blocked layout is required by the deterministic tie-break (its cross-CTA scan assumes CTA-rank order matches
  // ascending contiguous global-index ranges), so it is selected exactly when `need_determinism` is set.
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE chunk_partition
  make_chunk_partition(offset_t chunks, unsigned int cluster_rank, unsigned int cluster_blocks) const
  {
    // Idle ranks never reach here -- `compute_segment_layout` gives them an explicit empty partition.
    _CCCL_ASSERT(cluster_rank < cluster_blocks, "make_chunk_partition assumes a working rank (rank < cluster_blocks)");
    if constexpr (need_determinism)
    {
      const offset_t chunks_per_cta = ::cuda::ceil_div(chunks, static_cast<offset_t>(cluster_blocks));
      const offset_t first          = static_cast<offset_t>(cluster_rank) * chunks_per_cta;
      const offset_t count = (first < chunks) ? (::cuda::std::min) (chunks_per_cta, chunks - first) : offset_t{0};
      return {first, offset_t{1}, count};
    }
    else
    {
      // Strided count: rank `cluster_rank` owns the global indices `cluster_rank, cluster_rank + cluster_blocks, ...`
      // that stay `< chunks`, i.e. `ceil((chunks - cluster_rank) / cluster_blocks)`, written as
      // `(chunks - 1 - cluster_rank) / cluster_blocks + 1` for exact integer arithmetic. The `cluster_rank >= chunks`
      // guard (rank owns nothing) keeps the subtraction from underflowing the unsigned `offset_t`.
      const offset_t count =
        (cluster_rank < chunks) ? static_cast<offset_t>((chunks - 1 - cluster_rank) / cluster_blocks + 1) : offset_t{0};
      return {static_cast<offset_t>(cluster_rank), static_cast<offset_t>(cluster_blocks), count};
    }
  }

  // Stage a small (`< load_align_items` items) unaligned run -- a boundary edge (head prefix / tail suffix) -- from
  // gmem `src` into SMEM `dst` and fold it into the first pass in one strided sweep:
  // each thread copies *and* folds the same indices it owns (`local % threads_per_block == threadIdx.x`), folding from
  // the just-loaded register rather than from SMEM. These runs cannot go through the aligned (16-byte-aligned dst,
  // guard-free) BlockLoadToShared path. Fusing the copy and the fold means no thread ever reads a key another thread
  // wrote, so the first-pass fold needs no barrier after the staging *by construction* -- not by a coincidental match
  // between two separate traversals. Later passes and the final filter re-read `dst` (from `edge_keys` / the resident
  // span) only after a pass-boundary barrier, so their reads are independently safe.
  //
  // TODO(cccl): an asymmetric-alignment BlockLoadToShared API (independent begin/end alignment, e.g. an aligned begin
  // with an arbitrary end) would let a boundary chunk be loaded with a single aligned-bulk + in-place edge call and
  // remove these hand-rolled copies entirely.
  template <typename Apply>
  _CCCL_DEVICE _CCCL_FORCEINLINE void stage_and_fold_edge(key_t* dst, const key_t* src, int count, Apply&& apply) const
  {
    _CCCL_ASSERT(count >= 0 && count <= head_edge_cap_items, "a boundary edge must fit in its edge_keys slot");
    for (int local = tid; local < count; local += threads_per_block)
    {
      const key_t key = src[local];
      dst[local]      = key;
      apply(key);
    }
  }

  // Apply `apply(key, local)` to each key of a contiguous chunk (`local` is the key's strided lane index in
  // `[0, chunk_count)`), processing tiles of `Unroll * threads_per_block` keys. Each full tile is loaded into registers
  // by one unrolled loop, then handed to `apply` by a second. Splitting the loads out matters for the histogram passes:
  // `apply`'s SMEM atomics can't be proven disjoint from the SMEM key reads, so a fused loop would interleave each load
  // with its atomic instead of hoisting the whole load wave ahead. `Unroll` is the caller's clamped (floor) items per
  // thread, so the sub-tile remainder is handled by a single non-unrolled fused block-stride loop bounded by the count.
  template <int Unroll, typename Apply>
  _CCCL_DEVICE _CCCL_FORCEINLINE void for_each_chunk_key_impl(const key_t* keys, int chunk_count, Apply&& apply) const
  {
    constexpr int tile   = Unroll * threads_per_block;
    const int full_tiles = ::cuda::round_down(chunk_count, tile);

    _CCCL_PRAGMA_NOUNROLL()
    for (int tile_base = 0; tile_base < full_tiles; tile_base += tile)
    {
      key_t regs[Unroll];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < Unroll; ++i)
      {
        regs[i] = keys[tile_base + i * threads_per_block + tid];
      }
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < Unroll; ++i)
      {
        const int local = tile_base + i * threads_per_block + tid;
        apply(regs[i], local);
      }
    }

    // Sub-tile remainder
    _CCCL_PRAGMA_NOUNROLL()
    for (int local = full_tiles + tid; local < chunk_count; local += threads_per_block)
    {
      apply(keys[local], local);
    }
  }

  template <int Unroll, typename F>
  _CCCL_DEVICE _CCCL_FORCEINLINE void for_each_chunk_key(const key_t* chunk_keys, int count, F&& f) const
  {
    for_each_chunk_key_impl<Unroll>(chunk_keys, count, [&](const key_t& key, int) {
      f(key);
    });
  }

  // ---------------------------------------------------------------------------
  // Async bulk-copy pipeline helpers (raw mbarrier + cp.async.bulk via cuda::ptx)
  // ---------------------------------------------------------------------------
  // Inlines BlockLoadToShared's internals (one mbarrier per stage, a single elected thread issuing the TMA copy +
  // transaction arrival) without its reference member or per-call CommitToken: per-stage wait state is one bit of the
  // per-thread `load_phase` mask, so the pipeline loops spill nothing. SM 9.0+ only (gated by `NV_PROVIDES_SM_90`).

  // Front-load every stage mbarrier before any bulk copy uses it. Init needs no leader or uniform path, so all threads
  // init distinct stages in parallel via a block-stride loop. The caller's block `__syncthreads()` orders these
  // generic-proxy inits before the first `cp.async.bulk` issue, so no `fence.proxy.async` is needed (cf.
  // `BlockLoadToShared`, which front-loads init + `__syncthreads()` likewise).
  _CCCL_DEVICE _CCCL_FORCEINLINE void init_load_barriers()
  {
    _CCCL_PRAGMA_NOUNROLL()
    for (int stage = tid; stage < PipelineStages; stage += threads_per_block)
    {
      ::cuda::ptx::mbarrier_init(&temp_storage.load_mbar[stage], 1u);
    }
  }

  // Issue one aligned global->shared (TMA) bulk copy into `dst` on stage `stage`'s mbarrier from the block leader,
  // which also arrives with the transaction byte count (an empty copy arrives with zero so the phase still completes).
  // The stage mbarrier must already be initialized (see `init_load_barriers`). Each call must be paired, in issue order
  // per stage, with a matching `wait_stage(stage)`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void issue_bulk_copy(int stage, char* dst, ::cuda::std::span<const key_t> src)
  {
    _CCCL_ASSERT(stage >= 0 && stage < PipelineStages, "pipeline stage index out of range");
    // Only the block leader (see `is_load_leader`) drives the mbarrier, for a uniform branch and better mbarrier
    // codegen.
    if (!is_load_leader)
    {
      return;
    }
    const int num_bytes = static_cast<int>(::cuda::std::size(src)) * int{sizeof(key_t)};
    // A source is one chunk's aligned bulk (`round_down(count, load_align_items)`): a whole load-align unit, <= a slot.
    _CCCL_ASSERT(num_bytes % load_align_bytes == 0, "a bulk copy size must be a whole load-alignment unit");
    _CCCL_ASSERT(num_bytes <= ChunkBytes, "a bulk copy must not exceed one chunk slot");
    if (num_bytes > 0)
    {
      // The TMA path only requires `bulk_copy_min_align` (>= 16), but the aligned base + `load_align`-multiple bulk
      // packing give every destination the stronger `load_align_bytes` alignment (matching the gmem sources).
      _CCCL_ASSERT(::cuda::is_aligned(dst, load_align_bytes),
                   "block_tile destination must satisfy the shared-memory bulk-copy alignment");
#if __cccl_ptx_isa >= 860
      ::cuda::ptx::cp_async_bulk(
        ::cuda::ptx::space_shared,
        ::cuda::ptx::space_global,
        dst,
        ::cuda::std::data(src),
        num_bytes,
        &temp_storage.load_mbar[stage]);
#else // __cccl_ptx_isa < 860
      ::cuda::ptx::cp_async_bulk(
        ::cuda::ptx::space_cluster,
        ::cuda::ptx::space_global,
        dst,
        ::cuda::std::data(src),
        num_bytes,
        &temp_storage.load_mbar[stage]);
#endif // __cccl_ptx_isa >= 860
    }
    ::cuda::ptx::mbarrier_arrive_expect_tx(
      ::cuda::ptx::sem_release,
      ::cuda::ptx::scope_cta,
      ::cuda::ptx::space_shared,
      &temp_storage.load_mbar[stage],
      static_cast<::cuda::std::uint32_t>(num_bytes));
  }

  // Wait for stage `stage`'s copy to land (all threads spin on its current parity), then flip that stage's parity bit.
  _CCCL_DEVICE _CCCL_FORCEINLINE void wait_stage(int stage)
  {
    _CCCL_ASSERT(stage >= 0 && stage < PipelineStages, "pipeline stage index out of range");
    const ::cuda::std::uint32_t parity = (load_phase >> stage) & 1u;
    while (!::cuda::ptx::mbarrier_try_wait_parity(&temp_storage.load_mbar[stage], parity))
    {
    }
    load_phase ^= (::cuda::std::uint32_t{1} << stage);
  }

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  // Per-segment, per-rank geometry computed once by `compute_segment_layout` at the top of `run`: the head-aligned
  // chunking, the effective (non-idle) cluster width and this rank's partition of it, the leader/idle roles, and the
  // resident/streaming/edge split of this rank's chunks. All fields are loop-invariant for the segment; the
  // streaming-only intermediates (`overflow_base`, `resident_slots_cap`, `stream_slots`, `my_chunks`) are kept because
  // the overflow-streaming helpers (`init_overflow_stream`, `run_pass`, ...) read them.
  struct segment_layout
  {
    offset_t segment_size_off;
    key_it_t block_keys_in;
    const key_t* block_keys_base;
    offset_t head_items;
    offset_t chunks;
    unsigned int eff_cluster_blocks;
    bool is_idle_rank;
    chunk_partition part;
    unsigned int leader_rank;
    state_t* leader_state;
    offset_t my_chunks;
    offset_t my_resident_chunks;
    offset_t overflow_chunks;
    offset_t resident_base;
    offset_t overflow_base;
    offset_t resident_slots_cap;
    offset_t stream_slots;
    int head_edge_len_items;
    int tail_edge_len_items;
  };

  // ---------------------------------------------------------------------------
  // Per-thread members
  // ---------------------------------------------------------------------------
  _TempStorage& temp_storage;
  KeyInputItItT d_key_segments_it;
  KeyOutputItItT d_key_segments_out_it;
  ValueInputItItT d_value_segments_it;
  ValueOutputItItT d_value_segments_out_it;
  SegmentSizeParameterT segment_sizes;
  KParameterT k_param;
  SelectDirectionParameterT select_directions;
  NumSegmentsParameterT num_segments;
  char* key_slots;
  offset_t block_tile_capacity;
  // Per-thread mbarrier phase parity, one bit per pipeline stage (see `wait_stage`); the resident load and the
  // overflow stream keep their per-stage issue/wait calls balanced so each bit tracks its mbarrier's phase.
  ::cuda::std::uint32_t load_phase{};
  // The single block leader (warp 0's elected lane) that drives the bulk copies. Elected once at construction (the
  // block constructs convergently) and cached so the pipeline reuses it instead of re-electing per copy.
  const bool is_load_leader = ::cuda::device::__block_elect_one();
  // This thread's block-local index, signed for the agent's block-stride loops and cached to avoid re-casting
  // `threadIdx.x` at each site.
  const int tid = static_cast<int>(threadIdx.x);
  // Effective cluster geometry the radix/scan/filter path runs on, set once by `init_effective_cluster`: the launched
  // cluster, or rank 0 alone (`cluster_blocks == 1`) when a small segment collapsed onto a single resident CTA. The
  // select-all fast path predates the collapse and uses the raw hardware sregs instead.
  unsigned int cluster_rank   = 0u;
  unsigned int cluster_blocks = 1u;
  bool is_single_cta          = true;
  // Per-segment constants for this block, set once and then read directly by the helpers instead of being threaded
  // through their signatures. `segment_id`/`segment_size`/`k` are set in `process_impl`; `layout` is filled by
  // `compute_segment_layout` on the radix path (the select-all fast path and redundant CTAs never touch it).
  num_segments_val_t segment_id{};
  segment_size_val_t segment_size{};
  out_offset_t k{};
  segment_layout layout{};

  // Overflow-streaming slot geometry + ping-pong cursor, set once per segment by `init_overflow_stream` and carried
  // across every radix pass and the final filter (the "Overflow streaming" section documents the reuse scheme). Inert
  // when this rank has no overflow chunks (`init_overflow_stream` still runs, but streaming then touches no slot).
  int stream_slot_base = 0;
  int stream_stages    = 1;
  bool stream_is_forward{true};
  bool stream_is_primed{false};
  ::cuda::std::uint32_t stream_inflight_mask{0};

  _CCCL_DEVICE_API _CCCL_FORCEINLINE agent_batched_topk_cluster(
    TempStorage& temp_storage_,
    KeyInputItItT d_key_segments_it_,
    KeyOutputItItT d_key_segments_out_it_,
    ValueInputItItT d_value_segments_it_,
    ValueOutputItItT d_value_segments_out_it_,
    SegmentSizeParameterT segment_sizes_,
    KParameterT k_param_,
    SelectDirectionParameterT select_directions_,
    NumSegmentsParameterT num_segments_,
    char* key_slots_,
    offset_t block_tile_capacity_)
      : temp_storage(temp_storage_.Alias())
      , d_key_segments_it(d_key_segments_it_)
      , d_key_segments_out_it(d_key_segments_out_it_)
      , d_value_segments_it(d_value_segments_it_)
      , d_value_segments_out_it(d_value_segments_out_it_)
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
  // SM 9.0+ only. `NV_IF_TARGET` strips the call on NVCC's sub-SM-9.0 device passes, so `process_impl` (and the
  // cluster/async PTX in it and its callees) is never ODR-used there and never reaches ptxas.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Process()
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (process_impl();));
  }

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE void reset_hist()
  {
    for (int i = tid; i < num_buckets; i += threads_per_block)
    {
      temp_storage.hist[i] = 0;
    }
  }

  // ---------------------------------------------------------------------------
  // Block-private histogram atomics (shared-space `red` via inline PTX)
  // ---------------------------------------------------------------------------
  // The per-key histogram update is the hottest path. A builtin `atomicAdd(&temp_storage.hist[bucket], 1)` lowers to
  // the same warp-aggregated shared atomic (`ATOMS.POPC.INC.32`), but recomputes `&hist[bucket]` from the 64-bit
  // generic base of `temp_storage` on every key; the inline `red` adds a pre-hoisted 32-bit shared base instead
  // (measured ~3-4% fewer total instructions across this huge agent).

  // The 32-bit shared address of `hist[0]`, hoisted once per histogram region so the per-key update stays a pure
  // 32-bit add.
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::uint32_t hist_base32() const
  {
    return static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(temp_storage.hist));
  }

  // Cluster-scope `red.add` of `v` into this CTA's own shared `u32` at 32-bit `.shared::cta` address `addr` (e.g. from
  // `__cvta_generic_to_shared`). Cluster scope -- not the address space -- is what keeps it mutually atomic with peers'
  // DSMEM `red.add`s into the same location (`hist_fold_remote`, `add_remote_prefix`), so this CTA's own address needs
  // no `mapa`; it lowers to the same shared-atomic SASS as cta scope. `hist_inc` passes a compile-time `1`, which
  // current ptxas still folds into the warp-aggregated `ATOMS.POPC.INC.32` (verified in SASS) despite the register
  // operand.
  _CCCL_DEVICE _CCCL_FORCEINLINE static void add_local_shared_cluster(::cuda::std::uint32_t addr, offset_t v)
  {
    asm volatile("red.relaxed.cluster.shared::cta.add.u32 [%0], %1;" : : "r"(addr), "r"(v) : "memory");
  }

  // Increment this block's own histogram bucket by one. Applied unconditionally (non-leaders and the lone CTA
  // included): the cluster-scope add costs nothing there and it drops the leader branch.
  _CCCL_DEVICE _CCCL_FORCEINLINE void hist_inc(::cuda::std::uint32_t base32, int bucket)
  {
    _CCCL_ASSERT(bucket >= 0 && bucket < num_buckets, "histogram bucket index out of range");
    const ::cuda::std::uint32_t addr = base32 + static_cast<::cuda::std::uint32_t>(bucket) * sizeof(offset_t);
    add_local_shared_cluster(addr, offset_t{1});
  }

  // Step 2: a non-leader folds one bucket into the leader's histogram through DSMEM. `own_bucket_addr32` is the
  // bucket's address in this block's own window; since every CTA's shared window is laid out identically, `mapa` remaps
  // it to rank 0 to form the leader's `shared::cluster` address (no 64-bit pointer, no memory descriptor). Cluster
  // scope makes it mutually atomic with the leader's `hist_inc` adds.
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  hist_fold_remote(::cuda::std::uint32_t own_bucket_addr32, offset_t v, unsigned int leader_rank)
  {
    ::cuda::std::uint32_t remote;
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(own_bucket_addr32), "r"(leader_rank));
    asm volatile("red.relaxed.cluster.shared::cluster.add.u32 [%0], %1;" : : "r"(remote), "r"(v) : "memory");
  }

  // Generic pointer to this CTA's `state` as seen in the CTA at cluster rank `rank` (reached over DSMEM) -- the PTX
  // equivalent of cooperative_groups' `map_shared_rank`, via `mapa.u64` (generic-address form).
  _CCCL_DEVICE _CCCL_FORCEINLINE state_t* map_state_to_rank(unsigned int rank)
  {
    const ::cuda::std::uint64_t own = reinterpret_cast<::cuda::std::uint64_t>(&temp_storage.state);
    ::cuda::std::uint64_t remote;
    asm("mapa.u64 %0, %1, %2;" : "=l"(remote) : "l"(own), "r"(rank));
    return reinterpret_cast<state_t*>(remote);
  }

  // Adds `push_front`/`push_cand` into the front/back placement counters of the CTA at cluster rank `target_rank`
  // through DSMEM (mirrors `hist_fold_remote`: `mapa` to `target_rank`, then a cluster-scope `red.add` per counter).
  // Two independent 32-bit reductions rather than one 64-bit one: `red.add.u64` on shared memory is emulated with a
  // CAS spin loop, whereas `red.add.u32` is a native shared-memory atomic. Drives the cross-CTA selected/candidate
  // prefix scan (see `prime_placement_counters`).
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  add_remote_prefix(unsigned int target_rank, offset_t push_front, offset_t push_cand)
  {
    const ::cuda::std::uint32_t own_front =
      static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(&temp_storage.front_local_cnt));
    const ::cuda::std::uint32_t own_back =
      static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(&temp_storage.back_local_cnt));
    ::cuda::std::uint32_t remote_front, remote_back;
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_front) : "r"(own_front), "r"(target_rank));
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_back) : "r"(own_back), "r"(target_rank));
    asm volatile("red.relaxed.cluster.shared::cluster.add.u32 [%0], %1;"
                 :
                 : "r"(remote_front), "r"(push_front)
                 : "memory");
    asm volatile("red.relaxed.cluster.shared::cluster.add.u32 [%0], %1;"
                 :
                 : "r"(remote_back), "r"(push_cand)
                 : "memory");
  }

  // Folds the back region base (`num_selected`) into this CTA's own `back_local_cnt`, in parallel with the peers'
  // cluster-scope candidate-prefix pushes into the same counter (`add_remote_prefix`).
  _CCCL_DEVICE _CCCL_FORCEINLINE void seed_local_back(offset_t v)
  {
    add_local_shared_cluster(static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(&temp_storage.back_local_cnt)),
                             v);
  }

  // Parallel prefix sum (cub::BlockScan) over the leader's merged histogram
  // plus identification of the bucket holding the k-th item. Each thread of
  // the leader block contributes `buckets_per_thread` consecutive buckets in
  // a blocked arrangement; entries past `num_buckets` contribute zero. The
  // single (thread, slot) pair that owns the k-th bucket writes the per-pass
  // state. The caller must guarantee the leader block has finished its DSMEM
  // merge before invoking this.
  _CCCL_DEVICE _CCCL_FORCEINLINE void leader_identify_kth_bucket()
  {
    // Capture `state.k` before the scan: the owning thread overwrites it in the loop below, so any later read would
    // race with that write.
    const out_offset_t target_k = temp_storage.state.k;
    // Every pass keeps the remaining `k` inside the chosen bucket, so `1 <= target_k <= len`.
    _CCCL_ASSERT(target_k >= 1, "leader scan: remaining k must stay positive");
    _CCCL_ASSERT(target_k <= temp_storage.state.len, "leader scan: remaining k cannot exceed the candidate count");

    offset_t hist_vals[buckets_per_thread];
    offset_t prefixes[buckets_per_thread];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < buckets_per_thread; ++j)
    {
      const int bucket = tid * buckets_per_thread + j;
      hist_vals[j]     = (bucket < num_buckets) ? temp_storage.hist[bucket] : offset_t{0};
    }

    block_scan_t(temp_storage.scan_storage).ExclusiveSum(hist_vals, prefixes);

    // Exactly one (thread, slot) pair satisfies
    // `prefix < target_k <= prefix + hist_val`.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < buckets_per_thread; ++j)
    {
      const int bucket = tid * buckets_per_thread + j;
      if (bucket < num_buckets && prefixes[j] < target_k && prefixes[j] + hist_vals[j] >= target_k)
      {
        const out_offset_t new_k = target_k - static_cast<out_offset_t>(prefixes[j]);
        const offset_t new_len   = hist_vals[j];
        temp_storage.state.len   = new_len;
        temp_storage.state.k     = new_k;
        // Publish the splitter bucket and the early-stop flag (set when the bucket holds exactly the remaining `k`);
        // see `result_pair`.
        temp_storage.state.set_result(
          static_cast<::cuda::std::uint32_t>(bucket), static_cast<out_offset_t>(new_len) == new_k);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Overflow streaming
  // ---------------------------------------------------------------------------
  // Re-streams a rank's "overflow" chunks (those that do not fit its resident SMEM region) from gmem through a fixed,
  // round-robin set of `stream_stages` (<= `PipelineStages`) streaming slots, reused across every radix pass and the
  // final filter. It ping-pongs the iteration order across calls (`stream_is_forward`) so the `stream_stages`
  // turn-around chunks one pass leaves resident in the streaming slots are reused by the next with no reload; the
  // remaining `overflow_chunks - stream_stages` are reloaded from gmem each pass. `compute_segment_layout` sizes the
  // reservation `stream_slots = min(PipelineStages, excess)` (`excess = my_chunks - full_slots`) that
  // `init_overflow_stream` adopts as `stream_stages`, so a streaming rank reloads exactly `excess` chunks per pass --
  // the reserved slots only ever buy reuse of the turn-around chunks, never a reload-free pass. The resident region
  // occupies slots `[0, resident_slots)`, the streaming region `[stream_slot_base, stream_slot_base + stream_stages)`.
  // All geometry comes from `layout`; the only per-stage state
  // is one bit each of `stream_inflight_mask` (set while a copy is in flight) and the `load_phase` parity, so a
  // spillable per-stage array is avoided (the read span is recomputed on demand by `stage_span`).

  // Adopt the streaming-slot window (`stream_slots`, sized by `compute_segment_layout`) and reset the ping-pong/priming
  // cursor for this segment. `run` may then flip `stream_is_forward` for the deterministic filter's entry parity.
  _CCCL_DEVICE _CCCL_FORCEINLINE void init_overflow_stream()
  {
    stream_slot_base = static_cast<int>(layout.resident_slots_cap);
    // Use the whole reserved streaming region as pipeline stages for maximal ping-pong reuse; the `1` floor covers the
    // no-op (`overflow_chunks == 0`) case, where streaming never touches a slot (asserted below).
    stream_stages        = (layout.stream_slots > offset_t{0}) ? static_cast<int>(layout.stream_slots) : 1;
    stream_is_forward    = true;
    stream_is_primed     = false;
    stream_inflight_mask = 0;
    // Both chunk windows must lie inside this rank's `part.count` (resident first, streamed from `overflow_base`).
    _CCCL_ASSERT(layout.my_chunks == layout.part.count && layout.my_resident_chunks <= layout.my_chunks
                   && layout.overflow_base <= layout.my_chunks
                   && layout.overflow_chunks <= layout.my_chunks - layout.overflow_base,
                 "overflow stream chunk windows escape the rank's partition");
    // The streaming region is carved from the tile's slots and capped by the pipeline depth.
    _CCCL_ASSERT(stream_slot_base >= 0 && layout.stream_slots <= static_cast<offset_t>(PipelineStages)
                   && static_cast<offset_t>(stream_slot_base) + layout.stream_slots
                        <= block_tile_capacity / static_cast<offset_t>(chunk_items),
                 "overflow stream slots escape the block tile or pipeline");
    _CCCL_ASSERT(layout.overflow_chunks == 0 || stream_stages <= static_cast<int>(layout.overflow_chunks),
                 "streaming depth exceeds the overflow chunk count");
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void issue_load(int stage, offset_t overflow_idx)
  {
    _CCCL_ASSERT(overflow_idx < layout.overflow_chunks, "overflow chunk index out of range");
    _CCCL_ASSERT(stage >= 0 && stage < stream_stages, "overflow stage index exceeds the reserved streaming slots");
    _CCCL_ASSERT((stream_inflight_mask & (::cuda::std::uint32_t{1} << stage)) == 0,
                 "cannot issue a load into a streaming stage that is still in flight");
    const offset_t chunk_idx = layout.part.global_index(layout.overflow_base + overflow_idx);
    const auto chunk         = get_chunk(chunk_idx, layout.segment_size_off, layout.head_items);
    // Every chunk begins on a `load_align` boundary, so the guard-free aligned (TMA bulk) path applies. The global-
    // last chunk's unaligned suffix is always peeled into `edge_keys`, so streaming just its aligned bulk excludes
    // it. For every interior chunk `bulk == count`.
    const offset_t bulk =
      ::cuda::round_down(static_cast<offset_t>(chunk.count), static_cast<offset_t>(load_align_items));
    _CCCL_ASSERT(::cuda::is_aligned(layout.block_keys_base + chunk.offset, load_align_bytes),
                 "overflow stream received a chunk with an unaligned start");
    char* const dst = key_slots + (stream_slot_base + stage) * ChunkBytes;
    const ::cuda::std::span<const key_t> src{
      layout.block_keys_base + chunk.offset, static_cast<::cuda::std::size_t>(bulk)};
    issue_bulk_copy(stage, dst, src);
    stream_inflight_mask |= (::cuda::std::uint32_t{1} << stage);
  }

  // Shared-memory view of the chunk currently resident in `stage`'s slot: the slot index is a pure function of `stage`
  // and the length is recomputed from `overflow_idx` (no spillable per-stage array). Returns the aligned bulk only (the
  // always-peeled tail suffix is excluded).
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE smem_keys_t stage_span(int stage, offset_t overflow_idx) const
  {
    const auto chunk = get_chunk(
      layout.part.global_index(layout.overflow_base + overflow_idx), layout.segment_size_off, layout.head_items);
    return smem_keys_t(
      ::cuda::ptr_rebind<key_t>(key_slots + (stream_slot_base + stage) * ChunkBytes),
      static_cast<int>(::cuda::round_down(static_cast<offset_t>(chunk.count), static_cast<offset_t>(load_align_items))));
  }

  // Shared driver for one overflow pass. `block_apply(stage, overflow_idx)` folds the chunk `overflow_idx` resident
  // in streaming slot `stage` (block-load path); `generic_apply(chunk)` folds an overflow chunk read straight from
  // gmem (fallback). `mid()` runs at most once per pass, positioned to overlap the caller's resident-chunk work with
  // in-flight copies: after the first reload wave (`stream_stages` visits) is issued but before it is waited on
  // (block-load path), or before the gmem loop (fallback). A phase-1 early stop on the block-load path skips it, since
  // nothing is then left to place; it must be block-uniform with no unmatched barrier. `should_continue()` is polled
  // once after each consumed chunk (before its refill copy is issued); returning false breaks the stream so the final
  // filter bails once the top-k is placed. Its result must be block-uniform (all lanes break together, else the
  // post-break barrier deadlocks) and it is evaluated by every lane, so it may contain a barrier (both final filters'
  // do, to read their placement counters block-wide); the histogram passes an always-true predicate. An early break
  // can leave prefetches in flight, so the pass drains the remaining stages before returning (a full pass ends with an
  // empty `stream_inflight_mask`, so the drain is a no-op).
  template <typename BlockApply, typename GenericApply, typename Mid, typename Continue>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  run_pass(BlockApply&& block_apply, GenericApply&& generic_apply, Mid&& mid, Continue&& should_continue)
  {
    _CCCL_ASSERT(stream_inflight_mask == ::cuda::std::uint32_t{0},
                 "an overflow pass must begin with no copies in flight");
    if (layout.overflow_chunks == 0)
    {
      mid();
      return;
    }

    if constexpr (use_block_load_to_shared)
    {
      // First ever call: prime the streaming slots. Subsequent calls inherit the previous pass's resident tail, which
      // (because the order ping-pongs) is exactly the first `stream_stages` chunks of this direction.
      if (!stream_is_primed)
      {
        // Wait for all threads to leave the resident load's final wait before re-arming its shared mbarriers; else
        // the phase advances twice and a lagging thread misses the flip and spins forever.
        __syncthreads();
        for (int i = 0; i < stream_stages; ++i)
        {
          const offset_t overflow_idx =
            stream_is_forward ? static_cast<offset_t>(i) : (layout.overflow_chunks - 1 - static_cast<offset_t>(i));
          issue_load(static_cast<int>(overflow_idx % static_cast<offset_t>(stream_stages)), overflow_idx);
        }
        stream_is_primed = true;
      }

      // Consume the `i`-th visit (its ping-pong-ordered position is `overflow_idx`): wait for its slot, fold its keys
      // via `block_apply`, then prefetch the chunk `stream_stages` visits ahead into the slot just freed (a barrier
      // guards the slot before the async copy can overwrite the data the block was just reading). Returns false once
      // `should_continue()` reports the top-k fully placed -- polled before the prefetch so we never launch a copy we
      // would only drain again; the up-to-`stream_stages - 1` prefetches already in flight (from earlier visits or
      // priming) are drained after the loop.
      const auto consume = [&](offset_t i) -> bool {
        const offset_t overflow_idx = stream_is_forward ? i : (layout.overflow_chunks - 1 - i);
        const int stage             = static_cast<int>(overflow_idx % static_cast<offset_t>(stream_stages));
        if (stream_inflight_mask & (::cuda::std::uint32_t{1} << stage))
        {
          wait_stage(stage);
          stream_inflight_mask &= ~(::cuda::std::uint32_t{1} << stage);
        }
        block_apply(stage, overflow_idx);

        if (!should_continue())
        {
          return false;
        }

        const offset_t next_step = i + static_cast<offset_t>(stream_stages);
        if (next_step < layout.overflow_chunks)
        {
          const offset_t next_overflow_idx = stream_is_forward ? next_step : (layout.overflow_chunks - 1 - next_step);
          __syncthreads();
          issue_load(stage, next_overflow_idx);
        }
        return true;
      };

      // Phase 1: consume the first `stream_stages` visits (the chunks reused from the previous pass, already resident
      // in the streaming slots), which issues the prefetch loads for this pass's reload wave into the freed slots.
      bool is_stopped      = false;
      const offset_t split = (::cuda::std::min) (static_cast<offset_t>(stream_stages), layout.overflow_chunks);
      for (offset_t i = 0; i < split; ++i)
      {
        if (!consume(i))
        {
          is_stopped = true;
          break;
        }
      }

      if (!is_stopped)
      {
        // The reload wave is now in flight; run the caller's resident-chunk work to hide its latency before waiting.
        // Skipped on an early stop: the stream only breaks once this CTA's whole contribution is placed, so no resident
        // key can still be required (the non-deterministic filter folds its resident keys as `mid`).
        mid();

        // Phase 2: consume the remaining visits (their loads were issued in phase 1 and overlapped `mid`).
        for (offset_t i = split; i < layout.overflow_chunks; ++i)
        {
          if (!consume(i))
          {
            break;
          }
        }
      }

      // Drain prefetches still in flight before returning: an early break leaves outstanding bulk copies whose
      // mbarriers were never waited, and they must complete before the block can exit (their slots are never read).
      // `stream_inflight_mask` is block-uniform (set/cleared under uniform control flow), so the trip count and each
      // collective `wait_stage` are uniform across the block.
      while (stream_inflight_mask != ::cuda::std::uint32_t{0})
      {
        const int drain_stage = __ffs(static_cast<int>(stream_inflight_mask)) - 1;
        wait_stage(drain_stage);
        stream_inflight_mask &= ~(::cuda::std::uint32_t{1} << drain_stage);
      }
      stream_is_forward = !stream_is_forward;
    }
    else
    {
      // Generic fallback: no async SMEM pipeline, so resident work cannot hide load latency here. Fold the resident
      // chunks first (preserving the prior ordering), then read the overflow keys straight from gmem each pass (no
      // SMEM reuse), with the walk still snaking for L2 locality.
      mid();
      for (offset_t i = 0; i < layout.overflow_chunks; ++i)
      {
        const offset_t overflow_idx = stream_is_forward ? i : (layout.overflow_chunks - 1 - i);
        const offset_t chunk_idx    = layout.part.global_index(layout.overflow_base + overflow_idx);
        const auto chunk            = get_chunk(chunk_idx, layout.segment_size_off, layout.head_items);
        generic_apply(chunk);
        if (!should_continue())
        {
          break;
        }
      }
      stream_is_forward = !stream_is_forward;
    }
  }

  // Apply `f(key)` to every overflow key once in the current ping-pong direction. `UnrollFactor` partially unrolls the
  // generic (gmem fallback) fold loop; callers pass their clamped items-per-thread. See `run_pass` for the overlap
  // semantics of `mid`.
  template <int UnrollFactor, typename F, typename Mid>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_pass(F&& f, Mid&& mid)
  {
    run_pass(
      [&](int stage, offset_t overflow_idx) {
        const auto keys = stage_span(stage, overflow_idx);
        for_each_chunk_key<UnrollFactor>(keys.data(), static_cast<int>(keys.size()), f);
      },
      [&](const auto& chunk) {
        const int iterations = ::cuda::ceil_div(chunk.count, threads_per_block);
        _CCCL_PRAGMA_UNROLL(UnrollFactor)
        for (int j = 0; j < iterations; ++j)
        {
          const int local = j * threads_per_block + tid;
          if (local < chunk.count)
          {
            f(layout.block_keys_in[static_cast<segment_size_val_t>(chunk.offset + static_cast<offset_t>(local))]);
          }
        }
      },
      static_cast<Mid&&>(mid),
      [] {
        return true;
      });
  }

  // Overload with no interleaved work, for the fused first pass where the resident keys are still being streamed in
  // by the BlockLoadToShared pipeline (rather than already resident in SMEM).
  template <int UnrollFactor, typename F>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_pass(F&& f)
  {
    process_pass<UnrollFactor>(static_cast<F&&>(f), [] {});
  }

  // -------------------------------------------------------------------------
  // Per-direction implementation
  // -------------------------------------------------------------------------
  // Split halves of the cluster-wide barrier (replaces cooperative_groups' `cluster.sync()`). `arrive` releases this
  // CTA's prior writes and signals arrival; `wait` acquires all CTAs' writes once every CTA has arrived. Both are
  // `.aligned` since every thread reaches them under a uniform branch. Issuing independent, block-local work between
  // them hides the cluster-arrival latency.
  _CCCL_DEVICE _CCCL_FORCEINLINE static void cluster_arrive()
  {
    asm volatile("barrier.cluster.arrive.release.aligned;" : : : "memory");
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE static void cluster_wait()
  {
    asm volatile("barrier.cluster.wait.acquire.aligned;" : : : "memory");
  }

  // Cluster-wide barrier: arrive immediately followed by wait, no work window exploited.
  _CCCL_DEVICE _CCCL_FORCEINLINE static void cluster_barrier()
  {
    cluster_arrive();
    cluster_wait();
  }

  // Synchronize the segment's cluster. A single-CTA "cluster" keeps all state block-local, so `__syncthreads()` orders
  // it and the cluster-scoped barrier is unnecessary. `is_single_cta` is the agent's effective-cluster member (see
  // `init_effective_cluster`): 1 when a small segment collapsed onto rank 0, not the raw cluster size. It is
  // per-segment uniform across the surviving block(s), so the branch is reached uniformly.
  _CCCL_DEVICE _CCCL_FORCEINLINE static void cluster_or_block_sync(bool is_single_cta)
  {
    if (is_single_cta)
    {
      __syncthreads();
    }
    else
    {
      cluster_barrier();
    }
  }

  // Split form of `cluster_or_block_sync`, for overlapping the cluster-arrival latency with independent block-local
  // work: `arrive`, then the work, then `wait`. A single-CTA "cluster" has no arrival latency to hide and no cross-CTA
  // state, so it takes the whole `__syncthreads()` at `arrive` and its `wait` is a no-op. In the multi-CTA case the
  // work between the two must touch only block-local state (a cross-CTA read before `wait` may miss a peer's writes).
  _CCCL_DEVICE _CCCL_FORCEINLINE static void cluster_or_block_arrive(bool is_single_cta)
  {
    if (is_single_cta)
    {
      __syncthreads();
    }
    else
    {
      cluster_arrive();
    }
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE static void cluster_or_block_wait(bool is_single_cta)
  {
    if (!is_single_cta)
    {
      cluster_wait();
    }
  }

  // Exclusive cross-CTA prefix scan fused with priming the final-filter placement counters. Each working CTA pushes its
  // `push_front`/`push_cand` counts into every successor's front/back counter in `is_scan_descending` order (the leader
  // is last and pushes to nobody, so it holds the full predecessor sum) and folds its own back-region base
  // `num_selected` into its own back counter. All are commutative `red.add`s into the counters zeroed in `process_impl`
  // (so a single post-push barrier suffices), leaving `front_local_cnt = sel_prefix` and
  // `back_local_cnt = num_selected + cand_prefix` -- the absolute output-slot bases `place_one` expects.
  //
  // Returns this CTA's exclusive prefix packed as `(sel_prefix << 32) | cand_prefix` for the driver's region math;
  // `is_single_cta` yields 0 (front stays 0, back is just `num_selected`). The successor pushes are lane-parallel (each
  // thread owns a strided slice); all threads see CTA-uniform counts, so the guard and the barrier stay uniform.
  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::uint64_t
  prime_placement_counters(offset_t push_front, offset_t push_cand, offset_t num_selected)
  {
    if (is_single_cta)
    {
      // No peers: the front base is 0 (untouched since init) and the back base is just `num_selected`.
      if (tid == 0)
      {
        temp_storage.back_local_cnt = num_selected;
      }
      __syncthreads();
      return ::cuda::std::uint64_t{0};
    }
    // Fold this CTA's back base in parallel with the peers pushing their candidate counts into the same counter.
    if (tid == 0)
    {
      seed_local_back(num_selected);
    }
    if (push_front != offset_t{0} || push_cand != offset_t{0})
    {
      // Only working (non-leader) ranks reach here with a nonzero count; idle ranks / the leader push 0.
      _CCCL_ASSERT(cluster_rank < layout.eff_cluster_blocks, "a nonzero prefix count must come from a working rank");
      if constexpr (is_scan_descending)
      {
        _CCCL_PRAGMA_NOUNROLL()
        for (unsigned int rank = threadIdx.x; rank < cluster_rank; rank += threads_per_block) // lower ranks follow;
                                                                                              // leader last
        {
          add_remote_prefix(rank, push_front, push_cand);
        }
      }
      else
      {
        // Higher ranks follow. Stops at `eff_cluster_blocks` since idle ranks own nothing; the leader at the last
        // effective rank is last.
        _CCCL_PRAGMA_NOUNROLL()
        for (unsigned int rank = cluster_rank + 1u + threadIdx.x; rank < layout.eff_cluster_blocks;
             rank += threads_per_block)
        {
          add_remote_prefix(rank, push_front, push_cand);
        }
      }
    }
    // TODO(cccl): idle ranks arrive here only to keep this barrier reachable; a sub-cluster mbarrier over the working
    // ranks would let them exit (see the pass loop).
    cluster_or_block_sync(is_single_cta);
    // The local seed guarantees the back counter carries at least `num_selected` (postcondition of the fused prime).
    _CCCL_ASSERT(temp_storage.back_local_cnt >= num_selected,
                 "back counter must include the seeded region base after the scan");
    const offset_t sel_prefix  = temp_storage.front_local_cnt;
    const offset_t cand_prefix = temp_storage.back_local_cnt - num_selected;
    // Every thread must finish snapshotting the primed bases before any lane's `place_one` mutates the same counters:
    // the leading boundary edge in each driver has no barrier of its own before its first placement atomic.
    __syncthreads();
    return (static_cast<::cuda::std::uint64_t>(sel_prefix) << 32) | static_cast<::cuda::std::uint64_t>(cand_prefix);
  }

  // Deterministic final-filter state: the `run()`-local tie-break values (counts, prefixes, region extents) plus the
  // mutable `running`/`is_tie_active` scan cursor, bundled into one POD so the flattened filter methods below take a
  // single `state` argument instead of ~16. A deterministic-path-only local of `write_deterministic_topk`; never
  // instantiated on the non-deterministic path. Segment-invariant inputs (`segment_id`, `k`, and the `layout.*`
  // geometry) are read from the agent directly rather than duplicated here. `resident_front` carries the resident
  // window as a `smem_keys_t` view.
  template <detail::topk::select SelectDirection>
  struct det_filter_state
  {
    using identify_op_t = detail::topk::identify_candidates_op_t<key_t, SelectDirection, bits_per_pass, decomposer_t>;

    identify_op_t identify_op;
    it_value_t<KeyOutputItItT> block_keys_out;
    out_offset_t num_back;
    out_offset_t my_front;
    offset_t sel_prefix;
    offset_t cand_prefix;
    offset_t my_cand_count;
    offset_t front_seg_base;
    smem_keys_t resident_front;
    int front_count;
    bool is_select_all_cand_cta;
    bool is_resident_terminal;
    bool is_head_edge_terminal;
    bool is_tail_edge_terminal;
    // Mutable per-invocation tie-break scan cursor.
    offset_t running;
    bool is_tie_active;
  };

  // ---------------------------------------------------------------------------
  // Final-filter placement (shared) + deterministic filter
  // ---------------------------------------------------------------------------
  // The classify/place helpers below (`place_one`, `place_tile`, `process_tiles`, ...) are shared by both filters. The
  // deterministic-only sweeps that follow are named agent methods over a `det_filter_state` (`state`), driven by
  // `run_filter`; `write_deterministic_topk` computes `state` and calls `run_filter(state)`. See
  // `write_deterministic_topk` for the front/back placement scheme. All methods are `_CCCL_FORCEINLINE`.

  // Shared by both final filters: for each key written to `block_keys_out[pos]`, load the associated input value at the
  // key's segment-local index `seg_idx` from gmem and store it at the same slot. Compiled out (and never dereferences
  // the null value iterators) in keys-only builds; `segment_id` is loop-invariant, so the per-segment iterators hoist
  // out of the writes.
  _CCCL_DEVICE _CCCL_FORCEINLINE void final_filter_write_value(out_offset_t pos, offset_t seg_idx)
  {
    if constexpr (!is_keys_only)
    {
      _CCCL_ASSERT(pos < k && seg_idx < layout.segment_size_off,
                   "value write must land in the top-k output and read inside the segment");
      auto block_vals_in  = d_value_segments_it[segment_id];
      auto block_vals_out = d_value_segments_out_it[segment_id];
      block_vals_out[pos] = block_vals_in[static_cast<segment_size_val_t>(seg_idx)];
    }
  }

  // Per-item placement code stored in `flags[]` by the load/classify pass and reused by `place_tile`/`emit_indexed`, so
  // each key is classified exactly once.
  static constexpr offset_t flag_none      = 0; // rejected or out-of-range: no placement, absent from the tie scan
  static constexpr offset_t flag_candidate = 1; // tie candidate: routed to the back, counted by the boundary scan
  static constexpr offset_t flag_selected  = 2; // strictly selected: routed to the front

  // Map thread `tid`'s item slot `i` to its position within the current tile. Blocked is used by the straddling
  // deterministic CTA while it resolves ties (`process_tiles` phase A): `emit_indexed`'s `BlockScan` ranks candidates
  // in blocked order, which makes the tie ranks segment-index-ordered. Striped (consecutive threads hit consecutive
  // addresses: no SMEM bank conflicts on the staged reads, coalesced loads in the gmem fallback) is used everywhere
  // else -- the whole non-deterministic filter, every non-straddling deterministic CTA, and the straddling CTA's tiles
  // past the boundary (phase B). `classify_tile` and `place_tile` must use the same `Blocked` so `keys[i]`/`flags[i]`
  // denote the same element; `emit_indexed` is intrinsically blocked.
  template <bool Blocked, int ItemsPerThread>
  _CCCL_DEVICE _CCCL_FORCEINLINE int tile_item_pos(int tile_base, int i) const
  {
    if constexpr (Blocked)
    {
      return tile_base + tid * ItemsPerThread + i;
    }
    else
    {
      return tile_base + i * threads_per_block + tid;
    }
  }

  // Shared per-key arrival placement core (called by `place_tile` for both final filters): route one key by class to
  // the front (selected) or back (candidate) counter, each primed with its region base by `prime_placement_counters`
  // so the SMEM atomic returns the absolute output slot with no per-key offset. The uniform `out < k` guard drops
  // losing candidates but always accepts selected keys; pairs builds also copy the key's value from `seg_idx`.
  // (Deterministic index-ordered tie resolution goes through `emit_indexed` instead.)
  template <class KeyOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  place_one(KeyOutIt block_keys_out, bool is_cand, const key_t& key, offset_t seg_idx)
  {
    offset_t* const counter = is_cand ? &temp_storage.back_local_cnt : &temp_storage.front_local_cnt;
    const out_offset_t out  = static_cast<out_offset_t>(atomicAdd(counter, offset_t{1}));
    // The front counter never overruns the selected region, so only losing candidates may exceed `k`.
    _CCCL_ASSERT(is_cand || out < k, "a strictly-selected key must always land within the top-k output");
    if (out < k)
    {
      block_keys_out[out] = key;
      final_filter_write_value(out, seg_idx);
    }
  }

  // Uniform "all placed" predicate: true once this block has emitted all `my_front` strictly-selected keys and
  // resolved its ties. Callers must `__syncthreads()` first (the counter reads are block-wide and must resynchronize
  // lanes that raced ahead through the barrier-free tiles). Polled only at critical points -- between regions and
  // before each streaming bulk copy -- never per tile.
  template <detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE bool final_filter_should_stop(const det_filter_state<SelectDirection>& state)
  {
    // Counters run absolute (primed with the region base), so completion compares against each region's end. Every
    // region is visited at most once, so neither counter can pass its end.
    const offset_t num_selected = static_cast<offset_t>(k) - static_cast<offset_t>(state.num_back);
    const offset_t front_end    = state.sel_prefix + static_cast<offset_t>(state.my_front);
    const offset_t back_end     = num_selected + state.cand_prefix + state.my_cand_count;
    _CCCL_ASSERT(temp_storage.front_local_cnt <= front_end && temp_storage.back_local_cnt <= back_end,
                 "final-filter placement counters exceeded this CTA's assigned work");
    const bool is_front_done = temp_storage.front_local_cnt >= front_end;
    // Straddling/above CTAs finish the back when `is_tie_active` clears; an `is_select_all_cand_cta` (which never
    // clears it) finishes once all `my_cand_count` of its candidates are placed.
    const bool is_back_done = !state.is_tie_active || (temp_storage.back_local_cnt >= back_end);
    return is_front_done && is_back_done;
  }

  // Arrival-order placement for one tile: each in-range key is routed by `place_one` (both regions fill forward). The
  // candidate atomic advances win or lose; arrival order is fine because either every candidate wins (select-all) or
  // the lazy crossing tile is overwritten in index order later. `do_arrival == false` (terminal tile only) skips
  // candidates here and leaves them to `emit_indexed`; the lazy path passes `true` and its crossing tile is later
  // overwritten by `emit_indexed`.
  template <bool Reversed, bool Blocked, int ItemsPerThread, class State>
  _CCCL_DEVICE _CCCL_FORCEINLINE void place_tile(
    State& state,
    const key_t (&keys)[ItemsPerThread],
    const offset_t (&flags)[ItemsPerThread],
    offset_t seg_base,
    int count,
    int tile_base,
    bool do_arrival)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      const bool is_cand = flags[i] == flag_candidate;
      if (flags[i] == flag_none || (is_cand && !do_arrival))
      {
        continue;
      }
      const int pos          = tile_item_pos<Blocked, ItemsPerThread>(tile_base, i);
      const offset_t seg_idx = seg_base + static_cast<offset_t>(Reversed ? (count - 1 - pos) : pos);
      place_one(state.block_keys_out, is_cand, keys[i], seg_idx);
    }
  }

  // Index-ordered back placement for one tile: a block scan over the candidate mask gives each candidate a
  // deterministic rank (`base` plus the count of preceding candidates in the tile). A winner (rank below `num_back`)
  // lands at forward output slot `num_selected + rank`, matching `place_tile`. Returns the tile's candidate total. Used
  // where the K-boundary falls in this tile (terminal tile, or the lazy path's crossing tile).
  template <bool Reversed, int ItemsPerThread, class State>
  _CCCL_DEVICE _CCCL_FORCEINLINE offset_t emit_indexed(
    State& state,
    const key_t (&keys)[ItemsPerThread],
    const offset_t (&flags)[ItemsPerThread],
    offset_t seg_base,
    int count,
    int tile_base,
    offset_t base)
  {
    const offset_t num_selected = static_cast<offset_t>(k) - static_cast<offset_t>(state.num_back); // front region size
    // `flags[]` is 3-valued (none/candidate/selected); the scan must count candidates only, so reduce to a 0/1 mask.
    offset_t cand[ItemsPerThread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      cand[i] = (flags[i] == flag_candidate) ? offset_t{1} : offset_t{0};
    }
    offset_t excl[ItemsPerThread];
    offset_t tile_total = 0;
    block_scan_t(temp_storage.scan_storage).ExclusiveSum(cand, excl, tile_total);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      if (cand[i] != offset_t{0})
      {
        const offset_t global_rank = base + excl[i];
        if (global_rank < static_cast<offset_t>(state.num_back))
        {
          // `emit_indexed` is deterministic-only, and its `BlockScan` ranks assume a blocked arrangement.
          const int pos             = tile_item_pos<true, ItemsPerThread>(tile_base, i);
          const out_offset_t out    = static_cast<out_offset_t>(num_selected + global_rank);
          const offset_t seg_idx    = seg_base + static_cast<offset_t>(Reversed ? (count - 1 - pos) : pos);
          state.block_keys_out[out] = keys[i];
          final_filter_write_value(out, seg_idx);
        }
      }
    }
    return tile_total;
  }

  // Load and 3-way classify one tile's items into `keys`/`flags` (see `flag_*`), re-run by `place_tile`/`emit_indexed`
  // without touching `identify_op` again. `Blocked` picks the thread->element arrangement (`tile_item_pos`); the source
  // is `smem_src` (folded in-region position) or gmem `block_keys_in`. Out-of-range lanes stay `flag_none`.
  template <bool Blocked, int ItemsPerThread, bool FromSmem, bool Reversed, class State>
  _CCCL_DEVICE _CCCL_FORCEINLINE void classify_tile(
    State& state,
    [[maybe_unused]] const key_t* smem_src,
    offset_t seg_base,
    int count,
    int tile_base,
    key_t (&keys)[ItemsPerThread],
    offset_t (&flags)[ItemsPerThread])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      const int pos = tile_item_pos<Blocked, ItemsPerThread>(tile_base, i);
      flags[i]      = flag_none;
      if (pos < count)
      {
        // In-region index: forward, or (`Reversed`) counting down from the last element. Shared by the `FromSmem`
        // local read and the segment-local value index.
        const int folded_pos = Reversed ? (count - 1 - pos) : pos;
        if constexpr (FromSmem)
        {
          keys[i] = smem_src[folded_pos];
        }
        else
        {
          keys[i] = layout.block_keys_in[static_cast<segment_size_val_t>(seg_base + static_cast<offset_t>(folded_pos))];
        }
        const auto cls = state.identify_op(keys[i]);
        flags[i]       = (cls == detail::topk::candidate_class::candidate)
                         ? flag_candidate
                         : (cls == detail::topk::candidate_class::selected ? flag_selected : flag_none);
      }
    }
  }

  // Classify (`classify_tile`) and place each of `count` keys, tiled by `threads_per_block * ItemsPerThread`.
  // `FromSmem` picks the key source; `Reversed` walks the span high-to-low. Both final filters share this via
  // `Deterministic`:
  //   * `false` (a `nondet_filter_state`, always striped): every tile placed in arrival order (`place_tile` with
  //     `do_arrival == true`); no scan/tie-state/barrier, `region_is_terminal` unused.
  //   * `true` (a `det_filter_state`): resolves the boundary ties in segment-index order via `emit_indexed`, so callers
  //     walk the regions in global-index order and carry `state.running`/`state.is_tie_active` across tiles;
  //     `region_is_terminal` marks the CTA's last. Only the boundary-straddling CTA resolves ties and is dispatched
  //     `Blocked`; other deterministic CTAs never scan and are dispatched striped (see `write_deterministic_topk`).
  // `Blocked` is the *entry* arrangement (`tile_item_pos`). The blocked deterministic path is two-phase: it loads
  // blocked only while ties remain (phase A, `emit_indexed`'s scan needs it) and switches once to striped for the
  // remaining strictly-selected/rejected keys (phase B).
  template <int ItemsPerThread, bool FromSmem, bool Reversed, bool Deterministic, bool Blocked, class State>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tiles(
    State& state,
    [[maybe_unused]] const key_t* smem_src,
    offset_t seg_base,
    int count,
    [[maybe_unused]] bool region_is_terminal)
  {
    // The SMEM-source clause is exempt when reading from gmem (`FromSmem == false`, the generic overflow fallback
    // passes `nullptr`) or for an empty tile.
    _CCCL_ASSERT(count >= 0 && seg_base <= layout.segment_size_off
                   && static_cast<offset_t>(count) <= layout.segment_size_off - seg_base
                   && (!FromSmem || count == 0 || smem_src != nullptr),
                 "process_tiles: region must be a valid in-segment source span");
    constexpr int tile_size = threads_per_block * ItemsPerThread;
    int tile_base           = 0;

    if constexpr (Deterministic && Blocked)
    {
      // Blocked is dispatched only for the boundary-straddling CTA, which always has ties and is never select-all.
      _CCCL_ASSERT(!state.is_select_all_cand_cta, "blocked deterministic path is only for the straddling CTA");

      // Phase A: while ties remain, load blocked (`emit_indexed`'s scan needs it) and resolve them. A region entered
      // past the crossing (its ties already placed) has `is_tie_active == false`, so phase A is skipped and everything
      // falls through to phase B.
      for (; state.is_tie_active && tile_base < count; tile_base += tile_size)
      {
        key_t keys[ItemsPerThread];
        offset_t flags[ItemsPerThread];
        classify_tile</*Blocked=*/true, ItemsPerThread, FromSmem, Reversed>(
          state, smem_src, seg_base, count, tile_base, keys, flags);

        // The boundary tile places its candidates via `emit_indexed` (arrival placement skipped, `do_arrival ==
        // false`); every other tile uses `place_tile`'s arrival route:
        //   * terminal tile -- the CTA's last tile necessarily holds the boundary: scan it directly.
        //   * lazy crossing tile -- boundary not yet known: candidates land in arrival order first, then `B1` reads the
        //                           counter and, on the tile that crosses `num_back`, overwrites those slots in index
        //                           order.
        const bool is_terminal_tile = region_is_terminal && (tile_base + tile_size >= count);
        place_tile<Reversed, /*Blocked=*/true>(state, keys, flags, seg_base, count, tile_base, !is_terminal_tile);

        if (is_terminal_tile)
        {
          state.running += emit_indexed<Reversed>(state, keys, flags, seg_base, count, tile_base, state.running);
          state.is_tie_active = false;
        }
        else
        {
          // B1: order the arrival global writes (from `place_tile`) ahead of the index-order overwrite (same boundary
          // slots) and make the counter read race-free and block-uniform.
          __syncthreads();
          // `back_local_cnt` is primed with the back base (`num_selected + cand_prefix`), so subtracting the front
          // region size recovers this CTA's candidate-rank reached so far (`cand_prefix` plus placed candidates).
          const offset_t num_selected = static_cast<offset_t>(k) - static_cast<offset_t>(state.num_back);
          _CCCL_ASSERT(temp_storage.back_local_cnt >= num_selected,
                       "back counter must stay at or above its primed base (no unsigned underflow)");
          const offset_t reached = temp_storage.back_local_cnt - num_selected;
          if (reached > static_cast<offset_t>(state.num_back))
          {
            // Crossing tile: overwrite this tile's arrival slots `{num_selected+state.running, ...}` with the
            // index-ordered winners (identical slot set, different candidate->slot mapping).
            emit_indexed<Reversed>(state, keys, flags, seg_base, count, tile_base, state.running);
          }
          state.running = reached;
          if (state.running >= static_cast<offset_t>(state.num_back))
          {
            state.is_tie_active = false;
          }
          // Trailing barrier for the lazy-scan path only: separate this tile's counter read (B1 above) from the next
          // tile's `back_local_cnt` atomic. Other paths write disjoint slots and need no explicit barrier.
          __syncthreads();
        }
      }

      // Phase B: ties are placed, so any remaining candidate-class keys are losers (dropped via `do_arrival == false`);
      // only strictly-selected keys are placed, to the front in arrival order. Load striped (coalesced/conflict-free);
      // no scan or barrier -- `front_local_cnt` is order-independent, so the A->B switch needs none either.
      for (; tile_base < count; tile_base += tile_size)
      {
        key_t keys[ItemsPerThread];
        offset_t flags[ItemsPerThread];
        classify_tile</*Blocked=*/false, ItemsPerThread, FromSmem, Reversed>(
          state, smem_src, seg_base, count, tile_base, keys, flags);
        place_tile<Reversed, /*Blocked=*/false>(state, keys, flags, seg_base, count, tile_base, /*do_arrival=*/false);
      }
    }
    else
    {
      // Single striped pass: the non-deterministic filter (always) or a deterministic CTA that never scans.
      for (; tile_base < count; tile_base += tile_size)
      {
        key_t keys[ItemsPerThread];
        offset_t flags[ItemsPerThread];
        classify_tile<Blocked, ItemsPerThread, FromSmem, Reversed>(
          state, smem_src, seg_base, count, tile_base, keys, flags);
        if constexpr (Deterministic)
        {
          // Non-straddling CTA: a select-all CTA places its (all-winning) candidates in arrival order; a select-no or
          // already-resolved CTA has none. Never the index-ordered scan.
          _CCCL_ASSERT(state.is_select_all_cand_cta || !state.is_tie_active,
                       "striped deterministic tile must never need index-ordered tie resolution");
          const bool do_arrival = state.is_tie_active && state.is_select_all_cand_cta;
          place_tile<Reversed, Blocked>(state, keys, flags, seg_base, count, tile_base, do_arrival);
        }
        else
        {
          // Selected to the front, the first `num_back` candidates to the back via `place_one`'s `out < k` guard.
          place_tile<Reversed, Blocked>(state, keys, flags, seg_base, count, tile_base, /*do_arrival=*/true);
        }
      }
    }
  }

  // Resident-front region. Direction is the compile-time `is_residency_reversed` (== `is_tie_reversed` in
  // deterministic mode): ascending walks the low-index window forward, descending walks the high-index window
  // (`resident_base`) in reverse, so a single `process_tiles` call per span with the index folded at compile time
  // replaces the old fwd/rev pair.
  template <bool Blocked, detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_resident(det_filter_state<SelectDirection>& state)
  {
    if constexpr (use_block_load_to_shared)
    {
      // Whole contiguous resident span staged in SMEM.
      process_tiles<tie_break_items_per_thread_clamped,
                    /*FromSmem=*/true,
                    /*Reversed=*/is_residency_reversed,
                    /*Deterministic=*/true,
                    Blocked>(
        state, state.resident_front.data(), state.front_seg_base, state.front_count, state.is_resident_terminal);
    }
    else
    {
      const int resident_chunk_count = static_cast<int>(layout.my_resident_chunks);
      for (int slot = 0; slot < resident_chunk_count; ++slot)
      {
        const int local_slot     = is_residency_reversed ? (resident_chunk_count - 1 - slot) : slot;
        const offset_t chunk_idx = layout.part.global_index(layout.resident_base + static_cast<offset_t>(local_slot));
        const auto chunk         = get_chunk(chunk_idx, layout.segment_size_off, layout.head_items);
        // Generic multi-chunk resident loop reads the chunk's SMEM slot; never the terminal-tile fast path (the lazy
        // per-tile boundary detection handles any boundary), so pass `false`.
        process_tiles<tie_break_items_per_thread_clamped,
                      /*FromSmem=*/true,
                      /*Reversed=*/is_residency_reversed,
                      /*Deterministic=*/true,
                      Blocked>(
          state, ::cuda::ptr_rebind<key_t>(key_slots + local_slot * ChunkBytes), chunk.offset, chunk.count, false);
      }
    }
  }

  // Overflow chunks. On the block-load path each landed slot folds through `process_tiles` (reusing the TMA
  // pipeline); the generic fallback reads gmem chunk by chunk. `run_pass`'s `should_continue` breaks the stream once
  // the top-k is fully placed. The streaming direction is already correct on entry (preselected in `run`), and the
  // slots are already primed -- the histogram's first streaming pass primed the same persistent stream, so
  // `stream_is_primed` carries in as `true` and this pass reuses the resident turn-around chunks with no re-prime (the
  // generic fallback re-reads gmem each pass regardless).
  template <bool Blocked, detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_overflow(det_filter_state<SelectDirection>& state)
  {
    // Guards the straddling CTA (the only rank needing scan order): `run` preselected the direction so it enters at
    // `stream_is_forward == !is_tie_reversed` after the full `num_passes`. The escape disjuncts are the cases where
    // the streaming direction is moot: no overflow (`overflow_chunks == 0`); a CTA entirely at/below the boundary
    // (`is_select_all_cand_cta`, arrival-order atomics); or one with no back scan (`!is_tie_active`). Early stop is
    // subsumed -- it makes every CTA `is_select_all_cand_cta`, so the shorter (possibly mis-parity) pass count never
    // reaches a straddler.
    _CCCL_ASSERT(layout.overflow_chunks == 0 || state.is_select_all_cand_cta || !state.is_tie_active
                   || stream_is_forward == (!is_tie_reversed),
                 "preselected ping-pong parity mismatch: the straddling CTA entered the deterministic filter with "
                 "the wrong streaming direction");

    run_pass(
      // Block-load: fold the chunk `overflow_idx`, resident in streaming slot `stage`, straight from SMEM.
      // `stage_span` returns the slot's aligned-bulk view (a peeled tail suffix is handled by `process_tail_edge`).
      [&](int stage, offset_t overflow_idx) {
        const auto keys = stage_span(stage, overflow_idx);
        const offset_t base_off =
          get_chunk(
            layout.part.global_index(layout.overflow_base + overflow_idx), layout.segment_size_off, layout.head_items)
            .offset;
        // The multi-chunk overflow stream stays on the lazy per-tile boundary detection (`region_is_terminal ==
        // false`): a stray terminal direct scan here would need the stream to flag its last chunk, and the saving
        // is one barrier on one tile.
        process_tiles<tie_break_items_streamed,
                      /*FromSmem=*/true,
                      /*Reversed=*/is_tie_reversed,
                      /*Deterministic=*/true,
                      Blocked>(state, keys.data(), base_off, static_cast<int>(keys.size()), false);
      },
      // Generic fallback: read the overflow chunk straight from gmem (full count; the fallback never peels a tail).
      [&](const auto& chunk) {
        // FromSmem=false: `smem_src` is unread, so pass nullptr.
        process_tiles<tie_break_items_streamed,
                      /*FromSmem=*/false,
                      /*Reversed=*/is_tie_reversed,
                      /*Deterministic=*/true,
                      Blocked>(state, nullptr, chunk.offset, chunk.count, false);
      },
      // No interleaved resident work: the deterministic filter folds its resident span separately.
      [] {},
      // Break the stream once the whole top-k is placed. The barrier makes the counter reads block-wide and resyncs
      // lanes that drifted through the just-folded chunk's barrier-free tiles (polled before each refill copy).
      [&] {
        __syncthreads();
        return !final_filter_should_stop(state);
      });
  }

  // Fold one persistent boundary edge (head prefix or peeled tail suffix), both staged in `edge_keys`. A no-op on the
  // generic fallback (which reads boundary items straight from gmem) and for a zero-length edge. An edge spans fewer
  // than `load_align_items` items (it fits in its `edge_keys` slot), so it runs `process_tiles` at unroll factor 1
  // (non-unrolled) instead of the wider tie-break unroll the potentially large resident/overflow regions use.
  template <bool Blocked, detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_edge(
    det_filter_state<SelectDirection>& state, const key_t* keys, offset_t seg_base, int count, bool is_terminal)
  {
    if constexpr (use_block_load_to_shared)
    {
      _CCCL_ASSERT(count >= 0 && count <= head_edge_cap_items, "a boundary edge must fit in its edge_keys slot");
      process_tiles<1, /*FromSmem=*/true, /*Reversed=*/is_tie_reversed, /*Deterministic=*/true, Blocked>(
        state, keys, seg_base, count, is_terminal);
    }
  }

  // Head prefix edge (rank 0): the segment's lowest indices `[0, head_edge_len_items)`, staged at `edge_keys` (base
  // 0). In global-index order it is the leading region (ascending) / trailing region (descending); a non-head rank or
  // empty prefix is a no-op.
  template <bool Blocked, detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_head_edge(det_filter_state<SelectDirection>& state)
  {
    process_edge<Blocked>(
      state, temp_storage.edge_keys, offset_t{0}, layout.head_edge_len_items, state.is_head_edge_terminal);
  }

  // Peeled tail suffix edge (tail owner): the segment's highest indices, staged at `edge_keys + head_edge_cap_items`
  // (base `segment_size - tail_edge_len_items`). In global-index order it is the trailing region (ascending) /
  // leading region (descending); an aligned or non-owned tail is a no-op.
  template <bool Blocked, detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tail_edge(det_filter_state<SelectDirection>& state)
  {
    process_edge<Blocked>(
      state,
      temp_storage.edge_keys + head_edge_cap_items,
      layout.segment_size_off - static_cast<offset_t>(layout.tail_edge_len_items),
      layout.tail_edge_len_items,
      state.is_tail_edge_terminal);
  }

  // Drive the four regions in global-index order (ascending, or descending under `is_tie_reversed`), bailing between
  // regions once `final_filter_should_stop` reports the whole top-k placed. The two orders share the resident->overflow
  // middle and only swap which boundary edge leads: ascending is head, resident, overflow, tail; descending reverses
  // the edges. Both visit resident before overflow, so the stop check can skip re-streaming the overflow once the
  // top-k is placed; `is_residency_reversed` keeps the first-visited (high-index) chunks resident in the descending
  // order so this holds.
  template <bool Blocked, detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void run_filter(det_filter_state<SelectDirection>& state)
  {
    const auto step = [&](auto&& region) {
      // Barrier before polling: makes the placement-counter reads block-wide and resynchronizes lanes that raced
      // ahead through the previous region's barrier-free tiles.
      __syncthreads();
      if (!final_filter_should_stop(state))
      {
        region();
      }
    };
    if constexpr (is_tie_reversed)
    {
      process_tail_edge<Blocked>(state);
    }
    else
    {
      process_head_edge<Blocked>(state);
    }
    step([&] {
      process_resident<Blocked>(state);
    });
    step([&] {
      process_overflow<Blocked>(state);
    });
    step([&] {
      if constexpr (is_tie_reversed)
      {
        process_head_edge<Blocked>(state);
      }
      else
      {
        process_tail_edge<Blocked>(state);
      }
    });
  }

  // Fold the persistent boundary edges into `apply` (head prefix on rank 0; peeled tail suffix on the tail owner),
  // reading the keys already staged in `edge_keys`. Used by the histogram passes; both final filters fold the edges
  // through `process_tiles` instead (as separate regions for the deterministic filter, via `nondet_fold_resident`).
  template <class Apply>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  fold_boundary_edges(int head_edge_len_items, int tail_edge_len_items, Apply&& apply)
  {
    if constexpr (use_block_load_to_shared)
    {
      _CCCL_PRAGMA_NOUNROLL()
      for (int local = tid; local < head_edge_len_items; local += threads_per_block)
      {
        apply(temp_storage.edge_keys[local]);
      }
      _CCCL_PRAGMA_NOUNROLL()
      for (int local = tid; local < tail_edge_len_items; local += threads_per_block)
      {
        apply(temp_storage.edge_keys[head_edge_cap_items + local]);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Non-deterministic final filter
  // ---------------------------------------------------------------------------
  // Order-independent counterpart of the deterministic filter: every region is swept by
  // `process_tiles<..., Deterministic=false>` (arrival-order placement, no scan/tie-state/barrier). Since order is
  // irrelevant, `write_nondeterministic_topk` folds the resident keys as the overflow stream's `mid` work (overlapping
  // the first reloads) and bails out of the overflow stream once its contribution is fully placed.
  template <class IdentifyOp, class KeyOutIt>
  struct nondet_filter_state
  {
    IdentifyOp identify_op;
    KeyOutIt block_keys_out;
    out_offset_t num_back;
    offset_t sel_prefix;
    offset_t cand_prefix;
    offset_t my_front; // this CTA's front region size (see `write_deterministic_topk`'s `my_front`)
    smem_keys_t resident_keys;
  };

  // Fold this rank's resident keys (and boundary edges) through `process_tiles`, run as the overflow pass's `mid` work
  // so they overlap the first overflow reloads. Placement uses order-independent atomics into slots disjoint from the
  // streaming slots, so this never races the overflow apply. Each chunk's `chunk.offset` is passed as the region's
  // segment base (used for the pair value fetch; elided in keys-only).
  template <class IdentifyOp, class KeyOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void nondet_fold_resident(nondet_filter_state<IdentifyOp, KeyOutIt>& state)
  {
    if constexpr (use_block_load_to_shared)
    {
      // Resident keys are densely packed in slot order (aligned bulks only), so a running cursor recovers per-chunk
      // spans. Only the global-last chunk can be partial (its unaligned suffix is peeled into `edge_keys` and folded
      // below), so iterate the aligned bulk (`round_down(count, load_align_items)`), not `chunk.count`.
      key_t* const resident_ptr = state.resident_keys.data();
      int cursor_items          = 0;
      for (offset_t local_chunk = 0; local_chunk < layout.my_resident_chunks; ++local_chunk)
      {
        const auto chunk = get_chunk(
          layout.part.global_index(layout.resident_base + local_chunk), layout.segment_size_off, layout.head_items);
        const int bulk_count_items = static_cast<int>(
          ::cuda::round_down(static_cast<offset_t>(chunk.count), static_cast<offset_t>(load_align_items)));
        process_tiles<tie_break_items_per_thread_clamped,
                      /*FromSmem=*/true,
                      /*Reversed=*/false,
                      /*Deterministic=*/false,
                      /*Blocked=*/false>(state, resident_ptr + cursor_items, chunk.offset, bulk_count_items, false);
        cursor_items += bulk_count_items;
      }
      // Persistent boundary edges (unroll factor 1: an edge spans fewer than `load_align_items` items): head prefix at
      // `edge_keys` (segment index 0), peeled tail suffix at `edge_keys + head_edge_cap_items` (segment index
      // `segment_size - tail_edge_len_items`).
      if (layout.head_edge_len_items > 0)
      {
        process_tiles<1, /*FromSmem=*/true, /*Reversed=*/false, /*Deterministic=*/false, /*Blocked=*/false>(
          state, temp_storage.edge_keys, offset_t{0}, layout.head_edge_len_items, false);
      }
      if (layout.tail_edge_len_items > 0)
      {
        process_tiles<1, /*FromSmem=*/true, /*Reversed=*/false, /*Deterministic=*/false, /*Blocked=*/false>(
          state,
          temp_storage.edge_keys + head_edge_cap_items,
          layout.segment_size_off - static_cast<offset_t>(layout.tail_edge_len_items),
          layout.tail_edge_len_items,
          false);
      }
    }
    else
    {
      // Generic fallback: each resident chunk is staged in its own `key_slots` slot (indexed by `local_chunk`); no
      // edges are peeled (boundary items are read as ordinary chunks).
      for (offset_t local_chunk = 0; local_chunk < layout.my_resident_chunks; ++local_chunk)
      {
        const auto chunk = get_chunk(layout.part.global_index(local_chunk), layout.segment_size_off, layout.head_items);
        process_tiles<tie_break_items_per_thread_clamped,
                      /*FromSmem=*/true,
                      /*Reversed=*/false,
                      /*Deterministic=*/false,
                      /*Blocked=*/false>(
          state,
          ::cuda::ptr_rebind<key_t>(key_slots + static_cast<int>(local_chunk) * ChunkBytes),
          chunk.offset,
          chunk.count,
          false);
      }
    }
  }

  // Non-deterministic final filter driver. `prime_placement_counters` gives this block disjoint front/back bases
  // (`sel_prefix`/`cand_prefix`); overflow keys then stream through `run_pass` (resident keys folded as its `mid`) and
  // place into block-local SMEM atomics. `run_pass` breaks the stream once this CTA's contribution is fully placed
  // (all its selected keys in the front, and its back counter reached the region end `k`).
  template <class IdentifyOp, class KeyOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void write_nondeterministic_topk(
    out_offset_t num_kth, IdentifyOp identify_op, KeyOutIt block_keys_out, smem_keys_t resident_keys)
  {
    const bool participates = !layout.is_idle_rank && (cluster_rank != layout.leader_rank);
    const offset_t my_sel   = participates ? temp_storage.num_strictly_selected : offset_t{0};
    const offset_t my_cand  = participates ? temp_storage.my_candidates : offset_t{0};
    // The scan doubles as counter priming: it leaves this CTA's placement counters holding its absolute region bases
    // (front = `sel_prefix`, back = `num_selected + cand_prefix`), so no explicit priming follows.
    const offset_t num_selected               = static_cast<offset_t>(k) - static_cast<offset_t>(num_kth);
    const ::cuda::std::uint64_t packed_prefix = prime_placement_counters(my_sel, my_cand, num_selected);
    const offset_t sel_prefix                 = static_cast<offset_t>(packed_prefix >> 32);
    const offset_t cand_prefix                = static_cast<offset_t>(packed_prefix & 0xffffffffu);
    // The selected region has size `k - num_kth`, so the selected prefix leaves room for the `num_kth` tie-back slots.
    _CCCL_ASSERT(static_cast<::cuda::std::uint64_t>(sel_prefix) + static_cast<::cuda::std::uint64_t>(num_kth)
                   <= static_cast<::cuda::std::uint64_t>(k),
                 "selected prefix must fit before the tie-back output region");
    // This CTA's front region size (mirrors `write_deterministic_topk`): the leader is last in scan order and derives
    // it from the total (`num_selected - sel_prefix`) since its merged histogram can't self-count; others place
    // `my_sel`.
    const offset_t my_front = (cluster_rank == layout.leader_rank) ? (num_selected - sel_prefix) : my_sel;

    nondet_filter_state<IdentifyOp, KeyOutIt> state{
      identify_op, block_keys_out, num_kth, sel_prefix, cand_prefix, my_front, resident_keys};

    run_pass(
      // Block-load: fold the chunk `overflow_idx`, resident in streaming slot `stage`, straight from SMEM.
      [&](int stage, offset_t overflow_idx) {
        const auto keys = stage_span(stage, overflow_idx);
        const offset_t base_off =
          get_chunk(
            layout.part.global_index(layout.overflow_base + overflow_idx), layout.segment_size_off, layout.head_items)
            .offset;
        process_tiles<tie_break_items_streamed,
                      /*FromSmem=*/true,
                      /*Reversed=*/false,
                      /*Deterministic=*/false,
                      /*Blocked=*/false>(state, keys.data(), base_off, static_cast<int>(keys.size()), false);
      },
      // Generic fallback: read the overflow chunk straight from gmem.
      [&](const auto& chunk) {
        process_tiles<tie_break_items_streamed,
                      /*FromSmem=*/false,
                      /*Reversed=*/false,
                      /*Deterministic=*/false,
                      /*Blocked=*/false>(state, nullptr, chunk.offset, chunk.count, false);
      },
      // Fold the resident keys + boundary edges, overlapping the first overflow reloads.
      [&] {
        nondet_fold_resident(state);
      },
      // Break the stream once this CTA's whole contribution is placed: its front is full (`front_local_cnt` reached
      // `sel_prefix + my_front`) and its back is full (`back_local_cnt >= k`, so further candidates fail `place_one`'s
      // `out < k` guard). The barrier makes the counter reads block-wide, resyncing lanes that raced through the
      // barrier-free tiles. The stop needs both conditions, so a CTA whose candidates never fill its back to `k`
      // simply never stops early.
      [&] {
        __syncthreads();
        const offset_t front_end = state.sel_prefix + state.my_front;
        _CCCL_ASSERT(temp_storage.front_local_cnt <= front_end,
                     "front counter must stay within this CTA's front region");
        return !(temp_storage.front_local_cnt >= front_end && temp_storage.back_local_cnt >= static_cast<offset_t>(k));
      });
  }

  // Deterministic final filter: both regions fill forward -- strictly-selected keys into the front `[0, num_selected)`
  // via a SMEM atomic (primed with this CTA's `sel_prefix`), candidates into the back `[num_selected, k)`. Candidate
  // placement uses arrival-order atomics, with an index-ordered BlockScan only on the single boundary-crossing
  // (straddling) CTA. `prime_placement_counters` gives this block its disjoint front/back bases and lets it detect
  // whether all/none/some of its candidates win. This member computes the `det_filter_state` inputs and hands them to
  // `run_filter`, which drives the per-region sweeps.
  template <detail::topk::select SelectDirection, class IdentifyOp, class KeyOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void write_deterministic_topk(
    out_offset_t num_kth, IdentifyOp identify_op, KeyOutIt block_keys_out, smem_keys_t resident_keys)
  {
    // Early stop is not special-cased: `total_candidates == num_kth` then makes every CTA `is_select_all_cand_cta`.
    //
    // Cache `total_candidates` now, while every block is still tightly coupled to the pass loop's final cluster
    // barrier -- a post-scan re-read of `leader_state` could touch an already-returned leader (barrier gone).
    const offset_t total_candidates = layout.leader_state->len;
    // Guards the unsigned `k - num_back` below: the remaining tie count fits both `k` and the splitter bucket.
    _CCCL_ASSERT(num_kth <= k && static_cast<offset_t>(num_kth) <= total_candidates,
                 "remaining k must fit within both the top-k and the splitter bucket");
    const out_offset_t num_back     = num_kth; // all candidates go to the back; the front holds only selected keys
    const out_offset_t num_selected = k - num_back; // front region

    const bool participates = !layout.is_idle_rank && (cluster_rank != layout.leader_rank);
    const offset_t my_sel   = participates ? temp_storage.num_strictly_selected : offset_t{0};
    const offset_t my_cand  = participates ? temp_storage.my_candidates : offset_t{0};
    // Front count pushed by this block: its strictly-selected count. Candidates always route through the back, so
    // nothing folds into the front here. The leader and idle ranks push 0 -- the leader because its merged histogram
    // cannot self-count (it derives its own front from the total below), idle ranks because they own nothing.
    const offset_t push_front = my_sel;
    // The scan doubles as counter priming: it leaves this CTA's placement counters holding its absolute region bases
    // (front = `sel_prefix`, back = `num_selected + cand_prefix`), so no explicit priming follows.
    const ::cuda::std::uint64_t packed_prefix =
      prime_placement_counters(push_front, my_cand, static_cast<offset_t>(num_selected));
    const offset_t sel_prefix  = static_cast<offset_t>(packed_prefix >> 32);
    const offset_t cand_prefix = static_cast<offset_t>(packed_prefix & 0xffffffffu);
    // Guards the leader's remainder subtractions below (`total_candidates - cand_prefix`, `num_selected - sel_prefix`).
    _CCCL_ASSERT(cand_prefix <= total_candidates && sel_prefix <= static_cast<offset_t>(num_selected),
                 "cross-CTA prefixes must stay within their candidate and selected totals");
    // This block's own candidate count: non-leaders hold it in `my_cand`; the leader is last in scan order, so
    // `cand_prefix` already sums every other block's candidates and `total_candidates - cand_prefix` is its own. A
    // CTA is `is_select_all_cand_cta` when all of its candidates sit at or below the K-boundary
    // (`cand_prefix + my_cand_count <= num_back`): every one wins, so the back places them with arrival-order SMEM
    // atomics and skips the index-ordered scan. While `is_tie_active`, a non-`is_select_all_cand_cta` CTA is the
    // single boundary-crossing (straddling) CTA cluster-wide.
    const offset_t my_cand_count = (cluster_rank == layout.leader_rank) ? (total_candidates - cand_prefix) : my_cand;
    const bool is_select_all_cand_cta = (cand_prefix + my_cand_count) <= static_cast<offset_t>(num_back);
    // Mirror image: a CTA selects none of its candidates when the tie region is empty (`num_back == 0`) or all of its
    // candidates sort strictly after the K-boundary (`cand_prefix >= num_back`). Such a CTA seeds `is_tie_active`
    // false and skips the back placement entirely.
    const bool is_select_no_cand_cta =
      (num_back == out_offset_t{0}) || (cand_prefix >= static_cast<offset_t>(num_back));
    // This block's own front size: non-leaders know it directly (`push_front`); the leader is last in scan order, so
    // `sel_prefix` already sums every other block's front and `num_selected - sel_prefix` is the remainder it owns.
    const out_offset_t my_front =
      (cluster_rank == layout.leader_rank)
        ? static_cast<out_offset_t>(num_selected - static_cast<out_offset_t>(sel_prefix))
        : static_cast<out_offset_t>(push_front);
    // This CTA's front slots `[sel_prefix, sel_prefix + my_front)` (the range its primed `front_local_cnt` walks) must
    // stay within the selected region `[0, num_selected)`.
    _CCCL_ASSERT(sel_prefix + static_cast<offset_t>(my_front) <= static_cast<offset_t>(num_selected),
                 "this CTA's front slots must fit within the selected region");

    // Resident-front extent (bulk path): the whole contiguous resident span. The unaligned tail suffix (the
    // globally-last chunk's) is always peeled into `edge_keys` and folded by `process_tail_edge`, so it is never
    // part of this span.
    const int front_count = static_cast<int>(resident_keys.size());

    // Terminal-region flags for the last-tile direct-scan gate (block-load regions only; the generic resident loop
    // and overflow stream pass `false` and stay on lazy per-tile detection). A region is terminal when no later
    // region in the sweep (head/resident/overflow/tail-edge ascending, reversed descending) carries work.
    const bool has_head      = layout.head_edge_len_items > 0;
    const bool has_resident  = front_count > 0;
    const bool has_overflow  = layout.overflow_chunks > offset_t{0};
    const bool has_tail_edge = layout.tail_edge_len_items > 0;
    [[maybe_unused]] const bool is_resident_terminal =
      is_tie_reversed ? (!has_overflow && !has_head) : (!has_overflow && !has_tail_edge);
    [[maybe_unused]] const bool is_head_edge_terminal =
      is_tie_reversed ? true : (!has_resident && !has_overflow && !has_tail_edge);
    [[maybe_unused]] const bool is_tail_edge_terminal =
      is_tie_reversed ? (!has_resident && !has_overflow && !has_head) : true;

    // Segment-local base of the resident-front span (its lowest-index resident chunk; `resident_base` shifts it to
    // the high-index window under `is_residency_reversed`). The blocked partition packs the front contiguously, so
    // element `pos` maps to `front_seg_base + pos`. Guarded on `front_count > 0`: with a fully-streamed rank
    // (`stream_slots == full_slots`) there are no resident chunks and `resident_base` would name an out-of-range
    // local chunk under `is_residency_reversed`.
    const offset_t front_seg_base =
      (front_count > 0)
        ? get_chunk(layout.part.global_index(layout.resident_base), layout.segment_size_off, layout.head_items).offset
        : offset_t{0};

    // Positional aggregate init in `det_filter_state` declaration order. The last two initializers seed the tie-break
    // cursor: `running` at this CTA's exclusive back prefix (candidates owned by preceding CTAs), `is_tie_active` at
    // `!is_select_no_cand_cta` -- true unless this is a select-no-candidates CTA (empty back region, or all of this
    // CTA's candidates sort past the boundary).
    det_filter_state<SelectDirection> state{
      identify_op,
      block_keys_out,
      num_back,
      my_front,
      sel_prefix,
      cand_prefix,
      my_cand_count,
      front_seg_base,
      resident_keys,
      front_count,
      is_select_all_cand_cta,
      is_resident_terminal,
      is_head_edge_terminal,
      is_tail_edge_terminal,
      cand_prefix,
      !is_select_no_cand_cta};
    // Arrangement dispatch: only the boundary-straddling CTA (neither select-all nor select-no candidates) resolves
    // ties via `emit_indexed`, so it enters `Blocked` (then switches to striped past the boundary; see
    // `process_tiles`). Every other CTA places purely via arrival-order atomics and loads striped throughout.
    if (!is_select_all_cand_cta && !is_select_no_cand_cta)
    {
      run_filter</*Blocked=*/true>(state);
    }
    else
    {
      run_filter</*Blocked=*/false>(state);
    }
  }

  // Fused first pass: load this rank's resident chunks into the block_tile, stage the persistent boundary edges into
  // `edge_keys`, and fold every key (resident + edges + overflow) into pass 0's histogram in the same sweep (pass 0
  // needs no candidate filtering). Publishes the resident span as `resident_keys` for the later passes and the final
  // filter. The caller's `__syncthreads()` makes `edge_keys` and the block-local histogram block-visible; cross-CTA
  // visibility of the zeroed histogram comes later, from the deferred initial cluster wait in `run_radix_passes` (this
  // whole load runs in that arrive->wait window).
  template <detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void load_and_histogram_first_pass(smem_keys_t& resident_keys)
  {
    using extract_bin_op_t   = detail::topk::extract_bin_op_t<key_t, SelectDirection, bits_per_pass, decomposer_t>;
    constexpr int total_bits = int{sizeof(key_t)} * 8;

    extract_bin_op_t extract_op(0, total_bits, decomposer_t{});
    const ::cuda::std::uint32_t hist_smem32 = hist_base32();
    auto add_first_pass                     = [&](const key_t& key) {
      const int bucket = extract_op(key);
      hist_inc(hist_smem32, bucket);
    };

    if constexpr (use_block_load_to_shared)
    {
      if (layout.my_resident_chunks > 0)
      {
        // Stage mbarriers and the `load_phase` parity are shared with the overflow stream (no per-chunk token array
        // needed). Chunks are written densely in slot order from offset 0 and read back in the same order, so the read
        // cursor
        // (`read_off_bytes`) mirrors the write cursor (`next_off_bytes`) as a running prefix sum, avoiding a
        // dynamically-indexed `pending_spans` array that would anchor surrounding state to local memory. Every chunk
        // begins on a `load_align` boundary (zero prefix), so its aligned bulk is `round_down(count,
        // load_align_items)`: the whole `count` for interior chunks, and for the global-last chunk its `count` minus
        // the unaligned suffix that is always peeled into `edge_keys`.
        const int prologue = (::cuda::std::min) (PipelineStages, static_cast<int>(layout.my_resident_chunks));

        // Resident slot -> rank-local chunk index (`resident_base + slot`; identity unless `is_residency_reversed`
        // shifts the resident window to the high-index chunks).
        const auto resident_local = [&](offset_t slot) -> offset_t {
          return layout.resident_base + slot;
        };
        // Aligned bulk of the resident chunk in `slot` (its `count` minus any peeled tail suffix); empty when it has
        // none.
        const auto bulk_src = [&](offset_t slot) -> ::cuda::std::span<const key_t> {
          const offset_t chunk_idx = layout.part.global_index(resident_local(slot));
          const auto chunk         = get_chunk(chunk_idx, layout.segment_size_off, layout.head_items);
          const offset_t bulk =
            ::cuda::round_down(static_cast<offset_t>(chunk.count), static_cast<offset_t>(load_align_items));
          if (bulk == 0)
          {
            return {};
          }
          // Chunks start after the aligned head on an alignment-multiple stride (mirrors `issue_load`).
          _CCCL_ASSERT(::cuda::is_aligned(layout.block_keys_base + chunk.offset, load_align_bytes),
                       "resident loader received a chunk with an unaligned start");
          return {layout.block_keys_base + chunk.offset, static_cast<::cuda::std::size_t>(bulk)};
        };

        // Bulks are densely packed from the start of the block_tile. The head prefix is no longer kept here (it lives
        // in `edge_keys`), so there is no reserved front gap: the write cursor starts at offset 0.
        int next_off_bytes = 0;

        // Load every resident chunk's aligned bulk, densely packed in slot order.
        for (int stage = 0; stage < prologue; ++stage)
        {
          const auto src = bulk_src(static_cast<offset_t>(stage));
          issue_bulk_copy(stage, key_slots + next_off_bytes, src);
          next_off_bytes += static_cast<int>(::cuda::std::size(src)) * int{sizeof(key_t)};
        }

        // Read cursor trailing the write cursor: chunk `local_chunk`'s bulk was written at `read_off_bytes` (packed
        // and consumed in the same order), and `bulk_src(local_chunk)` recomputes its length, so the read span needs
        // no stored per-stage state.
        int read_off_bytes = 0;
        for (offset_t local_chunk = 0; local_chunk < layout.my_resident_chunks; ++local_chunk)
        {
          const int stage = static_cast<int>(local_chunk % static_cast<offset_t>(prologue));
          wait_stage(stage);
          const int read_len_items = static_cast<int>(::cuda::std::size(bulk_src(local_chunk)));
          for_each_chunk_key<histogram_items_per_thread_clamped>(
            ::cuda::ptr_rebind<key_t>(key_slots + read_off_bytes), read_len_items, add_first_pass);
          read_off_bytes += read_len_items * int{sizeof(key_t)};

          const offset_t next_local_chunk = local_chunk + static_cast<offset_t>(prologue);
          if (next_local_chunk < layout.my_resident_chunks)
          {
            const auto src = bulk_src(next_local_chunk);
            // Phase safety, not data safety (the target offset is fresh): re-arming this stage before all threads
            // leave the wait above would advance the phase twice, stranding a lagging waiter forever.
            __syncthreads();
            issue_bulk_copy(stage, key_slots + next_off_bytes, src);
            next_off_bytes += static_cast<int>(::cuda::std::size(src)) * int{sizeof(key_t)};
          }
        }

        // Read and write cursors sum the same aligned bulks in the same order, so they meet on a load-align boundary.
        _CCCL_ASSERT(read_off_bytes == next_off_bytes && next_off_bytes % load_align_bytes == 0,
                     "resident bulk read/write cursors must meet on a load-alignment boundary");
        // The resident region is one contiguous run of aligned bulks for the later passes; both boundary edges are
        // folded separately from `edge_keys`.
        resident_keys = smem_keys_t(::cuda::ptr_rebind<key_t>(key_slots), next_off_bytes / int{sizeof(key_t)});
      }
    }
    else
    {
      for (offset_t local_chunk = 0; local_chunk < layout.my_resident_chunks; ++local_chunk)
      {
        const offset_t chunk_idx = layout.part.global_index(layout.resident_base + local_chunk);
        const auto chunk         = get_chunk(chunk_idx, layout.segment_size_off, layout.head_items);
        key_t* const chunk_keys  = ::cuda::ptr_rebind<key_t>(key_slots + static_cast<int>(local_chunk) * ChunkBytes);
        const int iterations     = ::cuda::ceil_div(chunk.count, threads_per_block);
        _CCCL_PRAGMA_UNROLL(histogram_items_per_thread_clamped)
        for (int j = 0; j < iterations; ++j)
        {
          const int local = j * threads_per_block + tid;
          if (local < chunk.count)
          {
            const key_t key =
              layout.block_keys_in[static_cast<segment_size_val_t>(chunk.offset + static_cast<offset_t>(local))];
            chunk_keys[local] = key;
            add_first_pass(key);
          }
        }
      }
    }

    // Stage the persistent boundary edges into `edge_keys` and fold them into the first pass in the same sweep (see
    // `stage_and_fold_edge`: each thread folds keys it just wrote, so no barrier is needed here). The head prefix
    // (rank 0) precedes chunk 0; the peeled tail suffix (tail owner, always when unaligned) trails the last chunk.
    if constexpr (use_block_load_to_shared)
    {
      if (layout.head_edge_len_items > 0)
      {
        stage_and_fold_edge(temp_storage.edge_keys, layout.block_keys_base, layout.head_edge_len_items, add_first_pass);
      }
      if (layout.tail_edge_len_items > 0)
      {
        _CCCL_ASSERT(layout.chunks > offset_t{0}, "a peeled tail edge requires at least one aligned chunk");
        const auto tail_chunk = get_chunk(layout.chunks - offset_t{1}, layout.segment_size_off, layout.head_items);
        _CCCL_ASSERT(layout.tail_edge_len_items <= tail_chunk.count,
                     "peeled tail length cannot exceed its source chunk");
        stage_and_fold_edge(
          temp_storage.edge_keys + head_edge_cap_items,
          layout.block_keys_base + tail_chunk.offset + (tail_chunk.count - layout.tail_edge_len_items),
          layout.tail_edge_len_items,
          add_first_pass);
      }
    }

    // Fold the overflow chunks into the first-pass histogram, priming the streaming slots in the stream's initial
    // direction (preselected above; the histogram is order-independent, so the direction only sets up the leftover
    // parity for the final filter). The overflow stream reuses the resident load's stage mbarriers (all front-loaded at
    // `run` entry); `wait_stage` provides the producer/consumer sync.
    process_pass<histogram_items_per_thread_clamped>(add_first_pass);

    const int resident_count = static_cast<int>(resident_keys.size());
    _CCCL_ASSERT(resident_count == 0 || static_cast<offset_t>(resident_count) <= block_tile_capacity,
                 "Dynamic shared memory block_tile is too small");
  }

  // Fill the `layout` member for this block's `segment_id`/`segment_size`. Called once at the top of `run`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void compute_segment_layout()
  {
    layout.block_keys_in    = d_key_segments_it[segment_id];
    layout.segment_size_off = static_cast<offset_t>(segment_size);
    // A lone CTA (`is_single_cta`) routes barriers to `__syncthreads()` and keeps `state`/atomics block-local (no
    // cross-rank DSMEM folds); its histogram increments still use the unconditional cluster-scope `hist_inc`, which is
    // identical SASS at cluster size 1. For wider clusters, `eff_cluster_blocks` (below) further excludes ranks that
    // receive no chunks; they stay resident but idle.

    layout.block_keys_base = nullptr;
    layout.head_items      = 0;
    if constexpr (use_block_load_to_shared)
    {
      layout.block_keys_base = THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(layout.block_keys_in);
      // `run` is reached only for `0 < k < segment_size`, so a bulk-load segment is non-empty and has a real base.
      _CCCL_ASSERT(layout.block_keys_base != nullptr, "a bulk-load segment must have a valid input pointer");
      // Items from `block_keys_base` to the next `load_align_bytes` boundary (0 when already aligned), clamped to the
      // segment. `load_align_bytes` is a multiple of `sizeof(key_t)`, so the aligned pointer lands on a key boundary
      // and the pointer difference is already an exact item count; narrow it to 32-bit `offset_t` right away.
      const offset_t head_to_boundary =
        static_cast<offset_t>(::cuda::align_up(layout.block_keys_base, load_align_bytes) - layout.block_keys_base);
      layout.head_items = (::cuda::std::min) (head_to_boundary, layout.segment_size_off);
      // Distance to the next alignment boundary is < one alignment unit, so the head prefix fits its `edge_keys` slot.
      _CCCL_ASSERT(layout.head_items < static_cast<offset_t>(load_align_items),
                   "the unaligned head prefix must be shorter than one load-alignment unit");
    }
    // Number of aligned chunks covering the segment: the unaligned head prefix (`head_items`) is a separate edge, not a
    // chunk, so the aligned region `[head_items, segment_size)` chunks uniformly at `chunk_items` (a segment lying
    // entirely before the first boundary yields `head_items == segment_size` and zero chunks). The generic fallback
    // skips the alignment/peeling path and keeps `head_items == 0`; the two chunkings may assign keys to CTAs
    // differently, but top-k only depends on the multiset of keys the cluster covers.
    _CCCL_ASSERT(layout.head_items <= layout.segment_size_off, "head prefix cannot exceed the segment");
    layout.chunks =
      static_cast<offset_t>(::cuda::ceil_div(layout.segment_size_off - layout.head_items, offset_t{chunk_items}));

    // Effective cluster blocks: the CTAs that actually receive chunks (at least `min_chunks_per_block` each), <= the
    // launched `cluster_blocks`. Ranks at or beyond it are idle -- they own no chunks, fold nothing, and never lead --
    // but stay resident and still arrive at every cluster barrier (a returned CTA would hang the barrier; see the
    // TODOs at the barrier sites). Derived from this CTA's head-aligned `chunks` so it matches the partition exactly.
    // Stays at `cluster_blocks` for host-exact sizes (the dispatch already matched it) and on the single-CTA path.
    layout.eff_cluster_blocks = cluster_blocks;
    if constexpr (enable_runtime_single_cta)
    {
      if (!is_single_cta)
      {
        layout.eff_cluster_blocks = effective_cluster_blocks_from_chunks(
          static_cast<::cuda::std::uint64_t>(layout.chunks), min_chunks_per_block, cluster_blocks);
      }
    }
    _CCCL_ASSERT(layout.eff_cluster_blocks >= 1u && layout.eff_cluster_blocks <= cluster_blocks,
                 "effective cluster blocks must stay within [1, launched cluster blocks]");
    layout.is_idle_rank = cluster_rank >= layout.eff_cluster_blocks;

    // Idle ranks own no chunks; `make_chunk_partition` assumes `rank < size`, so hand them an explicit empty partition.
    layout.part      = layout.is_idle_rank ? chunk_partition{offset_t{0}, offset_t{1}, offset_t{0}}
                                           : make_chunk_partition(layout.chunks, cluster_rank, layout.eff_cluster_blocks);
    layout.my_chunks = layout.part.count;

    // Leader rank. The leader owns the cluster-merged histogram and the shared `state`, and is always a working rank
    // (`< eff_cluster_blocks`). The deterministic tie-break makes the leader the *last* CTA in scan order so it never
    // needs its own (merged-away) local candidate count: prefer-smallest scans ascending by rank (leader = last
    // effective rank), prefer-largest scans descending (leader = rank 0). The nondeterministic path keeps rank 0.
    layout.leader_rank = (need_determinism && !is_tie_reversed) ? (layout.eff_cluster_blocks - 1u) : 0u;
    _CCCL_ASSERT(layout.leader_rank < layout.eff_cluster_blocks, "leader must be a working rank");

    // DSMEM pointer into the leader block's shared memory. The Step 2 histogram fold reaches the leader's `hist`
    // through a `mapa`-formed `shared::cluster` address instead (see `hist_fold_remote`).
    layout.leader_state = is_single_cta ? &temp_storage.state : map_state_to_rank(layout.leader_rank);

    // Resident vs. streaming split, decided independently per CTA (CTAs need not agree -- cross-CTA traffic and every
    // cluster barrier is reached uniformly). A CTA whose chunks fit its resident slots (`my_chunks <= full_slots`)
    // keeps them all resident and streams nothing; an overflowing CTA reserves a round-robin streaming region at the
    // tail of its block_tile and re-streams its overflow chunks from gmem each pass via the overflow stream.
    //
    // Boundary edges (the unaligned head prefix on rank 0 and the unaligned tail suffix on the tail owner) cannot use
    // the aligned TMA stream, so both are always peeled into the persistent `edge_keys` buffer. The tail chunk's
    // aligned bulk is then a normal (possibly partial) aligned chunk that can be resident or streamed like any other;
    // no partial tail chunk is ever kept resident. Peeling both boundaries means streaming needs only `full_slots >=
    // 1`.
    //
    // `stream_slots` is right-sized: clamped into `[1, full_slots]` when streaming; deep overflows still get the full
    // `PipelineStages` depth. The generic fallback has no async pipeline (it re-reads overflow from gmem each pass and
    // peels nothing), so it reserves no streaming slots.
    // Dispatch guarantees >= 1 whole chunk slot of capacity, so the streaming clamp below is well-defined.
    _CCCL_ASSERT(block_tile_capacity >= static_cast<offset_t>(chunk_items)
                   && block_tile_capacity % static_cast<offset_t>(chunk_items) == offset_t{0},
                 "block tile capacity must be a positive whole number of chunk slots");
    const offset_t full_slots                   = block_tile_capacity / static_cast<offset_t>(chunk_items);
    [[maybe_unused]] const bool needs_streaming = layout.my_chunks > full_slots;

    // Does this rank own the global tail, and does that tail carry an unaligned suffix? (block-load path only; the
    // generic fallback reads any trailing items straight from gmem and never peels.)
    [[maybe_unused]] offset_t tail_suffix_items = offset_t{0};
    bool owns_suffix_tail                       = false;
    if constexpr (use_block_load_to_shared)
    {
      // This rank owns the global tail iff its last owned chunk is chunk `chunks-1` (its local index `my_chunks-1`,
      // true for both the strided and blocked partitions).
      if (layout.my_chunks > 0
          && layout.part.global_index(layout.my_chunks - offset_t{1}) == layout.chunks - offset_t{1})
      {
        const auto tail_chunk = get_chunk(layout.chunks - offset_t{1}, layout.segment_size_off, layout.head_items);
        tail_suffix_items =
          static_cast<offset_t>(tail_chunk.count)
          - ::cuda::round_down(static_cast<offset_t>(tail_chunk.count), static_cast<offset_t>(load_align_items));
        owns_suffix_tail = tail_suffix_items != offset_t{0};
      }
    }

    // `should_peel_tail` mirrors `owns_suffix_tail` (always peel, per above). `stream_slots` clamps into `[1,
    // full_slots]`.
    offset_t stream_slots = offset_t{0};
    bool should_peel_tail = false;
    if constexpr (use_block_load_to_shared)
    {
      should_peel_tail = owns_suffix_tail;
      if (needs_streaming)
      {
        const offset_t excess      = layout.my_chunks - full_slots;
        const offset_t want_stream = (::cuda::std::min) (static_cast<offset_t>(PipelineStages), excess);
        stream_slots               = ::cuda::std::clamp(want_stream, offset_t{1}, full_slots);
      }
    }
    layout.resident_slots_cap = full_slots - stream_slots;
    layout.my_resident_chunks = (::cuda::std::min) (layout.my_chunks, layout.resident_slots_cap);
    // Resident chunks stay within the first `resident_slots_cap` slots; the streaming region occupies the slots
    // `[resident_slots_cap, full_slots)`, so both regions live inside the allocated block_tile buffer.
    _CCCL_ASSERT(layout.my_resident_chunks <= layout.resident_slots_cap,
                 "Dynamic shared memory block_tile is too small");

    layout.overflow_chunks =
      (layout.my_chunks > layout.my_resident_chunks) ? (layout.my_chunks - layout.my_resident_chunks) : offset_t{0};
    // Rank-local base of the resident and overflow (streamed) chunk windows. Default: resident `[0,
    // my_resident_chunks)`, overflow the high-index rest. `is_residency_reversed` swaps them: resident
    // `[overflow_chunks, my_chunks)`, overflow `[0, overflow_chunks)`.
    layout.resident_base = is_residency_reversed ? layout.overflow_chunks : offset_t{0};
    layout.overflow_base = is_residency_reversed ? offset_t{0} : layout.my_resident_chunks;
    layout.stream_slots  = stream_slots;

    // Persistent boundary-edge lengths: the head prefix lives on rank 0 (`head_items` is 0 on the generic fallback and
    // for an aligned base); the peeled tail suffix lives on the tail owner whenever it is unaligned.
    layout.head_edge_len_items = (cluster_rank == 0u) ? static_cast<int>(layout.head_items) : 0;
    layout.tail_edge_len_items = should_peel_tail ? static_cast<int>(tail_suffix_items) : 0;
  }

  // Radix histogram/scan passes: refine the splitter one `bits_per_pass` digit at a time until the top-k prefix is
  // pinned down (or early stop fires). Folds `kth_key_bits_local` up digit by digit and returns the number of passes
  // that actually ran (`last_pass`), which the final filter uses to size its identify operator.
  template <detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE int run_radix_passes(smem_keys_t resident_keys, key_prefix_t& kth_key_bits_local)
  {
    using extract_bin_op_t = detail::topk::extract_bin_op_t<key_t, SelectDirection, bits_per_pass, decomposer_t>;
    using identify_candidates_op_t =
      detail::topk::identify_candidates_op_t<key_t, SelectDirection, bits_per_pass, decomposer_t>;

    constexpr int total_bits = int{sizeof(key_t)} * 8;
    constexpr int num_passes = detail::topk::calc_num_passes<key_t>(bits_per_pass);

    int last_pass = num_passes;
    for (int pass = 0; pass < num_passes; ++pass)
    {
      const bool is_first_pass = (pass == 0);

      // `kth_key_bits_local` already holds the splitter digits for passes `0..pass-1`: each block folded the leader's
      // published `kth_bucket` into it at the end of the previous pass (see the end of this loop). Pass 0 needs no
      // filtering (all keys are candidates), so it is handled by the fused first-pass load above.
      if (!is_first_pass)
      {
        identify_candidates_op_t identify_op(&kth_key_bits_local, pass, total_bits, decomposer_t{});
        extract_bin_op_t extract_op(pass, total_bits, decomposer_t{});

        // Step 1: block-private histogram via shared-space `red` at cluster scope (see `hist_inc`).
        const ::cuda::std::uint32_t hist_smem32 = hist_base32();
        auto add_hist                           = [&](const key_t& key) {
          if (identify_op(key) == detail::topk::candidate_class::candidate)
          {
            const int bucket = extract_op(key);
            hist_inc(hist_smem32, bucket);
          }
        };

        // Resident-chunk histogram, deferred into the overflow stream so it overlaps the stream's in-flight first
        // reload wave (see `process_pass`). The histogram is order-independent, so folding resident keys between the
        // stream's load issue and its wait does not change the result.
        const auto fold_resident_hist = [&] {
          if constexpr (use_block_load_to_shared)
          {
            for_each_chunk_key<histogram_items_per_thread_clamped>(
              resident_keys.data(), static_cast<int>(resident_keys.size()), add_hist);
          }
          else
          {
            for (offset_t local_chunk = 0; local_chunk < layout.my_resident_chunks; ++local_chunk)
            {
              const offset_t chunk_idx = layout.part.global_index(layout.resident_base + local_chunk);
              const auto chunk         = get_chunk(chunk_idx, layout.segment_size_off, layout.head_items);
              for_each_chunk_key<histogram_items_per_thread_clamped>(
                ::cuda::ptr_rebind<key_t>(key_slots + static_cast<int>(local_chunk) * ChunkBytes),
                static_cast<int>(chunk.count),
                add_hist);
            }
          }
        };

        // Re-stream the overflow chunks into this pass's histogram, overlapping the resident-chunk histogram with the
        // first wave of reload bulk copies. Ping-pongs direction and reuses the turn-around chunks left resident by the
        // previous pass.
        process_pass<histogram_items_per_thread_clamped>(add_hist, fold_resident_hist);

        // Fold the persistent boundary edges (loaded once in the first pass) into this pass's histogram, alongside the
        // resident and overflow keys. Keeps every owner's per-bucket counts (the source of its `num_strictly_selected`
        // and `my_candidates`, the cross-CTA scan inputs) inclusive of its edge candidates.
        fold_boundary_edges(layout.head_edge_len_items, layout.tail_edge_len_items, add_hist);
      }

      // Local barrier is enough here: all Step 1 / Step 2 writes to `hist[]` are atomic at compatible scopes (see Step
      // 1 dispatch). The cluster-wide ordering before Step 3's leader read of the merged `hist[]` is supplied by the
      // split post-fold cluster wait further below.
      __syncthreads();

      // First pass only: complete the deferred initial cluster barrier here. `process_impl` issued its matching
      // `cluster_arrive` right after zeroing `state`/`hist`, so the cluster-arrival latency overlapped the fused
      // first-pass load + histogram. The wait must precede the first cross-CTA access -- the Step-2 folds just below,
      // which need the leader's `hist` visibly zeroed.
      if (is_first_pass)
      {
        cluster_or_block_wait(is_single_cta);
      }

      // Step 2: non-leader blocks fold their per-bucket raw counts into
      // the leader's `hist` via cluster-scope DSMEM atomics (see
      // `hist_fold_remote`). The
      // leader skips this to avoid double-counting its own contribution;
      // idle ranks (`>= eff_cluster_blocks`) have an all-zero histogram, so
      // they skip the fold entirely (the loop would only read zeros).
      if (cluster_rank != layout.leader_rank && !layout.is_idle_rank)
      {
        const ::cuda::std::uint32_t hist_smem32 = hist_base32();
        for (int i = tid; i < num_buckets; i += threads_per_block)
        {
          const offset_t bucket_count = temp_storage.hist[i];
          if (bucket_count != 0)
          {
            hist_fold_remote(
              hist_smem32 + static_cast<::cuda::std::uint32_t>(i) * sizeof(offset_t), bucket_count, layout.leader_rank);
          }
        }
      }

      // Split the post-fold cluster barrier: arrive once the folds are released, then overlap the non-leaders'
      // own-histogram scan (block-local) with the cluster-arrival latency, and only wait before the leader reads the
      // merged `hist`.
      // TODO(cccl): idle ranks arrive here only because the cluster barrier spans the whole launched cluster. An
      // mbarrier over just the active ranks would let them exit and free their SM slots instead of spinning here.
      cluster_or_block_arrive(is_single_cta);

      // Step 3 (non-leader half, run in the arrive->wait window): each non-leader exclusive-scans its *own* (un-merged)
      // histogram into registers -- the leader is otherwise the only block doing useful work at Step 3. Once the leader
      // publishes `kth_bucket` below, the lane owning it reads its exclusive prefix (this block's keys strictly above
      // the splitter this pass -> `num_strictly_selected`) and its raw bucket count (-> `my_candidates`). Keeping the
      // scan in registers lets `hist` reset on the normal schedule (the regs survive the reset and the next sync).
      offset_t local_prefixes[buckets_per_thread]{};
      offset_t local_hist_vals[buckets_per_thread]{};
      if (cluster_rank != layout.leader_rank && !layout.is_idle_rank)
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < buckets_per_thread; ++j)
        {
          const int bucket   = tid * buckets_per_thread + j;
          local_hist_vals[j] = (bucket < num_buckets) ? temp_storage.hist[bucket] : offset_t{0};
        }
        block_scan_t(temp_storage.scan_storage).ExclusiveSum(local_hist_vals, local_prefixes);
      }

      cluster_or_block_wait(is_single_cta);

      // Step 3 (leader half): the leader prefix-scans the merged `hist` (raw counts, all folds now visible) and updates
      // the cluster-shared `state`. Subsequent reads (end-of-pass fold, last filter) observe these writes after the
      // next cluster sync.
      if (cluster_rank == layout.leader_rank)
      {
        leader_identify_kth_bucket();
      }

      if (pass + 1 < num_passes)
      {
        // Unconditional barrier: every working rank just read `hist` (the leader via its BlockScan, non-leaders via the
        // scan above), so order all of those reads before any rank resets `hist`.
        __syncthreads();
        reset_hist();
      }
      // TODO(cccl): see the barrier above -- idle ranks arrive here only to keep the cluster barrier reachable.
      cluster_or_block_sync(is_single_cta);

      // End-of-pass splitter fold. Every block pulls the leader's just-published `result_pair` once through DSMEM (a
      // single naturally-aligned `u64`, ordered by the `cluster_or_block_sync()` above) and decodes both halves from
      // that one load: working threads fold the `kth_bucket` digit into their own `kth_key_bits_local` (so the full
      // splitter key is reconstructed locally without a broadcast), and every block -- including idle ranks -- reuses
      // the same value for the uniform early-stop check below, so the whole cluster breaks together.
      const ::cuda::std::uint64_t pass_result = layout.leader_state->result_pair;
      if (!layout.is_idle_rank)
      {
        const int bucket = static_cast<int>(state_t::kth_bucket_of(pass_result));
        _CCCL_ASSERT(bucket >= 0 && bucket < num_buckets, "published splitter bucket index is out of range");
        detail::topk::set_kth_key_bits<key_t, bits_per_pass>(kth_key_bits_local, pass, bucket);
        last_pass = pass + 1;

        // Non-leader: the lane owning the splitter bucket holds its exclusive prefix and raw count in registers from
        // the scan above. Accumulate the strictly-selected count and overwrite `my_candidates` with this pass's
        // splitter-bucket count (the last pass's value is what the filter reads, however the loop exits). The leader's
        // `hist` is merged, so it derives its own counts from the scan total instead (see the filter).
        if (cluster_rank != layout.leader_rank)
        {
          const int owner = bucket / buckets_per_thread;
          if (tid == owner)
          {
            const int slot = bucket - owner * buckets_per_thread;
            atomicAdd(&temp_storage.num_strictly_selected, local_prefixes[slot]);
            temp_storage.my_candidates = local_hist_vals[slot];
          }
        }
      }
#ifndef CUB_DISABLE_CLUSTER_TOPK_EARLY_STOP
      // Early stop: the leader sets `early_stop` when the splitter bucket holds exactly the remaining `k` candidates,
      // so no further radix refinement can change the result. Every block decodes the same flag from the `pass_result`
      // it just loaded and breaks together; `last_pass`/`kth_key_bits_local` then match what the original
      // top-of-next-pass break produced.
      if (state_t::is_early_stop(pass_result))
      {
        break;
      }
#endif
    }
    return last_pass;
  }

  template <detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void run()
  {
    using identify_candidates_op_t =
      detail::topk::identify_candidates_op_t<key_t, SelectDirection, bits_per_pass, decomposer_t>;

    constexpr int total_bits = int{sizeof(key_t)} * 8;
    constexpr int num_passes = detail::topk::calc_num_passes<key_t>(bits_per_pass);

    // `process_impl` handles `k == 0` and select-all, so the radix path sees a strict `0 < k < segment_size`.
    _CCCL_ASSERT(k > out_offset_t{0} && static_cast<segment_size_val_t>(k) < segment_size,
                 "radix path requires 0 < k < segment size");
    compute_segment_layout();

    // Per-block local copy of `kth_key_bits` so each key check hits the block's own SMEM rather than DSMEM during the
    // histogram loop. Built up one splitter digit per pass from the leader's published `kth_bucket` (see
    // `run_radix_passes`), so the full key is never broadcast.
    key_prefix_t kth_key_bits_local = {};

    // Size the persistent overflow-streaming window for this segment; a no-op when this rank has no overflow.
    init_overflow_stream();

    // Preselect the streaming ping-pong direction. A streaming rank flips direction once per histogram pass, so the
    // leftover after the compile-time `num_passes` passes is `initial ^ (num_passes & 1)`; choosing `initial =
    // (!is_tie_reversed) ^ (num_passes & 1)` makes that leftover `== !is_tie_reversed` -- exactly what the
    // deterministic filter's straddling CTA needs to reuse its resident turn-around chunks with no re-prime (see
    // `process_overflow`; early exit runs fewer passes but then has no straddler, so direction is
    // moot). Non-deterministic filtering is order-independent, so leave its historical `stream_is_forward` start
    // untouched.
    if constexpr (need_determinism)
    {
      stream_is_forward = (!is_tie_reversed) ^ ((num_passes & 1) != 0);
    }

    // Contiguous resident-key window staged in SMEM (block-load path); read once per radix pass and in the final
    // filter. Rebound to the real resident window by `load_and_histogram_first_pass`.
    smem_keys_t resident_keys;

    // Front-load all stage mbarrier inits before any bulk copy issues; the barrier below orders them ahead of the first
    // issue (see `init_load_barriers`). The generic fallback has no mbarriers but keeps the same pass-start barrier.
    if constexpr (use_block_load_to_shared)
    {
      init_load_barriers();
    }
    __syncthreads();

    load_and_histogram_first_pass<SelectDirection>(resident_keys);

    // Publish the first pass's staged `edge_keys` and per-rank histogram block-wide before the radix passes read them.
    __syncthreads();

    const int last_pass = run_radix_passes<SelectDirection>(resident_keys, kth_key_bits_local);

    // -----------------------------------------------------------------------
    // Final filter pass: write the top-k keys for this segment. Strictly-
    // selected keys go to the front; the `num_kth` tied candidates fill the
    // back. `kth_key_bits_local` already holds the full splitter key (folded
    // from each pass's bucket above), so no broadcast is needed here.
    // -----------------------------------------------------------------------
    auto block_keys_out        = d_key_segments_out_it[segment_id];
    const out_offset_t num_kth = layout.leader_state->k; // remaining k after the radix passes
    // Each pass keeps `0 < num_kth <= k` inside the splitter bucket (same leader read as `num_kth`, equally safe).
    _CCCL_ASSERT(
      num_kth > out_offset_t{0} && num_kth <= k && static_cast<offset_t>(num_kth) <= layout.leader_state->len,
      "radix passes produced an invalid remaining-k count");

    // `last_pass` controls how many radix levels of `kth_key_bits_local` are significant. After an early-stop break,
    // only the first `last_pass` digits of the splitter were folded; comparing all bits would treat the (still-zero)
    // trailing digits as smaller and erroneously reject candidates that share the identified prefix.
    identify_candidates_op_t identify_op(&kth_key_bits_local, last_pass, total_bits, decomposer_t{});

    // Publish the final pass's per-rank `num_strictly_selected`/`my_candidates` (written by one lane after the last
    // cluster barrier) block-wide before the final-filter scan below reads them.
    __syncthreads();
    if constexpr (need_determinism)
    {
      write_deterministic_topk<SelectDirection>(num_kth, identify_op, block_keys_out, resident_keys);
    }
    else
    {
      write_nondeterministic_topk(num_kth, identify_op, block_keys_out, resident_keys);
    }

    // No cluster barrier after the final filter pass: both filter paths place output via block-local SMEM atomics into
    // gmem, so the last cross-CTA DSMEM access is the scan's counter push in `prime_placement_counters`, already fenced
    // by its post-push cluster barrier (and `early_stop` is cached pre-scan). With no shared-memory access to another
    // block after the scan, a block can return without risking a "cluster target block not present" fault from a
    // straggler.
  }

  // Copies an entire segment `input[i] -> output[i]` for the select-all fast path (`k >= segment_size`). Runs before
  // the effective-cluster collapse, so it takes the raw hardware rank/size (the `cluster_*` members are not set yet)
  // and reads the `segment_id`/`segment_size` members `process_impl` filled in.
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  copy_segment_select_all(unsigned int hw_cluster_rank, unsigned int hw_cluster_blocks)
  {
    constexpr int copy_items       = copy_items_per_thread_clamped;
    const offset_t num_items       = static_cast<offset_t>(segment_size);
    const offset_t cluster_tid     = hw_cluster_rank * static_cast<offset_t>(threads_per_block) + threadIdx.x;
    const offset_t cluster_threads = hw_cluster_blocks * static_cast<offset_t>(threads_per_block);
    const offset_t step            = cluster_threads * static_cast<offset_t>(copy_items);
    const offset_t full_tiles      = ::cuda::round_down(num_items, step);
    auto keys_in_it                = d_key_segments_it[segment_id];
    auto keys_out_it               = d_key_segments_out_it[segment_id];
    // Per-segment value iterators (pairs only), hoisted once like the key iterators and reused by both loops. For
    // keys-only the value iterators-of-iterators are null, so the discarded `if constexpr` branch keeps the indexing
    // out of those builds (the variables stay unused there).
    [[maybe_unused]] auto vals_in_it = [&]() -> value_it_t {
      if constexpr (!is_keys_only)
      {
        return d_value_segments_it[segment_id];
      }
      else
      {
        return value_it_t{};
      }
    }();
    [[maybe_unused]] auto vals_out_it = [&]() -> it_value_t<ValueOutputItItT> {
      if constexpr (!is_keys_only)
      {
        return d_value_segments_out_it[segment_id];
      }
      else
      {
        return it_value_t<ValueOutputItItT>{};
      }
    }();

    for (offset_t base = 0; base < full_tiles; base += step)
    {
      offset_t idx[copy_items];
      key_t keys[copy_items];
      [[maybe_unused]] value_t vals[copy_items];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < copy_items; ++j)
      {
        idx[j] = base + static_cast<offset_t>(j) * cluster_threads + cluster_tid;
      }
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < copy_items; ++j)
      {
        keys[j] = keys_in_it[static_cast<segment_size_val_t>(idx[j])];
      }
      if constexpr (!is_keys_only)
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < copy_items; ++j)
        {
          vals[j] = vals_in_it[static_cast<segment_size_val_t>(idx[j])];
        }
      }
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < copy_items; ++j)
      {
        keys_out_it[static_cast<segment_size_val_t>(idx[j])] = keys[j];
      }
      if constexpr (!is_keys_only)
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < copy_items; ++j)
        {
          vals_out_it[static_cast<segment_size_val_t>(idx[j])] = vals[j];
        }
      }
    }

    // Sub-tile remainder. Iterate the tail relative to zero and offset by `full_tiles`, so every index stays in
    // `[0, num_items)` without relying on `full_tiles + cluster_tid` staying within `offset_t`.
    const offset_t tail_items = num_items - full_tiles;
    _CCCL_PRAGMA_NOUNROLL()
    for (offset_t local = cluster_tid; local < tail_items; local += cluster_threads)
    {
      const auto seg_idx   = static_cast<segment_size_val_t>(full_tiles + local);
      keys_out_it[seg_idx] = keys_in_it[seg_idx];
      if constexpr (!is_keys_only)
      {
        vals_out_it[seg_idx] = vals_in_it[seg_idx];
      }
    }
  }

  // Set the effective-cluster members (`cluster_rank`/`cluster_blocks`/`is_single_cta`) from the hardware cluster and
  // this segment's size. For a per-segment (deferred) size argument the launch is sized for the maximum segment, so a
  // small segment that fits resident in one CTA and is at/below the single-CTA tuning threshold is served by rank 0
  // alone via the barrier-free path; the cluster's other CTAs are redundant and this returns false so they exit,
  // freeing their SM slots. The decision is per-segment uniform across the block, so a redundant CTA returns whole.
  // Compiled out for host-exact sizes, which the dispatch already sized to exact cluster blocks (returns true for all).
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE bool
  init_effective_cluster(unsigned int hw_cluster_rank, unsigned int hw_cluster_blocks)
  {
    cluster_rank   = hw_cluster_rank;
    cluster_blocks = hw_cluster_blocks;
    if constexpr (enable_runtime_single_cta)
    {
      const bool fits_single_cta = is_single_cta_eligible(
        static_cast<::cuda::std::uint64_t>(segment_size),
        static_cast<::cuda::std::uint64_t>(block_tile_capacity),
        single_block_max_seg_size);
      if (fits_single_cta)
      {
        if (hw_cluster_rank != 0u)
        {
          return false;
        }
        cluster_rank   = 0u;
        cluster_blocks = 1u;
      }
    }
    is_single_cta = (cluster_blocks == 1u);
    return true;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void process_impl()
  {
    // Hardware cluster rank/size from the PTX special registers (replaces cooperative_groups' `this_cluster()`). The
    // radix path runs on the *effective* geometry (the `cluster_rank`/`cluster_blocks` members set by
    // `init_effective_cluster` below); only the pre-collapse steps here (segment id, select-all fast path) use the raw
    // hardware values. Runtime cluster blocks match the launch attribute the dispatch passed to `cudaLaunchKernelExC`
    // (or the kernel's `__cluster_dims__` on CDP).
    const unsigned int hw_cluster_rank   = ::cuda::ptx::get_sreg_cluster_ctarank();
    const unsigned int hw_cluster_blocks = ::cuda::ptx::get_sreg_cluster_nctarank();
    _CCCL_ASSERT(hw_cluster_blocks > 0u && hw_cluster_rank < hw_cluster_blocks,
                 "hardware cluster rank must lie within a non-empty cluster");
    segment_id = static_cast<num_segments_val_t>(blockIdx.x / hw_cluster_blocks);

    if (segment_id >= detail::params::get_param(num_segments, num_segments_val_t{0}))
    {
      return;
    }

    segment_size = static_cast<segment_size_val_t>(detail::params::get_segment_size(segment_sizes, segment_id));
    // Precondition: sizes are clamped >= 0 and capped at 2^21, so a value exceeding 32-bit `offset_t` is a violation.
    _CCCL_ASSERT(static_cast<::cuda::std::uint64_t>(segment_size) <= ::cuda::std::uint64_t{0xffffffffu},
                 "segment size must be non-negative and fit the 32-bit cluster offset type");
    // Clamp the requested `k` to the segment size in a 64-bit width holding both operands, *before* narrowing to
    // `out_offset_t`. Clamping in a narrower type first would let a large "select all" `k` wrap to a small value and
    // silently truncate the output (or wrap to 0 and skip the segment).
    const auto k_clamped =
      (::cuda::std::min) (static_cast<::cuda::std::uint64_t>(detail::params::get_param(k_param, segment_id)),
                          static_cast<::cuda::std::uint64_t>(segment_size));

    if (k_clamped == 0)
    {
      return;
    }

    // Segments larger than the resident cluster_tile capacity are still handled -- the overflow chunks are re-streamed
    // from gmem (see the "Overflow streaming" section).

    // `k_clamped <= segment_size`, which now fits `out_offset_t`, so this narrowing is safe.
    k = static_cast<out_offset_t>(k_clamped);

    // Select-all fast path: when `k` reaches the full segment, every element wins, so we skip the radix passes,
    // histogram, and output-ordering and just copy. Runs on the full launched cluster (before the effective-cluster
    // collapse), so it uses the raw hardware rank/size; the decision is per-segment uniform, so the branch is
    // cluster-uniform.
    if (static_cast<segment_size_val_t>(k) == segment_size)
    {
      copy_segment_select_all(hw_cluster_rank, hw_cluster_blocks);
      return;
    }

    // Collapse the launched cluster onto the effective geometry the radix path runs on (sets the `cluster_rank`/
    // `cluster_blocks`/`is_single_cta` members). Returns false on the CTAs the collapse makes redundant, which exit
    // here.
    if (!init_effective_cluster(hw_cluster_rank, hw_cluster_blocks))
    {
      return;
    }

    // Every block's thread 0 initializes its local `state`. Only the
    // leader's copy is semantically read (non-leaders reach the cluster
    // state through `leader_state`), but mirroring the writes everywhere
    // keeps every block's unconditional `state.k` load safe under
    // compute-sanitizer.
    if (tid == 0)
    {
      temp_storage.state.len         = static_cast<offset_t>(segment_size);
      temp_storage.state.k           = k;
      temp_storage.state.result_pair = 0;
      // Front-load the scan accumulators so the initial cluster barrier below (arrive here, wait in the first pass)
      // publishes the zeros to all ranks: the final filter's `prime_placement_counters` then adds into already-zeroed
      // `front_local_cnt`/`back_local_cnt` (its own local seed + peers' DSMEM pushes, needing only a post-push
      // barrier). `my_candidates` is zeroed too so idle/leader ranks that never write it read 0.
      temp_storage.front_local_cnt       = 0;
      temp_storage.back_local_cnt        = 0;
      temp_storage.num_strictly_selected = 0;
      temp_storage.my_candidates         = 0;
    }
    reset_hist();
    // Only arrive here; the matching wait is deferred to just before the first cross-CTA fold in `run_radix_passes`'
    // first pass, so the cluster-arrival latency overlaps the fused first-pass load + histogram. Safe because nothing
    // between here and that wait touches another rank's DSMEM (the first-pass histogram writes only block-local
    // `hist`; `leader_state` and the scan counters are untouched until later).
    cluster_or_block_arrive(is_single_cta);

    [[maybe_unused]] const bool is_ok =
      detail::params::dispatch_discrete(select_directions, segment_id, [this](auto direction_tag) {
        constexpr detail::topk::select Direction = decltype(direction_tag)::value;
        this->template run<Direction>();
      });
    _CCCL_ASSERT(is_ok, "Unsupported select direction for cluster top-k");
  }
};
} // namespace detail::batched_topk_cluster

CUB_NAMESPACE_END
