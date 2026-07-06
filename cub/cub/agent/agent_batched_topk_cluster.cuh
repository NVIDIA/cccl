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
//!      shared-space `red` reductions (cheap, SMEM-local): cta scope for
//!      non-leaders, cluster scope for the leader (see `hist_inc`).
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
//! atomics, seeded by a single combined 64-bit cross-CTA prefix scan (each
//! block's selected-front and candidate-back base offsets); no cluster-wide
//! output cursor is kept in `state`.

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
#include <cuda/__cmath/round_up.h>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__memcpy_async/elect_one.h>
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
#include <cuda/std/limits>
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

// True for `cuda::args` segment-size arguments whose exact per-segment value the host cannot know at dispatch time, so
// the launch is sized for a static upper bound: the `deferred` forms. Combined with `!is_single_value` (any
// per-segment sequence) this gates the runtime effective-single-CTA path; host-exact `immediate`/`constant` singles,
// which the dispatch already sizes exactly, are excluded.
template <class>
inline constexpr bool __is_deferred_arg_v = false;
template <class _Arg, class _Bounds>
inline constexpr bool __is_deferred_arg_v<::cuda::args::deferred<_Arg, _Bounds>> = true;
template <class _Arg, class _Bounds>
inline constexpr bool __is_deferred_arg_v<::cuda::args::deferred_sequence<_Arg, _Bounds>> = true;

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
    (::cuda::std::clamp) (blocks, ::cuda::std::uint64_t{1}, static_cast<::cuda::std::uint64_t>(cluster_blocks_cap)));
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

  using offset_t     = ::cuda::std::uint32_t;
  using out_offset_t = ::cuda::std::uint32_t;
  using state_t      = cluster_topk_state<key_t, offset_t, out_offset_t>;
  using key_prefix_t = typename state_t::key_prefix_t;

  // See the class comment: deterministic guarantees select the index-ordered scan; `is_tie_reversed` flips its order.
  static constexpr bool need_determinism =
    Determinism != ::cuda::execution::determinism::__determinism_t::__not_guaranteed;
  static constexpr bool is_tie_reversed =
    TieBreak == ::cuda::execution::tie_break::__tie_break_t::__prefer_larger_index;

  // Push direction of the combined cross-CTA prefix scan (`combined_prefix_scan`). The leader must be *last* in scan
  // order so it derives its own (merged-away) counts from the predecessor sum: deterministic-prefer-smaller puts the
  // leader at the last effective rank and scans ascending; every other config (deterministic-prefer-larger and the
  // whole non-deterministic path) keeps the leader at rank 0 and scans descending, which makes rank 0 last. Matches the
  // `leader_rank` computed in `run`.
  static constexpr bool is_scan_descending = !(need_determinism && !is_tie_reversed);

  // The deterministic final scan visits chunks in global-index order and bails early (`should_stop`), so keeping the
  // *first-visited* chunks resident lets it skip re-reading the streamed overflow. Ascending visits low indices first
  // (the default low-resident split); descending (prefer-larger) visits high indices first, so we flip the split to
  // keep the high-index chunks resident (see `run`), restoring symmetry. Never set on the non-deterministic path.
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
    !::cuda::args::__traits<SegmentSizeParameterT>::is_single_value || __is_deferred_arg_v<SegmentSizeParameterT>;

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
  // keeps the whole resident segment inside one tile for fully-predicated loops (the deterministic filter).
  static constexpr int histogram_items_per_thread_clamped =
    clamp_unroll(segment_rounds_floor, histogram_items_per_thread);
  static constexpr int tie_break_items_per_thread_clamped =
    clamp_unroll(segment_rounds_ceil, tie_break_items_per_thread);
  static constexpr int tie_break_items_per_thread_floor_clamped =
    clamp_unroll(segment_rounds_floor, tie_break_items_per_thread);
  static constexpr int copy_items_per_thread_clamped = clamp_unroll(segment_rounds_floor, copy_items_per_thread);
  static_assert(histogram_items_per_thread_clamped >= 1, "histogram_items_per_thread_clamped must be positive");
  static_assert(tie_break_items_per_thread_clamped >= 1, "tie_break_items_per_thread_clamped must be positive");
  static_assert(tie_break_items_per_thread_floor_clamped >= 1,
                "tie_break_items_per_thread_floor_clamped must be positive");
  static_assert(copy_items_per_thread_clamped >= 1, "copy_items_per_thread_clamped must be positive");

  static constexpr int num_buckets      = 1 << bits_per_pass;
  using smem_layout_t                   = smem_block_tile_layout<key_t, ChunkBytes, LoadAlignBytes>;
  static constexpr int chunk_items      = smem_layout_t::chunk_items;
  static constexpr int load_align_items = smem_layout_t::load_align_items;
  static constexpr int slot_alignment   = smem_layout_t::slot_alignment;

  // Tie-break unroll for the deterministic filter's streamed overflow, which feeds `process_tiles` one chunk slot at a
  // time: clamp items so the tile (`threads_per_block * items`) stays within a chunk, bounding the per-tile early-exit
  // to <= chunk granularity. Resident/edge regions keep the full `tie_break_items_per_thread_clamped`. Floors at 1.
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
  // `prefix_pair` packs this block's exclusive cross-CTA scan result (high = `sel_prefix`, low = `cand_prefix`); peers
  // add into it through DSMEM (`add_remote_prefix`), so it must sit at an identical offset in every block's storage.
  // The remaining scalars are block-local: `front_local_cnt`/`back_local_cnt` hand out output slots via SMEM atomics,
  // `num_strictly_selected` accumulates this block's strictly-selected count across passes, and `my_candidates` holds
  // the last pass's splitter-bucket count.
  struct _TempStorage
  {
    offset_t hist[num_buckets];
    state_t state;
    ::cuda::std::uint64_t prefix_pair;
    offset_t front_local_cnt;
    offset_t back_local_cnt;
    offset_t num_strictly_selected;
    offset_t my_candidates;
    typename block_scan_t::TempStorage scan_storage;
    // One mbarrier handle per pipeline stage, shared by the resident load and the overflow streamer and reused
    // (ping-ponged) across radix passes; all are initialized once up front by `init_load_barriers`.
    ::cuda::std::uint64_t load_mbar[PipelineStages];
    // Persistent unaligned boundary edges (block-load path only): the head prefix (`[0, head_edge_cap_items)`, on rank
    // 0) and the peeled tail suffix (`[head_edge_cap_items, 2 * head_edge_cap_items)`, on the tail owner whenever it is
    // unaligned), each strictly `< load_align_items` keys. Loaded once in the first pass and folded into every pass +
    // the final filter. Block-local (never reached through DSMEM).
    key_t edge_keys[2 * load_align_items];
  };
  // The `red.add.u64` on `prefix_pair`'s `.shared::cluster` address needs 8-byte alignment; the `uint64_t` member is
  // already at an 8-aligned struct offset, so guarding the struct's alignment covers the absolute (and peer) address.
  static_assert(alignof(_TempStorage) >= 8, "prefix_pair must be 8-byte aligned for the u64 DSMEM atomic");
  // Split point of `edge_keys`: head edge in `[0, head_edge_cap_items)`, tail edge in `[head_edge_cap_items, 2 *
  // head_edge_cap_items)`.
  static constexpr int head_edge_cap_items = load_align_items;

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE key_t* slot_keys_unpadded(int slot) const
  {
    return reinterpret_cast<key_t*>(key_slots + slot * ChunkBytes);
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE int span_size(::cuda::std::span<key_t> keys) const
  {
    const int count = static_cast<int>(::cuda::std::size(keys));
    _CCCL_ASSERT(static_cast<::cuda::std::size_t>(count) == ::cuda::std::size(keys),
                 "Resident key span length must fit in int");
    return count;
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

  // Number of aligned chunks covering the segment. The unaligned head prefix (`head_items`) is handled as a separate
  // edge, not a chunk, so the aligned region `[head_items, segment_size)` chunks uniformly at `chunk_items` (a segment
  // lying entirely before the first boundary has no chunks). `head_items == 0` (aligned base or generic fallback) is
  // then plain uniform chunking from offset 0.
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE offset_t num_chunks(offset_t segment_size, offset_t head_items) const
  {
    const offset_t aligned_items = (segment_size > head_items) ? (segment_size - head_items) : offset_t{0};
    return static_cast<offset_t>(::cuda::ceil_div(aligned_items, offset_t{chunk_items}));
  }

  struct chunk_desc
  {
    offset_t offset;
    int count;
  };

  // Chunk `chunk_idx` of the aligned region: it begins on a `load_align` boundary (`head_items + chunk_idx *
  // chunk_items`, with `head_items` itself aligning the base), so every chunk has a zero prefix and only the last chunk
  // can carry an unaligned suffix (the segment's trailing `< load_align_items` items).
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE chunk_desc
  get_chunk(offset_t chunk_idx, offset_t segment_size, offset_t head_items) const
  {
    const offset_t offset    = head_items + chunk_idx * offset_t{chunk_items};
    const offset_t remaining = segment_size - offset;
    return {offset, static_cast<int>((::cuda::std::min) (remaining, offset_t{chunk_items}))};
  }

  // Splits a chunk into its unaligned front edge, aligned interior (bulk), and unaligned back edge, relative to the
  // gmem base. The interior begins and ends on a `load_align` boundary so it can be loaded with the aligned,
  // guard-free BlockLoadToShared path; the edges (each `< load_align_items` items) are staged via
  // `stage_and_fold_edge`. The head prefix is peeled as a separate edge before chunking, so a `get_chunk` chunk begins
  // on a boundary (zero prefix) and only the last chunk (tail) carries a nonzero suffix.
  struct chunk_split
  {
    offset_t prefix;
    offset_t bulk;
    offset_t suffix;
  };

  template <typename PtrT>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE chunk_split split_chunk(PtrT base, const chunk_desc chunk) const
  {
    const auto align_bytes   = static_cast<::cuda::std::uintptr_t>(load_align_bytes);
    const auto begin         = reinterpret_cast<::cuda::std::uintptr_t>(base + chunk.offset);
    const auto end           = begin + static_cast<::cuda::std::uintptr_t>(chunk.count) * sizeof(key_t);
    const auto aligned_begin = ::cuda::round_up(begin, align_bytes);
    const auto aligned_end   = ::cuda::round_down(end, align_bytes);
    if (aligned_begin > aligned_end)
    {
      // The chunk lies strictly between two load_align boundaries (no aligned point inside): the whole chunk is an
      // unaligned edge, attributed entirely to the front edge. `get_chunk` chunks begin on a boundary (the head is
      // peeled separately), so this only guards a degenerate sub-`load_align` chunk. A tail always begins on a
      // boundary, so it takes the `aligned_begin <= aligned_end` path below and its unaligned remainder is the suffix.
      return {static_cast<offset_t>(chunk.count), offset_t{0}, offset_t{0}};
    }
    const offset_t prefix = static_cast<offset_t>((aligned_begin - begin) / sizeof(key_t));
    const offset_t bulk   = static_cast<offset_t>((aligned_end - aligned_begin) / sizeof(key_t));
    return {prefix, bulk, static_cast<offset_t>(chunk.count) - prefix - bulk};
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE offset_t
  num_rank_chunks(offset_t chunks, unsigned int cluster_rank, unsigned int cluster_blocks) const
  {
    return (cluster_rank < chunks)
           ? static_cast<offset_t>((chunks - 1 - cluster_rank) / cluster_blocks + 1)
           : offset_t{0};
  }

  // Assignment of the cluster's global chunk indices `[0, chunks)` to its CTAs. A rank owns `count` chunks; its i-th
  // owned chunk has global index `global_index(i) = first + i * stride`. The single mapping point lets the rest of the
  // agent (resident load, streamer, per-pass scans) stay agnostic to the layout chosen by `make_chunk_partition`.
  struct chunk_partition
  {
    offset_t first; // global index of this rank's first owned chunk
    offset_t stride; // distance between consecutive owned chunks (`cluster_blocks` strided, `1` blocked)
    offset_t count; // number of chunks owned by this rank

    [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE offset_t global_index(offset_t local) const
    {
      return first + local * stride;
    }
  };

  // Decides which global chunks a cluster rank owns. Both layouts keep chunk 0 on rank 0 (which also stages the head
  // edge) and the tail (chunk `chunks-1`) on a single rank, and leave the per-chunk alignment, the resident/streaming
  // split, and the streamer ping-pong untouched, because all of those depend only on the global chunk index, not on
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
    if constexpr (need_determinism)
    {
      const offset_t chunks_per_cta = ::cuda::ceil_div(chunks, static_cast<offset_t>(cluster_blocks));
      const offset_t first          = static_cast<offset_t>(cluster_rank) * chunks_per_cta;
      const offset_t count = (first < chunks) ? (::cuda::std::min) (chunks_per_cta, chunks - first) : offset_t{0};
      return {first, offset_t{1}, count};
    }
    else
    {
      return {static_cast<offset_t>(cluster_rank),
              static_cast<offset_t>(cluster_blocks),
              num_rank_chunks(chunks, cluster_rank, cluster_blocks)};
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
    for (int local = static_cast<int>(threadIdx.x); local < count; local += threads_per_block)
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
  _CCCL_DEVICE _CCCL_FORCEINLINE void for_each_chunk_key_impl(const key_t* data, int chunk_count, Apply&& apply) const
  {
    constexpr int tile   = Unroll * threads_per_block;
    const int tid        = static_cast<int>(threadIdx.x);
    const int full_tiles = ::cuda::round_down(chunk_count, tile);

    _CCCL_PRAGMA_NOUNROLL()
    for (int tile_base = 0; tile_base < full_tiles; tile_base += tile)
    {
      key_t regs[Unroll];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < Unroll; ++i)
      {
        regs[i] = data[tile_base + i * threads_per_block + tid];
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
      apply(data[local], local);
    }
  }

  template <int Unroll, typename F>
  _CCCL_DEVICE _CCCL_FORCEINLINE void for_each_chunk_key(::cuda::std::span<key_t> chunk_keys, F&& f) const
  {
    for_each_chunk_key_impl<Unroll>(::cuda::std::data(chunk_keys), span_size(chunk_keys), [&](const key_t& key, int) {
      f(key);
    });
  }

  // Like `for_each_chunk_key`, but also hands `f` each key's segment-local index `base_off + local`, where `base_off`
  // is the segment-local offset of the chunk's first element. The pair path uses that index to fetch the key's value
  // payload from gmem, so overflow keys can be reused from the streaming SMEM pipeline instead of re-read from gmem.
  template <int Unroll, typename F>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  for_each_chunk_key_indexed(::cuda::std::span<key_t> chunk_keys, offset_t base_off, F&& f) const
  {
    for_each_chunk_key_impl<Unroll>(
      ::cuda::std::data(chunk_keys), span_size(chunk_keys), [&](const key_t& key, int local) {
        f(key, base_off + static_cast<offset_t>(local));
      });
  }

  // A bulk in the block_tile as a 32-bit shared address + length. A spilled 32-bit shared address (rebuilt with
  // `__cvta_shared_to_generic`) keeps the key reads `LDS`; a spilled 64-bit generic pointer would demote them to `LD`.
  struct shared_bulk
  {
    ::cuda::std::uint32_t smem32;
    int len;
  };

  // Rebuild a bulk's span from its 32-bit shared address at the point of use (spill-proof `LDS`).
  [[nodiscard]] static _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::span<key_t> bulk_span(shared_bulk b)
  {
    return {reinterpret_cast<key_t*>(__cvta_shared_to_generic(b.smem32)), static_cast<::cuda::std::size_t>(b.len)};
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
    for (int stage = static_cast<int>(threadIdx.x); stage < PipelineStages; stage += threads_per_block)
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
    // Only the block leader (see `is_load_leader`) drives the mbarrier, for a uniform branch and better mbarrier
    // codegen.
    if (!is_load_leader)
    {
      return;
    }
    const int num_bytes = static_cast<int>(::cuda::std::size(src)) * int{sizeof(key_t)};
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
    const ::cuda::std::uint32_t parity = (load_phase >> stage) & 1u;
    while (!::cuda::ptx::mbarrier_try_wait_parity(&temp_storage.load_mbar[stage], parity))
    {
    }
    load_phase ^= (::cuda::std::uint32_t{1} << stage);
  }

  struct TempStorage : Uninitialized<_TempStorage>
  {};

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
  // overflow streamer keep their per-stage issue/wait calls balanced so each bit tracks its mbarrier's phase.
  ::cuda::std::uint32_t load_phase{};
  // The single block leader (warp 0's elected lane) that drives the bulk copies. Elected once at construction (the
  // block constructs convergently) and cached so the pipeline reuses it instead of re-electing per copy.
  const bool is_load_leader = ::cuda::device::__block_elect_one();

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
    for (int i = static_cast<int>(threadIdx.x); i < num_buckets; i += threads_per_block)
    {
      temp_storage.hist[i] = 0;
    }
  }

  // ---------------------------------------------------------------------------
  // Block-private histogram atomics (shared-space `red` via cuda::ptx-style inline PTX)
  // ---------------------------------------------------------------------------
  // A builtin `atomicAdd(&temp_storage.hist[bucket], 1)` compiles to a generic atomic (`ATOM.E`) whose 64-bit base is
  // spilled and reloaded (`LDL.64`) at every update across this huge agent. A shared-space `red` instead addresses with
  // the 32-bit shared address (no base to spill). Shared atomics only allow cta/cluster scope, which is exactly right:
  // every writer of a given `hist` is in the same cluster. `red` (no return) matches the discarded `atomicAdd` result.

  // The 32-bit shared address of `hist[0]`. Hoisted once per histogram region so the per-key address math is a pure
  // 32-bit add; recomputing it per key would reload the 64-bit generic base of `temp_storage` from the stack.
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::uint32_t hist_base32() const
  {
    return static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(temp_storage.hist));
  }

  // Increment this block's own histogram bucket by one via its 32-bit shared address. The leader (rank 0) also receives
  // remote folds from the other cluster blocks (Step 2, `hist_fold_remote`), so its add must be cluster-scoped to be
  // mutually atomic with them; non-leaders only touch their own `hist` before the fold, so cta scope suffices.
  _CCCL_DEVICE _CCCL_FORCEINLINE void hist_inc(::cuda::std::uint32_t base32, int bucket, bool is_leader)
  {
    const ::cuda::std::uint32_t addr = base32 + static_cast<::cuda::std::uint32_t>(bucket) * sizeof(offset_t);
    if (is_leader)
    {
      asm volatile("red.relaxed.cluster.shared::cta.add.u32 [%0], 1;" : : "r"(addr) : "memory");
    }
    else
    {
      asm volatile("red.relaxed.cta.shared::cta.add.u32 [%0], 1;" : : "r"(addr) : "memory");
    }
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

  // Adds the packed 64-bit `v` to the `prefix_pair` of the CTA at cluster rank `target_rank` through DSMEM (mirrors
  // `hist_fold_remote`: `mapa` to `target_rank`, then a cluster-scope `red.add`). Exact because the two 32-bit lanes
  // never carry into each other. Drives the combined cross-CTA selected/candidate prefix scan.
  _CCCL_DEVICE _CCCL_FORCEINLINE void add_remote_prefix(unsigned int target_rank, ::cuda::std::uint64_t v)
  {
    const ::cuda::std::uint32_t own =
      static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(&temp_storage.prefix_pair));
    ::cuda::std::uint32_t remote;
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(own), "r"(target_rank));
    asm volatile("red.relaxed.cluster.shared::cluster.add.u64 [%0], %1;" : : "r"(remote), "l"(v) : "memory");
  }

  // Block-local SMEM atomics for the final filter: `front_local_inc`/`back_local_inc` hand out this block's next
  // front/back output slot (pre-increment value), `add_local_selected` accumulates its strictly-selected count across
  // passes. Each addresses the 32-bit shared slot directly (like `hist_inc`) to dodge the generic-atomic base spill of
  // `atomicAdd(&shared)`, at cta scope since every writer of a block's counter lives in that block.
  _CCCL_DEVICE _CCCL_FORCEINLINE offset_t front_local_inc()
  {
    const ::cuda::std::uint32_t addr =
      static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(&temp_storage.front_local_cnt));
    offset_t old;
    asm volatile("atom.relaxed.cta.shared::cta.add.u32 %0, [%1], 1;" : "=r"(old) : "r"(addr) : "memory");
    return old;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE offset_t back_local_inc()
  {
    const ::cuda::std::uint32_t addr =
      static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(&temp_storage.back_local_cnt));
    offset_t old;
    asm volatile("atom.relaxed.cta.shared::cta.add.u32 %0, [%1], 1;" : "=r"(old) : "r"(addr) : "memory");
    return old;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void add_local_selected(offset_t count)
  {
    const ::cuda::std::uint32_t addr =
      static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(&temp_storage.num_strictly_selected));
    asm volatile("red.relaxed.cta.shared::cta.add.u32 [%0], %1;" : : "r"(addr), "r"(count) : "memory");
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
        // Publish the splitter bucket and the early-stop flag (set when the bucket holds exactly the remaining `k`);
        // see `result_pair`.
        temp_storage.state.set_result(
          static_cast<::cuda::std::uint32_t>(bucket), static_cast<out_offset_t>(new_len) == new_k);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Overflow streamer
  // ---------------------------------------------------------------------------
  // Re-streams a rank's "overflow" chunks (those that do not fit its resident SMEM region) from gmem through a fixed,
  // round-robin set of `eff_stages` (<= `PipelineStages`) streaming slots, reused across every radix pass and the final
  // filter. It ping-pongs the iteration order across calls so the `eff_stages` turn-around chunks that one pass leaves
  // resident in the streaming slots are reused by the next with no reload; the remaining `overflow_chunks - eff_stages`
  // are reloaded from gmem each pass. The caller sizes the reservation to `eff_stages = min(PipelineStages, excess)`
  // (`excess = my_chunks - full_slots`), so a streaming rank reloads exactly `excess` chunks per pass -- the reserved
  // slots only ever buy reuse of the turn-around chunks, never a reload-free pass. The resident region occupies slots
  // `[0, resident_slots)`, the streaming region `[stream_slot_base, stream_slot_base + eff_stages)`.
  struct overflow_streamer
  {
    agent_batched_topk_cluster& agent;
    key_it_t block_keys_in;
    const key_t* block_keys_base; // unwrapped contiguous base (pipeline path only; null otherwise)
    offset_t segment_size;
    offset_t head_items;
    chunk_partition part; // rank -> global chunk index mapping (strided or blocked)
    offset_t resident_chunks; // number of rank-local chunks kept resident
    offset_t overflow_base; // rank-local chunk index of the first overflow (streamed) chunk
    int stream_slot_base; // SMEM slot index at which the streaming region begins
    offset_t overflow_chunks; // number of overflow chunks for this rank (M)
    int eff_stages; // streaming region size = reserved streaming slots (<= PipelineStages, <= M, >= 1)
    bool is_forward = true;
    bool is_primed  = false;

    // Stage mbarriers are shared with the resident load (`agent.temp_storage.load_mbar`); stage `stage` targets slot
    // `stream_slot_base + stage`. `inflight_mask` bit `stage` is set only while a copy is in flight (issued, not yet
    // waited). The slot/stage mapping is fixed, so the read span is recomputed on demand by `stage_span` rather than
    // held in a spillable per-stage array; the only per-stage state is one bit each of `inflight_mask` (here) and the
    // agent's `load_phase` parity.
    ::cuda::std::uint32_t inflight_mask = 0;

    _CCCL_DEVICE _CCCL_FORCEINLINE overflow_streamer(
      agent_batched_topk_cluster& agent_,
      key_it_t block_keys_in_,
      const key_t* block_keys_base_,
      offset_t segment_size_,
      offset_t head_items_,
      chunk_partition part_,
      offset_t resident_chunks_,
      offset_t overflow_base_,
      int stream_slot_base_,
      int stream_slots_,
      offset_t my_chunks_)
        : agent(agent_)
        , block_keys_in(block_keys_in_)
        , block_keys_base(block_keys_base_)
        , segment_size(segment_size_)
        , head_items(head_items_)
        , part(part_)
        , resident_chunks(resident_chunks_)
        , overflow_base(overflow_base_)
        , stream_slot_base(stream_slot_base_)
        , overflow_chunks((my_chunks_ > resident_chunks_) ? (my_chunks_ - resident_chunks_) : offset_t{0})
    {
      // `eff_stages` is the streaming region size the caller reserved at `[stream_slot_base, stream_slot_base +
      // stream_slots)`; using all of it as pipeline stages keeps the ping-pong reuse maximal. It is `<= M` whenever
      // there is overflow (`M = excess + stream_slots >= stream_slots`); the `>= 1` floor only matters for the no-op
      // (`M == 0`) case, where the streamer never touches a slot.
      eff_stages = (stream_slots_ > 0) ? stream_slots_ : 1;
      _CCCL_ASSERT(overflow_chunks == 0 || eff_stages <= static_cast<int>(overflow_chunks),
                   "streaming depth exceeds the overflow chunk count");
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE offset_t chunk_index_of(offset_t overflow_idx) const
    {
      return part.global_index(overflow_base + overflow_idx);
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE void issue_load(int stage, offset_t overflow_idx)
    {
      const offset_t chunk_idx = chunk_index_of(overflow_idx);
      const auto chunk         = agent.get_chunk(chunk_idx, segment_size, head_items);
      // Every chunk begins on a `load_align` boundary (zero prefix), so the guard-free aligned (TMA bulk) path applies.
      // The global-last chunk's unaligned suffix is always peeled into `edge_keys`, so streaming just its aligned bulk
      // excludes it. For every interior chunk `bulk == count`.
      const auto split = agent.split_chunk(block_keys_base, chunk);
      _CCCL_ASSERT(split.prefix == offset_t{0}, "overflow streamer received a chunk with an unaligned start");
      char* const dst = agent.key_slots + (stream_slot_base + stage) * ChunkBytes;
      const ::cuda::std::span<const key_t> src{
        block_keys_base + chunk.offset, static_cast<::cuda::std::size_t>(split.bulk)};
      agent.issue_bulk_copy(stage, dst, src);
      inflight_mask |= (::cuda::std::uint32_t{1} << stage);
    }

    // Rebuild the shared span for the chunk currently resident in `stage`'s slot without storing per-stage state: the
    // slot address is a pure function of `stage` and the length is recomputed from `overflow_idx`, so there is no
    // spillable `pending[]` array (see `bulk_span`). Returns the aligned bulk only (the always-peeled tail suffix is
    // excluded).
    [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::span<key_t>
    stage_span(int stage, offset_t overflow_idx) const
    {
      char* const dst  = agent.key_slots + (stream_slot_base + stage) * ChunkBytes;
      const auto chunk = agent.get_chunk(chunk_index_of(overflow_idx), segment_size, head_items);
      const auto split = agent.split_chunk(block_keys_base, chunk);
      return agent.bulk_span(
        {static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(dst)), static_cast<int>(split.bulk)});
    }

    // Shared driver for one overflow pass. `block_apply(stage, overflow_idx)` folds the chunk `overflow_idx` resident
    // in streaming slot `stage` (block-load path); `generic_apply(chunk)` folds an overflow chunk read straight from
    // gmem (fallback). `mid()` runs once on a full pass -- after the first reload wave (`eff_stages` visits) is issued
    // but before it is waited on, overlapping the caller's resident-chunk work with those in-flight copies (skipped if
    // phase 1 stops early); it must be block-uniform with no unmatched barrier. `should_continue()` is polled once
    // after each consumed chunk (before its refill copy is issued); returning false breaks the stream so the final
    // filter bails once the top-k is placed. Its result must be block-uniform (all lanes break together, else the
    // post-break barrier deadlocks) and it is evaluated by every lane, so it may contain a barrier (the deterministic
    // filter's does, to read its placement counters block-wide); the histogram and non-deterministic filter pass an
    // always-true predicate. An early break can leave prefetches in flight, so the pass drains the
    // remaining stages before returning (a full pass ends with an empty `inflight_mask`, so the drain is a no-op).
    template <typename BlockApply, typename GenericApply, typename Mid, typename Continue>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    run_pass(BlockApply&& block_apply, GenericApply&& generic_apply, Mid&& mid, Continue&& should_continue)
    {
      if (overflow_chunks == 0)
      {
        mid();
        return;
      }

      if constexpr (use_block_load_to_shared)
      {
        // First ever call: prime the streaming slots. Subsequent calls inherit
        // the previous pass's resident tail, which (because the order
        // ping-pongs) is exactly the first `eff_stages` chunks of this direction.
        if (!is_primed)
        {
          // Wait for all threads to leave the resident load's final wait before re-arming its shared mbarriers; else
          // the phase advances twice and a lagging thread misses the flip and spins forever.
          __syncthreads();
          for (int i = 0; i < eff_stages; ++i)
          {
            const offset_t overflow_idx =
              is_forward ? static_cast<offset_t>(i) : (overflow_chunks - 1 - static_cast<offset_t>(i));
            issue_load(static_cast<int>(overflow_idx % static_cast<offset_t>(eff_stages)), overflow_idx);
          }
          is_primed = true;
        }

        // Consume the `i`-th visit (its ping-pong-ordered position is `overflow_idx`): wait for its slot, fold its keys
        // via `block_apply`, then prefetch the chunk `eff_stages` visits ahead into the slot just freed (a barrier
        // guards the slot before the async copy can overwrite the data the block was just reading). Returns false once
        // `should_continue()` reports the top-k fully placed -- polled before the prefetch so we never launch a copy we
        // would only drain again; the up-to-`eff_stages - 1` prefetches already in flight (from earlier visits or
        // priming) are drained after the loop.
        const auto consume = [&](offset_t i) -> bool {
          const offset_t overflow_idx = is_forward ? i : (overflow_chunks - 1 - i);
          const int stage             = static_cast<int>(overflow_idx % static_cast<offset_t>(eff_stages));
          if (inflight_mask & (::cuda::std::uint32_t{1} << stage))
          {
            agent.wait_stage(stage);
            inflight_mask &= ~(::cuda::std::uint32_t{1} << stage);
          }
          block_apply(stage, overflow_idx);

          if (!should_continue())
          {
            return false;
          }

          const offset_t next_step = i + static_cast<offset_t>(eff_stages);
          if (next_step < overflow_chunks)
          {
            const offset_t next_overflow_idx = is_forward ? next_step : (overflow_chunks - 1 - next_step);
            __syncthreads();
            issue_load(stage, next_overflow_idx);
          }
          return true;
        };

        // Phase 1: consume the first `eff_stages` visits (the chunks reused from the previous pass, already resident in
        // the streaming slots), which issues the prefetch loads for this pass's reload wave into the freed slots.
        bool is_stopped      = false;
        const offset_t split = (::cuda::std::min) (static_cast<offset_t>(eff_stages), overflow_chunks);
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
          mid();

          // Phase 2: consume the remaining visits (their loads were issued in phase 1 and overlapped `mid`).
          for (offset_t i = split; i < overflow_chunks; ++i)
          {
            if (!consume(i))
            {
              break;
            }
          }
        }

        // Drain prefetches still in flight before returning: an early break leaves outstanding bulk copies whose
        // mbarriers were never waited, and they must complete before the block can exit (their slots are never read).
        // `inflight_mask` is block-uniform (set/cleared under uniform control flow), so the trip count and each
        // collective `wait_stage` are uniform across the block.
        while (inflight_mask != ::cuda::std::uint32_t{0})
        {
          const int drain_stage = __ffs(static_cast<int>(inflight_mask)) - 1;
          agent.wait_stage(drain_stage);
          inflight_mask &= ~(::cuda::std::uint32_t{1} << drain_stage);
        }
        is_forward = !is_forward;
      }
      else
      {
        // Generic fallback: no async SMEM pipeline, so resident work cannot hide load latency here. Fold the resident
        // chunks first (preserving the prior ordering), then read the overflow keys straight from gmem each pass (no
        // SMEM reuse), with the walk still snaking for L2 locality.
        mid();
        for (offset_t i = 0; i < overflow_chunks; ++i)
        {
          const offset_t overflow_idx = is_forward ? i : (overflow_chunks - 1 - i);
          const offset_t chunk_idx    = chunk_index_of(overflow_idx);
          const auto chunk            = agent.get_chunk(chunk_idx, segment_size, head_items);
          generic_apply(chunk);
          if (!should_continue())
          {
            break;
          }
        }
        is_forward = !is_forward;
      }
    }

    // Fold every overflow key once in the current ping-pong direction. `Indexed` selects the callable shape: keys-only
    // (`Indexed == false`) applies `f(key)`; the pair filter (`Indexed == true`) applies `f(key, seg_idx)` with each
    // key's segment-local index, needed to fetch its value payload from gmem while still reusing the streamed overflow
    // keys. The index math lives entirely inside the `if constexpr (Indexed)` arms, so it is elided in the keys-only
    // instantiation. See `run_pass` for the overlap semantics of `mid`; `UnrollFactor` partially unrolls the generic
    // (gmem fallback) fold loop.
    template <int UnrollFactor, bool Indexed, typename F, typename Mid>
    _CCCL_DEVICE _CCCL_FORCEINLINE void process_pass_dispatch(F&& f, Mid&& mid)
    {
      run_pass(
        [&](int stage, offset_t overflow_idx) {
          if constexpr (Indexed)
          {
            const offset_t base_off = agent.get_chunk(chunk_index_of(overflow_idx), segment_size, head_items).offset;
            agent.template for_each_chunk_key_indexed<UnrollFactor>(stage_span(stage, overflow_idx), base_off, f);
          }
          else
          {
            agent.template for_each_chunk_key<UnrollFactor>(stage_span(stage, overflow_idx), f);
          }
        },
        [&](const auto& chunk) {
          const int iterations = ::cuda::ceil_div(chunk.count, threads_per_block);
          _CCCL_PRAGMA_UNROLL(UnrollFactor)
          for (int j = 0; j < iterations; ++j)
          {
            const int local = j * threads_per_block + static_cast<int>(threadIdx.x);
            if (local < chunk.count)
            {
              if constexpr (Indexed)
              {
                const offset_t seg_idx = chunk.offset + static_cast<offset_t>(local);
                f(block_keys_in[static_cast<segment_size_val_t>(seg_idx)], seg_idx);
              }
              else
              {
                f(block_keys_in[static_cast<segment_size_val_t>(chunk.offset + static_cast<offset_t>(local))]);
              }
            }
          }
        },
        static_cast<Mid&&>(mid),
        [] {
          return true;
        });
    }

    // `f(key)` over every overflow key. `UnrollFactor` partially unrolls the generic (gmem fallback) fold loop; callers
    // pass their clamped items-per-thread.
    template <int UnrollFactor, typename F, typename Mid>
    _CCCL_DEVICE _CCCL_FORCEINLINE void process_pass(F&& f, Mid&& mid)
    {
      process_pass_dispatch<UnrollFactor, /*Indexed=*/false>(static_cast<F&&>(f), static_cast<Mid&&>(mid));
    }

    // Overload with no interleaved work, for the fused first pass where the resident keys are still being streamed in
    // by the BlockLoadToShared pipeline (rather than already resident in SMEM).
    template <int UnrollFactor, typename F>
    _CCCL_DEVICE _CCCL_FORCEINLINE void process_pass(F&& f)
    {
      process_pass<UnrollFactor>(static_cast<F&&>(f), [] {});
    }

    // Like `process_pass`, but applies `f(key, seg_idx)` where `seg_idx` is the key's segment-local index.
    template <int UnrollFactor, typename F, typename Mid>
    _CCCL_DEVICE _CCCL_FORCEINLINE void process_pass_indexed(F&& f, Mid&& mid)
    {
      process_pass_dispatch<UnrollFactor, /*Indexed=*/true>(static_cast<F&&>(f), static_cast<Mid&&>(mid));
    }
  };

  // -------------------------------------------------------------------------
  // Per-direction implementation
  // -------------------------------------------------------------------------
  // Cluster-wide barrier via PTX (replaces cooperative_groups' `cluster.sync()`): `.release` on arrive, `.acquire` on
  // wait, both `.aligned` since every thread reaches it under a uniform branch.
  _CCCL_DEVICE _CCCL_FORCEINLINE static void cluster_barrier()
  {
    asm volatile("barrier.cluster.arrive.release.aligned;" : : : "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" : : : "memory");
  }

  // Synchronize the segment's cluster. A single-CTA "cluster" keeps all state block-local, so `__syncthreads()` orders
  // it and the cluster-scoped barrier is unnecessary. `is_single_cta` is computed in `run()` from the collapsed cluster
  // blocks `process_impl` passes in (1 when a small segment collapsed onto rank 0), not the raw cluster size. It is
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

  // Combined 64-bit exclusive cross-CTA prefix scan over each working CTA's packed `(front_count << 32) | cand_count`.
  // Each CTA pushes its counts into every successor's `prefix_pair` in `is_scan_descending` order; the leader is last
  // and pushes to nobody, so it ends up holding the full predecessor sum. `prefix_pair` is pre-zeroed in `process_impl`
  // and untouched until here, so a single post-push barrier suffices. Idle ranks and the leader pass `packed == 0`.
  // Returns this CTA's exclusive prefix packed the same way; `is_single_cta` returns 0.
  //
  // The successor pushes are lane-parallel: the `red.add` reductions are commutative and target distinct remote ranks,
  // so each thread owns a strided slice of the successor range (one push per thread for the usual small cluster). All
  // threads see the same CTA-uniform `packed`, so the guard and the post-push barrier stay uniform.
  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::uint64_t combined_prefix_scan(
    bool is_single_cta, unsigned int cluster_rank, unsigned int eff_cluster_blocks, ::cuda::std::uint64_t packed)
  {
    if (is_single_cta)
    {
      return ::cuda::std::uint64_t{0};
    }
    if (packed != ::cuda::std::uint64_t{0})
    {
      if constexpr (is_scan_descending)
      {
        _CCCL_PRAGMA_NOUNROLL()
        for (unsigned int rank = threadIdx.x; rank < cluster_rank; rank += threads_per_block) // lower ranks follow;
                                                                                              // leader last
        {
          add_remote_prefix(rank, packed);
        }
      }
      else
      {
        // Higher ranks follow. Stops at `eff_cluster_blocks` since idle ranks own nothing; the leader at the last
        // effective rank is last.
        _CCCL_PRAGMA_NOUNROLL()
        for (unsigned int rank = cluster_rank + 1u + threadIdx.x; rank < eff_cluster_blocks; rank += threads_per_block)
        {
          add_remote_prefix(rank, packed);
        }
      }
    }
    // TODO(cccl): idle ranks arrive here only to keep this barrier reachable; a sub-cluster mbarrier over the working
    // ranks would let them exit (see the pass loop).
    cluster_or_block_sync(is_single_cta);
    return temp_storage.prefix_pair;
  }

  // Deterministic final-filter driver. The `run()`-local tie-break state (counts, prefixes, region extents, and the
  // mutable `running`/`is_tie_active` scan cursor) is hoisted here so the per-region index-ordered sweeps are named
  // member functions instead of a nest of `[&]` lambdas. Constructed once per `run()` in the deterministic branch. Kept
  // an aggregate (no user constructor) so its members are initialized positionally at the single call site.
  //
  // Codegen: methods are `_CCCL_FORCEINLINE` and no SMEM key pointer is stored -- the resident window is carried as its
  // 32-bit shared address (`resident_smem32`) and rebuilt with `__cvta_shared_to_generic` at use, so the reads stay
  // `LDS` (see the `resident_smem32` note in `run`).
  template <detail::topk::select SelectDirection>
  struct det_final_filter
  {
    using identify_candidates_op_t =
      detail::topk::identify_candidates_op_t<key_t, SelectDirection, bits_per_pass, decomposer_t>;

    agent_batched_topk_cluster& agent;
    num_segments_val_t segment_id;
    identify_candidates_op_t identify_op;
    it_value_t<KeyOutputItItT> block_keys_out;
    key_it_t block_keys_in;
    overflow_streamer& streamer;
    chunk_partition part;
    out_offset_t k;
    out_offset_t num_back;
    out_offset_t my_front;
    offset_t sel_prefix;
    offset_t cand_prefix;
    offset_t my_cand_count;
    offset_t resident_base;
    offset_t my_resident_chunks;
    offset_t segment_size_u32;
    offset_t head_items;
    offset_t front_seg_base;
    ::cuda::std::uint32_t resident_smem32;
    int front_count;
    int head_edge_len_items;
    int tail_edge_len_items;
    bool is_select_all_cand_cta;
    bool is_resident_terminal;
    bool is_head_edge_terminal;
    bool is_tail_edge_terminal;
    // Mutable per-invocation tie-break scan cursor.
    offset_t running;
    bool is_tie_active;

    // For each key written to `block_keys_out[pos]`, load the associated input value at the key's segment-local index
    // `seg_idx` from gmem and store it at the same slot. Compiled out (and never dereferences the null value iterators)
    // in keys-only builds; `segment_id` is loop-invariant, so the per-segment iterators hoist out of the writes.
    _CCCL_DEVICE _CCCL_FORCEINLINE void write_value(out_offset_t pos, offset_t seg_idx) const
    {
      if constexpr (!is_keys_only)
      {
        auto block_vals_in  = agent.d_value_segments_it[segment_id];
        auto block_vals_out = agent.d_value_segments_out_it[segment_id];
        block_vals_out[pos] = block_vals_in[static_cast<segment_size_val_t>(seg_idx)];
      }
    }

    // Uniform "all placed" predicate: true once this block has emitted all `my_front` strictly-selected keys and
    // resolved its ties. The leading barrier makes the counter reads block-wide (and resynchronizes lanes that raced
    // ahead through the barrier-free tiles). Polled only at critical points -- between regions and before each
    // streaming bulk copy -- never per tile.
    _CCCL_DEVICE _CCCL_FORCEINLINE bool should_stop()
    {
      __syncthreads();
      const bool is_front_done = agent.temp_storage.front_local_cnt >= static_cast<offset_t>(my_front);
      // Straddling/above CTAs finish the back when `is_tie_active` clears; an `is_select_all_cand_cta` (which never
      // clears it) finishes once all `my_cand_count` of its candidates are placed.
      const bool is_back_done = !is_tie_active || (agent.temp_storage.back_local_cnt >= my_cand_count);
      return is_front_done && is_back_done;
    }

    // Fold a flat scan position `pos` into its in-region index: forward, or (`Reversed`) counting down from the
    // region's last element. The `FromSmem` local read and the segment-local value index share this one folded index.
    template <bool Reversed>
    [[nodiscard]] static _CCCL_DEVICE _CCCL_FORCEINLINE int fold_pos(int pos, int count)
    {
      return Reversed ? (count - 1 - pos) : pos;
    }

    // Emit one back/tie candidate: if its `global_rank` (arrival- or scan-ordered by the caller) is a winner it lands
    // in the top-k output at slot `k-1-global_rank`, with its value pulled from segment-local index `seg_idx`; ranks at
    // or past `num_back` are losers and dropped (a no-op). Forced-inline with explicit args (no capture) so it folds
    // into the hot back-placement loops with no extra live ranges; the caller keeps any counter side effects (e.g.
    // `back_local_inc`) in the `global_rank` argument so they still run for every candidate.
    _CCCL_DEVICE _CCCL_FORCEINLINE void emit_back_one(offset_t global_rank, const key_t& key, offset_t seg_idx)
    {
      if (global_rank < static_cast<offset_t>(num_back))
      {
        const out_offset_t out = static_cast<out_offset_t>(k - 1) - static_cast<out_offset_t>(global_rank);
        block_keys_out[out]    = key;
        write_value(out, seg_idx);
      }
    }

    // Process a flat span of `count` keys in scan order, tiled by `threads_per_block * Items`. `FromSmem` selects the
    // key source: the SMEM buffer `smem_src` (indexed by the folded in-region position) or gmem `block_keys_in`
    // (indexed by the segment-local index `seg_base + folded`). `Reversed` walks the span high-to-low. Strictly-
    // selected keys go to the front via a SMEM atomic (offset by `sel_prefix`); candidates go to the back (see the
    // per-tile logic). `region_is_terminal` marks this region as the CTA's last, so its last tile -- if ties are
    // unresolved -- holds the boundary and scans directly. `running` carries across tiles/regions. No per-tile early-
    // exit or barrier here except the lazy-scan `else` branch; early exit is decided at critical points via
    // `should_stop`. The SMEM base is passed in (rebuilt via `__cvta_shared_to_generic` at the call site) rather than
    // stored, so the reads stay `LDS`.
    template <int Items, bool FromSmem, bool Reversed>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    process_tiles([[maybe_unused]] const key_t* smem_src, offset_t seg_base, int count, bool region_is_terminal)
    {
      constexpr int items = Items;
      constexpr int tile  = threads_per_block * items;
      for (int tile_base = 0; tile_base < count; tile_base += tile)
      {
        key_t keys[items];
        offset_t flags[items];
        detail::topk::candidate_class cls[items];
        bool is_valid[items];
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < items; ++i)
        {
          const int pos = tile_base + static_cast<int>(threadIdx.x) * items + i;
          is_valid[i]   = pos < count;
          flags[i]      = offset_t{0};
          if (is_valid[i])
          {
            const int folded_pos = fold_pos<Reversed>(pos, count);
            if constexpr (FromSmem)
            {
              keys[i] = smem_src[folded_pos];
            }
            else
            {
              keys[i] = block_keys_in[static_cast<segment_size_val_t>(seg_base + static_cast<offset_t>(folded_pos))];
            }
            cls[i]   = identify_op(keys[i]);
            flags[i] = (cls[i] == detail::topk::candidate_class::candidate) ? offset_t{1} : offset_t{0};
          }
        }

        // Strictly-selected keys go to this block's front slice via a block-local SMEM atomic offset by `sel_prefix`.
        // The per-block slices (disjoint by `sel_prefix`) together fill `[0, num_selected)`. Candidates never fold in
        // here -- they always route through the back below, even on early stop.
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < items; ++i)
        {
          const bool is_front_key = is_valid[i] && (cls[i] == detail::topk::candidate_class::selected);
          if (is_front_key)
          {
            const int pos          = tile_base + static_cast<int>(threadIdx.x) * items + i;
            const offset_t local   = agent.front_local_inc();
            const out_offset_t out = static_cast<out_offset_t>(sel_prefix + local);
            const offset_t seg_idx = seg_base + static_cast<offset_t>(fold_pos<Reversed>(pos, count));
            block_keys_out[out]    = keys[i];
            write_value(out, seg_idx);
          }
        }

        // Back/tie placement (only while this CTA still has unresolved ties). Three block-uniform sub-paths:
        //   * is_select_all_cand_cta -- every candidate wins: arrival-order SMEM atomics, no scan, no barrier.
        //   * terminal tile -- the straddling CTA's last tile necessarily holds the boundary: scan it directly.
        //   * other tiles   -- straddling CTA, boundary not yet known: place in arrival order, then `B1` to read the
        //                      counter and, on the crossing tile only, overwrite the arrival slots in index order.
        if (is_tie_active)
        {
          const bool is_terminal_tile = region_is_terminal && (tile_base + tile >= count);
          if (is_select_all_cand_cta)
          {
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int i = 0; i < items; ++i)
            {
              if (is_valid[i] && flags[i] != offset_t{0})
              {
                const int pos = tile_base + static_cast<int>(threadIdx.x) * items + i;
                emit_back_one(cand_prefix + agent.back_local_inc(),
                              keys[i],
                              seg_base + static_cast<offset_t>(fold_pos<Reversed>(pos, count)));
              }
            }
          }
          else if (is_terminal_tile)
          {
            offset_t excl[items];
            offset_t tile_total = 0;
            block_scan_t(agent.temp_storage.scan_storage).ExclusiveSum(flags, excl, tile_total);
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int i = 0; i < items; ++i)
            {
              if (is_valid[i] && flags[i] != offset_t{0})
              {
                const int pos = tile_base + static_cast<int>(threadIdx.x) * items + i;
                emit_back_one(
                  running + excl[i], keys[i], seg_base + static_cast<offset_t>(fold_pos<Reversed>(pos, count)));
              }
            }
            running += tile_total;
            is_tie_active = false;
          }
          else
          {
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int i = 0; i < items; ++i)
            {
              if (is_valid[i] && flags[i] != offset_t{0})
              {
                const int pos = tile_base + static_cast<int>(threadIdx.x) * items + i;
                emit_back_one(cand_prefix + agent.back_local_inc(),
                              keys[i],
                              seg_base + static_cast<offset_t>(fold_pos<Reversed>(pos, count)));
              }
            }
            // B1: order the arrival global writes ahead of the index-order overwrite (same boundary slots) and make
            // the counter read race-free and block-uniform.
            __syncthreads();
            const offset_t placed = agent.temp_storage.back_local_cnt;
            if ((cand_prefix + placed) > static_cast<offset_t>(num_back))
            {
              // Boundary tile: overwrite this tile's arrival slots `{k-1-running, ...}` with the index-ordered
              // winners (identical slot set, different candidate->slot mapping).
              offset_t excl[items];
              offset_t tile_total = 0;
              block_scan_t(agent.temp_storage.scan_storage).ExclusiveSum(flags, excl, tile_total);
              _CCCL_PRAGMA_UNROLL_FULL()
              for (int i = 0; i < items; ++i)
              {
                if (is_valid[i] && flags[i] != offset_t{0})
                {
                  const int pos = tile_base + static_cast<int>(threadIdx.x) * items + i;
                  emit_back_one(
                    running + excl[i], keys[i], seg_base + static_cast<offset_t>(fold_pos<Reversed>(pos, count)));
                }
              }
            }
            running = cand_prefix + placed;
            if (running >= static_cast<offset_t>(num_back))
            {
              is_tie_active = false;
            }
            // Trailing barrier for the lazy-scan path only: separate this tile's `placed` read (B1 above) from the
            // next tile's `back_local_inc`. The other sub-paths write disjoint slots and need no per-tile barrier.
            __syncthreads();
          }
        }
      }
    }

    // Resident-front region. Direction is the compile-time `is_residency_reversed` (== `is_tie_reversed` in
    // deterministic mode): ascending walks the low-index window forward, descending walks the high-index window
    // (`resident_base`) in reverse, so a single `process_tiles` call per span with the index folded at compile time
    // replaces the old fwd/rev pair.
    _CCCL_DEVICE _CCCL_FORCEINLINE void process_resident()
    {
      if constexpr (use_block_load_to_shared)
      {
        // Whole contiguous resident span staged in SMEM; rebuild the base from its 32-bit address at use.
        key_t* const rfront = reinterpret_cast<key_t*>(__cvta_shared_to_generic(resident_smem32));
        process_tiles<tie_break_items_per_thread_clamped, /*FromSmem=*/true, /*Reversed=*/is_residency_reversed>(
          rfront, front_seg_base, front_count, is_resident_terminal);
      }
      else
      {
        const int resident_chunk_count = static_cast<int>(my_resident_chunks);
        for (int slot = 0; slot < resident_chunk_count; ++slot)
        {
          const int local_slot     = is_residency_reversed ? (resident_chunk_count - 1 - slot) : slot;
          const offset_t chunk_idx = part.global_index(resident_base + static_cast<offset_t>(local_slot));
          const auto chunk         = agent.get_chunk(chunk_idx, segment_size_u32, head_items);
          // Generic multi-chunk resident loop reads the chunk's SMEM slot; never the terminal-tile fast path (the lazy
          // per-tile boundary detection handles any boundary), so pass `false`.
          process_tiles<tie_break_items_per_thread_clamped, /*FromSmem=*/true, /*Reversed=*/is_residency_reversed>(
            agent.slot_keys_unpadded(local_slot), chunk.offset, chunk.count, false);
        }
      }
    }

    // Overflow chunks. On the block-load path each landed slot folds through `process_tiles` (reusing the TMA
    // pipeline); the generic fallback reads gmem chunk by chunk. `run_pass`'s `should_continue` breaks the stream once
    // the top-k is fully placed. The streaming direction is already correct on entry (preselected in `run`), and the
    // slots are already primed -- the histogram's first streaming pass primed the same persistent streamer, so
    // `streamer.is_primed` carries in as `true` and this pass reuses the resident turn-around chunks with no re-prime
    // (the generic fallback re-reads gmem each pass regardless).
    _CCCL_DEVICE _CCCL_FORCEINLINE void process_overflow()
    {
      // Guards the straddling CTA (the only rank needing scan order): `run` preselected the direction so it enters at
      // `is_forward == !is_tie_reversed` after the full `num_passes`. The escape disjuncts are the cases where the
      // streaming direction is moot: no overflow (`overflow_chunks == 0`); a CTA entirely at/below the boundary
      // (`is_select_all_cand_cta`, arrival-order atomics); or one with no back scan (`!is_tie_active`). Early stop is
      // subsumed -- it makes every CTA `is_select_all_cand_cta`, so the shorter (possibly mis-parity) pass count never
      // reaches a straddler.
      _CCCL_ASSERT(streamer.overflow_chunks == 0 || is_select_all_cand_cta || !is_tie_active
                     || streamer.is_forward == (!is_tie_reversed),
                   "preselected ping-pong parity mismatch: the straddling CTA entered the deterministic filter with "
                   "the wrong streaming direction");

      streamer.run_pass(
        // Block-load: fold the chunk `overflow_idx`, resident in streaming slot `stage`, straight from SMEM.
        // `stage_span` rebuilds the slot pointer from its 32-bit shared address (spill-proof `LDS`) and returns only
        // the aligned bulk (a peeled tail suffix is handled by `process_tail_edge`).
        [&](int stage, offset_t overflow_idx) {
          const auto span = streamer.stage_span(stage, overflow_idx);
          const offset_t base_off =
            agent.get_chunk(streamer.chunk_index_of(overflow_idx), segment_size_u32, head_items).offset;
          // The multi-chunk overflow stream stays on the lazy per-tile boundary detection (`region_is_terminal ==
          // false`): a stray terminal direct scan here would need the streamer to flag its last chunk, and the saving
          // is one barrier on one tile.
          process_tiles<tie_break_items_streamed, /*FromSmem=*/true, /*Reversed=*/is_tie_reversed>(
            span.data(), base_off, static_cast<int>(span.size()), false);
        },
        // Generic fallback: read the overflow chunk straight from gmem (full count; the fallback never peels a tail).
        [&](const auto& chunk) {
          process_tiles<tie_break_items_streamed, /*FromSmem=*/false, /*Reversed=*/is_tie_reversed>(
            nullptr, chunk.offset, chunk.count, false);
        },
        // No interleaved resident work: the deterministic filter folds its resident span separately.
        [] {},
        // Checked before each refill bulk copy: break the stream once the whole top-k is placed. `should_stop`'s
        // barrier also resynchronizes the lanes that drifted through the just-folded chunk's barrier-free tiles.
        [&] {
          return !should_stop();
        });
    }

    // Fold one persistent boundary edge (head prefix or peeled tail suffix), both staged in `edge_keys`. A no-op on the
    // generic fallback (which reads boundary items straight from gmem) and for a zero-length edge.
    _CCCL_DEVICE _CCCL_FORCEINLINE void process_edge(const key_t* keys, offset_t seg_base, int count, bool is_terminal)
    {
      if constexpr (use_block_load_to_shared)
      {
        process_tiles<tie_break_items_per_thread_clamped, /*FromSmem=*/true, /*Reversed=*/is_tie_reversed>(
          keys, seg_base, count, is_terminal);
      }
    }

    // Head prefix edge (rank 0): the segment's lowest indices `[0, head_edge_len_items)`, staged at `edge_keys` (base
    // 0). In global-index order it is the leading region (ascending) / trailing region (descending); a non-head rank or
    // empty prefix is a no-op.
    _CCCL_DEVICE _CCCL_FORCEINLINE void process_head_edge()
    {
      process_edge(agent.temp_storage.edge_keys, offset_t{0}, head_edge_len_items, is_head_edge_terminal);
    }

    // Peeled tail suffix edge (tail owner): the segment's highest indices, staged at `edge_keys + head_edge_cap_items`
    // (base `segment_size - tail_edge_len_items`). In global-index order it is the trailing region (ascending) /
    // leading region (descending); an aligned or non-owned tail is a no-op.
    _CCCL_DEVICE _CCCL_FORCEINLINE void process_tail_edge()
    {
      process_edge(agent.temp_storage.edge_keys + head_edge_cap_items,
                   segment_size_u32 - static_cast<offset_t>(tail_edge_len_items),
                   tail_edge_len_items,
                   is_tail_edge_terminal);
    }

    // Drive the four regions in global-index order (ascending, or descending under `is_tie_reversed`), bailing between
    // regions once `should_stop` reports the whole top-k placed. The two orders share the resident->overflow middle and
    // only swap which boundary edge leads: ascending is head, resident, overflow, tail; descending reverses the edges.
    // Both visit resident before overflow, so `should_stop` can skip re-streaming the overflow once the top-k is
    // placed; `is_residency_reversed` keeps the first-visited (high-index) chunks resident in the descending order so
    // this holds.
    _CCCL_DEVICE _CCCL_FORCEINLINE void run_filter()
    {
      const auto step = [&](auto&& region) {
        if (!should_stop())
        {
          region();
        }
      };
      if constexpr (is_tie_reversed)
      {
        process_tail_edge();
      }
      else
      {
        process_head_edge();
      }
      step([&] {
        process_resident();
      });
      step([&] {
        process_overflow();
      });
      step([&] {
        if constexpr (is_tie_reversed)
        {
          process_head_edge();
        }
        else
        {
          process_tail_edge();
        }
      });
    }
  };

  template <detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  run(num_segments_val_t segment_id,
      unsigned int cluster_rank,
      unsigned int cluster_blocks,
      segment_size_val_t segment_size,
      out_offset_t k)
  {
    using extract_bin_op_t = detail::topk::extract_bin_op_t<key_t, SelectDirection, bits_per_pass, decomposer_t>;
    using identify_candidates_op_t =
      detail::topk::identify_candidates_op_t<key_t, SelectDirection, bits_per_pass, decomposer_t>;

    constexpr int total_bits = int{sizeof(key_t)} * 8;
    constexpr int num_passes = detail::topk::calc_num_passes<key_t>(bits_per_pass);

    auto block_keys_in          = d_key_segments_it[segment_id];
    const auto segment_size_u32 = static_cast<offset_t>(segment_size);
    // `cluster_blocks` is what `process_impl` runs at: the launched cluster size, or `1` when it collapsed a
    // small segment onto rank 0. A lone CTA routes barriers to `__syncthreads()`, keeps `state`/atomics block-local,
    // and uses CTA-scope histogram atomics (no cross-rank DSMEM folds to be mutually atomic with). For wider clusters,
    // `eff_cluster_blocks` (below) further excludes ranks that receive no chunks; they stay resident but idle.
    const bool is_single_cta = (cluster_blocks == 1u);

    // Per-block local copy of `kth_key_bits` so each key check hits the
    // block's own SMEM rather than DSMEM during the histogram loop. Built up one splitter digit per pass from the
    // leader's published `kth_bucket` (see the pass loop), so the full key is never broadcast.
    key_prefix_t kth_key_bits_local = {};

    // Tracks the highest pass count that actually executed. Without early
    // stop this stays at `num_passes`; with early stop it captures the pass
    // at which we broke out so the final filter can construct its identify
    // operator at the matching radix level.
    int last_pass = num_passes;

    const key_t* block_keys_base = nullptr;
    offset_t head_items          = 0;
    if constexpr (use_block_load_to_shared)
    {
      block_keys_base = THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(block_keys_in);
      head_items      = aligned_head_items(block_keys_base, segment_size_u32);
    }
    // The generic fallback does not use BlockLoadToShared's alignment hint or peeling path, so it can keep a simple
    // uniform chunking (`head_items == 0`). The two chunkings may assign keys to CTAs differently, but top-k only
    // depends on the multiset of keys covered by the cluster.
    const offset_t chunks = num_chunks(segment_size_u32, head_items);

    // Effective cluster blocks: the CTAs that actually receive chunks (at least `min_chunks_per_block` each), <= the
    // launched `cluster_blocks`. Ranks at or beyond it are idle -- they own no chunks, fold nothing, and never lead --
    // but stay resident and still arrive at every cluster barrier (a returned CTA would hang the barrier; see the
    // TODOs at the barrier sites). Derived from this CTA's head-aligned `chunks` so it matches the partition exactly.
    // Stays at `cluster_blocks` for host-exact sizes (the dispatch already matched it) and on the single-CTA path.
    unsigned int eff_cluster_blocks = cluster_blocks;
    if constexpr (enable_runtime_single_cta)
    {
      if (!is_single_cta)
      {
        eff_cluster_blocks = effective_cluster_blocks_from_chunks(
          static_cast<::cuda::std::uint64_t>(chunks), min_chunks_per_block, cluster_blocks);
      }
    }
    const bool is_idle_rank = cluster_rank >= eff_cluster_blocks;

    // Idle ranks own no chunks; `make_chunk_partition` assumes `rank < size`, so hand them an explicit empty partition.
    const chunk_partition part = is_idle_rank ? chunk_partition{offset_t{0}, offset_t{1}, offset_t{0}}
                                              : make_chunk_partition(chunks, cluster_rank, eff_cluster_blocks);
    const offset_t my_chunks   = part.count;

    // Leader rank. The leader owns the cluster-merged histogram and the shared `state`, and is always a working rank
    // (`< eff_cluster_blocks`). The deterministic tie-break makes the leader the *last* CTA in scan order so it never
    // needs its own (merged-away) local candidate count: prefer-smallest scans ascending by rank (leader = last
    // effective rank), prefer-largest scans descending (leader = rank 0). The nondeterministic path keeps rank 0.
    const unsigned int leader_rank = (need_determinism && !is_tie_reversed) ? (eff_cluster_blocks - 1u) : 0u;

    // DSMEM pointer into the leader block's shared memory. The Step 2 histogram fold reaches the leader's `hist`
    // through a `mapa`-formed `shared::cluster` address instead (see `hist_fold_remote`).
    state_t* leader_state = is_single_cta ? &temp_storage.state : map_state_to_rank(leader_rank);

    // Resident vs. streaming split, decided independently per CTA (CTAs need not agree -- cross-CTA traffic and every
    // cluster barrier is reached uniformly). A CTA whose chunks fit its resident slots (`my_chunks <= full_slots`)
    // keeps them all resident and streams nothing; an overflowing CTA reserves a round-robin streaming region at the
    // tail of its block_tile and re-streams its overflow chunks from gmem each pass via `streamer`.
    //
    // Boundary edges (the unaligned head prefix on rank 0 and the unaligned tail suffix on the tail owner) cannot use
    // the aligned TMA streamer, so both are always peeled into the persistent `edge_keys` buffer. The tail chunk's
    // aligned bulk is then a normal (possibly partial) aligned chunk that can be resident or streamed like any other;
    // no partial tail chunk is ever kept resident. Peeling both boundaries means streaming needs only `full_slots >=
    // 1`.
    //
    // `stream_slots` is right-sized: clamped into `[1, full_slots]` when streaming; deep overflows still get the full
    // `PipelineStages` depth. The generic fallback has no async pipeline (it re-reads overflow from gmem each pass and
    // peels nothing), so it reserves no streaming slots.
    const offset_t full_slots                   = block_tile_capacity / static_cast<offset_t>(chunk_items);
    [[maybe_unused]] const bool needs_streaming = my_chunks > full_slots;

    // Does this rank own the global tail, and does that tail carry an unaligned suffix? (block-load path only; the
    // generic fallback reads any trailing items straight from gmem and never peels.)
    [[maybe_unused]] offset_t tail_suffix_items = offset_t{0};
    bool owns_suffix_tail                       = false;
    if constexpr (use_block_load_to_shared)
    {
      // This rank owns the global tail iff its last owned chunk is chunk `chunks-1` (its local index `my_chunks-1`,
      // true for both the strided and blocked partitions).
      if (my_chunks > 0 && part.global_index(my_chunks - offset_t{1}) == chunks - offset_t{1})
      {
        const auto tail_chunk = get_chunk(chunks - offset_t{1}, segment_size_u32, head_items);
        tail_suffix_items     = split_chunk(block_keys_base, tail_chunk).suffix;
        owns_suffix_tail      = tail_suffix_items != offset_t{0};
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
        const offset_t excess      = my_chunks - full_slots;
        const offset_t want_stream = (::cuda::std::min) (static_cast<offset_t>(PipelineStages), excess);
        stream_slots               = (::cuda::std::max) (offset_t{1}, (::cuda::std::min) (want_stream, full_slots));
      }
    }
    const offset_t resident_slots_cap = full_slots - stream_slots;
    const offset_t my_resident_chunks = (::cuda::std::min) (my_chunks, resident_slots_cap);
    // Resident chunks stay within the first `resident_slots_cap` slots; the streaming region occupies the slots
    // `[resident_slots_cap, full_slots)`, so both regions live inside the allocated block_tile buffer.
    _CCCL_ASSERT(my_resident_chunks <= resident_slots_cap, "Dynamic shared memory block_tile is too small");

    const offset_t overflow_chunks = (my_chunks > my_resident_chunks) ? (my_chunks - my_resident_chunks) : offset_t{0};
    // Rank-local base of the resident and overflow (streamed) chunk windows. Default: resident `[0,
    // my_resident_chunks)`, overflow the high-index rest. `is_residency_reversed` swaps them: resident
    // `[overflow_chunks, my_chunks)`, overflow `[0, overflow_chunks)`.
    const offset_t resident_base = is_residency_reversed ? overflow_chunks : offset_t{0};
    const offset_t overflow_base = is_residency_reversed ? offset_t{0} : my_resident_chunks;

    // Persistent boundary-edge lengths: the head prefix lives on rank 0 (`head_items` is 0 on the generic fallback and
    // for an aligned base); the peeled tail suffix lives on the tail owner whenever it is unaligned.
    [[maybe_unused]] const int head_edge_len_items = (cluster_rank == 0u) ? static_cast<int>(head_items) : 0;
    [[maybe_unused]] const int tail_edge_len_items = should_peel_tail ? static_cast<int>(tail_suffix_items) : 0;

    // Persistent streamer for the overflow chunks; a no-op (constructs nothing) when this rank has no overflow.
    overflow_streamer streamer(
      *this,
      block_keys_in,
      block_keys_base,
      segment_size_u32,
      head_items,
      part,
      my_resident_chunks,
      overflow_base,
      static_cast<int>(resident_slots_cap),
      static_cast<int>(stream_slots),
      my_chunks);

    // Preselect the streamer's initial ping-pong direction. A streaming rank flips direction once per histogram pass,
    // so the leftover after the compile-time `num_passes` passes is `initial ^ (num_passes & 1)`; choosing `initial =
    // (!is_tie_reversed) ^ (num_passes & 1)` makes that leftover `== !is_tie_reversed` -- exactly what the
    // deterministic filter's straddling CTA needs to reuse its resident turn-around chunks with no re-prime (see
    // `det_final_filter::process_overflow`; early exit runs fewer passes but then has no straddler, so direction is
    // moot). Non-deterministic filtering is order-independent, so leave its historical `is_forward` start untouched.
    if constexpr (need_determinism)
    {
      streamer.is_forward = (!is_tie_reversed) ^ ((num_passes & 1) != 0);
    }

    ::cuda::std::span<key_t> resident_keys;
    // 32-bit shared-window address of `resident_keys.data()`. The resident span is read once per radix pass and in
    // the final filter; a 64-bit generic pointer kept live across that loop spills and reloads as a *generic*
    // pointer (`LDL.64`), which demotes every key read from `LDS` to a generic `LD`. Carrying the base as a 32-bit
    // shared address and rebuilding the pointer with `__cvta_shared_to_generic` at each use keeps the reads `LDS`
    // even when the value spills (the cvta intrinsic re-confers shared provenance; a spilled 64-bit generic pointer
    // cannot be re-anchored after the fact).
    ::cuda::std::uint32_t resident_smem32 = 0;

    // Fold the persistent boundary edges into `apply` (head prefix on rank 0; peeled tail suffix on the tail owner),
    // reading the keys already staged in `edge_keys`. Used by the histogram passes and the non-deterministic final
    // filter; the deterministic filter folds the edges as separate index-ordered regions instead (see below).
    const auto fold_edges = [&](auto&& apply) {
      if constexpr (use_block_load_to_shared)
      {
        const int tid = static_cast<int>(threadIdx.x);
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
    };

    // Front-load all stage mbarrier inits before any bulk copy issues; the barrier below orders them ahead of the first
    // issue (see `init_load_barriers`). The generic fallback has no mbarriers but keeps the same pass-start barrier.
    if constexpr (use_block_load_to_shared)
    {
      init_load_barriers();
    }
    __syncthreads();

    {
      extract_bin_op_t extract_op(0, total_bits, decomposer_t{});
      const ::cuda::std::uint32_t hist_smem32 = hist_base32();
      // Cluster-scope leader atomics are only needed to stay mutually atomic with the non-leaders' DSMEM folds; a lone
      // CTA has none, so it uses the cheaper CTA scope.
      const bool is_leader_cluster_scope = (cluster_rank == leader_rank) && !is_single_cta;
      auto add_first_pass                = [&](const key_t& key) {
        const int bucket = extract_op(key);
        hist_inc(hist_smem32, bucket, is_leader_cluster_scope);
      };

      if constexpr (use_block_load_to_shared)
      {
        if (my_resident_chunks > 0)
        {
          // Stage mbarriers and the `load_phase` parity are shared with the streamer (no per-chunk token array needed).
          // Chunks are written densely in slot order from offset 0 and read back in the same order, so the read cursor
          // (`read_off_bytes`) mirrors the write cursor (`next_off_bytes`) as a running prefix sum, avoiding a
          // dynamically-indexed `pending_spans` array that would anchor surrounding state to local memory. Every chunk
          // begins on a `load_align` boundary (zero prefix), so its whole `count` is the aligned bulk - except the
          // global-last chunk, whose unaligned suffix is always peeled into `edge_keys`, leaving only its aligned bulk
          // here. The read span is rebuilt from a rooted 32-bit shared address (see `bulk_span`) so a spilled cursor
          // cannot demote the reads to `LD`.
          const int prologue = (::cuda::std::min) (PipelineStages, static_cast<int>(my_resident_chunks));

          // Resident slot -> rank-local chunk index (`resident_base + slot`; identity unless `is_residency_reversed`
          // shifts the resident window to the high-index chunks).
          const auto resident_local = [&](offset_t slot) -> offset_t {
            return resident_base + slot;
          };
          // Aligned bulk of the resident chunk in `slot` (its `count` minus any peeled tail suffix); empty when it has
          // none.
          const auto bulk_src = [&](offset_t slot) -> ::cuda::std::span<const key_t> {
            const offset_t chunk_idx = part.global_index(resident_local(slot));
            const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
            const auto split         = split_chunk(block_keys_base, chunk);
            if (split.bulk == 0)
            {
              return {};
            }
            return {block_keys_base + chunk.offset + split.prefix, static_cast<::cuda::std::size_t>(split.bulk)};
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
          for (offset_t local_chunk = 0; local_chunk < my_resident_chunks; ++local_chunk)
          {
            const int stage = static_cast<int>(local_chunk % static_cast<offset_t>(prologue));
            wait_stage(stage);
            const int read_len_items = static_cast<int>(::cuda::std::size(bulk_src(local_chunk)));
            for_each_chunk_key<histogram_items_per_thread_clamped>(
              bulk_span({static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(key_slots + read_off_bytes)),
                         read_len_items}),
              add_first_pass);
            read_off_bytes += read_len_items * int{sizeof(key_t)};

            const offset_t next_local_chunk = local_chunk + static_cast<offset_t>(prologue);
            if (next_local_chunk < my_resident_chunks)
            {
              const auto src = bulk_src(next_local_chunk);
              // Phase safety, not data safety (the target offset is fresh): re-arming this stage before all threads
              // leave the wait above would advance the phase twice, stranding a lagging waiter forever.
              __syncthreads();
              issue_bulk_copy(stage, key_slots + next_off_bytes, src);
              next_off_bytes += static_cast<int>(::cuda::std::size(src)) * int{sizeof(key_t)};
            }
          }

          // The resident region is one contiguous span of aligned bulks for the later passes; both boundary edges are
          // folded separately from `edge_keys`.
          resident_keys   = {reinterpret_cast<key_t*>(key_slots),
                             static_cast<::cuda::std::size_t>(next_off_bytes / int{sizeof(key_t)})};
          resident_smem32 = static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(key_slots));
        }
      }
      else
      {
        for (offset_t local_chunk = 0; local_chunk < my_resident_chunks; ++local_chunk)
        {
          const offset_t chunk_idx = part.global_index(resident_base + local_chunk);
          const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
          key_t* const chunk_keys  = slot_keys_unpadded(static_cast<int>(local_chunk));
          const int iterations     = ::cuda::ceil_div(chunk.count, threads_per_block);
          _CCCL_PRAGMA_UNROLL(histogram_items_per_thread_clamped)
          for (int j = 0; j < iterations; ++j)
          {
            const int local = j * threads_per_block + static_cast<int>(threadIdx.x);
            if (local < chunk.count)
            {
              const key_t key =
                block_keys_in[static_cast<segment_size_val_t>(chunk.offset + static_cast<offset_t>(local))];
              chunk_keys[local] = key;
              add_first_pass(key);
            }
          }
        }
      }

      // Stage the persistent boundary edges into `edge_keys` and fold them into the first pass in the same sweep (see
      // `stage_and_fold_edge`: each thread folds keys it just wrote, so no barrier is needed here). The head prefix
      // (rank 0) precedes chunk 0; the peeled tail suffix (tail owner, always when unaligned) trails the last chunk.
      // The `__syncthreads()` below publishes `edge_keys` for the later passes and the final filter (which read it via
      // `fold_edges` after a barrier).
      if constexpr (use_block_load_to_shared)
      {
        if (head_edge_len_items > 0)
        {
          stage_and_fold_edge(temp_storage.edge_keys, block_keys_base, head_edge_len_items, add_first_pass);
        }
        if (tail_edge_len_items > 0)
        {
          const auto tail_chunk = get_chunk(chunks - offset_t{1}, segment_size_u32, head_items);
          stage_and_fold_edge(temp_storage.edge_keys + head_edge_cap_items,
                              block_keys_base + tail_chunk.offset + (tail_chunk.count - tail_edge_len_items),
                              tail_edge_len_items,
                              add_first_pass);
        }
      }

      // Fold the overflow chunks into the first-pass histogram, priming the streaming slots in the streamer's initial
      // direction (preselected above; the histogram is order-independent, so the direction only sets up the leftover
      // parity for the final filter). The streamer reuses the resident load's stage mbarriers (all front-loaded at
      // `run` entry); `wait_stage` provides the producer/consumer sync.
      streamer.process_pass<histogram_items_per_thread_clamped>(add_first_pass);

      const int resident_count = span_size(resident_keys);
      _CCCL_ASSERT(resident_count == 0 || static_cast<offset_t>(resident_count) <= block_tile_capacity,
                   "Dynamic shared memory block_tile is too small");
      __syncthreads();
    }

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

        // Step 1: block-private histogram via shared-space `red` (see `hist_inc`): leader uses cluster scope to be
        // mutually atomic with the non-leaders' Step 2 DSMEM folds, non-leaders use the cheaper cta scope.
        const ::cuda::std::uint32_t hist_smem32 = hist_base32();
        const bool is_leader_cluster_scope      = (cluster_rank == leader_rank) && !is_single_cta;
        auto add_hist                           = [&](const key_t& key) {
          if (identify_op(key) == detail::topk::candidate_class::candidate)
          {
            const int bucket = extract_op(key);
            hist_inc(hist_smem32, bucket, is_leader_cluster_scope);
          }
        };

        // Resident-chunk histogram, deferred into the streamer so it overlaps the streamer's in-flight first reload
        // wave (see `process_pass`). The histogram is order-independent, so folding resident keys between the
        // streamer's load issue and its wait does not change the result.
        const auto fold_resident_hist = [&] {
          if constexpr (use_block_load_to_shared)
          {
            // Rebuild the resident pointer from its 32-bit shared address so the reads stay `LDS` even if the value
            // spilled across the pass loop (see `resident_smem32`).
            key_t* const resident_ptr = reinterpret_cast<key_t*>(__cvta_shared_to_generic(resident_smem32));
            for_each_chunk_key<histogram_items_per_thread_clamped>(
              {resident_ptr, static_cast<::cuda::std::size_t>(span_size(resident_keys))}, add_hist);
          }
          else
          {
            for (offset_t local_chunk = 0; local_chunk < my_resident_chunks; ++local_chunk)
            {
              const offset_t chunk_idx = part.global_index(resident_base + local_chunk);
              const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
              key_t* const chunk_keys  = slot_keys_unpadded(static_cast<int>(local_chunk));
              for_each_chunk_key<histogram_items_per_thread_clamped>(
                {chunk_keys, static_cast<::cuda::std::size_t>(chunk.count)}, add_hist);
            }
          }
        };

        // Re-stream the overflow chunks into this pass's histogram, overlapping the resident-chunk histogram with the
        // first wave of reload bulk copies. Ping-pongs direction and reuses the turn-around chunks left resident by the
        // previous pass.
        streamer.process_pass<histogram_items_per_thread_clamped>(add_hist, fold_resident_hist);

        // Fold the persistent boundary edges (loaded once in the first pass) into this pass's histogram, alongside the
        // resident and overflow keys. Keeps every owner's per-bucket counts (the source of its `num_strictly_selected`
        // and `my_candidates`, the cross-CTA scan inputs) inclusive of its edge candidates.
        fold_edges(add_hist);
      }

      // Local barrier is enough: all Step 1 / Step 2 writes to `hist[]`
      // are atomic at compatible scopes (see Step 1 dispatch). The
      // cluster-wide ordering before Step 3's leader read of `hist[]`
      // is supplied by the cluster barrier further below.
      __syncthreads();

      // Step 2: non-leader blocks fold their per-bucket raw counts into
      // the leader's `hist` via cluster-scope DSMEM atomics (see
      // `hist_fold_remote`). The
      // leader skips this to avoid double-counting its own contribution;
      // idle ranks (`>= eff_cluster_blocks`) have an all-zero histogram, so
      // they skip the fold entirely (the loop would only read zeros).
      if (cluster_rank != leader_rank && !is_idle_rank)
      {
        const ::cuda::std::uint32_t hist_smem32 = hist_base32();
        for (int i = static_cast<int>(threadIdx.x); i < num_buckets; i += threads_per_block)
        {
          const offset_t bucket_count = temp_storage.hist[i];
          if (bucket_count != 0)
          {
            hist_fold_remote(
              hist_smem32 + static_cast<::cuda::std::uint32_t>(i) * sizeof(offset_t), bucket_count, leader_rank);
          }
        }
      }

      // TODO(cccl): idle ranks arrive here only because the cluster barrier spans the whole launched cluster. An
      // mbarrier over just the active ranks would let them exit and free their SM slots instead of spinning here.
      cluster_or_block_sync(is_single_cta);

      // Step 3: the leader prefix-scans the merged `hist` (raw counts) and
      // updates the cluster-shared `state`. Subsequent reads (end-of-pass
      // fold, last filter) all observe these writes after the next cluster sync.
      //
      // In parallel, each non-leader exclusive-scans its *own* (un-merged) histogram into registers (the leader was
      // otherwise the only block doing useful work here). Once the leader publishes `kth_bucket` below, the lane that
      // owns it reads its exclusive prefix (= this block's keys strictly above the splitter this pass -> accumulated
      // into `num_strictly_selected`) and its raw bucket count (-> `my_candidates`). Keeping the scan in registers lets
      // `hist` reset on the normal schedule (the regs survive the reset and the next cluster sync).
      offset_t local_prefixes[buckets_per_thread]{};
      offset_t local_hist_vals[buckets_per_thread]{};
      if (cluster_rank == leader_rank)
      {
        leader_identify_kth_bucket();
      }
      else if (!is_idle_rank)
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < buckets_per_thread; ++j)
        {
          const int bucket   = static_cast<int>(threadIdx.x) * buckets_per_thread + j;
          local_hist_vals[j] = (bucket < num_buckets) ? temp_storage.hist[bucket] : offset_t{0};
        }
        block_scan_t(temp_storage.scan_storage).ExclusiveSum(local_hist_vals, local_prefixes);
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
      const ::cuda::std::uint64_t pass_result = leader_state->result_pair;
      if (!is_idle_rank)
      {
        const int bucket = static_cast<int>(state_t::kth_bucket_of(pass_result));
        detail::topk::set_kth_key_bits<key_t, bits_per_pass>(kth_key_bits_local, pass, bucket);
        last_pass = pass + 1;

        // Non-leader: the lane owning the splitter bucket holds its exclusive prefix and raw count in registers from
        // the scan above. Accumulate the strictly-selected count and overwrite `my_candidates` with this pass's
        // splitter-bucket count (the last pass's value is what the filter reads, however the loop exits). The leader's
        // `hist` is merged, so it derives its own counts from the scan total instead (see the filter).
        if (cluster_rank != leader_rank)
        {
          const int owner = bucket / buckets_per_thread;
          if (static_cast<int>(threadIdx.x) == owner)
          {
            const int slot = bucket - owner * buckets_per_thread;
            add_local_selected(local_prefixes[slot]);
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

    // -----------------------------------------------------------------------
    // Final filter pass: write the top-k keys for this segment. Strictly-
    // selected keys go to the front; the `num_kth` tied candidates fill the
    // back. `kth_key_bits_local` already holds the full splitter key (folded
    // from each pass's bucket above), so no broadcast is needed here.
    // -----------------------------------------------------------------------
    auto block_keys_out        = d_key_segments_out_it[segment_id];
    const out_offset_t num_kth = leader_state->k; // remaining k after the radix passes

    // Store the value for a key written to `block_keys_out[pos]`, fetched from gmem at its segment-local index
    // `seg_idx`. Deriving the per-segment value iterators *inside* the `is_keys_only` guard avoids indexing the null
    // `cub::NullType**` value-iterators-of-iterators in keys-only builds; `segment_id` is loop-invariant, so they hoist
    // out of the writes. Serves only the non-deterministic branch below (the deterministic path uses
    // `det_final_filter::write_value`), so it is unused but still instantiated in deterministic builds.
    [[maybe_unused]] const auto write_value = [&](out_offset_t pos, offset_t seg_idx) {
      if constexpr (!is_keys_only)
      {
        auto block_vals_in  = d_value_segments_it[segment_id];
        auto block_vals_out = d_value_segments_out_it[segment_id];
        block_vals_out[pos] = block_vals_in[static_cast<segment_size_val_t>(seg_idx)];
      }
    };

    // `last_pass` controls how many radix levels of `kth_key_bits_local` are significant. After an early-stop break,
    // only the first `last_pass` digits of the splitter were folded; comparing all bits would treat the (still-zero)
    // trailing digits as smaller and erroneously reject candidates that share the identified prefix.
    identify_candidates_op_t identify_op(&kth_key_bits_local, last_pass, total_bits, decomposer_t{});

    if constexpr (need_determinism)
    {
      // Deterministic tie-break: strictly-selected keys fill the front via a SMEM atomic (offset by `sel_prefix`);
      // candidates fill the back -- arrival-order atomics away from the K-boundary, an index-ordered BlockScan only on
      // the boundary-crossing tile (see `is_select_all_cand_cta` and the per-tile back logic). Early stop is not
      // special-cased: `total_candidates == num_kth` then makes every CTA `is_select_all_cand_cta`.
      //
      // Cache `total_candidates` now, while every block is still tightly coupled to the pass loop's final cluster
      // barrier -- a post-scan re-read of `leader_state` could touch an already-returned leader (barrier gone).
      const offset_t total_candidates = leader_state->len;
      const out_offset_t num_back     = num_kth; // all candidates go to the back; the front holds only selected keys
      const out_offset_t num_selected = k - num_back; // front region

      // Publish the last pass's `num_strictly_selected`/`my_candidates` (written by the owning lane after the final
      // cluster barrier) block-wide before they feed the scan and `front_count`.
      __syncthreads();
      const bool participates = !is_idle_rank && (cluster_rank != leader_rank);
      const offset_t my_sel   = participates ? temp_storage.num_strictly_selected : offset_t{0};
      const offset_t my_cand  = participates ? temp_storage.my_candidates : offset_t{0};
      // Front count pushed by this block: its strictly-selected count. Candidates always route through the back, so
      // nothing folds into the front here. The leader and idle ranks push 0 -- the leader because its merged histogram
      // cannot self-count (it derives its own front from the total below), idle ranks because they own nothing.
      const offset_t push_front = my_sel;
      const ::cuda::std::uint64_t packed =
        (static_cast<::cuda::std::uint64_t>(push_front) << 32) | static_cast<::cuda::std::uint64_t>(my_cand);
      const ::cuda::std::uint64_t packed_prefix =
        combined_prefix_scan(is_single_cta, cluster_rank, eff_cluster_blocks, packed);
      const offset_t sel_prefix  = static_cast<offset_t>(packed_prefix >> 32);
      const offset_t cand_prefix = static_cast<offset_t>(packed_prefix & 0xffffffffu);
      // This block's own candidate count: non-leaders hold it in `my_cand`; the leader is last in scan order, so
      // `cand_prefix` already sums every other block's candidates and `total_candidates - cand_prefix` is its own. A
      // CTA is `is_select_all_cand_cta` when all of its candidates sit at or below the K-boundary
      // (`cand_prefix + my_cand_count <= num_back`): every one wins, so the back places them with arrival-order SMEM
      // atomics and skips the index-ordered scan. While `is_tie_active`, a non-`is_select_all_cand_cta` CTA is the
      // single boundary-crossing (straddling) CTA cluster-wide.
      const offset_t my_cand_count      = (cluster_rank == leader_rank) ? (total_candidates - cand_prefix) : my_cand;
      const bool is_select_all_cand_cta = (cand_prefix + my_cand_count) <= static_cast<offset_t>(num_back);
      // Mirror image: a CTA selects none of its candidates when the tie region is empty (`num_back == 0`) or all of its
      // candidates sort strictly after the K-boundary (`cand_prefix >= num_back`). Such a CTA seeds `is_tie_active`
      // false and skips the back placement entirely.
      const bool is_select_no_cand_cta =
        (num_back == out_offset_t{0}) || (cand_prefix >= static_cast<offset_t>(num_back));
      // This block's own front size: non-leaders know it directly (`push_front`); the leader is last in scan order, so
      // `sel_prefix` already sums every other block's front and `num_selected - sel_prefix` is the remainder it owns.
      const out_offset_t my_front =
        (cluster_rank == leader_rank)
          ? static_cast<out_offset_t>(num_selected - static_cast<out_offset_t>(sel_prefix))
          : static_cast<out_offset_t>(push_front);

      // Resident-front extent (bulk path): the whole contiguous resident span. The unaligned tail suffix (the
      // globally-last chunk's) is always peeled into `edge_keys` and folded by `process_tail_edge`, so it is never
      // part of this span.
      const int front_count = span_size(resident_keys);

      // Terminal-region flags for the last-tile direct-scan gate (block-load regions only; the generic resident loop
      // and overflow stream pass `false` and stay on lazy per-tile detection). A region is terminal when no later
      // region in the sweep (head/resident/overflow/tail-edge ascending, reversed descending) carries work.
      const bool has_head      = head_edge_len_items > 0;
      const bool has_resident  = front_count > 0;
      const bool has_overflow  = overflow_chunks > offset_t{0};
      const bool has_tail_edge = tail_edge_len_items > 0;
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
          ? get_chunk(part.global_index(resident_base), segment_size_u32, head_items).offset
          : offset_t{0};

      // Positional aggregate init in declaration order. The last two initializers seed the tie-break cursor: `running`
      // at this CTA's exclusive back prefix (candidates owned by preceding CTAs), `is_tie_active` at
      // `!is_select_no_cand_cta` -- true unless this is a select-no-candidates CTA (empty back region, or all of this
      // CTA's candidates sort past the boundary).
      det_final_filter<SelectDirection> filt{
        *this,
        segment_id,
        identify_op,
        block_keys_out,
        block_keys_in,
        streamer,
        part,
        k,
        num_back,
        my_front,
        sel_prefix,
        cand_prefix,
        my_cand_count,
        resident_base,
        my_resident_chunks,
        segment_size_u32,
        head_items,
        front_seg_base,
        resident_smem32,
        front_count,
        head_edge_len_items,
        tail_edge_len_items,
        is_select_all_cand_cta,
        is_resident_terminal,
        is_head_edge_terminal,
        is_tail_edge_terminal,
        cand_prefix,
        !is_select_no_cand_cta};
      filt.run_filter();
    }
    else
    {
      // Non-deterministic tie-break: strictly-selected keys fill the front `[0, num_selected)`, the first `num_kth`
      // candidates (arrival order) fill the back. The same combined scan as the deterministic path gives this block
      // disjoint front/back bases (`sel_prefix`/`cand_prefix`); placement then uses block-local SMEM atomics since
      // output order is not preserved. A perf change over the old cluster-wide `out_cnt`/`out_back_cnt` DSMEM atomics,
      // not a correctness one.
      __syncthreads();
      const bool participates = !is_idle_rank && (cluster_rank != leader_rank);
      const offset_t my_sel   = participates ? temp_storage.num_strictly_selected : offset_t{0};
      const offset_t my_cand  = participates ? temp_storage.my_candidates : offset_t{0};
      const ::cuda::std::uint64_t packed =
        (static_cast<::cuda::std::uint64_t>(my_sel) << 32) | static_cast<::cuda::std::uint64_t>(my_cand);
      const ::cuda::std::uint64_t packed_prefix =
        combined_prefix_scan(is_single_cta, cluster_rank, eff_cluster_blocks, packed);
      const offset_t sel_prefix  = static_cast<offset_t>(packed_prefix >> 32);
      const offset_t cand_prefix = static_cast<offset_t>(packed_prefix & 0xffffffffu);

      // Placement sink shared by both value modes: strictly-selected keys go to the front (`sel_prefix` + a block-local
      // atomic), the first `num_kth` candidates (arrival order) to the back; output order is not preserved. In pair
      // mode each written key additionally pulls its value from gmem at `seg_idx`. Keys-only elides that write via the
      // `if constexpr` below and drives the sink from index-free traversals, passing a dummy `seg_idx` that goes
      // unread.
      const auto sink = [&](const key_t& key, [[maybe_unused]] offset_t seg_idx) {
        const auto res = identify_op(key);
        if (res == detail::topk::candidate_class::selected)
        {
          const out_offset_t pos = static_cast<out_offset_t>(sel_prefix + front_local_inc());
          block_keys_out[pos]    = key;
          if constexpr (!is_keys_only)
          {
            write_value(pos, seg_idx);
          }
        }
        else if (res == detail::topk::candidate_class::candidate)
        {
          const out_offset_t back_pos = static_cast<out_offset_t>(cand_prefix + back_local_inc());
          if (back_pos < num_kth)
          {
            const out_offset_t pos = k - 1 - back_pos;
            block_keys_out[pos]    = key;
            if constexpr (!is_keys_only)
            {
              write_value(pos, seg_idx);
            }
          }
        }
      };

      if constexpr (is_keys_only)
      {
        // Keys-only: no value payload, so drive the sink through the index-free traversal with a dummy `seg_idx`.
        const auto write_selected = [&](const key_t& key) {
          sink(key, offset_t{0});
        };
        // Fold the resident keys as the streamer's `mid` work so they overlap the first overflow reloads. Writes are
        // order-independent atomics, and resident SMEM slots are disjoint from streaming slots, so `mid` never races.
        const auto fold_resident = [&] {
          if constexpr (use_block_load_to_shared)
          {
            key_t* const resident_ptr = reinterpret_cast<key_t*>(__cvta_shared_to_generic(resident_smem32));
            for_each_chunk_key<tie_break_items_per_thread_floor_clamped>(
              {resident_ptr, static_cast<::cuda::std::size_t>(span_size(resident_keys))}, write_selected);
          }
          else
          {
            for (offset_t local_chunk = 0; local_chunk < my_resident_chunks; ++local_chunk)
            {
              const offset_t chunk_idx = part.global_index(local_chunk);
              const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
              key_t* const chunk_keys  = slot_keys_unpadded(static_cast<int>(local_chunk));
              for_each_chunk_key<tie_break_items_per_thread_floor_clamped>(
                {chunk_keys, static_cast<::cuda::std::size_t>(chunk.count)}, write_selected);
            }
          }
          // Scan the persistent boundary edges alongside the resident keys (order-independent atomic writes).
          fold_edges(write_selected);
        };
        streamer.process_pass<tie_break_items_per_thread_floor_clamped>(write_selected, fold_resident);
      }
      else
      {
        // Pair (key + value) path: the shared `sink` above additionally stores each written key's value (fetched from
        // gmem at its `seg_idx`). Unlike keys-only, the traversal must recover each key's segment-local index, so it
        // folds resident/edge keys through `write_run` (below) and overflow keys through `process_pass_indexed`.

        // Iterate a contiguous run of `count` keys staged in SMEM at `smem`, whose element `local` has segment-local
        // index `base_off + local`. Every source folded here is SMEM (resident slots and the persistent boundary
        // edges); overflow chunks fold through the streamer's own indexed callback below. Materialize the key into a
        // register first: `sink` binds it by `const&` and reads it several times, so passing `smem[local]` directly
        // would re-issue a narrow `LDS` per use instead of reusing the loaded value.
        auto write_run = [&](const key_t* smem, offset_t base_off, int count) {
          const int iterations = ::cuda::ceil_div(count, threads_per_block);
          _CCCL_PRAGMA_UNROLL(tie_break_items_per_thread_clamped)
          for (int j = 0; j < iterations; ++j)
          {
            const int local = j * threads_per_block + static_cast<int>(threadIdx.x);
            if (local < count)
            {
              const key_t key = smem[local];
              sink(key, base_off + static_cast<offset_t>(local));
            }
          }
        };

        // Fold the resident keys (and their values) as the streamer's `mid` work, exactly as in the keys-only path.
        const auto fold_resident = [&] {
          if constexpr (use_block_load_to_shared)
          {
            // Resident keys are densely packed in slot order (aligned bulks only), so a running cursor recovers
            // per-chunk spans. Only the global-last chunk can be partial (its unaligned suffix is peeled into
            // `edge_keys` and folded below), so iterate `split.bulk`, not `chunk.count`.
            key_t* const resident_ptr = reinterpret_cast<key_t*>(__cvta_shared_to_generic(resident_smem32));
            int cursor_items          = 0;
            for (offset_t local_chunk = 0; local_chunk < my_resident_chunks; ++local_chunk)
            {
              const auto chunk =
                get_chunk(part.global_index(resident_base + local_chunk), segment_size_u32, head_items);
              const auto split           = split_chunk(block_keys_base, chunk);
              const offset_t base_off    = chunk.offset + split.prefix;
              const int bulk_count_items = split.bulk;
              write_run(resident_ptr + cursor_items, base_off, bulk_count_items);
              cursor_items += bulk_count_items;
            }
          }
          else
          {
            for (offset_t local_chunk = 0; local_chunk < my_resident_chunks; ++local_chunk)
            {
              const auto chunk        = get_chunk(part.global_index(local_chunk), segment_size_u32, head_items);
              const offset_t base_off = chunk.offset;
              write_run(slot_keys_unpadded(static_cast<int>(local_chunk)), base_off, chunk.count);
            }
          }
          // Scan the persistent boundary edges with their segment-local indices: the head prefix starts at index 0, the
          // peeled tail suffix at `segment_size - tail_edge_len_items`. Value fetched per key.
          if constexpr (use_block_load_to_shared)
          {
            if (head_edge_len_items > 0)
            {
              write_run(temp_storage.edge_keys, offset_t{0}, head_edge_len_items);
            }
            if (tail_edge_len_items > 0)
            {
              write_run(temp_storage.edge_keys + head_edge_cap_items,
                        segment_size_u32 - static_cast<offset_t>(tail_edge_len_items),
                        tail_edge_len_items);
            }
          }
        };

        // Overflow chunks: reuse the streamed keys (the generic fallback re-reads from gmem) and fetch each selected
        // key's value at index `seg_idx`. The resident keys above fold in as the streamer's `mid` work.
        streamer.process_pass_indexed<tie_break_items_per_thread_floor_clamped>(sink, fold_resident);
      }
    }

    // No cluster barrier after the final filter pass: both filter paths place output via block-local SMEM atomics into
    // gmem, so the last cross-CTA DSMEM access is the combined scan's `prefix_pair` push, already fenced by its
    // post-push cluster barrier (and `early_stop` is cached pre-scan). With no shared-memory access to another block
    // after the scan, a block can return without risking a "cluster target block not present" fault from a straggler.
  }

  // Copies an entire segment `input[i] -> output[i]` for the select-all fast path (`k >= segment_size`).
  _CCCL_DEVICE _CCCL_FORCEINLINE void copy_segment_select_all(
    num_segments_val_t segment_id,
    segment_size_val_t segment_size,
    unsigned int cluster_rank,
    unsigned int cluster_blocks)
  {
    constexpr int copy_items       = copy_items_per_thread_clamped;
    const offset_t num_items       = static_cast<offset_t>(segment_size);
    const offset_t cluster_tid     = cluster_rank * static_cast<offset_t>(threads_per_block) + threadIdx.x;
    const offset_t cluster_threads = cluster_blocks * static_cast<offset_t>(threads_per_block);
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

    // Sub-tile remainder
    _CCCL_PRAGMA_NOUNROLL()
    for (offset_t idx = full_tiles + cluster_tid; idx < num_items; idx += cluster_threads)
    {
      const auto seg_idx   = static_cast<segment_size_val_t>(idx);
      keys_out_it[seg_idx] = keys_in_it[seg_idx];
      if constexpr (!is_keys_only)
      {
        vals_out_it[seg_idx] = vals_in_it[seg_idx];
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void process_impl()
  {
    // Cluster rank/size from the PTX special registers (replaces cooperative_groups' `this_cluster()`).
    const unsigned int cluster_rank = ::cuda::ptx::get_sreg_cluster_ctarank();
    // Runtime cluster blocks match the launch attribute the dispatch passed
    // to `cudaLaunchKernelExC` (or the kernel's `__cluster_dims__` on CDP).
    const unsigned int cluster_blocks = ::cuda::ptx::get_sreg_cluster_nctarank();
    const auto segment_id             = static_cast<num_segments_val_t>(blockIdx.x / cluster_blocks);

    if (segment_id >= detail::params::get_param(num_segments, num_segments_val_t{0}))
    {
      return;
    }

    const auto segment_size = static_cast<segment_size_val_t>(detail::params::get_param(segment_sizes, segment_id));
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

    // Segments larger than the resident cluster_tile capacity are handled by re-streaming the overflow chunks from
    // gmem (see `overflow_streamer`), so the only hard limit left is the 32-bit offset range used internally.
    bool segment_fits_offset = true;
    if constexpr (sizeof(segment_size_val_t) > sizeof(offset_t))
    {
      segment_fits_offset =
        segment_size <= static_cast<segment_size_val_t>(::cuda::std::numeric_limits<offset_t>::max());
    }
    if (!segment_fits_offset)
    {
      _CCCL_ASSERT(false, "Segment exceeds the 32-bit offset range supported by cluster top-k");
      return;
    }

    // `k_clamped <= segment_size`, which now fits `out_offset_t`, so this narrowing is safe.
    const auto k = static_cast<out_offset_t>(k_clamped);

    // Select-all fast path: when `k` reaches the full segment, every element wins, so we skip the radix passes,
    // histogram, and output-ordering and just copy. Runs on the full launched cluster (before the effective-cluster
    // collapse); the decision is per-segment uniform, so the branch is cluster-uniform.
    if (static_cast<segment_size_val_t>(k) == segment_size)
    {
      copy_segment_select_all(segment_id, segment_size, cluster_rank, cluster_blocks);
      return;
    }

    // Effective cluster blocks/rank. For a per-segment (deferred) size argument the launch is sized for the maximum
    // segment, so a small segment that fits resident in one CTA and is at/below the single-CTA tuning threshold is
    // served by rank 0 alone via the barrier-free path; the cluster's other CTAs exit immediately, freeing their SM
    // slots. The decision is per-segment uniform across the block, so a redundant CTA returns whole. Compiled out for
    // host-exact sizes, which the dispatch already sized to exact cluster blocks.
    unsigned int eff_cluster_blocks = cluster_blocks;
    unsigned int eff_cluster_rank   = cluster_rank;
    if constexpr (enable_runtime_single_cta)
    {
      const bool fits_single_cta = is_single_cta_eligible(
        static_cast<::cuda::std::uint64_t>(segment_size),
        static_cast<::cuda::std::uint64_t>(block_tile_capacity),
        single_block_max_seg_size);
      if (fits_single_cta)
      {
        if (cluster_rank != 0u)
        {
          return;
        }
        eff_cluster_blocks = 1u;
        eff_cluster_rank   = 0u;
      }
    }
    const bool is_single_cta = (eff_cluster_blocks == 1u);

    // Every block's thread 0 initializes its local `state`. Only the
    // leader's copy is semantically read (non-leaders reach the cluster
    // state through `leader_state`), but mirroring the writes everywhere
    // keeps every block's unconditional `state.k` load safe under
    // compute-sanitizer.
    if (threadIdx.x == 0)
    {
      temp_storage.state.len         = static_cast<offset_t>(segment_size);
      temp_storage.state.k           = k;
      temp_storage.state.result_pair = 0;
      // Front-load every counter the final filter relies on so the first cluster barrier below publishes the zeros to
      // all ranks (the combined scan's DSMEM pushes then add into an already-zeroed `prefix_pair`, needing only a
      // post-push barrier). `my_candidates` is zeroed too so idle/leader ranks (which never write it) read 0.
      temp_storage.prefix_pair           = 0;
      temp_storage.front_local_cnt       = 0;
      temp_storage.back_local_cnt        = 0;
      temp_storage.num_strictly_selected = 0;
      temp_storage.my_candidates         = 0;
    }
    reset_hist();
    cluster_or_block_sync(is_single_cta);

    [[maybe_unused]] const bool is_ok = detail::params::dispatch_discrete(
      select_directions,
      segment_id,
      [this, segment_id, eff_cluster_rank, eff_cluster_blocks, segment_size, k](auto direction_tag) {
        constexpr detail::topk::select Direction = decltype(direction_tag)::value;
        this->template run<Direction>(segment_id, eff_cluster_rank, eff_cluster_blocks, segment_size, k);
      });
    _CCCL_ASSERT(is_ok, "Unsupported select direction for cluster top-k");
  }
};
} // namespace detail::batched_topk_cluster

CUB_NAMESPACE_END
