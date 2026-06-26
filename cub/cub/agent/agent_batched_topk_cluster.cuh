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
//!      shared-space `red` reductions (cheap, SMEM-local): cta scope for
//!      non-leaders, cluster scope for the leader (see `hist_inc`).
//!   2. After a cluster-wide barrier, every non-leader block walks its
//!      histogram and folds its bucket counts into the leader block's `hist`
//!      via cluster-scope DSMEM atomics. The leader's `hist` therefore plays
//!      a dual role: its own block-private histogram in step 1, then the
//!      cluster-merged histogram after the second cluster sync.
//!   3. The leader's thread 0 prefix-scans `hist`, identifies the bucket of
//!      the k-th key, and updates the cluster-shared `state`. Every block
//!      reads `state.kth_bucket` from the leader via DSMEM at the end of each
//!      pass and folds it into its own local splitter key.
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
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/round_down.h>
#include <cuda/__cmath/round_up.h>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__ptx/instructions/cp_async_bulk.h>
#include <cuda/__ptx/instructions/mbarrier_arrive.h>
#include <cuda/__ptx/instructions/mbarrier_init.h>
#include <cuda/__ptx/instructions/mbarrier_wait.h>
#include <cuda/argument>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
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
  // Splitter bucket identified by the leader for the current pass. Replaces broadcasting the full multi-word
  // `kth_key_bits`: every block reads this single digit through DSMEM at the end of each pass and folds it into its own
  // `kth_key_bits_local` via `set_kth_key_bits`, so the full splitter key is reconstructed locally and never broadcast.
  ::cuda::std::uint32_t kth_bucket;
  OutOffsetT out_cnt;
  OutOffsetT out_back_cnt;
  // Set by the leader after `leader_identify_kth_bucket` whenever the
  // identified bucket holds exactly `k` items (every candidate is part of
  // the top-k). Read by every block of the cluster at the end of each radix
  // pass through DSMEM. Carried in the cluster-shared state so the value
  // survives the cluster sync that ends the current pass.
  ::cuda::std::uint32_t early_stop;
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
[[nodiscard]] _CCCL_HOST_DEVICE constexpr bool single_cta_eligible(
  ::cuda::std::uint64_t segment_size,
  ::cuda::std::uint64_t block_tile_capacity,
  int single_cta_max_segment_size) noexcept
{
  return segment_size <= block_tile_capacity
      && segment_size <= static_cast<::cuda::std::uint64_t>(single_cta_max_segment_size);
}

// Effective cluster blocks implied by a chunk count: a CTA joins the effective cluster iff it would own at least
// `min_chunks_per_cta` chunks, clamped to `[1, cluster_blocks_cap]`. Identical arithmetic on both sides; only the cap
// differs (live `cluster.num_blocks()` on the device, max launchable blocks on the host). `min_chunks_per_cta` is
// `static_assert`ed positive, so the divide is well-defined. 64-bit math.
[[nodiscard]] _CCCL_HOST_DEVICE constexpr unsigned int effective_cluster_blocks_from_chunks(
  ::cuda::std::uint64_t chunks, int min_chunks_per_cta, unsigned int cluster_blocks_cap) noexcept
{
  const auto blocks = chunks / static_cast<::cuda::std::uint64_t>(min_chunks_per_cta);
  return static_cast<unsigned int>(
    (::cuda::std::clamp) (blocks, ::cuda::std::uint64_t{1}, static_cast<::cuda::std::uint64_t>(cluster_blocks_cap)));
}

// -----------------------------------------------------------------------------
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
          int SingleCtaMaxSegmentSize,
          int MinChunksPerCta,
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

  // See the class comment: deterministic guarantees select the index-ordered scan; `tie_reversed` flips its order.
  static constexpr bool deterministic =
    Determinism != ::cuda::execution::determinism::__determinism_t::__not_guaranteed;
  static constexpr bool tie_reversed = TieBreak == ::cuda::execution::tie_break::__tie_break_t::__prefer_larger_index;

  // Segments at or below this size that also fit resident in one CTA take the single-CTA fast path (see
  // `process_impl`).
  static constexpr int single_cta_max_segment_size = SingleCtaMaxSegmentSize;

  // A CTA joins a segment's effective cluster only if it would own at least this many chunks (see `run`). At 1 the
  // effective cluster is just the CTAs that receive any chunk. Must be positive: it is the divisor in
  // `effective_cluster_blocks_from_chunks` and a zero-chunk CTA has no work.
  static constexpr int min_chunks_per_cta = MinChunksPerCta;
  static_assert(min_chunks_per_cta >= 1, "min_chunks_per_cta must be positive");

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
  static constexpr int num_buckets                = 1 << bits_per_pass;
  using smem_layout_t                             = smem_block_tile_layout<key_t, ChunkBytes, LoadAlignBytes>;
  static constexpr int chunk_items                = smem_layout_t::chunk_items;
  static constexpr int load_align_items           = smem_layout_t::load_align_items;
  static constexpr int slot_alignment             = smem_layout_t::slot_alignment;

  static_assert(PipelineStages > 0);
  static_assert(HistogramItemsPerThread > 0, "histogram_items_per_thread must be positive");
  static_assert(ChunkBytes > 0);
  static_assert(LoadAlignBytes > 0);
  static_assert(ChunkBytes % LoadAlignBytes == 0, "ChunkBytes must be a multiple of LoadAlignBytes");
  // The hybrid load relies on the aligned bulk-copy path being exact (no scalar guard), which requires the load
  // alignment to be at least the bulk-copy minimum alignment.
  static_assert(LoadAlignBytes >= detail::bulk_copy_min_align, "LoadAlignBytes must be >= bulk_copy_min_align");
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
  // `cand_prefix` is each block's own exclusive cross-CTA candidate count for the deterministic tie-break; other blocks
  // add into it through DSMEM (`add_remote_prefix`), so it must sit at an identical offset in every block's storage.
  struct _TempStorage
  {
    offset_t hist[num_buckets];
    state_t state;
    offset_t cand_prefix;
    typename block_scan_t::TempStorage scan_storage;
    // One mbarrier handle per pipeline stage, shared by the resident load and the overflow streamer and reused
    // (ping-ponged) across radix passes; initialized once by `init_load_barriers`.
    ::cuda::std::uint64_t load_mbar[PipelineStages];
    // Persistent unaligned boundary edges (block-load path only): the head prefix (`[0, head_edge_cap)`, on rank 0) and
    // the peeled tail suffix (`[head_edge_cap, 2 * head_edge_cap)`, on the tail owner when `full_slots == 1`), each
    // strictly `< load_align_items` keys. Loaded once in the first pass and folded into every pass + the final filter.
    // Block-local (never reached through DSMEM).
    key_t edge_keys[2 * load_align_items];
  };
  // Split point of `edge_keys`: head edge in `[0, head_edge_cap)`, tail edge in `[head_edge_cap, 2 * head_edge_cap)`.
  static constexpr int head_edge_cap = load_align_items;

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::span<char> block_tile_buffer() const
  {
    const int slots = static_cast<int>(block_tile_capacity / chunk_items);
    return {key_slots, static_cast<::cuda::std::size_t>(slots * ChunkBytes)};
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE key_t* slot_keys_unpadded(int slot) const
  {
    return reinterpret_cast<key_t*>(key_slots + slot * ChunkBytes);
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
    const auto la            = static_cast<::cuda::std::uintptr_t>(load_align_bytes);
    const auto begin         = reinterpret_cast<::cuda::std::uintptr_t>(base + chunk.offset);
    const auto end           = begin + static_cast<::cuda::std::uintptr_t>(chunk.count) * sizeof(key_t);
    const auto aligned_begin = ::cuda::round_up(begin, la);
    const auto aligned_end   = ::cuda::round_down(end, la);
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

  template <typename PtrT>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE bool is_aligned_chunk(PtrT base, const chunk_desc chunk) const
  {
    const auto s = split_chunk(base, chunk);
    return s.prefix == 0 && s.suffix == 0;
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
  // ascending contiguous global-index ranges), so it is selected exactly when `deterministic` is set.
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE chunk_partition
  make_chunk_partition(offset_t chunks, unsigned int cluster_rank, unsigned int cluster_blocks) const
  {
    if constexpr (deterministic)
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

  // Stage a small (`< load_align_items` items) unaligned run -- a boundary edge (head prefix / tail suffix) or a
  // resident chunk's suffix -- from gmem `src` into SMEM `dst` and fold it into the first pass in one strided sweep:
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
  // `[0, chunk_count)`), processing tiles of `histogram_items_per_thread * threads_per_block` keys. Each tile is loaded
  // into registers by one unrolled loop, then handed to `apply` by a second. Splitting the loads out matters for the
  // histogram passes: `apply`'s SMEM atomics can't be proven disjoint from the SMEM key reads, so a fused loop would
  // interleave each load with its atomic instead of hoisting the whole load wave ahead.
  template <typename Apply>
  _CCCL_DEVICE _CCCL_FORCEINLINE void for_each_chunk_key_impl(const key_t* data, int chunk_count, Apply&& apply) const
  {
    constexpr int tile   = histogram_items_per_thread * threads_per_block;
    const int tid        = static_cast<int>(threadIdx.x);
    const int full_tiles = ::cuda::round_down(chunk_count, tile);

    _CCCL_PRAGMA_NOUNROLL()
    for (int tile_base = 0; tile_base < full_tiles; tile_base += tile)
    {
      key_t regs[histogram_items_per_thread];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int t = 0; t < histogram_items_per_thread; ++t)
      {
        regs[t] = data[tile_base + t * threads_per_block + tid];
      }
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int t = 0; t < histogram_items_per_thread; ++t)
      {
        const int local = tile_base + t * threads_per_block + tid;
        apply(regs[t], local);
      }
    }

    if (full_tiles < chunk_count)
    {
      key_t regs[histogram_items_per_thread];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int t = 0; t < histogram_items_per_thread; ++t)
      {
        const int local = full_tiles + t * threads_per_block + tid;
        if (local < chunk_count)
        {
          regs[t] = data[local];
        }
      }
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int t = 0; t < histogram_items_per_thread; ++t)
      {
        const int local = full_tiles + t * threads_per_block + tid;
        if (local < chunk_count)
        {
          apply(regs[t], local);
        }
      }
    }
  }

  template <typename F>
  _CCCL_DEVICE _CCCL_FORCEINLINE void for_each_chunk_key(::cuda::std::span<key_t> chunk_keys, F&& f) const
  {
    for_each_chunk_key_impl(::cuda::std::data(chunk_keys), span_size(chunk_keys), [&](const key_t& key, int) {
      f(key);
    });
  }

  // Like `for_each_chunk_key`, but also hands `f` each key's segment-local index `base_off + local`, where `base_off`
  // is the segment-local offset of the chunk's first element. The pair path uses that index to fetch the key's value
  // payload from gmem, so overflow keys can be reused from the streaming SMEM pipeline instead of re-read from gmem.
  template <typename F>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  for_each_chunk_key_indexed(::cuda::std::span<key_t> chunk_keys, offset_t base_off, F&& f) const
  {
    for_each_chunk_key_impl(::cuda::std::data(chunk_keys), span_size(chunk_keys), [&](const key_t& key, int local) {
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
  // transaction arrival) without its reference member or per-call CommitToken: the only per-stage wait state is one
  // bit of the per-thread `load_phase` mask, so the pipeline loops spill nothing per stage. SM 9.0+ only (the agent is
  // gated behind `NV_PROVIDES_SM_90` in `Process`).

  // Init each stage mbarrier with arrival count 1: only the elected thread arrives (registering the tx byte count) and
  // the `cp.async.bulk` delivers the matching count, so the phase completes. Call once, followed by a block sync.
  _CCCL_DEVICE _CCCL_FORCEINLINE void init_load_barriers()
  {
    if (threadIdx.x == 0)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int s = 0; s < PipelineStages; ++s)
      {
        ::cuda::ptx::mbarrier_init(&temp_storage.load_mbar[s], 1u);
      }
    }
  }

  // Issue one aligned global->shared (TMA) bulk copy into `dst` on stage `stage`'s mbarrier from the elected thread,
  // which also arrives with the transaction byte count (an empty copy arrives with zero so the phase still completes).
  // Each call must be paired, in issue order per stage, with a matching `wait_stage(stage)`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void issue_bulk_copy(int stage, char* dst, ::cuda::std::span<const key_t> src)
  {
    if (threadIdx.x != 0)
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
  _CCCL_DEVICE _CCCL_FORCEINLINE void hist_inc(::cuda::std::uint32_t base32, int bucket, bool leader)
  {
    const ::cuda::std::uint32_t addr = base32 + static_cast<::cuda::std::uint32_t>(bucket) * sizeof(offset_t);
    if (leader)
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

  // Adds `v` to the `cand_prefix` of the CTA at cluster rank `target_rank` through DSMEM (mirrors `hist_fold_remote`:
  // `mapa` this block's own `cand_prefix` address to `target_rank`, then a cluster-scope `red.add`). Used by the
  // deterministic tie-break's exclusive cross-CTA candidate-count scan.
  _CCCL_DEVICE _CCCL_FORCEINLINE void add_remote_prefix(unsigned int target_rank, offset_t v)
  {
    const ::cuda::std::uint32_t own =
      static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(&temp_storage.cand_prefix));
    ::cuda::std::uint32_t remote;
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(own), "r"(target_rank));
    asm volatile("red.relaxed.cluster.shared::cluster.add.u32 [%0], %1;" : : "r"(remote), "r"(v) : "memory");
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
        // Publish only the splitter bucket; every block folds it into its own `kth_key_bits_local` at the end of the
        // pass (see the pass loop), so the full splitter key is never broadcast.
        temp_storage.state.kth_bucket = static_cast<::cuda::std::uint32_t>(bucket);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Overflow streamer
  // ---------------------------------------------------------------------------
  // Re-streams the per-rank "overflow" chunks (those that do not fit in the
  // resident SMEM region) from gmem through a small, fixed, round-robin set of
  // `p_eff` (<= `PipelineStages`) streaming slots. The same object is reused for
  // every radix pass and the final filter. It ping-pongs the iteration order
  // across calls so the `p_eff` turn-around chunks that one pass leaves resident
  // in the streaming slots are reused by the next pass with no reload; the
  // remaining `overflow_chunks - p_eff` chunks are reloaded from gmem on each pass.
  // The caller right-sizes the reservation to `p_eff = min(PipelineStages, excess)`
  // (where `excess = my_chunks - full_slots`), so a streaming rank always has
  // `overflow_chunks = excess + p_eff` and reloads exactly `excess` chunks per
  // pass - the reserved slots only ever buy reuse of those `p_eff` turn-around
  // chunks, never a reload-free pass. The resident region is unaffected: it lives in the
  // slots `[0, resident_slots)`, the streaming region in `[stream_slot_base,
  // stream_slot_base + p_eff)`.
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
    int p_eff; // streaming region size = reserved streaming slots (<= PipelineStages, <= M, >= 1)
    bool forward = true;
    bool primed  = false;

    // Stage mbarriers are shared with the resident load (`agent.temp_storage.load_mbar`); stage `stage` targets slot
    // `stream_slot_base + stage`. `inflight_mask` bit `stage` is set only while a copy is in flight (issued, not yet
    // waited). The slot/stage mapping is fixed, so the read span is recomputed on demand by `stage_span` rather than
    // held in a spillable per-stage array; the only per-stage state is one bit each of `inflight_mask` and
    // `load_phase`.
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
      // `p_eff` is the streaming region size the caller reserved at `[stream_slot_base, stream_slot_base +
      // stream_slots)`; using all of it as pipeline stages keeps the ping-pong reuse maximal. It is `<= M` whenever
      // there is overflow (`M = excess + stream_slots >= stream_slots`); the `>= 1` floor only matters for the no-op
      // (`M == 0`) case, where the streamer never touches a slot.
      p_eff = (stream_slots_ > 0) ? stream_slots_ : 1;
      _CCCL_ASSERT(overflow_chunks == 0 || p_eff <= static_cast<int>(overflow_chunks),
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
      // Only the global-last chunk can carry an unaligned suffix, and it only reaches the streamer when that tail was
      // peeled (its suffix lives in `edge_keys`); streaming just the aligned bulk excludes it. For every interior chunk
      // `bulk == count`.
      const auto split = agent.split_chunk(block_keys_base, chunk);
      _CCCL_ASSERT(split.prefix == offset_t{0}, "overflow streamer received a chunk with an unaligned start");
      char* const dst = agent.key_slots + (stream_slot_base + stage) * ChunkBytes;
      const ::cuda::std::span<const key_t> src{
        block_keys_base + chunk.offset, static_cast<::cuda::std::size_t>(split.bulk)};
      agent.issue_bulk_copy(stage, dst, src);
      inflight_mask |= (::cuda::std::uint32_t{1} << stage);
    }

    // Rebuild the shared span for the chunk currently resident in `stage`'s slot without storing per-stage state: the
    // slot address is a pure function of `stage` and the length is recomputed from chunk index `o`, so there is no
    // spillable `pending[]` array (see `bulk_span`). Returns the aligned bulk only (a peeled tail's suffix is
    // excluded).
    [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::span<key_t> stage_span(int stage, offset_t o) const
    {
      char* const dst  = agent.key_slots + (stream_slot_base + stage) * ChunkBytes;
      const auto chunk = agent.get_chunk(chunk_index_of(o), segment_size, head_items);
      const auto split = agent.split_chunk(block_keys_base, chunk);
      return agent.bulk_span(
        {static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(dst)), static_cast<int>(split.bulk)});
    }

    // Shared driver for one overflow pass. `block_apply(stage, o)` folds the chunk for visit `o` currently resident in
    // the streaming slot `stage` (block-load path); `generic_apply(chunk)` folds an overflow chunk read straight from
    // gmem (generic fallback). `mid()` is invoked exactly once, after the prefetch loads for this pass's first reload
    // wave (the first `p_eff` visits) have been issued but before they are waited on, so the caller's resident-chunk
    // work overlaps those in-flight bulk copies. The two public entry points (`process_pass` / `process_pass_indexed`)
    // only differ in whether the applied callable receives the key alone or the key plus its segment-local index, so
    // they share this loop verbatim. `mid` must be uniform across the block and contain no unmatched block barrier.
    template <typename BlockApply, typename GenericApply, typename Mid>
    _CCCL_DEVICE _CCCL_FORCEINLINE void run_pass(BlockApply&& block_apply, GenericApply&& generic_apply, Mid&& mid)
    {
      if (overflow_chunks == 0)
      {
        mid();
        return;
      }

      const offset_t m = overflow_chunks;

      if constexpr (use_block_load_to_shared)
      {
        const offset_t pe = static_cast<offset_t>(p_eff);

        // First ever call: prime the streaming slots. Subsequent calls inherit
        // the previous pass's resident tail, which (because the order
        // ping-pongs) is exactly the first `p_eff` chunks of this direction.
        if (!primed)
        {
          // Wait for all threads to leave the resident load's final wait before re-arming its shared mbarriers; else
          // the phase advances twice and a lagging thread misses the flip and spins forever.
          __syncthreads();
          for (int i = 0; i < p_eff; ++i)
          {
            const offset_t o = forward ? static_cast<offset_t>(i) : (m - 1 - static_cast<offset_t>(i));
            issue_load(static_cast<int>(o % pe), o);
          }
          primed = true;
        }

        // Consume overflow visit `i`: wait for its slot, fold its keys via `block_apply`, then prefetch the chunk
        // `p_eff` visits ahead into the slot just freed (a barrier guards the slot before the async copy can overwrite
        // the data the block was just reading).
        const auto consume = [&](offset_t i) {
          const offset_t o = forward ? i : (m - 1 - i);
          const int stage  = static_cast<int>(o % pe);
          if (inflight_mask & (::cuda::std::uint32_t{1} << stage))
          {
            agent.wait_stage(stage);
            inflight_mask &= ~(::cuda::std::uint32_t{1} << stage);
          }
          block_apply(stage, o);

          const offset_t ni = i + pe;
          if (ni < m)
          {
            const offset_t no = forward ? ni : (m - 1 - ni);
            __syncthreads();
            issue_load(stage, no);
          }
        };

        // Phase 1: consume the first `p_eff` visits (the chunks reused from the previous pass, already resident in the
        // streaming slots), which issues the prefetch loads for this pass's reload wave into the freed slots.
        const offset_t split = (::cuda::std::min) (pe, m);
        for (offset_t i = 0; i < split; ++i)
        {
          consume(i);
        }

        // The reload wave is now in flight; run the caller's resident-chunk work to hide its latency before waiting.
        mid();

        // Phase 2: consume the remaining visits (their loads were issued in phase 1 and overlapped `mid`).
        for (offset_t i = split; i < m; ++i)
        {
          consume(i);
        }
        forward = !forward;
      }
      else
      {
        // Generic fallback: no async SMEM pipeline, so resident work cannot hide load latency here. Fold the resident
        // chunks first (preserving the prior ordering), then read the overflow keys straight from gmem each pass (no
        // SMEM reuse), with the walk still snaking for L2 locality.
        mid();
        for (offset_t i = 0; i < m; ++i)
        {
          const offset_t o         = forward ? i : (m - 1 - i);
          const offset_t chunk_idx = chunk_index_of(o);
          const auto chunk         = agent.get_chunk(chunk_idx, segment_size, head_items);
          generic_apply(chunk);
        }
        forward = !forward;
      }
    }

    // Apply `f(key)` to every overflow key once, in the current ping-pong direction. See `run_pass` for the overlap
    // semantics of `mid`. `UnrollFactor` partially unrolls the generic (gmem fallback) fold loop; callers pass the
    // tuning parameter matching their context (`histogram_items_per_thread` for the histogram passes,
    // `tie_break_items_per_thread` for the non-deterministic final filter).
    template <int UnrollFactor, typename F, typename Mid>
    _CCCL_DEVICE _CCCL_FORCEINLINE void process_pass(F&& f, Mid&& mid)
    {
      run_pass(
        [&](int stage, offset_t o) {
          agent.for_each_chunk_key(stage_span(stage, o), f);
        },
        [&](const auto& chunk) {
          const int iterations = ::cuda::ceil_div(chunk.count, threads_per_block);
          _CCCL_PRAGMA_UNROLL(UnrollFactor)
          for (int j = 0; j < iterations; ++j)
          {
            const int local = j * threads_per_block + static_cast<int>(threadIdx.x);
            if (local < chunk.count)
            {
              f(block_keys_in[static_cast<segment_size_val_t>(chunk.offset + static_cast<offset_t>(local))]);
            }
          }
        },
        static_cast<Mid&&>(mid));
    }

    // Overload with no interleaved work, for the fused first pass where the resident keys are still being streamed in
    // by the BlockLoadToShared pipeline (rather than already resident in SMEM).
    template <int UnrollFactor, typename F>
    _CCCL_DEVICE _CCCL_FORCEINLINE void process_pass(F&& f)
    {
      process_pass<UnrollFactor>(static_cast<F&&>(f), [] {});
    }

    // Like `process_pass`, but applies `f(key, seg_idx)` where `seg_idx` is the key's segment-local index. The pair
    // final filter needs that index to fetch each selected key's value payload from gmem, while still reusing the
    // overflow keys from the streaming SMEM pipeline (block-load path) instead of re-reading them from gmem.
    template <int UnrollFactor, typename F, typename Mid>
    _CCCL_DEVICE _CCCL_FORCEINLINE void process_pass_indexed(F&& f, Mid&& mid)
    {
      run_pass(
        [&](int stage, offset_t o) {
          const offset_t base_off = agent.get_chunk(chunk_index_of(o), segment_size, head_items).offset;
          agent.for_each_chunk_key_indexed(stage_span(stage, o), base_off, f);
        },
        [&](const auto& chunk) {
          const int iterations = ::cuda::ceil_div(chunk.count, threads_per_block);
          _CCCL_PRAGMA_UNROLL(UnrollFactor)
          for (int j = 0; j < iterations; ++j)
          {
            const int local = j * threads_per_block + static_cast<int>(threadIdx.x);
            if (local < chunk.count)
            {
              const offset_t seg_idx = chunk.offset + static_cast<offset_t>(local);
              f(block_keys_in[static_cast<segment_size_val_t>(seg_idx)], seg_idx);
            }
          }
        },
        static_cast<Mid&&>(mid));
    }
  };

  // -------------------------------------------------------------------------
  // Per-direction implementation
  // -------------------------------------------------------------------------
  // Stripped on sub-SM-9.0 device passes; uses `cluster_group`, which is only
  // declared when `_CG_HAS_CLUSTER_GROUP` is set.
#if defined(_CG_HAS_CLUSTER_GROUP)
  // Synchronize the segment's cluster. A single-CTA "cluster" keeps all state block-local, so `__syncthreads()` orders
  // it and the cluster-scoped barrier is unnecessary. `single_cta` is computed in `run()` from the collapsed cluster
  // blocks `process_impl` passes in (1 when a small segment collapsed onto rank 0), not the raw `cluster.num_blocks()`.
  // It is per-segment uniform across the surviving block(s), so the branch is reached uniformly.
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  cluster_or_block_sync(::cooperative_groups::cluster_group& cluster, bool single_cta)
  {
    if (single_cta)
    {
      __syncthreads();
    }
    else
    {
      cluster.sync();
    }
  }

  template <detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  run(::cooperative_groups::cluster_group& cluster,
      num_segments_val_t segment_id,
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
    // `cluster_blocks` is what `process_impl` runs at: the launched `cluster.num_blocks()`, or `1` when it collapsed a
    // small segment onto rank 0. A lone CTA routes barriers to `__syncthreads()`, keeps `state`/atomics block-local,
    // and uses CTA-scope histogram atomics (no cross-rank DSMEM folds to be mutually atomic with). For wider clusters,
    // `eff_cluster_blocks` (below) further excludes ranks that receive no chunks; they stay resident but idle.
    const bool single_cta = (cluster_blocks == 1u);

    // Per-block local copy of `kth_key_bits` so each key check hits the
    // block's own SMEM rather than DSMEM during the histogram loop. Built up one splitter digit per pass from the
    // leader's published `kth_bucket` (see the pass loop), so the full key is never broadcast.
    key_prefix_t kth_key_bits_local = {};

    // Last splitter bucket folded into `kth_key_bits_local` (the last executed pass's bucket). Used by the
    // deterministic tie-break to read this CTA's local candidate count from `hist[last_bucket]`.
    [[maybe_unused]] int last_bucket = 0;

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

    // Effective cluster blocks: the CTAs that actually receive chunks (at least `min_chunks_per_cta` each), <= the
    // launched `cluster_blocks`. Ranks at or beyond it are idle -- they own no chunks, fold nothing, and never lead --
    // but stay resident and still arrive at every `cluster.sync()` (a returned CTA would hang the barrier; see the
    // TODOs at the barrier sites). Derived from this CTA's head-aligned `chunks` so it matches the partition exactly.
    // Stays at `cluster_blocks` for host-exact sizes (the dispatch already matched it) and on the single-CTA path.
    unsigned int eff_cluster_blocks = cluster_blocks;
    if constexpr (enable_runtime_single_cta)
    {
      if (!single_cta)
      {
        eff_cluster_blocks = effective_cluster_blocks_from_chunks(
          static_cast<::cuda::std::uint64_t>(chunks), min_chunks_per_cta, cluster_blocks);
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
    const unsigned int leader_rank = (deterministic && !tie_reversed) ? (eff_cluster_blocks - 1u) : 0u;

    // DSMEM pointer into the leader block's shared memory. The Step 2 histogram fold reaches the leader's `hist`
    // through a `mapa`-formed `shared::cluster` address instead (see `hist_fold_remote`).
    state_t* leader_state =
      single_cta ? &temp_storage.state : cluster.map_shared_rank(&temp_storage.state, leader_rank);

    // Resident vs. streaming split, decided independently per CTA (CTAs need not agree -- cross-CTA traffic and every
    // `cluster.sync()` are reached uniformly). A CTA whose chunks fit its resident slots (`my_chunks <= full_slots`)
    // keeps them all resident and streams nothing; an overflowing CTA reserves a round-robin streaming region at the
    // tail of its block_tile and re-streams its overflow chunks from gmem each pass via `streamer`.
    //
    // Boundary edges (the unaligned head prefix on rank 0 and the unaligned tail suffix on the tail owner) cannot use
    // the aligned TMA streamer. The head is always peeled into the persistent `edge_keys` buffer; the tail suffix is
    // kept resident in its (partial, single-slot) tail chunk, peeled into `edge_keys` only when a single slot fits
    // (`full_slots == 1`). Peeling the head removes the dual-boundary pressure, so streaming needs only `full_slots >=
    // 1`.
    //
    // `stream_slots` is right-sized: clamped to leave a resident slot for a non-peeled tail without exceeding the
    // available slots; deep overflows still get the full `PipelineStages` depth. The generic fallback has no async
    // pipeline (it re-reads overflow from gmem each pass and peels nothing), so it reserves no streaming slots.
    const offset_t full_slots                   = block_tile_capacity / static_cast<offset_t>(chunk_items);
    [[maybe_unused]] const bool needs_streaming = my_chunks > full_slots;

    // Does this rank own the global tail, and does that tail carry an unaligned suffix? (block-load path only; the
    // generic fallback reads any trailing items straight from gmem and never peels.)
    [[maybe_unused]] offset_t tail_local        = offset_t{0};
    [[maybe_unused]] offset_t tail_suffix_items = offset_t{0};
    bool owns_suffix_tail                       = false;
    if constexpr (use_block_load_to_shared)
    {
      // This rank owns the global tail iff its last owned chunk is chunk `chunks-1` (its local index `my_chunks-1`,
      // true for both the strided and blocked partitions).
      if (my_chunks > 0 && part.global_index(my_chunks - offset_t{1}) == chunks - offset_t{1})
      {
        tail_local            = my_chunks - offset_t{1};
        const auto tail_chunk = get_chunk(chunks - offset_t{1}, segment_size_u32, head_items);
        tail_suffix_items     = split_chunk(block_keys_base, tail_chunk).suffix;
        owns_suffix_tail      = tail_suffix_items != offset_t{0};
      }
    }

    // Peel the suffix tail only when streaming and no resident slot can be spared for it (`full_slots == 1`); otherwise
    // keep it resident. `reserved_resident` holds back one slot for a non-peeled tail; `stream_slots` is then clamped
    // into `[1, full_slots - reserved_resident]`.
    offset_t stream_slots    = offset_t{0};
    bool peel_tail           = false;
    bool force_tail_resident = false;
    if constexpr (use_block_load_to_shared)
    {
      if (needs_streaming)
      {
        peel_tail                        = owns_suffix_tail && full_slots < offset_t{2};
        const offset_t reserved_resident = (owns_suffix_tail && !peel_tail) ? offset_t{1} : offset_t{0};
        const offset_t excess            = my_chunks - full_slots;
        const offset_t want_stream       = (::cuda::std::min) (static_cast<offset_t>(PipelineStages), excess);
        const offset_t max_stream        = full_slots - reserved_resident; // >= 1 (full_slots >= 1; reserve <= f-1)
        stream_slots = (::cuda::std::max) (offset_t{1}, (::cuda::std::min) (want_stream, max_stream));
      }
    }
    const offset_t resident_slots_cap = full_slots - stream_slots;
    const offset_t my_resident_chunks = (::cuda::std::min) (my_chunks, resident_slots_cap);
    // Resident chunks stay within the first `resident_slots_cap` slots; the streaming region occupies the slots
    // `[resident_slots_cap, full_slots)`, so both regions live inside the allocated block_tile buffer.
    _CCCL_ASSERT(my_resident_chunks <= resident_slots_cap, "Dynamic shared memory block_tile is too small");

    // A non-peeled suffix tail is forced into the last resident slot iff it would otherwise stream (it falls in the
    // overflow region). The overflow then begins one chunk earlier so the displaced middle chunk streams in its place.
    const offset_t overflow_count = (my_chunks > my_resident_chunks) ? (my_chunks - my_resident_chunks) : offset_t{0};
    if constexpr (use_block_load_to_shared)
    {
      force_tail_resident = owns_suffix_tail && !peel_tail && overflow_count > 0 && (tail_local >= my_resident_chunks);
    }
    const offset_t overflow_base = force_tail_resident ? (my_resident_chunks - offset_t{1}) : my_resident_chunks;
    _CCCL_ASSERT(!force_tail_resident || my_resident_chunks >= offset_t{1},
                 "a forced-resident tail needs at least one resident slot");

    // Persistent boundary-edge lengths: the head prefix lives on rank 0 (`head_items` is 0 on the generic fallback and
    // for an aligned base); the peeled tail suffix lives on the tail owner only when `peel_tail`.
    [[maybe_unused]] const int head_edge_len = (cluster_rank == 0u) ? static_cast<int>(head_items) : 0;
    [[maybe_unused]] const int tail_edge_len = peel_tail ? static_cast<int>(tail_suffix_items) : 0;

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
        if (head_edge_len > 0)
        {
          for_each_chunk_key({temp_storage.edge_keys, static_cast<::cuda::std::size_t>(head_edge_len)}, apply);
        }
        if (tail_edge_len > 0)
        {
          for_each_chunk_key({temp_storage.edge_keys + head_edge_cap, static_cast<::cuda::std::size_t>(tail_edge_len)},
                             apply);
        }
      }
    };

    if constexpr (use_block_load_to_shared)
    {
      // Arm the stage barriers once; reused (ping-ponged) by the resident load and the overflow streamer across passes.
      init_load_barriers();
    }
    __syncthreads();

    {
      extract_bin_op_t extract_op(0, total_bits, decomposer_t{});
      const ::cuda::std::uint32_t hist_smem32 = hist_base32();
      // Cluster-scope leader atomics are only needed to stay mutually atomic with the non-leaders' DSMEM folds; a lone
      // CTA has none, so it uses the cheaper CTA scope.
      const bool leader_cluster_scope = (cluster_rank == leader_rank) && !single_cta;
      auto add_first_pass             = [&](const key_t& key) {
        const int bucket = extract_op(key);
        hist_inc(hist_smem32, bucket, leader_cluster_scope);
      };

      if constexpr (use_block_load_to_shared)
      {
        if (my_resident_chunks > 0)
        {
          // Stage mbarriers and the `load_phase` parity are shared with the streamer (no per-chunk token array needed).
          // Chunks are written densely in slot order from offset 0 and read back in the same order, so the read cursor
          // (`read_off`) mirrors the write cursor (`next_off`) as a running prefix sum, avoiding a dynamically-indexed
          // `pending_spans` array that would anchor surrounding state to local memory. Every chunk begins on a
          // `load_align` boundary (zero prefix), so its whole `count` is the aligned bulk - except a resident suffix
          // tail, whose bulk is loaded here and whose suffix is appended afterwards. The read span is rebuilt from a
          // rooted 32-bit shared address (see `bulk_span`) so a spilled cursor cannot demote the reads to `LD`.
          const int prologue = (::cuda::std::min) (PipelineStages, static_cast<int>(my_resident_chunks));

          // Resident slot -> rank-local chunk index. Identity, except the last slot holds the forced-resident tail.
          const auto resident_local = [&](offset_t slot) -> offset_t {
            return (force_tail_resident && slot == my_resident_chunks - offset_t{1}) ? tail_local : slot;
          };
          // Aligned bulk of the resident chunk in `slot` (its `count` minus any tail suffix); empty when it has none.
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
          int next_off = 0;

          // Load every resident chunk's aligned bulk, densely packed in slot order, so the last slot's bulk (the tail
          // bulk in the forced case) ends the packed region and its suffix can be appended right after it.
          for (int stage = 0; stage < prologue; ++stage)
          {
            const auto src = bulk_src(static_cast<offset_t>(stage));
            issue_bulk_copy(stage, key_slots + next_off, src);
            next_off += static_cast<int>(::cuda::std::size(src)) * int{sizeof(key_t)};
          }

          // Read cursor trailing the write cursor: chunk `p`'s bulk was written at `read_off` (packed and consumed in
          // the same order), and `bulk_src(p)` recomputes its length, so the read span needs no stored per-stage state.
          int read_off = 0;
          for (offset_t p = 0; p < my_resident_chunks; ++p)
          {
            const int stage = static_cast<int>(p % static_cast<offset_t>(prologue));
            wait_stage(stage);
            const int read_len = static_cast<int>(::cuda::std::size(bulk_src(p)));
            for_each_chunk_key(
              bulk_span({static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(key_slots + read_off)), read_len}),
              add_first_pass);
            read_off += read_len * int{sizeof(key_t)};

            const offset_t next_p = p + static_cast<offset_t>(prologue);
            if (next_p < my_resident_chunks)
            {
              const auto src = bulk_src(next_p);
              // Phase safety, not data safety (the target offset is fresh): re-arming this stage before all threads
              // leave the wait above would advance the phase twice, stranding a lagging waiter forever.
              __syncthreads();
              issue_bulk_copy(stage, key_slots + next_off, src);
              next_off += static_cast<int>(::cuda::std::size(src)) * int{sizeof(key_t)};
            }
          }

          // Tail suffix of a resident (non-peeled) suffix tail: append right after the last (tail) bulk. Nothing
          // follows, so its non-16 length is harmless. A peeled tail's suffix lives in `edge_keys` instead.
          const offset_t last_slot = my_resident_chunks - offset_t{1};
          const auto last_chunk = get_chunk(part.global_index(resident_local(last_slot)), segment_size_u32, head_items);
          const auto last_split = split_chunk(block_keys_base, last_chunk);
          if (last_split.suffix > 0)
          {
            key_t* const edge_dst = reinterpret_cast<key_t*>(key_slots + next_off);
            stage_and_fold_edge(edge_dst,
                                block_keys_base + last_chunk.offset + last_split.prefix + last_split.bulk,
                                static_cast<int>(last_split.suffix),
                                add_first_pass);
            next_off += static_cast<int>(last_split.suffix) * int{sizeof(key_t)};
          }

          // The resident region is one contiguous span [bulks | tail suffix] for the later passes; the head prefix is
          // folded separately from `edge_keys`.
          resident_keys   = {reinterpret_cast<key_t*>(key_slots),
                             static_cast<::cuda::std::size_t>(next_off / int{sizeof(key_t)})};
          resident_smem32 = static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(key_slots));
        }
      }
      else
      {
        for (offset_t p = 0; p < my_resident_chunks; ++p)
        {
          const offset_t chunk_idx = part.global_index(p);
          const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
          key_t* const chunk_keys  = slot_keys_unpadded(static_cast<int>(p));
          const int iterations     = ::cuda::ceil_div(chunk.count, threads_per_block);
          _CCCL_PRAGMA_UNROLL(histogram_items_per_thread)
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
      // (rank 0) precedes chunk 0; the peeled tail suffix (tail owner, `full_slots == 1`) trails the last chunk. The
      // `__syncthreads()` below publishes `edge_keys` for the later passes and the final filter (which read it via
      // `fold_edges` after a barrier).
      if constexpr (use_block_load_to_shared)
      {
        if (head_edge_len > 0)
        {
          stage_and_fold_edge(temp_storage.edge_keys, block_keys_base, head_edge_len, add_first_pass);
        }
        if (tail_edge_len > 0)
        {
          const auto tail_chunk = get_chunk(chunks - offset_t{1}, segment_size_u32, head_items);
          stage_and_fold_edge(temp_storage.edge_keys + head_edge_cap,
                              block_keys_base + tail_chunk.offset + (tail_chunk.count - tail_edge_len),
                              tail_edge_len,
                              add_first_pass);
        }
      }

      // Fold the overflow chunks into the first-pass histogram. Forward direction; primes the streaming slots. The
      // streamer reuses the resident load's stage barriers (no re-init); `wait_stage` provides the producer/consumer
      // sync.
      streamer.process_pass<histogram_items_per_thread>(add_first_pass);

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
        const bool leader_cluster_scope         = (cluster_rank == leader_rank) && !single_cta;
        auto add_hist                           = [&](const key_t& key) {
          if (identify_op(key) == detail::topk::candidate_class::candidate)
          {
            const int bucket = extract_op(key);
            hist_inc(hist_smem32, bucket, leader_cluster_scope);
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
            key_t* const rk = reinterpret_cast<key_t*>(__cvta_shared_to_generic(resident_smem32));
            for_each_chunk_key({rk, static_cast<::cuda::std::size_t>(span_size(resident_keys))}, add_hist);
          }
          else
          {
            for (offset_t p = 0; p < my_resident_chunks; ++p)
            {
              const offset_t chunk_idx = part.global_index(p);
              const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
              key_t* const chunk_keys  = slot_keys_unpadded(static_cast<int>(p));
              for_each_chunk_key({chunk_keys, static_cast<::cuda::std::size_t>(chunk.count)}, add_hist);
            }
          }
        };

        // Re-stream the overflow chunks into this pass's histogram, overlapping the resident-chunk histogram with the
        // first wave of reload bulk copies. Ping-pongs direction and reuses the turn-around chunks left resident by the
        // previous pass.
        streamer.process_pass<histogram_items_per_thread>(add_hist, fold_resident_hist);

        // Fold the persistent boundary edges (loaded once in the first pass) into this pass's histogram, alongside the
        // resident and overflow keys. Keeps every owner's `hist[last_bucket]` (the deterministic cross-CTA prefix
        // input) inclusive of its edge candidates.
        fold_edges(add_hist);
      }

      // Local barrier is enough: all Step 1 / Step 2 writes to `hist[]`
      // are atomic at compatible scopes (see Step 1 dispatch). The
      // cluster-wide ordering before Step 3's leader read of `hist[]`
      // is supplied by the `cluster.sync()` further below.
      __syncthreads();

      // Step 2: non-leader blocks fold their per-bucket values
      // (raw counts in the reduce-then-scan path, block-local inclusive
      // scans in the scan-then-reduce path) into the leader's `hist`
      // via cluster-scope DSMEM atomics (see `hist_fold_remote`). The
      // leader skips this to avoid double-counting its own contribution;
      // idle ranks (`>= eff_cluster_blocks`) have an all-zero histogram, so
      // they skip the fold entirely (the loop would only read zeros).
      if (cluster_rank != leader_rank && !is_idle_rank)
      {
        const ::cuda::std::uint32_t hist_smem32 = hist_base32();
        for (int i = static_cast<int>(threadIdx.x); i < num_buckets; i += threads_per_block)
        {
          const offset_t v = temp_storage.hist[i];
          if (v != 0)
          {
            hist_fold_remote(hist_smem32 + static_cast<::cuda::std::uint32_t>(i) * sizeof(offset_t), v, leader_rank);
          }
        }
      }

      // TODO(cccl): idle ranks arrive here only because `cluster.sync()` spans the whole launched cluster. An mbarrier
      // over just the active ranks would let them exit and free their SM slots instead of spinning on this barrier.
      cluster_or_block_sync(cluster, single_cta);

      // Step 3: the leader walks the merged `hist` (raw counts in the
      // reduce-then-scan path, cluster-wide inclusive scan in the
      // scan-then-reduce path) and updates the cluster-shared `state`.
      // Subsequent reads (end-of-pass fold, last filter) all observe
      // these writes after the next cluster sync.
      if (cluster_rank == leader_rank)
      {
        leader_identify_kth_bucket();
      }

      if (pass + 1 < num_passes)
      {
        if (cluster_rank == leader_rank)
        {
          __syncthreads(); // order the leader's `hist` reads (BlockScan in `leader_identify_kth_bucket`) before resets
        }
        reset_hist();
      }
      // TODO(cccl): see the barrier above -- idle ranks arrive here only to keep the cluster barrier reachable.
      cluster_or_block_sync(cluster, single_cta);

      // End-of-pass splitter fold. Every working thread reads the leader's just-published `kth_bucket` through DSMEM (a
      // single `u32`, ordered by the `cluster.sync()` above) and folds it into its own `kth_key_bits_local`, so the
      // full splitter key is reconstructed locally without a broadcast. Idle ranks produce no output, so they skip the
      // read (trimming the leader's fan-out); the early-stop check below stays uniform so every block breaks together.
      if (!is_idle_rank)
      {
        const int bucket = static_cast<int>(leader_state->kth_bucket);
        detail::topk::set_kth_key_bits<key_t, bits_per_pass>(kth_key_bits_local, pass, bucket);
        last_bucket = bucket;
        last_pass   = pass + 1;
      }
#  ifndef CUB_DISABLE_CLUSTER_TOPK_EARLY_STOP
      // Early stop: the leader sets `early_stop` when the splitter bucket holds exactly the remaining `k` candidates,
      // so no further radix refinement can change the result. Every block reads the same flag from the leader and
      // breaks together; `last_pass`/`kth_key_bits_local` then match what the original top-of-next-pass break produced.
      if (leader_state->early_stop != ::cuda::std::uint32_t{0})
      {
        break;
      }
#  endif
    }

    // -----------------------------------------------------------------------
    // Final filter pass: write the top-k keys for this segment. Strictly-
    // selected keys go to the front; the `num_kth` tied candidates fill the
    // back. `kth_key_bits_local` already holds the full splitter key (folded
    // from each pass's bucket above), so no broadcast is needed here.
    // -----------------------------------------------------------------------
    auto block_keys_out        = d_key_segments_out_it[segment_id];
    const out_offset_t num_kth = leader_state->k; // remaining k after the radix passes

    // For each key written to `block_keys_out[pos]`, the associated input value at the key's segment-local index
    // `seg_idx` is loaded naively from gmem and written to `block_vals_out[pos]`. `seg_idx` is recomputed per region in
    // the sweeps below. The per-segment value iterators are derived *inside* the `is_keys_only` guard: in keys-only
    // builds the value iterators-of-iterators are `cub::NullType**` (null), so indexing them with `segment_id` here
    // would dereference a null pointer; `segment_id` is loop-invariant, so the compiler hoists these out of the writes.
    const auto write_value = [&](out_offset_t pos, offset_t seg_idx) {
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

    if constexpr (deterministic)
    {
      // Deterministic tie-break: select the `num_kth` tied candidates with the smallest (or largest, if `tie_reversed`)
      // global indices via a cluster-wide index-ordered scan. Strictly-selected keys go to the front; ties fill the
      // back by rank. Early stop means the splitter bucket holds *exactly* the remaining `k` candidates, so all are
      // winners (`num_back == 0`, `num_selected == k`) and we skip the scan and its `hist[last_bucket]` read (the
      // end-of-pass reset may already have cleared `hist`). Valid even when the early-stop *break* is compiled out.
      const out_offset_t num_back = (leader_state->early_stop != ::cuda::std::uint32_t{0}) ? out_offset_t{0} : num_kth;
      const out_offset_t num_selected = k - num_back; // front region; == k on early stop

      offset_t running = 0;
      bool tie_active  = false;
      if (num_back != out_offset_t{0})
      {
        // Exclusive cross-CTA candidate-count scan: each non-leader adds its candidate count (`hist[last_bucket]`) into
        // every later CTA's `cand_prefix` in scan order; the leader is last (see `leader_rank`) and receives the sum.
        const offset_t local_count = (cluster_rank == leader_rank) ? offset_t{0} : temp_storage.hist[last_bucket];
        if (threadIdx.x == 0)
        {
          temp_storage.cand_prefix = 0;
        }
        // TODO(cccl): idle ranks (`local_count == 0`) arrive only to keep this barrier reachable; a sub-cluster
        // mbarrier over the working ranks would let them exit (see the pass loop).
        cluster_or_block_sync(cluster, single_cta);
        if (threadIdx.x == 0 && local_count != offset_t{0})
        {
          if constexpr (tie_reversed)
          {
            for (unsigned int r = 0; r < cluster_rank; ++r) // descending scan order: lower ranks follow
            {
              add_remote_prefix(r, local_count);
            }
          }
          else
          {
            // Ascending scan order: higher ranks follow. Stops at `eff_cluster_blocks` since idle ranks own nothing.
            for (unsigned int r = cluster_rank + 1u; r < eff_cluster_blocks; ++r)
            {
              add_remote_prefix(r, local_count);
            }
          }
        }
        cluster_or_block_sync(cluster, single_cta);

        running    = temp_storage.cand_prefix; // candidates owned by preceding CTAs (this CTA's exclusive prefix)
        tie_active = running < static_cast<offset_t>(num_back);
      }

      // Process a flat span of `count` keys already in scan order, tiled by `threads_per_block * items`. Front keys go
      // via `out_cnt`; surplus ties get a BlockScan-exclusive rank (seeded by `running`) and, if `rank < num_back`, are
      // written reversed at `block_keys_out[k - 1 - rank]`. `running` carries across tiles and regions. `get_idx(pos)`
      // gives the segment-local index, used only to load the value payload (compiled out in keys-only builds).
      auto process_flat = [&](auto get_key, auto get_idx, int count) {
        constexpr int items = tie_break_items_per_thread;
        constexpr int tile  = threads_per_block * items;
        for (int tile_base = 0; tile_base < count; tile_base += tile)
        {
          key_t keys[items];
          offset_t flags[items];
          detail::topk::candidate_class cls[items];
          bool valid[items];
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int i = 0; i < items; ++i)
          {
            const int pos = tile_base + static_cast<int>(threadIdx.x) * items + i;
            valid[i]      = pos < count;
            flags[i]      = offset_t{0};
            if (valid[i])
            {
              keys[i]  = get_key(pos);
              cls[i]   = identify_op(keys[i]);
              flags[i] = (cls[i] == detail::topk::candidate_class::candidate) ? offset_t{1} : offset_t{0};
            }
          }

          // Strictly-selected keys go to the front via `out_cnt`; on early stop (`num_back == 0`) the splitter bucket's
          // candidates are winners too and fold in here. Exactly `num_selected` are placed in `[0, num_selected)`.
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int i = 0; i < items; ++i)
          {
            const bool to_front =
              valid[i]
              && (cls[i] == detail::topk::candidate_class::selected
                  || (num_back == out_offset_t{0} && cls[i] == detail::topk::candidate_class::candidate));
            if (to_front)
            {
              const int pos          = tile_base + static_cast<int>(threadIdx.x) * items + i;
              const out_offset_t out = atomicAdd(&leader_state->out_cnt, out_offset_t{1});
              block_keys_out[out]    = keys[i];
              write_value(out, get_idx(pos));
            }
          }

          // Surplus ties (`num_back > 0`): each candidate gets a BlockScan rank (seeded by `running`), written reversed
          // at `block_keys_out[k - 1 - rank]` when `rank < num_back`. Skipped on early stop (`tie_active == false`).
          if (tie_active)
          {
            offset_t excl[items];
            offset_t tile_total = 0;
            block_scan_t(temp_storage.scan_storage).ExclusiveSum(flags, excl, tile_total);
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int i = 0; i < items; ++i)
            {
              if (valid[i] && flags[i] != offset_t{0})
              {
                const offset_t global_rank = running + excl[i];
                if (global_rank < static_cast<offset_t>(num_back))
                {
                  const int pos          = tile_base + static_cast<int>(threadIdx.x) * items + i;
                  const out_offset_t out = static_cast<out_offset_t>(k - 1) - static_cast<out_offset_t>(global_rank);
                  block_keys_out[out]    = keys[i];
                  write_value(out, get_idx(pos));
                }
              }
            }
            running += tile_total;
            if (running >= static_cast<offset_t>(num_back))
            {
              tie_active = false;
            }
            // The next tile reuses `scan_storage`; order this tile's reads against the next ExclusiveSum's writes.
            __syncthreads();
          }
        }
      };

      // Uniform early-exit between regions: stop once this CTA's ties are placed and all selected are placed
      // cluster-wide. Lets the common prefer-smallest case skip the (expensive) overflow re-stream entirely.
      auto should_stop = [&]() -> bool {
        if (tie_active)
        {
          return false;
        }
        const out_offset_t out_now = *static_cast<volatile out_offset_t*>(&leader_state->out_cnt);
        return __syncthreads_or(static_cast<int>(out_now >= num_selected)) != 0;
      };

      // Resident-front extent (bulk path): the contiguous resident span minus the forced-resident tail chunk, which is
      // visited separately after the overflow because it is the globally-last chunk.
      int tail_count  = 0;
      int front_count = span_size(resident_keys);
      if constexpr (use_block_load_to_shared)
      {
        if (force_tail_resident)
        {
          tail_count  = get_chunk(chunks - offset_t{1}, segment_size_u32, head_items).count;
          front_count = front_count - tail_count;
        }
      }

      // Segment-local base of the resident-front span. The deterministic path's blocked partition keeps the front
      // chunks contiguous and densely packed, so element `pos` of the front maps to `front_seg_base + pos`.
      const offset_t front_seg_base = get_chunk(part.first, segment_size_u32, head_items).offset;

      auto process_resident = [&](bool reversed) {
        if constexpr (use_block_load_to_shared)
        {
          key_t* const rfront = reinterpret_cast<key_t*>(__cvta_shared_to_generic(resident_smem32));
          const int fc        = front_count;
          if (reversed)
          {
            process_flat(
              [&](int pos) {
                return rfront[fc - 1 - pos];
              },
              [&](int pos) {
                return front_seg_base + static_cast<offset_t>(fc - 1 - pos);
              },
              fc);
          }
          else
          {
            process_flat(
              [&](int pos) {
                return rfront[pos];
              },
              [&](int pos) {
                return front_seg_base + static_cast<offset_t>(pos);
              },
              fc);
          }
        }
        else
        {
          const int rc = static_cast<int>(my_resident_chunks);
          for (int s = 0; s < rc; ++s)
          {
            const int local_slot     = reversed ? (rc - 1 - s) : s;
            const offset_t chunk_idx = part.global_index(static_cast<offset_t>(local_slot));
            const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
            const int cc             = chunk.count;
            const offset_t base_off  = chunk.offset;
            key_t* const ck          = slot_keys_unpadded(local_slot);
            if (reversed)
            {
              process_flat(
                [&](int pos) {
                  return ck[cc - 1 - pos];
                },
                [&](int pos) {
                  return base_off + static_cast<offset_t>(cc - 1 - pos);
                },
                cc);
            }
            else
            {
              process_flat(
                [&](int pos) {
                  return ck[pos];
                },
                [&](int pos) {
                  return base_off + static_cast<offset_t>(pos);
                },
                cc);
            }
          }
        }
      };

      auto process_overflow = [&](bool reversed) {
        for (offset_t oo = 0; oo < overflow_count; ++oo)
        {
          const offset_t o         = reversed ? (overflow_count - 1 - oo) : oo;
          const offset_t chunk_idx = part.global_index(overflow_base + o);
          const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
          // Visit only the aligned bulk: a peeled tail suffix is handled by `process_tail_edge`; every other overflow
          // chunk has `bulk == count`. (The generic fallback never peels, so it uses the full count.)
          int cc;
          if constexpr (use_block_load_to_shared)
          {
            cc = static_cast<int>(split_chunk(block_keys_base, chunk).bulk);
          }
          else
          {
            cc = chunk.count;
          }
          const offset_t base_off = chunk.offset;
          if (reversed)
          {
            process_flat(
              [&](int pos) {
                return block_keys_in[static_cast<segment_size_val_t>(base_off + static_cast<offset_t>(cc - 1 - pos))];
              },
              [&](int pos) {
                return base_off + static_cast<offset_t>(cc - 1 - pos);
              },
              cc);
          }
          else
          {
            process_flat(
              [&](int pos) {
                return block_keys_in[static_cast<segment_size_val_t>(base_off + static_cast<offset_t>(pos))];
              },
              [&](int pos) {
                return base_off + static_cast<offset_t>(pos);
              },
              cc);
          }
        }
      };

      auto process_tail = [&](bool reversed) {
        if constexpr (use_block_load_to_shared)
        {
          if (!force_tail_resident)
          {
            return;
          }
          key_t* const rfront = reinterpret_cast<key_t*>(__cvta_shared_to_generic(resident_smem32));
          key_t* const tptr   = rfront + front_count;
          const int tc        = tail_count;
          // The forced-resident tail is the globally-last chunk `chunks-1`; its segment-local base is that chunk's
          // offset.
          const offset_t tail_seg_base = get_chunk(chunks - offset_t{1}, segment_size_u32, head_items).offset;
          if (reversed)
          {
            process_flat(
              [&](int pos) {
                return tptr[tc - 1 - pos];
              },
              [&](int pos) {
                return tail_seg_base + static_cast<offset_t>(tc - 1 - pos);
              },
              tc);
          }
          else
          {
            process_flat(
              [&](int pos) {
                return tptr[pos];
              },
              [&](int pos) {
                return tail_seg_base + static_cast<offset_t>(pos);
              },
              tc);
          }
        }
      };

      // Head prefix edge (rank 0): the segment's lowest indices `[0, head_edge_len)`, staged in `edge_keys`. Processed
      // before every chunk (ascending) / after (descending) to keep global index order. `count == 0` is a no-op, so
      // non-head ranks skip it.
      auto process_head_edge = [&](bool reversed) {
        if constexpr (use_block_load_to_shared)
        {
          key_t* const he = temp_storage.edge_keys;
          const int hc    = head_edge_len;
          if (reversed)
          {
            process_flat(
              [&](int pos) {
                return he[hc - 1 - pos];
              },
              [&](int pos) {
                return static_cast<offset_t>(hc - 1 - pos);
              },
              hc);
          }
          else
          {
            process_flat(
              [&](int pos) {
                return he[pos];
              },
              [&](int pos) {
                return static_cast<offset_t>(pos);
              },
              hc);
          }
        }
      };

      // Peeled tail suffix edge (tail owner, `full_slots == 1`): the segment's highest indices, staged in `edge_keys`.
      // Processed after every chunk (ascending) / before (descending). `tail_edge_len == 0` (tail kept resident or not
      // owned here) makes it a no-op.
      auto process_tail_edge = [&](bool reversed) {
        if constexpr (use_block_load_to_shared)
        {
          key_t* const te          = temp_storage.edge_keys + head_edge_cap;
          const int tc             = tail_edge_len;
          const offset_t tail_base = segment_size_u32 - static_cast<offset_t>(tc);
          if (reversed)
          {
            process_flat(
              [&](int pos) {
                return te[tc - 1 - pos];
              },
              [&](int pos) {
                return tail_base + static_cast<offset_t>(tc - 1 - pos);
              },
              tc);
          }
          else
          {
            process_flat(
              [&](int pos) {
                return te[pos];
              },
              [&](int pos) {
                return tail_base + static_cast<offset_t>(pos);
              },
              tc);
          }
        }
      };

      if constexpr (tie_reversed)
      {
        process_tail_edge(true);
        if (!should_stop())
        {
          process_tail(true);
        }
        if (!should_stop())
        {
          process_overflow(true);
        }
        if (!should_stop())
        {
          process_resident(true);
        }
        if (!should_stop())
        {
          process_head_edge(true);
        }
      }
      else
      {
        process_head_edge(false);
        if (!should_stop())
        {
          process_resident(false);
        }
        if (!should_stop())
        {
          process_overflow(false);
        }
        if (!should_stop())
        {
          process_tail(false);
        }
        if (!should_stop())
        {
          process_tail_edge(false);
        }
      }
    }
    else
    {
      if constexpr (is_keys_only)
      {
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
        // Fold the resident keys as the streamer's `mid` work so they overlap the first overflow reloads. Writes are
        // order-independent atomics, and resident SMEM slots are disjoint from streaming slots, so `mid` never races.
        const auto fold_resident = [&] {
          if constexpr (use_block_load_to_shared)
          {
            key_t* const rk = reinterpret_cast<key_t*>(__cvta_shared_to_generic(resident_smem32));
            for_each_chunk_key({rk, static_cast<::cuda::std::size_t>(span_size(resident_keys))}, write_selected);
          }
          else
          {
            for (offset_t p = 0; p < my_resident_chunks; ++p)
            {
              const offset_t chunk_idx = part.global_index(p);
              const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
              key_t* const chunk_keys  = slot_keys_unpadded(static_cast<int>(p));
              for_each_chunk_key({chunk_keys, static_cast<::cuda::std::size_t>(chunk.count)}, write_selected);
            }
          }
          // Scan the persistent boundary edges alongside the resident keys (order-independent atomic writes).
          fold_edges(write_selected);
        };
        streamer.process_pass<tie_break_items_per_thread>(write_selected, fold_resident);
      }
      else
      {
        // Pair (key + value) path. Same `out_cnt`/`out_back_cnt` atomics and key reuse as the keys-only path; for each
        // written key we additionally load its value from gmem at the segment-local index `seg_idx` and store it at the
        // same slot (values are never streamed). Output order is not preserved, as the non-deterministic path permits.
        auto write_selected_idx = [&](const key_t& key, offset_t seg_idx) {
          const auto res = identify_op(key);
          if (res == detail::topk::candidate_class::selected)
          {
            const out_offset_t pos = atomicAdd(&leader_state->out_cnt, out_offset_t{1});
            block_keys_out[pos]    = key;
            write_value(pos, seg_idx);
          }
          else if (res == detail::topk::candidate_class::candidate)
          {
            const out_offset_t back_pos = atomicAdd(&leader_state->out_back_cnt, out_offset_t{1});
            if (back_pos < num_kth)
            {
              const out_offset_t pos = k - 1 - back_pos;
              block_keys_out[pos]    = key;
              write_value(pos, seg_idx);
            }
          }
        };

        // Iterate a contiguous run of `count` keys whose element `local` has segment-local index `base_off + local`.
        // `get_key(local)` reads the key (from SMEM for resident chunks, from gmem for overflow chunks).
        auto write_run = [&](auto get_key, offset_t base_off, int count) {
          const int iterations = ::cuda::ceil_div(count, threads_per_block);
          _CCCL_PRAGMA_UNROLL(tie_break_items_per_thread)
          for (int j = 0; j < iterations; ++j)
          {
            const int local = j * threads_per_block + static_cast<int>(threadIdx.x);
            if (local < count)
            {
              write_selected_idx(get_key(local), base_off + static_cast<offset_t>(local));
            }
          }
        };

        // Fold the resident keys (and their values) as the streamer's `mid` work, exactly as in the keys-only path.
        const auto fold_resident = [&] {
          if constexpr (use_block_load_to_shared)
          {
            // Resident keys are densely packed in slot order, so a running cursor recovers per-chunk spans. The last
            // slot holds the forced-resident tail when applicable.
            key_t* const rk = reinterpret_cast<key_t*>(__cvta_shared_to_generic(resident_smem32));
            int cursor      = 0;
            for (offset_t s = 0; s < my_resident_chunks; ++s)
            {
              const offset_t rl       = (force_tail_resident && s == my_resident_chunks - offset_t{1}) ? tail_local : s;
              const auto chunk        = get_chunk(part.global_index(rl), segment_size_u32, head_items);
              const offset_t base_off = chunk.offset;
              const int cc            = chunk.count;
              write_run(
                [&](int local) {
                  return rk[cursor + local];
                },
                base_off,
                cc);
              cursor += cc;
            }
          }
          else
          {
            for (offset_t p = 0; p < my_resident_chunks; ++p)
            {
              const auto chunk        = get_chunk(part.global_index(p), segment_size_u32, head_items);
              const offset_t base_off = chunk.offset;
              key_t* const chunk_keys = slot_keys_unpadded(static_cast<int>(p));
              write_run(
                [&](int local) {
                  return chunk_keys[local];
                },
                base_off,
                chunk.count);
            }
          }
          // Scan the persistent boundary edges with their segment-local indices: the head prefix starts at index 0, the
          // peeled tail suffix at `segment_size - tail_edge_len`. Value fetched per key.
          if constexpr (use_block_load_to_shared)
          {
            if (head_edge_len > 0)
            {
              write_run(
                [&](int local) {
                  return temp_storage.edge_keys[local];
                },
                offset_t{0},
                head_edge_len);
            }
            if (tail_edge_len > 0)
            {
              write_run(
                [&](int local) {
                  return temp_storage.edge_keys[head_edge_cap + local];
                },
                segment_size_u32 - static_cast<offset_t>(tail_edge_len),
                tail_edge_len);
            }
          }
        };

        // Overflow chunks: reuse the streamed keys (the generic fallback re-reads from gmem) and fetch each selected
        // key's value at index `seg_idx`. The resident keys above fold in as the streamer's `mid` work.
        streamer.process_pass_indexed<tie_break_items_per_thread>(write_selected_idx, fold_resident);
      }
    }

    // Final cluster barrier: hold every block in the cluster until all DSMEM
    // atomics into the leader's state are complete. Without this, a fast
    // block (e.g. one whose block_tile is entirely padding) can return while another
    // block is still writing to leader-resident memory through DSMEM, which
    // surfaces as a "cluster target block not present" exception. A lone CTA has no remote writers, so a block barrier
    // suffices (and orders its own final writes).
    cluster_or_block_sync(cluster, single_cta);
  }

  // Copies an entire segment `input[i] -> output[i]` (keys, plus values for pairs) for the select-all fast path
  // (`k == segment_size`); top-k output order is unspecified, so a straight copy is valid. The whole launched cluster
  // grid-strides the segment with no chunk partition or cluster barriers -- each CTA copies its part and exits. Each
  // thread issues all `copy_items` loads of a step before its stores to keep many requests in flight (ILP/MLP). Input
  // and output bases are mutually unaligned in general, so we use per-element (not vectorized/TMA) coalesced accesses.
  _CCCL_DEVICE _CCCL_FORCEINLINE void copy_segment_select_all(
    num_segments_val_t segment_id,
    segment_size_val_t segment_size,
    unsigned int cluster_rank,
    unsigned int cluster_blocks)
  {
    constexpr int copy_items    = 8;
    const offset_t n            = static_cast<offset_t>(segment_size);
    const offset_t cluster_tid  = cluster_rank * static_cast<offset_t>(blockDim.x) + threadIdx.x;
    const offset_t cluster_tids = cluster_blocks * static_cast<offset_t>(blockDim.x);
    const offset_t step         = cluster_tids * static_cast<offset_t>(copy_items);
    auto keys_in_it             = d_key_segments_it[segment_id];
    auto keys_out_it            = d_key_segments_out_it[segment_id];
    for (offset_t tile = 0; tile < n; tile += step)
    {
      // `idx[]` stays in `offset_t` (not the possibly-narrower `segment_size_val_t`) so the `idx[j] < n` bound check
      // happens before any narrowing; the cast at each indexing site is then safe.
      offset_t idx[copy_items];
      key_t keys[copy_items];
      [[maybe_unused]] value_t vals[copy_items];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < copy_items; ++j)
      {
        idx[j] = tile + static_cast<offset_t>(j) * cluster_tids + cluster_tid;
      }
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < copy_items; ++j)
      {
        if (idx[j] < n)
        {
          keys[j] = keys_in_it[static_cast<segment_size_val_t>(idx[j])];
        }
      }
      if constexpr (!is_keys_only)
      {
        auto vals_in_it = d_value_segments_it[segment_id];
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < copy_items; ++j)
        {
          if (idx[j] < n)
          {
            vals[j] = vals_in_it[static_cast<segment_size_val_t>(idx[j])];
          }
        }
      }
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < copy_items; ++j)
      {
        if (idx[j] < n)
        {
          keys_out_it[static_cast<segment_size_val_t>(idx[j])] = keys[j];
        }
      }
      if constexpr (!is_keys_only)
      {
        auto vals_out_it = d_value_segments_out_it[segment_id];
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < copy_items; ++j)
        {
          if (idx[j] < n)
          {
            vals_out_it[static_cast<segment_size_val_t>(idx[j])] = vals[j];
          }
        }
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void process_impl()
  {
    ::cooperative_groups::cluster_group cluster = ::cooperative_groups::this_cluster();
    const unsigned int cluster_rank             = cluster.block_rank();
    // Runtime cluster blocks match the launch attribute the dispatch passed
    // to `cudaLaunchKernelExC` (or the kernel's `__cluster_dims__` on CDP).
    const unsigned int cluster_blocks = cluster.num_blocks();
    const auto segment_id             = static_cast<num_segments_val_t>(blockIdx.x / cluster_blocks);

    if (segment_id >= detail::params::get_param(num_segments, num_segments_val_t{0}))
    {
      return;
    }

    const auto segment_size = static_cast<segment_size_val_t>(detail::params::get_param(segment_sizes, segment_id));
    const auto k_requested  = static_cast<out_offset_t>(detail::params::get_param(k_param, segment_id));
    const auto k =
      static_cast<out_offset_t>((::cuda::std::min) (static_cast<segment_size_val_t>(k_requested), segment_size));

    if (k == 0)
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
      const bool fits_single_cta = single_cta_eligible(
        static_cast<::cuda::std::uint64_t>(segment_size),
        static_cast<::cuda::std::uint64_t>(block_tile_capacity),
        single_cta_max_segment_size);
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
    const bool single_cta = (eff_cluster_blocks == 1u);

    // Every block's thread 0 initializes its local `state`. Only the
    // leader's copy is semantically read (non-leaders reach the cluster
    // state through `leader_state`), but mirroring the writes everywhere
    // keeps the scan-then-reduce path's unconditional `state.k` load
    // safe under compute-sanitizer.
    if (threadIdx.x == 0)
    {
      temp_storage.state.len          = static_cast<offset_t>(segment_size);
      temp_storage.state.k            = k;
      temp_storage.state.kth_bucket   = 0;
      temp_storage.state.out_cnt      = 0;
      temp_storage.state.out_back_cnt = 0;
      temp_storage.state.early_stop   = 0;
    }
    reset_hist();
    cluster_or_block_sync(cluster, single_cta);

    [[maybe_unused]] const bool ok = detail::params::dispatch_discrete(
      select_directions,
      segment_id,
      [this, &cluster, segment_id, eff_cluster_rank, eff_cluster_blocks, segment_size, k](auto direction_tag) {
        constexpr detail::topk::select Direction = decltype(direction_tag)::value;
        this->template run<Direction>(cluster, segment_id, eff_cluster_rank, eff_cluster_blocks, segment_size, k);
      });
    _CCCL_ASSERT(ok, "Unsupported select direction for cluster top-k");
  }
#endif // _CG_HAS_CLUSTER_GROUP
};
} // namespace detail::batched_topk_cluster

CUB_NAMESPACE_END
