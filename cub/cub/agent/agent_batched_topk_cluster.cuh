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
#include <cuda/__ptx/instructions/cp_async_bulk.h>
#include <cuda/__ptx/instructions/mbarrier_arrive.h>
#include <cuda/__ptx/instructions/mbarrier_init.h>
#include <cuda/__ptx/instructions/mbarrier_wait.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
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
  static constexpr int chunk_items      = ChunkBytes / int{sizeof(KeyT)};
  static constexpr int load_align_items = LoadAlignBytes / int{sizeof(KeyT)};
  static constexpr int slot_alignment =
    (::cuda::std::max) (LoadAlignBytes, detail::LoadToSharedBufferAlignBytes<KeyT>());
  // Each chunk maps to exactly one slot of ChunkBytes: boundary chunks are loaded as an aligned bulk plus a small
  // hand-rolled edge, so no chunk ever needs the scalar-load guard headroom.
  static constexpr int slot_stride_bytes  = ::cuda::round_up(ChunkBytes, slot_alignment); // == ChunkBytes normally
  static constexpr int base_padding_bytes = (alignof(KeyT) > 16) ? slot_alignment : 0;
  static_assert(chunk_items >= load_align_items, "ChunkBytes must hold at least one load_align unit");
  // The aligned full-chunk buffer (no guard, since LoadAlignBytes >= bulk_copy_min_align) is exactly its byte count
  // and must fit one slot.
  static_assert(detail::LoadToSharedBufferSizeBytes<KeyT, LoadAlignBytes>(chunk_items) <= slot_stride_bytes,
                "An aligned full chunk must fit one slot");

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
  static constexpr int load_align_items  = smem_layout_t::load_align_items;
  static constexpr int slot_alignment    = smem_layout_t::slot_alignment;
  static constexpr int slot_stride_bytes = smem_layout_t::slot_stride_bytes;

  static_assert(PipelineStages > 0);
  static_assert(ChunkBytes > 0);
  static_assert(LoadAlignBytes > 0);
  static_assert(ChunkBytes % LoadAlignBytes == 0);
  static_assert(LoadAlignBytes % int{sizeof(key_t)} == 0);
  // The hybrid load relies on the aligned bulk-copy path being exact (no scalar guard), which requires the load
  // alignment to be at least the bulk-copy minimum alignment.
  static_assert(LoadAlignBytes >= detail::bulk_copy_min_align, "LoadAlignBytes must be >= bulk_copy_min_align");
  static_assert(chunk_items > 0);

  using decomposer_t = detail::identity_decomposer_t;
  // Resident/streamed chunks are pulled into the block_tile with elected-thread `cp.async.bulk`/TMA copies against
  // raw per-stage mbarriers; see the async bulk-copy pipeline helpers below.

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
    // One mbarrier handle per pipeline stage, shared by the resident load and the overflow streamer and reused
    // (ping-ponged) across radix passes; initialized once by `init_load_barriers`.
    ::cuda::std::uint64_t load_mbar[PipelineStages];
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

  // Item count of the near-full head chunk (chunk 0) for an unaligned base. The unaligned prefix (`head_items`) plus
  // an aligned bulk filling the rest of the slot up to a `load_align` boundary, so chunk 0 ends aligned and every
  // later chunk begins aligned. Clamped to the segment for tiny segments.
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE offset_t
  head_chunk_items(offset_t segment_size, offset_t head_items) const
  {
    const offset_t bulk0 = offset_t{chunk_items} - offset_t{load_align_items};
    return (::cuda::std::min) (segment_size, head_items + bulk0);
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE offset_t num_chunks(offset_t segment_size, offset_t head_items) const
  {
    if (head_items == 0)
    {
      return static_cast<offset_t>(::cuda::ceil_div(segment_size, offset_t{chunk_items}));
    }
    const offset_t head0     = head_chunk_items(segment_size, head_items);
    const offset_t remaining = segment_size - head0;
    return offset_t{1} + static_cast<offset_t>(::cuda::ceil_div(remaining, offset_t{chunk_items}));
  }

  struct chunk_desc
  {
    offset_t offset;
    int count;
  };

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE chunk_desc
  get_chunk(offset_t chunk_idx, offset_t segment_size, offset_t head_items) const
  {
    offset_t offset;
    if (head_items != 0)
    {
      const offset_t head0 = head_chunk_items(segment_size, head_items);
      if (chunk_idx == 0)
      {
        return {offset_t{0}, static_cast<int>(head0)};
      }
      offset = head0 + (chunk_idx - 1) * offset_t{chunk_items};
    }
    else
    {
      offset = chunk_idx * offset_t{chunk_items};
    }
    const offset_t remaining = segment_size - offset;
    return {offset, static_cast<int>((::cuda::std::min) (remaining, offset_t{chunk_items}))};
  }

  // Splits a chunk into its unaligned front edge, aligned interior (bulk), and unaligned back edge, relative to the
  // gmem base. The interior begins and ends on a `load_align` boundary so it can be loaded with the aligned,
  // guard-free BlockLoadToShared path; the edges (each `< load_align_items` items) are loaded with `copy_edge`. Only
  // chunk 0 (head) can have a nonzero prefix and only the last chunk (tail) a nonzero suffix.
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
    const auto aligned_end   = (end / la) * la;
    if (aligned_begin > aligned_end)
    {
      // The chunk lies strictly between two load_align boundaries (no aligned point inside): the whole chunk is an
      // unaligned edge. This only happens for a tiny chunk with an unaligned begin (the head or a single-chunk
      // segment), so attributing it entirely to the front edge is correct. A tail always begins on a boundary, so it
      // takes the `aligned_begin <= aligned_end` path below and its unaligned remainder becomes the suffix.
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

  // Hand-rolled per-thread copy of a small (`< load_align_items` items) unaligned edge from gmem to smem. Used for
  // the head prefix and tail suffix, which cannot go through the aligned (16-byte-aligned dst, guard-free)
  // BlockLoadToShared path. Each thread copies the same indices it later reads in the first-pass histogram, so no
  // extra synchronization is needed for the edge's own pass.
  //
  // TODO(cccl): an asymmetric-alignment BlockLoadToShared API (independent begin/end alignment, e.g. an aligned begin
  // with an arbitrary end) would let a boundary chunk be loaded with a single aligned-bulk + in-place edge call and
  // remove these hand-rolled copies entirely.
  _CCCL_DEVICE _CCCL_FORCEINLINE void copy_edge(key_t* dst, const key_t* src, int count) const
  {
    for (int local = static_cast<int>(threadIdx.x); local < count; local += threads_per_block)
    {
      dst[local] = src[local];
    }
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
      // The TMA path requires `bulk_copy_min_align` (>= 16); the cursor carries the stronger `load_align_bytes`.
      _CCCL_ASSERT(::cuda::is_aligned(dst, detail::bulk_copy_min_align),
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
  _CCCL_DEVICE _CCCL_FORCEINLINE void hist_fold_remote(::cuda::std::uint32_t own_bucket_addr32, offset_t v)
  {
    ::cuda::std::uint32_t remote;
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(own_bucket_addr32), "n"(0));
    asm volatile("red.relaxed.cluster.shared::cluster.add.u32 [%0], %1;" : : "r"(remote), "r"(v) : "memory");
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

  // ---------------------------------------------------------------------------
  // Overflow streamer
  // ---------------------------------------------------------------------------
  // Re-streams the per-rank "overflow" chunks (those that do not fit in the
  // resident SMEM region) from gmem through a small, fixed, round-robin set of
  // `PipelineStages` streaming slots. The same object is reused for every radix
  // pass and the final filter. It ping-pongs the iteration order across calls so
  // the `PipelineStages` boundary chunks that one pass leaves resident in the
  // streaming slots are reused by the next pass with no reload; in the limit
  // where the overflow fits entirely in the streaming slots (`overflow <=
  // PipelineStages`), the chunks are loaded once and never reloaded. The
  // resident region is unaffected: it lives in the slots `[0, resident_slots)`,
  // the streaming region in `[stream_slot_base, stream_slot_base +
  // PipelineStages)`.
  struct overflow_streamer
  {
    agent_batched_topk_cluster& agent;
    key_it_t block_keys_in;
    const key_t* block_keys_base; // unwrapped contiguous base (pipeline path only; null otherwise)
    offset_t segment_size;
    offset_t head_items;
    unsigned int cluster_rank;
    unsigned int cluster_size;
    offset_t resident_chunks; // number of rank-local chunks kept resident
    offset_t overflow_base; // rank-local chunk index of the first overflow (streamed) chunk
    int stream_slot_base; // SMEM slot index at which the streaming region begins
    offset_t overflow_chunks; // number of overflow chunks for this rank (M)
    int p_eff; // active streaming depth = min(PipelineStages, M) (>= 1)
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
      unsigned int cluster_rank_,
      unsigned int cluster_size_,
      offset_t resident_chunks_,
      offset_t overflow_base_,
      int stream_slot_base_,
      offset_t my_chunks_)
        : agent(agent_)
        , block_keys_in(block_keys_in_)
        , block_keys_base(block_keys_base_)
        , segment_size(segment_size_)
        , head_items(head_items_)
        , cluster_rank(cluster_rank_)
        , cluster_size(cluster_size_)
        , resident_chunks(resident_chunks_)
        , overflow_base(overflow_base_)
        , stream_slot_base(stream_slot_base_)
        , overflow_chunks((my_chunks_ > resident_chunks_) ? (my_chunks_ - resident_chunks_) : offset_t{0})
    {
      const int m = static_cast<int>((::cuda::std::min) (overflow_chunks, static_cast<offset_t>(PipelineStages)));
      p_eff       = (m > 0) ? m : 1;
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE offset_t chunk_index_of(offset_t overflow_idx) const
    {
      return cluster_rank + (overflow_base + overflow_idx) * static_cast<offset_t>(cluster_size);
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE void issue_load(int stage, offset_t overflow_idx)
    {
      const offset_t chunk_idx = chunk_index_of(overflow_idx);
      const auto chunk         = agent.get_chunk(chunk_idx, segment_size, head_items);
      // The boundary chunks (unaligned head and tail) are kept resident, so every streamed chunk is fully aligned
      // and uses the guard-free aligned (TMA bulk) path.
      _CCCL_ASSERT(agent.is_aligned_chunk(block_keys_base, chunk), "overflow streamer received an unaligned chunk");
      char* const dst = agent.key_slots + (stream_slot_base + stage) * slot_stride_bytes;
      const ::cuda::std::span<const key_t> src{
        block_keys_base + chunk.offset, static_cast<::cuda::std::size_t>(chunk.count)};
      agent.issue_bulk_copy(stage, dst, src);
      inflight_mask |= (::cuda::std::uint32_t{1} << stage);
    }

    // Rebuild the shared span for the chunk currently resident in `stage`'s slot without storing per-stage state: the
    // slot address is a pure function of `stage` and the length is recomputed from chunk index `o`, so there is no
    // spillable `pending[]` array (see `bulk_span`).
    [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::span<key_t> stage_span(int stage, offset_t o) const
    {
      char* const dst  = agent.key_slots + (stream_slot_base + stage) * slot_stride_bytes;
      const auto chunk = agent.get_chunk(chunk_index_of(o), segment_size, head_items);
      return agent.bulk_span({static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(dst)), chunk.count});
    }

    // Apply `f` to every overflow key once, in the current ping-pong direction.
    template <typename F>
    _CCCL_DEVICE _CCCL_FORCEINLINE void process_pass(F&& f)
    {
      if (overflow_chunks == 0)
      {
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

        for (offset_t i = 0; i < m; ++i)
        {
          const offset_t o = forward ? i : (m - 1 - i);
          const int stage  = static_cast<int>(o % pe);
          if (inflight_mask & (::cuda::std::uint32_t{1} << stage))
          {
            agent.wait_stage(stage);
            inflight_mask &= ~(::cuda::std::uint32_t{1} << stage);
          }
          agent.for_each_chunk_key(stage_span(stage, o), f);

          // Prefetch the chunk `p_eff` visits ahead in this direction. It maps
          // to the slot we just finished, so a barrier is required before the
          // async copy can overwrite the data the block was just reading.
          const offset_t ni = i + pe;
          if (ni < m)
          {
            const offset_t no = forward ? ni : (m - 1 - ni);
            __syncthreads();
            issue_load(stage, no);
          }
        }
        forward = !forward;
      }
      else
      {
        // Generic fallback: overflow keys are read straight from gmem each pass
        // (no SMEM reuse), but the walk still snakes for L2 locality.
        for (offset_t i = 0; i < m; ++i)
        {
          const offset_t o         = forward ? i : (m - 1 - i);
          const offset_t chunk_idx = chunk_index_of(o);
          const auto chunk         = agent.get_chunk(chunk_idx, segment_size, head_items);
          const int iterations     = ::cuda::ceil_div(chunk.count, threads_per_block);
          detail::transform::unrolled_for<UnrollFactor>(iterations, [&](int j) {
            const int local = j * threads_per_block + static_cast<int>(threadIdx.x);
            if (local < chunk.count)
            {
              const key_t key =
                block_keys_in[static_cast<segment_size_val_t>(chunk.offset + static_cast<offset_t>(local))];
              f(key);
            }
          });
        }
        forward = !forward;
      }
    }
  };

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

    // DSMEM pointer into the leader block's shared memory. The Step 2 histogram fold reaches the leader's `hist`
    // through a `mapa`-formed `shared::cluster` address instead (see `hist_fold_remote`).
    state_t* leader_state = cluster.map_shared_rank(&temp_storage.state, 0);

    // Per-block local copy of `kth_key_bits` so each key check hits the
    // block's own SMEM rather than DSMEM during the histogram loop.
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
    const offset_t chunks    = num_chunks(segment_size_u32, head_items);
    const offset_t my_chunks = num_rank_chunks(chunks, cluster_rank, cluster_size);

    // Resident vs. streaming split. Segments that fit the all-resident coverage behave exactly as before
    // (`resident_slots_cap == full_slots`, no streaming). Larger segments reserve the last `PipelineStages` slots of
    // the block_tile as a round-robin streaming region and keep `full_slots - PipelineStages` slots resident; the
    // overflow chunks are re-streamed from gmem on every pass by `streamer`. The launch coverage check still reserves
    // one extra chunk for the possible unaligned head.
    const offset_t full_slots = block_tile_capacity / static_cast<offset_t>(chunk_items);
    const offset_t all_resident_capacity =
      smem_layout_t::template cluster_tile_capacity<offset_t>(static_cast<int>(cluster_size), block_tile_capacity);
    const bool needs_streaming = segment_size_u32 > all_resident_capacity;
    _CCCL_ASSERT(!needs_streaming || full_slots > static_cast<offset_t>(PipelineStages),
                 "block_tile too small to reserve a streaming region");
    const offset_t resident_slots_cap =
      needs_streaming
        ? ((full_slots > static_cast<offset_t>(PipelineStages))
             ? full_slots - static_cast<offset_t>(PipelineStages)
             : offset_t{1})
        : full_slots;
    const offset_t my_resident_chunks = (::cuda::std::min) (my_chunks, resident_slots_cap);
    // Resident chunks stay within the first `resident_slots_cap` slots; the streaming region occupies the slots
    // `[resident_slots_cap, full_slots)`, so both regions live inside the allocated block_tile buffer.
    _CCCL_ASSERT(my_resident_chunks * offset_t{chunk_items} <= resident_slots_cap * offset_t{chunk_items},
                 "Dynamic shared memory block_tile is too small");

    // Boundary chunks (the unaligned head = chunk 0 and the unaligned tail = chunk chunks-1) carry hand-rolled edges
    // and are kept resident so the overflow streamer only ever sees fully-aligned middle chunks. The head is already
    // resident (local index 0 of its owning rank). The tail is forced into its owner's resident set - occupying the
    // last resident slot - when this rank owns it, it has an unaligned end, and it would otherwise stream. The
    // overflow then begins at `overflow_base`, shifted back by one in the forced case so the tail is skipped while the
    // middle chunk it displaces is streamed instead.
    const offset_t overflow_count = (my_chunks > my_resident_chunks) ? (my_chunks - my_resident_chunks) : offset_t{0};
    offset_t tail_local           = offset_t{0};
    bool force_tail_resident      = false;
    if constexpr (use_block_load_to_shared)
    {
      if (overflow_count > 0
          && (chunks - offset_t{1}) % static_cast<offset_t>(cluster_size) == static_cast<offset_t>(cluster_rank))
      {
        tail_local                 = (chunks - offset_t{1}) / static_cast<offset_t>(cluster_size);
        const auto tail_chunk      = get_chunk(chunks - offset_t{1}, segment_size_u32, head_items);
        const bool tail_has_suffix = split_chunk(block_keys_base, tail_chunk).suffix != offset_t{0};
        force_tail_resident        = tail_has_suffix && (tail_local >= my_resident_chunks);
      }
    }
    const offset_t overflow_base = force_tail_resident ? (my_resident_chunks - offset_t{1}) : my_resident_chunks;
    // A rank that owns both boundary chunks (the head on rank 0 and a forced tail) needs two distinct resident slots.
    _CCCL_ASSERT(!(force_tail_resident && head_items != 0 && cluster_rank == 0) || my_resident_chunks >= offset_t{2},
                 "streaming needs >= 2 resident slots to keep both head and tail resident");

    // Persistent streamer for the overflow chunks; a no-op (constructs nothing) when this rank has no overflow.
    overflow_streamer streamer(
      *this,
      block_keys_in,
      block_keys_base,
      segment_size_u32,
      head_items,
      cluster_rank,
      cluster_size,
      my_resident_chunks,
      overflow_base,
      static_cast<int>(resident_slots_cap),
      my_chunks);

    ::cuda::std::span<key_t> resident_keys;
    // 32-bit shared-window address of `resident_keys.data()`. The resident span is read once per radix pass and in
    // the final filter; a 64-bit generic pointer kept live across that loop spills and reloads as a *generic*
    // pointer (`LDL.64`), which demotes every key read from `LDS` to a generic `LD`. Carrying the base as a 32-bit
    // shared address and rebuilding the pointer with `__cvta_shared_to_generic` at each use keeps the reads `LDS`
    // even when the value spills (the cvta intrinsic re-confers shared provenance; a spilled 64-bit generic pointer
    // cannot be re-anchored after the fact).
    ::cuda::std::uint32_t resident_smem32 = 0;

    reset_hist();
    if constexpr (use_block_load_to_shared)
    {
      // Arm the stage barriers once; reused (ping-ponged) by the resident load and the overflow streamer across passes.
      init_load_barriers();
    }
    __syncthreads();

    {
      extract_bin_op_t extract_op(0, total_bits, decomposer_t{});
      const ::cuda::std::uint32_t hist_smem32 = hist_base32();
      const bool is_leader_block              = cluster_rank == 0;
      auto add_first_pass                     = [&](const key_t& key) {
        const int bucket = extract_op(key);
        hist_inc(hist_smem32, bucket, is_leader_block);
      };

      if constexpr (use_block_load_to_shared)
      {
        if (my_resident_chunks > 0)
        {
          // Stage mbarriers and the `load_phase` parity are shared with the streamer (no per-chunk token array needed).
          // Chunks are written densely in slot order and read back in the same order, so the read cursor (`read_off`)
          // mirrors the write cursor (`next_off`) as a running prefix sum, avoiding a dynamically-indexed
          // `pending_spans` array that would anchor surrounding state to local memory. Each chunk loads its aligned
          // bulk only (boundary edges are filled afterwards); the read span is rebuilt from a rooted 32-bit shared
          // address (see `bulk_span`) so a spilled cursor cannot demote the first-pass reads from `LDS` to `LD`.
          const int prologue = (::cuda::std::min) (PipelineStages, static_cast<int>(my_resident_chunks));

          // Resident slot -> rank-local chunk index. Identity, except the last slot holds the forced-resident tail.
          const auto resident_local = [&](offset_t slot) -> offset_t {
            return (force_tail_resident && slot == my_resident_chunks - offset_t{1}) ? tail_local : slot;
          };
          // Aligned bulk of the resident chunk in `slot`; empty when the chunk has no aligned interior.
          const auto bulk_src = [&](offset_t slot) -> ::cuda::std::span<const key_t> {
            const offset_t chunk_idx =
              static_cast<offset_t>(cluster_rank) + resident_local(slot) * static_cast<offset_t>(cluster_size);
            const auto chunk = get_chunk(chunk_idx, segment_size_u32, head_items);
            const auto split = split_chunk(block_keys_base, chunk);
            if (split.bulk == 0)
            {
              return {};
            }
            return {block_keys_base + chunk.offset + split.prefix, static_cast<::cuda::std::size_t>(split.bulk)};
          };

          // First resident slot's unaligned front edge (the head prefix on rank 0). Reserve an aligned-up gap in
          // front of its bulk so the bulk stays 16-aligned and the edge sits contiguously right before it.
          const auto first_chunk = get_chunk(
            static_cast<offset_t>(cluster_rank) + resident_local(offset_t{0}) * static_cast<offset_t>(cluster_size),
            segment_size_u32,
            head_items);
          const auto first_split  = split_chunk(block_keys_base, first_chunk);
          const int front_edge    = static_cast<int>(first_split.prefix);
          const int sba           = detail::LoadToSharedBufferAlignBytes<key_t>();
          const int front_bytes   = front_edge * int{sizeof(key_t)};
          const int head_bulk_off = ::cuda::round_up(front_bytes, sba);
          // Write cursor as a byte offset from `key_slots`; the destination is always `key_slots + offset` (rooted at
          // the extern-shared base) so it keeps shared address-space provenance.
          const int resident_begin_off = head_bulk_off - front_bytes;
          int next_off                 = head_bulk_off;

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
          int read_off = head_bulk_off;
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

          // Head prefix: copy into the reserved front gap, contiguous right before the first bulk.
          if (front_edge > 0)
          {
            key_t* const edge_dst = reinterpret_cast<key_t*>(key_slots + resident_begin_off);
            copy_edge(edge_dst, block_keys_base + first_chunk.offset, front_edge);
            for_each_chunk_key({edge_dst, static_cast<::cuda::std::size_t>(front_edge)}, add_first_pass);
          }

          // Tail suffix: append right after the last (tail) bulk. Nothing follows, so its non-16 length is harmless.
          const offset_t last_slot = my_resident_chunks - offset_t{1};
          const auto last_chunk    = get_chunk(
            static_cast<offset_t>(cluster_rank) + resident_local(last_slot) * static_cast<offset_t>(cluster_size),
            segment_size_u32,
            head_items);
          const auto last_split = split_chunk(block_keys_base, last_chunk);
          if (last_split.suffix > 0)
          {
            key_t* const edge_dst = reinterpret_cast<key_t*>(key_slots + next_off);
            copy_edge(edge_dst,
                      block_keys_base + last_chunk.offset + last_split.prefix + last_split.bulk,
                      static_cast<int>(last_split.suffix));
            for_each_chunk_key({edge_dst, static_cast<::cuda::std::size_t>(last_split.suffix)}, add_first_pass);
            next_off += static_cast<int>(last_split.suffix) * int{sizeof(key_t)};
          }

          // The resident region is one contiguous span [head edge | bulks | tail edge] for the later passes.
          resident_keys = {reinterpret_cast<key_t*>(key_slots + resident_begin_off),
                           static_cast<::cuda::std::size_t>((next_off - resident_begin_off) / int{sizeof(key_t)})};
          resident_smem32 =
            static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(key_slots + resident_begin_off));
        }
      }
      else
      {
        for (offset_t p = 0; p < my_resident_chunks; ++p)
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

      // Fold the overflow chunks into the first-pass histogram. Forward direction; primes the streaming slots. The
      // streamer reuses the resident load's stage barriers (no re-init); `wait_stage` provides the producer/consumer
      // sync.
      streamer.process_pass(add_first_pass);

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

        // Step 1: block-private histogram via shared-space `red` (see `hist_inc`): leader uses cluster scope to be
        // mutually atomic with the non-leaders' Step 2 DSMEM folds, non-leaders use the cheaper cta scope.
        const ::cuda::std::uint32_t hist_smem32 = hist_base32();
        const bool is_leader_block              = cluster_rank == 0;
        auto add_hist                           = [&](const key_t& key) {
          if (identify_op(key) == detail::topk::candidate_class::candidate)
          {
            const int bucket = extract_op(key);
            hist_inc(hist_smem32, bucket, is_leader_block);
          }
        };

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
            const offset_t chunk_idx = static_cast<offset_t>(cluster_rank) + p * static_cast<offset_t>(cluster_size);
            const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
            key_t* const chunk_keys  = slot_keys_unpadded(static_cast<int>(p));
            for_each_chunk_key({chunk_keys, static_cast<::cuda::std::size_t>(chunk.count)}, add_hist);
          }
        }

        // Re-stream the overflow chunks into this pass's histogram. Ping-pongs direction and reuses the boundary
        // chunks left resident by the previous pass.
        streamer.process_pass(add_hist);
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
      // leader skips this to avoid double-counting its own contribution.
      if (cluster_rank != 0)
      {
        const ::cuda::std::uint32_t hist_smem32 = hist_base32();
        for (int i = static_cast<int>(threadIdx.x); i < num_buckets; i += threads_per_block)
        {
          const offset_t v = temp_storage.hist[i];
          if (v != 0)
          {
            hist_fold_remote(hist_smem32 + static_cast<::cuda::std::uint32_t>(i) * sizeof(offset_t), v);
          }
        }
      }

      cluster.sync();

      // Step 3: the leader walks the merged `hist` (raw counts in the
      // reduce-then-scan path, cluster-wide inclusive scan in the
      // scan-then-reduce path) and updates the cluster-shared `state`.
      // Subsequent reads (next-pass refresh, last filter) all observe
      // these writes after the next cluster sync.
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
      key_t* const rk = reinterpret_cast<key_t*>(__cvta_shared_to_generic(resident_smem32));
      for_each_chunk_key({rk, static_cast<::cuda::std::size_t>(span_size(resident_keys))}, write_selected);
    }
    else
    {
      for (offset_t p = 0; p < my_resident_chunks; ++p)
      {
        const offset_t chunk_idx = static_cast<offset_t>(cluster_rank) + p * static_cast<offset_t>(cluster_size);
        const auto chunk         = get_chunk(chunk_idx, segment_size_u32, head_items);
        key_t* const chunk_keys  = slot_keys_unpadded(static_cast<int>(p));
        for_each_chunk_key({chunk_keys, static_cast<::cuda::std::size_t>(chunk.count)}, write_selected);
      }
    }
    streamer.process_pass(write_selected);

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
