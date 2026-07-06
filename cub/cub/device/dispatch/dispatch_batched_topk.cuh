// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. SPDX-License-Identifier:
// Apache-2.0 WITH LLVM-exception

//! @file
//! Internal device-wide dispatch for cub::DeviceBatchedTopK: selects between the baseline (worker-per-segment) and
//! cluster (SM 9.0+) backends and launches them through a single kernel symbol.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_batched_topk_cluster.cuh>
#include <cub/detail/cc_dispatch.cuh>
#include <cub/detail/choose_offset.cuh>
#include <cub/detail/env_dispatch.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_batched_topk.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh>
#include <cub/util_device.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/round_up.h>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__iterator/transform_iterator.h>
#include <cuda/__numeric/narrow.h>
#include <cuda/argument>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <nv/target>

#include <cuda_runtime.h>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// -----------------------------------------------------------------------------
// Internal: wrap the compile-time select direction into a discrete param for dispatch
// -----------------------------------------------------------------------------

// The selection direction is compile-time only: callers pass `::cuda::args::constant<Dir>`, which maps to a
// value-less static_discrete_param. Because the direction is fixed at compile time and carries no runtime value, it
// can never disagree with its only supported option, so dispatch can never silently degrade to a no-op.
template <detail::topk::select Dir, class _Tp>
[[nodiscard]] _CCCL_HOST_DEVICE auto wrap_select_direction(::cuda::args::constant<Dir, _Tp>)
{
  return params::static_discrete_param<detail::topk::select, Dir>{};
}

// The selection direction is intentionally a compile-time constant: only `::cuda::args::constant<Dir>` is
// accepted (the overload above maps it to a value-less static_discrete_param). This catch-all documents that
// deliberate limitation and rejects anything else (e.g. a runtime `detail::topk::select` or a per-segment iterator of
// directions) with a clear diagnostic. It is an intent/documentation guard rather than a user-facing one: callers
// reach the algorithm through the min/max device entry points (DeviceBatchedTopK::{Max,Min}{Keys,Pairs}), which
// construct the matching `constant<Dir>` internally, so `dispatch` is only ever invoked with a direction we create.
template <typename SelectDirectionT>
[[nodiscard]] _CCCL_HOST_DEVICE auto wrap_select_direction(SelectDirectionT)
{
  static_assert(::cuda::std::__always_false_v<SelectDirectionT>,
                "DeviceBatchedTopK currently supports only compile-time selection directions: the min/max entry "
                "points (DeviceBatchedTopK::{Max,Min}{Keys,Pairs}) dispatch with a "
                "::cuda::args::constant<Dir>; runtime or per-segment directions are "
                "intentionally not supported");
  // Unreachable (the static_assert above always fires); keeps the return type well-formed so the only diagnostic is
  // the message above.
  return params::static_discrete_param<detail::topk::select, detail::topk::select::min>{};
}

// -----------------------------------------------------------------------------
// Helper: turn a segment ID into the number of large-segment-agent tiles needed
// to cover that segment. Wrapped in a transform_iterator, this produces the
// per-segment tile counts that we exclusive-scan to obtain per-segment tile
// offsets.
// -----------------------------------------------------------------------------
template <class SegmentSizeParameterT, class TotalNumItemsValueType>
struct segment_size_to_tile_count_op
{
  SegmentSizeParameterT segment_sizes;
  int large_segment_agent_tile_size;

  template <typename SegmentIndexT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr TotalNumItemsValueType operator()(SegmentIndexT segment_id) const
  {
    return static_cast<TotalNumItemsValueType>(
      ::cuda::ceil_div(params::get_param(segment_sizes, segment_id), large_segment_agent_tile_size));
  }
};

// -----------------------------------------------------------------------------
// Segmented Top-K Dispatch
// -----------------------------------------------------------------------------
//
// NOTE: `baseline_dispatch` / `baseline_dispatch_with_env` are the legacy baseline-only entry points. The public API
// now routes through the unified `dispatch` (baseline + cluster) below and no longer reaches these; they are retained
// for now and must stay in sync with the baseline arm of `dispatch`.

//! @param d_temp_storage Device-accessible allocation of temporary storage. When `nullptr`, the required allocation
//!        size is written to `temp_storage_bytes` and no work is done.
//! @param temp_storage_bytes Reference to size in bytes of `d_temp_storage` allocation
//! @param d_key_segments_it d_key_segments_it[segment_index] -> iterator to the input sequence of key data for segment
//!        `segment_index`
//! @param d_key_segments_out_it d_key_segments_out_it[segment_index] -> iterator to the output sequence of key data for
//!        segment `segment_index`
//! @param d_value_segments_it d_value_segments_it[segment_index] -> iterator to the input sequence of associated value
//!        items for segment `segment_index`. When cub::NullType**, only keys are provided.
//! @param d_value_segments_out_it d_value_segments_out_it[segment_index] -> iterator to the output sequence of
//!        associated value items for segment `segment_index`
//! @param segment_sizes Parameter providing segment sizes for each segment
//! @param k Parameter providing K for each segment
//! @param select_directions Parameter providing the selection direction for each segment
//! @param num_segments Number of segments
//! @param total_num_items_guarantee Allows the user to provide a guarantee on the upper bound of the total number of
//!        items
template <typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionT,
          typename NumSegmentsParameterT,
          typename TotalNumItemsGuaranteeT,
          typename PolicySelector = baseline_policy_selector_from_types<it_value_t<it_value_t<KeyInputItItT>>,
                                                                        it_value_t<it_value_t<ValueInputItItT>>,
                                                                        ::cuda::std::int64_t,
                                                                        ::cuda::args::__traits<KParameterT>::highest>>
#if _CCCL_HAS_CONCEPTS()
  requires baseline_topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t baseline_dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k,
  SelectDirectionT select_direction,
  NumSegmentsParameterT num_segments,
  [[maybe_unused]] TotalNumItemsGuaranteeT total_num_items_guarantee,
  cudaStream_t stream                             = nullptr,
  [[maybe_unused]] PolicySelector policy_selector = {})
{
  using large_segment_tile_offset_t = typename ::cuda::args::__traits<TotalNumItemsGuaranteeT>::element_type;

  // Wrap the raw enum into the internal discrete param type
  auto select_directions          = wrap_select_direction(select_direction);
  using SelectDirectionParameterT = decltype(select_directions);

  // Helper that determines (a) whether there's any one-worker-per-segment policy supporting the range of segment
  // sizes and k, and (b) if so, which set of one-worker-per-segment policies to use
  constexpr auto policy = find_smallest_covering_policy<
    PolicySelector,
    SegmentSizeParameterT,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT,
    large_segment_tile_offset_t>::policy;
  constexpr worker_policy worker_per_segment_policy             = policy.worker_per_segment_policy;
  constexpr multi_worker_policy multi_worker_per_segment_policy = policy.multi_worker_per_segment_policy;

  static constexpr int worker_per_segment_tile_size =
    worker_per_segment_policy.threads_per_block * worker_per_segment_policy.items_per_thread;
  static constexpr bool any_small_segments =
    ::cuda::args::__traits<SegmentSizeParameterT>::lowest <= worker_per_segment_tile_size;
  static constexpr bool only_small_segments =
    ::cuda::args::__traits<SegmentSizeParameterT>::highest <= worker_per_segment_tile_size;

  // Allocation layout:
  //   only_small_segments: [0] dummy.
  //   any_small_segments && !only_small_segments (mixed): [0] tile offsets, [1] counters struct,
  //                                                       [2] large-segment ids.
  //   !any_small_segments (large-only): [0] tile offsets, [1] segment-size transform-scan temp storage.
  static constexpr int allocations_array_size     = only_small_segments ? 1 : (any_small_segments ? 3 : 2);
  size_t allocation_sizes[allocations_array_size] = {1};

  using num_segments_val_t         = typename ::cuda::args::__traits<NumSegmentsParameterT>::element_type;
  using counters_t                 = batched_topk_counters<num_segments_val_t>;
  using segment_size_scan_offset_t = detail::choose_offset_t<num_segments_val_t>;
  using segment_size_scan_input_op_t =
    segment_size_to_tile_count_op<SegmentSizeParameterT, large_segment_tile_offset_t>;
  static constexpr auto multi_worker_per_segment_tile_size =
    multi_worker_per_segment_policy.threads_per_block * multi_worker_per_segment_policy.items_per_thread;
  const segment_size_scan_input_op_t segment_size_scan_input_op{segment_sizes, multi_worker_per_segment_tile_size};
  // Transform iterator over [0, num_segments) producing the tile-count for each segment.
  [[maybe_unused]] const auto segment_size_scan_input_it = ::cuda::transform_iterator(
    ::cuda::counting_iterator<num_segments_val_t>{num_segments_val_t{0}}, segment_size_scan_input_op);

  if constexpr (!only_small_segments)
  {
    const auto num_segments_val = params::get_param(num_segments, 0);
    // Scan output
    allocation_sizes[0] = num_segments_val * sizeof(large_segment_tile_offset_t);
    if constexpr (any_small_segments)
    {
      allocation_sizes[1] = sizeof(counters_t);
      // Large segment ids for indirectly accessing the large segment parameters
      allocation_sizes[2] = num_segments_val * sizeof(num_segments_val_t);
    }
    else
    {
      // Query the temporary storage requirement of the segment-size transform-scan.
      if (const auto error = CubDebug(detail::scan::dispatch(
            nullptr,
            allocation_sizes[1],
            segment_size_scan_input_it,
            static_cast<large_segment_tile_offset_t*>(nullptr),
            ::cuda::std::plus<>{},
            detail::InputValue<large_segment_tile_offset_t>(large_segment_tile_offset_t{0}),
            static_cast<segment_size_scan_offset_t>(num_segments_val),
            stream)))
      {
        return error;
      }
    }
  }

  // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
  void* allocations[allocations_array_size] = {};
  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  {
    return cudaSuccess;
  }

  // TODO (elstehle): support number of segments provided by device-accessible iterator
  // Only uniform number of segments are supported (i.e., we need to resolve the number of segments on the host)
  static_assert(::cuda::args::__traits<NumSegmentsParameterT>::is_single_value,
                "Only a uniform number of segments is currently supported.");

  if constexpr (any_small_segments)
  {
    if constexpr (!only_small_segments)
    {
      // Zero-initialize the counters struct that holds the large-segment queue length and the block retirement
      // counter; both are read by the agent's atomic operations and must start at 0.
      if (const auto error = CubDebug(cudaMemsetAsync(allocations[1], 0, sizeof(counters_t), stream)))
      {
        return error;
      }
    }
    const int grid_dim      = static_cast<int>(params::get_param(num_segments, 0));
    constexpr int block_dim = worker_per_segment_policy.threads_per_block;
    if (const auto error = CubDebug(
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(grid_dim, block_dim, 0, stream)
            .doit(
              device_segmented_topk_kernel<
                PolicySelector,
                KeyInputItItT,
                KeyOutputItItT,
                ValueInputItItT,
                ValueOutputItItT,
                SegmentSizeParameterT,
                KParameterT,
                SelectDirectionParameterT,
                NumSegmentsParameterT,
                large_segment_tile_offset_t>,
              d_key_segments_it,
              d_key_segments_out_it,
              d_value_segments_it,
              d_value_segments_out_it,
              segment_sizes,
              k,
              select_directions,
              num_segments,
              only_small_segments ? nullptr : static_cast<counters_t*>(allocations[1]),
              only_small_segments ? nullptr : static_cast<num_segments_val_t*>(allocations[2]),
              only_small_segments ? nullptr : static_cast<large_segment_tile_offset_t*>(allocations[0]))))
    {
      return error;
    }
  }
  else
  {
    // No small segments: the small-kernel epilogue (which would otherwise produce the per-segment tile offsets) does
    // not run. Compute the per-segment tile offsets directly via a transform-scan over all segment sizes.
    // The large segment agent will either consume these offsets directly (segment_id -> tile offset) or, when going
    // through the large-segment queue, via a transform iterator over `d_large_segments_ids` (level of indirection).
    if (const auto error = CubDebug(detail::scan::dispatch(
          allocations[1],
          allocation_sizes[1],
          segment_size_scan_input_it,
          static_cast<large_segment_tile_offset_t*>(allocations[0]),
          ::cuda::std::plus<>{},
          detail::InputValue<large_segment_tile_offset_t>(large_segment_tile_offset_t{0}),
          static_cast<segment_size_scan_offset_t>(params::get_param(num_segments, 0)),
          stream)))
    {
      return error;
    }
  }

  if constexpr (!only_small_segments)
  {
    // TODO (elstehle): support larger number of segments through multiple kernel launches
    // Depending on any_small_segments, we need to either:
    // - Indirectly get the large segment parameters via the queued large segment IDs
    // - Directly take the segment parameters since all segments are large
  }
  return CubDebug(detail::DebugSyncStream(stream));
}
// Env-based dispatch function handling memory allocation as well. This is usually done by the device-layer, but there
// is no public API for segmented topk yet.
template <typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename TotalNumItemsGuaranteeT,
          typename EnvT = ::cuda::std::execution::env<>>
[[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t baseline_dispatch_with_env(
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments,
  TotalNumItemsGuaranteeT total_num_items_guarantee,
  EnvT env = {})
{
  using default_policy_selector =
    baseline_policy_selector_from_types<it_value_t<it_value_t<KeyInputItItT>>,
                                        it_value_t<it_value_t<ValueInputItItT>>,
                                        ::cuda::std::int64_t,
                                        ::cuda::args::__traits<KParameterT>::highest>;
  return detail::dispatch_with_env_and_tuning<default_policy_selector>(
    env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
      return baseline_dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_key_segments_it,
        d_key_segments_out_it,
        d_value_segments_it,
        d_value_segments_out_it,
        segment_sizes,
        k,
        select_directions,
        num_segments,
        total_num_items_guarantee,
        stream,
        policy_selector);
    });
}

// -----------------------------------------------------------------------------
// Dispatch (both backends behind one kernel symbol)
// -----------------------------------------------------------------------------
// Tightest upper bound carried by the segment-size argument. Mirrors `args::__traits<>::highest` semantics:
// the compile-time bound for `constant`/bounded sequence arguments and the runtime value for a uniform
// `immediate`. For a per-segment sequence with only a static bound this can be the loose `numeric_limits<T>::max()`.
template <typename SegmentSizeParameterT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto runtime_max_segment_size(SegmentSizeParameterT segment_sizes) noexcept
{
  return ::cuda::args::__highest_(segment_sizes);
}

// Host launches go through the single kernel symbol (`device_batched_topk_kernel`); the CDP path uses a dedicated
// static-cluster kernel symbol (`device_segmented_topk_cluster_kernel_static`) because device-side launches cannot opt
// in to dynamic cluster dimensions. Both kernels live in kernel_batched_topk.cuh.

// CDP launch body, empty when CDP is disabled. Wrapped in a macro because
// `#ifdef` can't sit inside `NV_IF_TARGET`.
#ifndef CUB_RDC_ENABLED
// Without CDP/RDC a device-side launch is impossible; surface that instead of silently returning success (no-op).
#  define CUB_TOPK_CLUSTER_DEVICE_LAUNCH return cudaErrorNotSupported;
#else // CUB_RDC_ENABLED
#  define CUB_TOPK_CLUSTER_DEVICE_LAUNCH                                                    \
    auto static_kernel = detail::batched_topk::device_segmented_topk_cluster_kernel_static< \
      ThreadsPerBlock,                                                                      \
      HistogramItemsPerThread,                                                              \
      PipelineStages,                                                                       \
      ChunkBytes,                                                                           \
      LoadAlignBytes,                                                                       \
      BitsPerPass,                                                                          \
      TieBreakItemsPerThread,                                                               \
      SingleBlockMaxSegSize,                                                                \
      MinChunksPerBlock,                                                                    \
      CopyItemsPerThread,                                                                   \
      Determinism,                                                                          \
      TieBreak,                                                                             \
      KeyInputItItT,                                                                        \
      KeyOutputItItT,                                                                       \
      ValueInputItItT,                                                                      \
      ValueOutputItItT,                                                                     \
      SegmentSizeParameterT,                                                                \
      KParameterT,                                                                          \
      SelectDirectionParameterT,                                                            \
      NumSegmentsParameterT>;                                                               \
    if (const auto error = CubDebug(                                                        \
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(                            \
            static_cast<int>(grid_blocks), ThreadsPerBlock, dynamic_smem_bytes, stream)     \
            .doit(static_kernel,                                                            \
                  d_key_segments_it,                                                        \
                  d_key_segments_out_it,                                                    \
                  d_value_segments_it,                                                      \
                  d_value_segments_out_it,                                                  \
                  segment_sizes,                                                            \
                  k_param,                                                                  \
                  select_directions,                                                        \
                  num_segments,                                                             \
                  block_tile_capacity)))                                                    \
    {                                                                                       \
      return error;                                                                         \
    }
#endif // CUB_RDC_ENABLED

// Cluster host-launch arm of the dispatch. Launches the single kernel symbol
// (`device_batched_topk_kernel`, passing an empty `baseline_kernel_args` and the resident `cluster_kernel_args`).
// `select_directions` arrives already wrapped; the cluster tuning is taken from `policy_getter` (the resolved-CC
// policy) and the requested `Determinism`/`TieBreak` from the `PolicySelector`. The CDP arm still launches the
// dedicated static-cluster kernel symbol.
template <class PolicySelector,
          class LargeSegmentTileOffsetT,
          class PolicyGetter,
          class KeyInputItItT,
          class KeyOutputItItT,
          class ValueInputItItT,
          class ValueOutputItItT,
          class SegmentSizeParameterT,
          class KParameterT,
          class SelectDirectionParameterT,
          class NumSegmentsParameterT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t launch_cluster_arm(
  PolicyGetter policy_getter,
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k_param,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments,
  cudaStream_t stream)
{
  constexpr auto Determinism = PolicySelector::determinism;
  constexpr auto TieBreak    = PolicySelector::tie_break;
  // Use the cluster sub-policy for the *resolved* architecture (the one `dispatch_compute_cap` matched), i.e. exactly
  // what the device kernel instantiates via `current_policy<PolicySelector>()`. Sourcing it from `policy_getter` keeps
  // the host launch config (block size, shared-memory math, agent instantiation) in lock-step with the device policy
  // per CC. `policy_getter()` is a constant expression in AOT builds, so `policy` is usable as a non-type template arg.
  constexpr cluster_topk_policy policy  = policy_getter().cluster;
  constexpr int ThreadsPerBlock         = policy.threads_per_block;
  constexpr int HistogramItemsPerThread = policy.histogram_items_per_thread;
  constexpr int PipelineStages          = policy.pipeline_stages;
  constexpr int ChunkBytes              = policy.chunk_bytes;
  constexpr int LoadAlignBytes          = policy.load_align_bytes;
  constexpr int BitsPerPass             = policy.bits_per_pass;
  constexpr int TieBreakItemsPerThread  = policy.tie_break_items_per_thread;
  constexpr int SingleBlockMaxSegSize   = policy.single_block_max_seg_size;
  constexpr int MinChunksPerBlock       = policy.min_chunks_per_block;
  constexpr int CopyItemsPerThread      = policy.copy_items_per_thread;

  using key_it_t = it_value_t<KeyInputItItT>;
  using key_t    = it_value_t<key_it_t>;
  using layout_t = batched_topk_cluster::smem_block_tile_layout<key_t, ChunkBytes, LoadAlignBytes>;
  using agent_t  = batched_topk_cluster::agent_batched_topk_cluster<
     ThreadsPerBlock,
     HistogramItemsPerThread,
     PipelineStages,
     ChunkBytes,
     LoadAlignBytes,
     BitsPerPass,
     TieBreakItemsPerThread,
     SingleBlockMaxSegSize,
     MinChunksPerBlock,
     CopyItemsPerThread,
     Determinism,
     TieBreak,
     KeyInputItItT,
     KeyOutputItItT,
     ValueInputItItT,
     ValueOutputItItT,
     SegmentSizeParameterT,
     KParameterT,
     SelectDirectionParameterT,
     NumSegmentsParameterT>;

  // TODO: This should be taken care of in the public env-based interface.
  // A tie-break preference is only meaningful once the result set itself is deterministic.
  static_assert(Determinism != ::cuda::execution::determinism::__determinism_t::__not_guaranteed
                  || TieBreak == ::cuda::execution::tie_break::__tie_break_t::__unspecified,
                "A tie-break preference requires a deterministic execution requirement");

  // Validate the block_tile byte geometry (load alignment power-of-two / >= 16, chunk a multiple of it) in one place.
  static_assert(is_valid_cluster_policy(policy));
  static_assert(LoadAlignBytes % int{sizeof(key_t)} == 0);
  // Static-footprint estimate for the device-side CDP fallback, which cannot query `cudaFuncGetAttributes`.
  // The host path instead uses the driver-reported `sharedSizeBytes` (see below), which is padding-aware.
  constexpr int static_smem_bytes = static_cast<int>(sizeof(typename agent_t::TempStorage));

  const auto max_seg_size = runtime_max_segment_size(segment_sizes);

  // The harness expects temp_storage_bytes > 0. Allocate a single byte placeholder.
  size_t allocation_sizes[1] = {1};
  void* allocations[1]       = {};
  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  {
    return cudaSuccess;
  }

  static_assert(::cuda::args::__traits<NumSegmentsParameterT>::is_single_value,
                "Number of segments must be resolved on the host.");

  using num_segments_val_t = typename ::cuda::args::__traits<NumSegmentsParameterT>::element_type;
  const auto num_seg_val   = detail::params::get_param(num_segments, num_segments_val_t{0});
  if (num_seg_val == 0)
  {
    return cudaSuccess;
  }

  // A zero bound would drive `clusterDim.x = 0`, which the runtime rejects.
  if (max_seg_size == 0)
  {
    return cudaSuccess;
  }

  // Cluster launches require compute capability 9.0+.
  int sm_version = 0;
  if (const auto error = CubDebug(SmVersionUncached(sm_version)))
  {
    return error;
  }
  if (sm_version < 900)
  {
    return cudaErrorNotSupported;
  }

  // Single kernel symbol; its cluster arm is selected device-side via `current_policy<PolicySelector>()`. The baseline
  // arm is pruned per-arch, so no baseline symbol is emitted here.
  constexpr auto dynamic_kernel = &device_batched_topk_kernel<
    PolicySelector,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT,
    LargeSegmentTileOffsetT>;

  NV_IF_TARGET(
    NV_IS_HOST,
    ({
      // The launcher's `doit` carries the triple-chevron that makes NVCC emit `dynamic_kernel` for this TU, and
      // performs the cluster launch via `cudaLaunchKernelEx`. The factory also wraps the pre-launch driver queries.
      detail::TripleChevronFactory launcher_factory{};

      // Opt in to non-portable cluster blocks (>8 on Hopper).
      if (const auto error = launcher_factory.set_non_portable_cluster_allowed(dynamic_kernel))
      {
        return error;
      }

      // Config used only for the occupancy probe below; the final launch goes through `launcher_factory`.
      // `clusterDim.x` is a placeholder since `cudaOccupancyMaxPotentialClusterSize` ignores it.
      ::cudaLaunchAttribute cluster_attr{};
      cluster_attr.id               = ::cudaLaunchAttributeClusterDimension;
      cluster_attr.val.clusterDim.x = 1;
      cluster_attr.val.clusterDim.y = 1;
      cluster_attr.val.clusterDim.z = 1;

      ::cudaLaunchConfig_t cfg{};
      cfg.gridDim          = dim3(1, 1, 1);
      cfg.blockDim         = dim3(static_cast<unsigned int>(ThreadsPerBlock), 1, 1);
      cfg.dynamicSmemBytes = 0;
      cfg.stream           = stream;
      cfg.attrs            = &cluster_attr;
      cfg.numAttrs         = 1;

      // Resolve the per-block opt-in shared-memory budget and the kernel's static footprint from the driver so
      // the dynamic-SMEM math below matches exactly what the launch permits. The opt-in budget
      // (`cudaDevAttrMaxSharedMemoryPerBlockOptin`) is the documented total per-block budget; the usable dynamic
      // portion (`max_dynamic_smem_bytes`) is that budget minus the static footprint.
      int device_id = 0;
      if (const auto error = CubDebug(cudaGetDevice(&device_id)))
      {
        return error;
      }
      int max_smem_optin_bytes = 0;
      if (const auto error =
            CubDebug(cudaDeviceGetAttribute(&max_smem_optin_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id)))
      {
        return error;
      }
      // Use the driver-reported static footprint (`sharedSizeBytes`) rather than `sizeof(TempStorage)`: it reflects
      // any padding the toolchain inserts to align the dynamic shared-memory section after the static one, so the
      // derived dynamic sizes neither overshoot the budget nor conservatively drop the top table tier.
      cudaFuncAttributes kernel_attrs{};
      if (const auto error = CubDebug(cudaFuncGetAttributes(&kernel_attrs, dynamic_kernel)))
      {
        return error;
      }
      // `cudaDevAttrMaxSharedMemoryPerBlockOptin` already excludes the driver's per-block reserved shared memory
      // (opt-in == per-SM - reserved), so the dynamic budget is just the opt-in budget minus the static footprint;
      // reserved must not be subtracted a second time.
      const int nondynamic_smem_bytes = static_cast<int>(kernel_attrs.sharedSizeBytes);
      const int max_dynamic_smem_bytes =
        (max_smem_optin_bytes > nondynamic_smem_bytes) ? max_smem_optin_bytes - nondynamic_smem_bytes : 0;

      // Raise the kernel's dynamic-SMEM opt-in lazily: occupancy queries and the launch must not request more than the
      // currently configured `cudaFuncAttributeMaxDynamicSharedMemorySize`. The kernel's compile-time default already
      // permits the portable 48 KiB total, i.e. that budget minus the static footprint.
      constexpr int portable_total_smem_bytes = 48 * 1024;
      int configured_dynamic_smem_limit =
        (portable_total_smem_bytes > nondynamic_smem_bytes) ? portable_total_smem_bytes - nondynamic_smem_bytes : 0;
      const auto ensure_dynamic_smem_limit = [&](int dynamic_smem_bytes) {
        if (dynamic_smem_bytes <= configured_dynamic_smem_limit)
        {
          return cudaSuccess;
        }

        if (const auto error = CubDebug(cudaFuncSetAttribute(
              reinterpret_cast<const void*>(dynamic_kernel),
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              dynamic_smem_bytes)))
        {
          return error;
        }
        configured_dynamic_smem_limit = dynamic_smem_bytes;
        return cudaSuccess;
      };

      // Wave-aware cluster-blocks selection. The free variable is the cluster blocks `C` (one cluster per segment);
      // each `C` is paired with the smallest dynamic SMEM that keeps a segment fully resident. A smaller `C` needs more
      // SMEM (fewer clusters-per-wave, less L1); a larger `C` needs less SMEM (more clusters-per-wave, more L1). We
      // pick the `C` that minimizes the number of waves, breaking ties toward the largest `C` (= smallest SMEM = most
      // L1), which matches the profiled fast configs. We enumerate `C` analytically rather than discovering SMEM tiers
      // via occupancy, so a register-limited occupancy (e.g. 1 CTA/SM) cannot collapse the candidate set.
      const auto seg                    = static_cast<::cuda::std::uint64_t>(max_seg_size);
      const auto chunk_items_u64        = static_cast<::cuda::std::uint64_t>(layout_t::chunk_items);
      const int max_block_tile_capacity = static_cast<int>(layout_t::block_tile_capacity(max_dynamic_smem_bytes));
      if (max_block_tile_capacity <= 0)
      {
        // Not even one load-aligned chunk fits in the opt-in budget; the kernel cannot run.
        return cudaErrorInvalidValue;
      }

      // `S_res(items)`: smallest chunk-granular dynamic SMEM whose per-CTA capacity reaches `items`.
      const auto smem_for_block_capacity = [&](::cuda::std::uint64_t items) {
        const auto slots = ::cuda::ceil_div(items, chunk_items_u64);
        return layout_t::base_padding_bytes + static_cast<int>(slots) * layout_t::chunk_bytes;
      };

      // `C_full`: at the 1-chunk SMEM each CTA holds `chunk_items`, so full residency needs this many CTAs (cap HW
      // max). `C_lo`: at the largest SMEM each CTA holds `max_block_tile_capacity`, the smallest fully-resident `C`.
      // Both are computed and compared in 64-bit, because `max_seg_size` may be a loose bound (e.g.
      // `numeric_limits<T>::max()` for an unbounded deferred sequence); narrowing such a `C_lo` to `int` could wrap to
      // a small (or negative) value and wrongly enter the resident branch instead of the oversize/streaming fallback.
      const int c_full = static_cast<int>(
        (::cuda::std::min) (static_cast<::cuda::std::uint64_t>(max_supported_cluster_blocks),
                            ::cuda::ceil_div(seg, chunk_items_u64)));
      const auto c_lo = ::cuda::ceil_div(seg, static_cast<::cuda::std::uint64_t>(max_block_tile_capacity));
      // Cluster blocks the max segment actually needs (shared with the device so the launch is never wider than
      // necessary). At `min_chunks_per_block == 1` this equals `c_full`; a larger knob shrinks it.
      const int desired_cluster_blocks = static_cast<int>(batched_topk_cluster::effective_cluster_blocks_from_chunks(
        ::cuda::ceil_div(seg, chunk_items_u64),
        MinChunksPerBlock,
        static_cast<unsigned int>(max_supported_cluster_blocks)));

      int cluster_blocks   = 0;
      int dynamic_smem_sel = 0;

      if (batched_topk_cluster::is_single_cta_eligible(
            seg, static_cast<::cuda::std::uint64_t>(max_block_tile_capacity), SingleBlockMaxSegSize))
      {
        // Single-CTA fast path: the segment fits resident in one CTA and is small enough that the agent's
        // cluster-barrier-free path beats spreading it across more CTAs. `S_res(seg)` is within budget and one CTA is
        // always launchable, so the occupancy probe is skipped (the shared `ensure_dynamic_smem_limit` below raises the
        // opt-in for the selected SMEM). Larger fully-resident segments fall through to the wave-aware search below.
        cluster_blocks   = 1;
        dynamic_smem_sel = smem_for_block_capacity(seg);
      }
      else if (c_lo <= static_cast<::cuda::std::uint64_t>(max_supported_cluster_blocks))
      {
        // Full residency is achievable. `seg <= C_lo * max_block_tile_capacity` with `C_lo <= HW max`, so every
        // per-CTA capacity (and thus its slot count and SMEM bytes) below stays well within `int` -- no overflow.
        // Scan `C` in `[max(C_lo, 2), C_end]`, minimize waves, tie-break largest `C`. `C = 1` is handled above. The
        // upper bound is capped at the cluster blocks the max segment needs (`desired_cluster_blocks`), so the host
        // never launches a wider cluster than necessary; at `min_chunks_per_block == 1` the cap equals `c_full`.
        const int c_begin = (::cuda::std::max) (2, static_cast<int>(c_lo));
        const int c_end   = (::cuda::std::max) (c_begin, (::cuda::std::min) (c_full, desired_cluster_blocks));
        ::cuda::std::uint64_t best_waves = (::cuda::std::numeric_limits<::cuda::std::uint64_t>::max)();
        for (int c = c_begin; c <= c_end; ++c)
        {
          const auto per_block_items = ::cuda::ceil_div(seg, static_cast<::cuda::std::uint64_t>(c));
          const int s_res            = smem_for_block_capacity(per_block_items);
          if (s_res > max_dynamic_smem_bytes)
          {
            continue; // unreachable for c >= C_lo, but guards the SMEM budget regardless.
          }

          if (const auto error = ensure_dynamic_smem_limit(s_res))
          {
            return error;
          }

          // `cudaOccupancyMaxActiveClusters` needs the cluster dimension and the matching dynamic SMEM; the grid must
          // be a multiple of the cluster blocks. The returned value is the device-wide clusters-per-wave (capacity),
          // independent of grid size, and accounts for the static footprint and register pressure internally.
          cluster_attr.val.clusterDim.x = static_cast<unsigned int>(c);
          cfg.gridDim                   = dim3(static_cast<unsigned int>(c), 1, 1);
          cfg.dynamicSmemBytes          = static_cast<unsigned int>(s_res);
          int clusters_per_wave         = 0;
          if (const auto error = launcher_factory.max_active_clusters(clusters_per_wave, dynamic_kernel, &cfg))
          {
            return error;
          }
          if (clusters_per_wave <= 0)
          {
            continue; // cluster blocks not launchable at this SMEM.
          }

          const auto waves = ::cuda::ceil_div(
            static_cast<::cuda::std::uint64_t>(num_seg_val), static_cast<::cuda::std::uint64_t>(clusters_per_wave));
          // Min waves; tie-break largest `C`. The loop ascends in `C`, so `<=` keeps the largest at equal waves.
          if (cluster_blocks == 0 || waves <= best_waves)
          {
            best_waves       = waves;
            cluster_blocks   = c;
            dynamic_smem_sel = s_res;
          }
        }

        if (cluster_blocks == 0 && c_lo == 1)
        {
          // No multi-CTA config was launchable; fall back to single-CTA full residency. Slower for large segments,
          // but `C_lo == 1` guarantees `S_res(seg)` fits the budget and one CTA is always launchable.
          cluster_blocks   = 1;
          dynamic_smem_sel = smem_for_block_capacity(seg);
        }
      }

      if (cluster_blocks == 0)
      {
        // Oversize (`C_lo > HW max`) or nothing launchable in range: full residency is impossible, so maximize
        // residency with the largest launchable cluster at the largest SMEM and let the agent stream the overflow.
        if (const auto error = ensure_dynamic_smem_limit(max_dynamic_smem_bytes))
        {
          return error;
        }
        cluster_attr.val.clusterDim.x = 1; // ignored by `max_potential_cluster_size`
        cfg.gridDim                   = dim3(1, 1, 1);
        cfg.dynamicSmemBytes          = static_cast<unsigned int>(max_dynamic_smem_bytes);
        int hw_max_cluster_blocks     = 0;
        if (const auto error = launcher_factory.max_potential_cluster_size(hw_max_cluster_blocks, dynamic_kernel, &cfg))
        {
          return error;
        }
        hw_max_cluster_blocks = (::cuda::std::min) (hw_max_cluster_blocks, max_supported_cluster_blocks);
        if (hw_max_cluster_blocks <= 0)
        {
          return cudaErrorInvalidValue;
        }
        cluster_blocks   = hw_max_cluster_blocks;
        dynamic_smem_sel = max_dynamic_smem_bytes;
      }

      // The launch needs `MaxDynamicSharedMemorySize >= dynamic_smem_sel`; the scan already raised the limit past the
      // largest probed SMEM, so this is a no-op unless the selected config skipped the scan.
      if (const auto error = ensure_dynamic_smem_limit(dynamic_smem_sel))
      {
        return error;
      }

      const int dynamic_smem_bytes   = dynamic_smem_sel;
      const auto block_tile_capacity = layout_t::block_tile_capacity(dynamic_smem_bytes);

      const auto grid_blocks =
        static_cast<::cuda::std::uint64_t>(num_seg_val) * static_cast<::cuda::std::uint64_t>(cluster_blocks);
      if (grid_blocks > static_cast<::cuda::std::uint64_t>(::cuda::std::numeric_limits<int>::max()))
      {
        return cudaErrorInvalidValue;
      }

      // The cluster dimension routes the host launch through `cudaLaunchKernelEx`; passing `dynamic_kernel` to `.doit`
      // below forces NVCC to emit that kernel symbol for this TU.
      if (const auto error = CubDebug(
            launcher_factory(dim3(static_cast<unsigned int>(grid_blocks), 1, 1),
                             dim3(static_cast<unsigned int>(ThreadsPerBlock), 1, 1),
                             static_cast<::cuda::std::size_t>(dynamic_smem_bytes),
                             stream,
                             /*dependent_launch=*/false,
                             dim3(static_cast<unsigned int>(cluster_blocks), 1, 1))
              .doit(dynamic_kernel,
                    d_key_segments_it,
                    d_key_segments_out_it,
                    d_value_segments_it,
                    d_value_segments_out_it,
                    segment_sizes,
                    k_param,
                    select_directions,
                    num_segments,
                    baseline_kernel_args<num_segments_val_t, LargeSegmentTileOffsetT>{},
                    cluster_kernel_args{static_cast<::cuda::std::uint32_t>(block_tile_capacity)})))
      {
        return error;
      }
    }),
    ({
      // CDP path: device-side launches cannot opt in to more than portable total SMEM or non-portable cluster blocks.
      // Segments exceeding the portable resident coverage are still handled: the agent re-streams overflow from gmem.
      constexpr int portable_total_smem_bytes = 48 * 1024;
      constexpr int dynamic_smem_bytes =
        (portable_total_smem_bytes > static_smem_bytes) ? portable_total_smem_bytes - static_smem_bytes : 0;

      // The compile-time `ChunkBytes` is reused verbatim; the agent peels unaligned boundary edges into a tiny
      // per-block buffer and re-streams overflow from gmem, so the only hard requirement is that one load-aligned chunk
      // fits the worst-case portable SMEM block tile.
      constexpr auto block_tile_capacity = layout_t::block_tile_capacity(dynamic_smem_bytes);
      static_assert(block_tile_capacity >= static_cast<::cuda::std::uint32_t>(layout_t::chunk_items),
                    "Portable SMEM is too small to fit even one load-aligned chunk for the device-launch (CDP) path");

      const auto grid_blocks = static_cast<::cuda::std::uint64_t>(num_seg_val)
                             * static_cast<::cuda::std::uint64_t>(max_portable_cluster_blocks);
      if (grid_blocks > static_cast<::cuda::std::uint64_t>(::cuda::std::numeric_limits<int>::max()))
      {
        return cudaErrorInvalidValue;
      }

      CUB_TOPK_CLUSTER_DEVICE_LAUNCH
    }));

  // Cluster launches can fail on the device while reporting success; sync.
  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  return CubDebug(detail::DebugSyncStream(stream));
}

#undef CUB_TOPK_CLUSTER_DEVICE_LAUNCH

// Baseline host-launch arm of the dispatch. Launches the single kernel symbol
// (`device_batched_topk_kernel`, packing the large-segment bookkeeping into `baseline_kernel_args` and passing an empty
// `cluster_kernel_args`). `select_directions` arrives already wrapped and the baseline tuning is taken from the
// `PolicySelector` (via `baseline_policy_selector_adaptor`).
template <class PolicySelector,
          class LargeSegmentTileOffsetT,
          class KeyInputItItT,
          class KeyOutputItItT,
          class ValueInputItItT,
          class ValueOutputItItT,
          class SegmentSizeParameterT,
          class KParameterT,
          class SelectDirectionParameterT,
          class NumSegmentsParameterT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t launch_baseline_arm(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments,
  cudaStream_t stream)
{
  if constexpr (!PolicySelector::baseline_can_cover)
  {
    // The policy selector never routes to the baseline backend when it cannot cover the static max segment size, so
    // this arm is pruned per-arch in AOT builds. Kept assert-free (no `find_smallest_covering_policy`) so it also
    // compiles under runtime-policies mode, where both host arms are instantiated.
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }
    return cudaErrorNotSupported;
  }
  else
  {
    using large_segment_tile_offset_t = LargeSegmentTileOffsetT;
    using baseline_selector_t         = baseline_policy_selector_adaptor<PolicySelector>;

    // Determine which one-worker-per-segment policy covers the segment-size range and k.
    constexpr auto policy = find_smallest_covering_policy<
      baseline_selector_t,
      SegmentSizeParameterT,
      KeyInputItItT,
      KeyOutputItItT,
      ValueInputItItT,
      ValueOutputItItT,
      SegmentSizeParameterT,
      KParameterT,
      SelectDirectionParameterT,
      NumSegmentsParameterT,
      large_segment_tile_offset_t>::policy;
    constexpr worker_policy worker_per_segment_policy             = policy.worker_per_segment_policy;
    constexpr multi_worker_policy multi_worker_per_segment_policy = policy.multi_worker_per_segment_policy;

    static constexpr int worker_per_segment_tile_size =
      worker_per_segment_policy.threads_per_block * worker_per_segment_policy.items_per_thread;
    static constexpr bool any_small_segments =
      ::cuda::args::__traits<SegmentSizeParameterT>::lowest <= worker_per_segment_tile_size;
    static constexpr bool only_small_segments =
      ::cuda::args::__traits<SegmentSizeParameterT>::highest <= worker_per_segment_tile_size;

    // Allocation layout:
    //   only_small_segments: [0] dummy.
    //   any_small_segments && !only_small_segments (mixed): [0] tile offsets, [1] counters struct,
    //                                                       [2] large-segment ids.
    //   !any_small_segments (large-only): [0] tile offsets, [1] segment-size transform-scan temp storage.
    static constexpr int allocations_array_size     = only_small_segments ? 1 : (any_small_segments ? 3 : 2);
    size_t allocation_sizes[allocations_array_size] = {1};

    using num_segments_val_t         = typename ::cuda::args::__traits<NumSegmentsParameterT>::element_type;
    using counters_t                 = batched_topk_counters<num_segments_val_t>;
    using segment_size_scan_offset_t = detail::choose_offset_t<num_segments_val_t>;
    using segment_size_scan_input_op_t =
      segment_size_to_tile_count_op<SegmentSizeParameterT, large_segment_tile_offset_t>;
    static constexpr auto multi_worker_per_segment_tile_size =
      multi_worker_per_segment_policy.threads_per_block * multi_worker_per_segment_policy.items_per_thread;
    const segment_size_scan_input_op_t segment_size_scan_input_op{segment_sizes, multi_worker_per_segment_tile_size};
    // Transform iterator over [0, num_segments) producing the tile-count for each segment.
    [[maybe_unused]] const auto segment_size_scan_input_it = ::cuda::transform_iterator(
      ::cuda::counting_iterator<num_segments_val_t>{num_segments_val_t{0}}, segment_size_scan_input_op);

    if constexpr (!only_small_segments)
    {
      const auto num_segments_val = params::get_param(num_segments, 0);
      allocation_sizes[0]         = num_segments_val * sizeof(large_segment_tile_offset_t);
      if constexpr (any_small_segments)
      {
        allocation_sizes[1] = sizeof(counters_t);
        allocation_sizes[2] = num_segments_val * sizeof(num_segments_val_t);
      }
      else
      {
        // Query the temporary storage requirement of the segment-size transform-scan.
        if (const auto error = CubDebug(detail::scan::dispatch(
              nullptr,
              allocation_sizes[1],
              segment_size_scan_input_it,
              static_cast<large_segment_tile_offset_t*>(nullptr),
              ::cuda::std::plus<>{},
              detail::InputValue<large_segment_tile_offset_t>(large_segment_tile_offset_t{0}),
              static_cast<segment_size_scan_offset_t>(num_segments_val),
              stream)))
        {
          return error;
        }
      }
    }

    void* allocations[allocations_array_size] = {};
    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      return cudaSuccess;
    }

    static_assert(::cuda::args::__traits<NumSegmentsParameterT>::is_single_value,
                  "Only a uniform number of segments is currently supported.");

    if constexpr (any_small_segments)
    {
      if constexpr (!only_small_segments)
      {
        // Zero-initialize the counters struct read by the agent's atomics.
        if (const auto error = CubDebug(cudaMemsetAsync(allocations[1], 0, sizeof(counters_t), stream)))
        {
          return error;
        }
      }
      const int grid_dim      = static_cast<int>(params::get_param(num_segments, 0));
      constexpr int block_dim = worker_per_segment_policy.threads_per_block;
      if (const auto error = CubDebug(
            THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(grid_dim, block_dim, 0, stream)
              .doit(
                device_batched_topk_kernel<PolicySelector,
                                           KeyInputItItT,
                                           KeyOutputItItT,
                                           ValueInputItItT,
                                           ValueOutputItItT,
                                           SegmentSizeParameterT,
                                           KParameterT,
                                           SelectDirectionParameterT,
                                           NumSegmentsParameterT,
                                           large_segment_tile_offset_t>,
                d_key_segments_it,
                d_key_segments_out_it,
                d_value_segments_it,
                d_value_segments_out_it,
                segment_sizes,
                k,
                select_directions,
                num_segments,
                baseline_kernel_args<num_segments_val_t, large_segment_tile_offset_t>{
                  only_small_segments ? nullptr : static_cast<counters_t*>(allocations[1]),
                  only_small_segments ? nullptr : static_cast<num_segments_val_t*>(allocations[2]),
                  only_small_segments ? nullptr : static_cast<large_segment_tile_offset_t*>(allocations[0])},
                cluster_kernel_args{})))
      {
        return error;
      }
    }
    else
    {
      // No small segments: compute the per-segment tile offsets directly via a transform-scan over all segment sizes.
      if (const auto error = CubDebug(detail::scan::dispatch(
            allocations[1],
            allocation_sizes[1],
            segment_size_scan_input_it,
            static_cast<large_segment_tile_offset_t*>(allocations[0]),
            ::cuda::std::plus<>{},
            detail::InputValue<large_segment_tile_offset_t>(large_segment_tile_offset_t{0}),
            static_cast<segment_size_scan_offset_t>(params::get_param(num_segments, 0)),
            stream)))
      {
        return error;
      }
    }

    return CubDebug(detail::DebugSyncStream(stream));
  }
}

#if _CCCL_CUDA_COMPILATION() && !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)
// Returns true if at least one architecture in the compile target list (`CMAKE_CUDA_ARCHITECTURES`, exposed as
// `::cuda::__target_compute_capabilities()`) resolves to the `unsupported` backend for `PolicySelector` -- e.g. a
// deterministic request while a pre-SM90 target is present in the list. Used to turn a would-be runtime
// `cudaErrorNotSupported` into a compile-time diagnostic (see the static_assert in `dispatch`).
template <class PolicySelector>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL bool any_target_cc_unsupported()
{
  bool any = false;
  for (const auto cc : ::cuda::__target_compute_capabilities())
  {
    any = any || (PolicySelector{}(cc).backend == topk_backend::unsupported);
  }
  return any;
}
#endif // _CCCL_CUDA_COMPILATION() && !CUB_DEFINE_RUNTIME_POLICIES && !NVRTC

// Internal entry point: the single dispatch that replaces the standalone baseline / cluster dispatches. It resolves the
// runtime compute capability, then uses `dispatch_compute_cap` to pick, per architecture, the backend chosen by
// `policy_selector` (deterministic -> cluster; otherwise the arch+size crossover). Both host arms launch the same
// kernel symbol. `Determinism`/`TieBreak` are compile-time selection inputs; `Mode` lets a caller force a backend.
//
// A non-`no_override` `PolicySelectorOverride` (threaded through the tuning environment) fully replaces the automatic
// selector -- its `.backend` chooses the arm and its `.baseline`/`.cluster` carry the tunings -- so a benchmark or
// tuning test can pick the backend and its knobs in one selector.
template <
  ::cuda::execution::determinism::__determinism_t Determinism =
    ::cuda::execution::determinism::__determinism_t::__not_guaranteed,
  ::cuda::execution::tie_break::__tie_break_t TieBreak = ::cuda::execution::tie_break::__tie_break_t::__unspecified,
  backend_mode Mode                                    = backend_mode::automatic,
  class PolicySelectorOverride                         = no_override,
  class KeyInputItItT,
  class KeyOutputItItT,
  class ValueInputItItT,
  class ValueOutputItItT,
  class SegmentSizeParameterT,
  class KParameterT,
  class SelectDirectionT,
  class NumSegmentsParameterT,
  class TotalNumItemsGuaranteeT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k,
  SelectDirectionT select_direction,
  NumSegmentsParameterT num_segments,
  [[maybe_unused]] TotalNumItemsGuaranteeT total_num_items_guarantee,
  cudaStream_t stream)
{
  // The selection direction is a compile-time constant carried as `::cuda::args::constant<Dir>`. Wrap it into the
  // internal discrete param the kernel/agent expect (both host arms take the wrapped form).
  auto select_directions          = wrap_select_direction(select_direction);
  using SelectDirectionParameterT = decltype(select_directions);

  using key_t                   = it_value_t<it_value_t<KeyInputItItT>>;
  using value_t                 = it_value_t<it_value_t<ValueInputItItT>>;
  using LargeSegmentTileOffsetT = typename ::cuda::args::__traits<TotalNumItemsGuaranteeT>::element_type;

  constexpr ::cuda::std::int64_t max_k          = ::cuda::args::__traits<KParameterT>::highest;
  constexpr ::cuda::std::int64_t static_max_seg = ::cuda::args::__traits<SegmentSizeParameterT>::highest;

  // Baseline sub-policy the kernel will actually instantiate: the tuning override's `.baseline` when an override is
  // present (its sub-policies are forwarded verbatim by `selector_override_adaptor`), otherwise the default
  // baseline sub-selector. Coverage must be computed from this same policy so the backend decision (and the
  // override adaptor's `baseline_can_cover` it borrows) matches the policy `find_smallest_covering_policy` resolves.
  using launched_baseline_selector_t =
    ::cuda::std::conditional_t<::cuda::std::is_same_v<PolicySelectorOverride, no_override>,
                               baseline_policy_selector_from_types<key_t, value_t, ::cuda::std::int64_t, max_k>,
                               baseline_policy_selector_adaptor<PolicySelectorOverride>>;

  // Assert-free coverage predicate (never instantiates `find_smallest_covering_policy`'s hard static_assert).
  constexpr bool baseline_can_cover = baseline_can_cover_v<
    launched_baseline_selector_t,
    SegmentSizeParameterT,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT,
    LargeSegmentTileOffsetT>;

  // Default automatic selector from the compile-time inputs; a non-`no_override` override replaces it wholesale.
  using default_selector_t = policy_selector_from_types<
    key_t,
    value_t,
    ::cuda::std::int64_t,
    max_k,
    static_max_seg,
    Determinism,
    TieBreak,
    baseline_can_cover,
    Mode>;
  using selector_t = ::cuda::std::conditional_t<::cuda::std::is_same_v<PolicySelectorOverride, no_override>,
                                                default_selector_t,
                                                selector_override_adaptor<PolicySelectorOverride, default_selector_t>>;

#if _CCCL_CUDA_COMPILATION() && !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC) \
  && !defined(_CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT)
  // Strict mode (default): fail at compile time if the request cannot be served on *any* architecture in
  // `CMAKE_CUDA_ARCHITECTURES` (e.g. a deterministic / large-segment request while a pre-SM90 target is present, since
  // the cluster backend requires SM90+). This is the least-surprising UX for callers building the default multi-arch
  // preset. Define `_CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT` to defer the diagnosis to runtime instead (the dispatch
  // then returns `cudaErrorNotSupported` on unsupported devices); CUB's own tests and benchmarks do this so they can
  // compile the full configuration space across all target architectures and skip at runtime where unsupported.
  static_assert(
    !any_target_cc_unsupported<selector_t>(),
    "cub::DeviceBatchedTopK: the requested top-k configuration is not supported on at least one architecture in "
    "CMAKE_CUDA_ARCHITECTURES (the deterministic / large-segment cluster backend requires SM90+). Remove the "
    "unsupported architecture(s), relax the request, or define _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT to defer the "
    "diagnosis to runtime (cudaErrorNotSupported).");
#endif // strict unsupported-arch check

  detail::TripleChevronFactory launcher_factory{};
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  return detail::dispatch_compute_cap(selector_t{}, cc, [&](auto policy_getter) -> cudaError_t {
    CUB_DETAIL_CONSTEXPR_ISH auto active_policy = policy_getter();
    if CUB_DETAIL_CONSTEXPR_ISH (active_policy.backend == topk_backend::baseline)
    {
      return launch_baseline_arm<selector_t, LargeSegmentTileOffsetT>(
        d_temp_storage,
        temp_storage_bytes,
        d_key_segments_it,
        d_key_segments_out_it,
        d_value_segments_it,
        d_value_segments_out_it,
        segment_sizes,
        k,
        select_directions,
        num_segments,
        stream);
    }
    else if CUB_DETAIL_CONSTEXPR_ISH (active_policy.backend == topk_backend::cluster)
    {
      return launch_cluster_arm<selector_t, LargeSegmentTileOffsetT>(
        policy_getter,
        d_temp_storage,
        temp_storage_bytes,
        d_key_segments_it,
        d_key_segments_out_it,
        d_value_segments_it,
        d_value_segments_out_it,
        segment_sizes,
        k,
        select_directions,
        num_segments,
        stream);
    }
    else
    {
      // Unsupported on this architecture (e.g. a deterministic request on pre-SM90). Report a positive temp-storage
      // size so the two-phase protocol proceeds, then fail the launch explicitly.
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 1;
        return cudaSuccess;
      }
      return cudaErrorNotSupported;
    }
  });
}
} // namespace detail::batched_topk

CUB_NAMESPACE_END
