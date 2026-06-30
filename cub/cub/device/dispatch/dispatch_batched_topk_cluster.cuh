// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Cluster-based batched top-k dispatch.
//!
//! Prototype that launches a grid of thread block clusters to compute a
//! segmented top-k. Each cluster processes one segment end-to-end: private
//! histograms are reduced into the leader block via DSMEM atomics, then every
//! block reads the merged histogram back through DSMEM, locally identifies the
//! k-th bucket, and refines its shared-memory cluster_tile across radix passes.
//!
//! Two kernels share the agent body:
//!   * Host: no `__cluster_dims__`, launched via `cudaLaunchKernelExC` with the
//!     cluster blocks chosen at runtime (up to 16 on Hopper).
//!   * CDP: static `__cluster_dims__(max_portable_cluster_blocks, 1, 1)`,
//!     gated on `CUB_RDC_ENABLED` so CDP-disabled builds skip emitting it
//!     (same pattern as `dispatch_segmented_sort.cuh`).

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
#include <cub/detail/env_dispatch.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk_cluster.cuh>
#include <cub/util_device.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/round_up.h>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__numeric/narrow.h>
#include <cuda/argument>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <nv/target>

#include <cuda_runtime.h>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk_cluster
{
// -----------------------------------------------------------------------------
// Cluster-size / dynamic-SMEM selection
// -----------------------------------------------------------------------------
// Tightest upper bound carried by the segment-size argument. Mirrors `args::__traits<>::highest` semantics:
// the compile-time bound for `constant`/bounded sequence arguments and the runtime value for a uniform
// `immediate`. For a per-segment sequence with only a static bound this can be the loose `numeric_limits<T>::max()`.
template <typename SegmentSizeParameterT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto runtime_max_segment_size(SegmentSizeParameterT segment_sizes) noexcept
{
  return ::cuda::args::__highest_(segment_sizes);
}

// -----------------------------------------------------------------------------
// Kernel entry points
// -----------------------------------------------------------------------------
// Dynamic-cluster kernel for host launches; the agent reads the active cluster
// width via cooperative groups.
template <int ThreadsPerBlock,
          int MinBlocksPerSm,
          int HistogramItemsPerThread,
          int PipelineStages,
          int ChunkBytes,
          int LoadAlignBytes,
          int BitsPerPass,
          int TieBreakItemsPerThread,
          int SingleCtaMaxSegmentSize,
          int MinChunksPerCta,
          int CopyItemsPerThread,
          int FirstResidentSlotSubdivision,
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
__launch_bounds__(ThreadsPerBlock, MinBlocksPerSm) _CCCL_KERNEL_ATTRIBUTES void device_segmented_topk_cluster_kernel(
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k_param,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments,
  ::cuda::std::uint32_t block_tile_capacity)
{
  using agent_t = agent_batched_topk_cluster<
    ThreadsPerBlock,
    HistogramItemsPerThread,
    PipelineStages,
    ChunkBytes,
    LoadAlignBytes,
    BitsPerPass,
    TieBreakItemsPerThread,
    SingleCtaMaxSegmentSize,
    MinChunksPerCta,
    CopyItemsPerThread,
    FirstResidentSlotSubdivision,
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

  __shared__ typename agent_t::TempStorage temp_storage;
  extern __shared__ char topk_cluster_smem[];
  char* key_slots = topk_cluster_smem;
  // Align the base up to `slot_alignment` (>= load_align) so every bulk-copy destination gets the same `load_align`
  // alignment the gmem sources have (peak TMA throughput on Hopper). The layout reserves `base_padding_bytes` for this.
  {
    ::cuda::std::uint32_t smem32 = __cvta_generic_to_shared(key_slots);
    smem32 = ::cuda::round_up(smem32, static_cast<::cuda::std::uint32_t>(agent_t::slot_alignment));
    asm("" : "+r"(smem32));
    key_slots = static_cast<char*>(__cvta_shared_to_generic(smem32));
  }

  agent_t agent(
    temp_storage,
    d_key_segments_it,
    d_key_segments_out_it,
    d_value_segments_it,
    d_value_segments_out_it,
    segment_sizes,
    k_param,
    select_directions,
    num_segments,
    key_slots,
    block_tile_capacity);

  agent.Process();
}

#ifdef CUB_RDC_ENABLED
// CDP-only static-cluster kernel: compile-time `__cluster_dims__` so the
// triple-chevron launch from device code needs no `cudaFuncSetAttribute`.
template <int ThreadsPerBlock,
          int HistogramItemsPerThread,
          int PipelineStages,
          int ChunkBytes,
          int LoadAlignBytes,
          int BitsPerPass,
          int TieBreakItemsPerThread,
          int SingleCtaMaxSegmentSize,
          int MinChunksPerCta,
          int CopyItemsPerThread,
          int FirstResidentSlotSubdivision,
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
__launch_bounds__(ThreadsPerBlock) __cluster_dims__(max_portable_cluster_blocks, 1, 1)
  _CCCL_KERNEL_ATTRIBUTES void device_segmented_topk_cluster_kernel_static(
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    ValueInputItItT d_value_segments_it,
    ValueOutputItItT d_value_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k_param,
    SelectDirectionParameterT select_directions,
    NumSegmentsParameterT num_segments,
    ::cuda::std::uint32_t block_tile_capacity)
{
  using agent_t = agent_batched_topk_cluster<
    ThreadsPerBlock,
    HistogramItemsPerThread,
    PipelineStages,
    ChunkBytes,
    LoadAlignBytes,
    BitsPerPass,
    TieBreakItemsPerThread,
    SingleCtaMaxSegmentSize,
    MinChunksPerCta,
    CopyItemsPerThread,
    FirstResidentSlotSubdivision,
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

  __shared__ typename agent_t::TempStorage temp_storage;
  extern __shared__ char topk_cluster_smem[];
  char* key_slots = topk_cluster_smem;
  // Align the base up to `slot_alignment` (>= load_align) so every bulk-copy destination gets the same `load_align`
  // alignment the gmem sources have (peak TMA throughput on Hopper). The layout reserves `base_padding_bytes` for this.
  {
    ::cuda::std::uint32_t smem32 = __cvta_generic_to_shared(key_slots);
    smem32 = ::cuda::round_up(smem32, static_cast<::cuda::std::uint32_t>(agent_t::slot_alignment));
    asm("" : "+r"(smem32));
    key_slots = static_cast<char*>(__cvta_shared_to_generic(smem32));
  }

  agent_t agent(
    temp_storage,
    d_key_segments_it,
    d_key_segments_out_it,
    d_value_segments_it,
    d_value_segments_out_it,
    segment_sizes,
    k_param,
    select_directions,
    num_segments,
    key_slots,
    block_tile_capacity);

  agent.Process();
}
#endif // CUB_RDC_ENABLED

// -----------------------------------------------------------------------------
// Dispatch
// -----------------------------------------------------------------------------
// Keys and key/value pairs; segments larger than the resident block tile are streamed from gmem by the agent. The host
// path picks `(cluster_blocks, dynamic_smem_bytes)` at runtime via a wave-aware cluster-size search (see below); CDP
// uses the static kernel at `max_portable_cluster_blocks` and portable SMEM.

// CDP launch body, empty when CDP is disabled. Wrapped in a macro because
// `#ifdef` can't sit inside `NV_IF_TARGET`.
#ifndef CUB_RDC_ENABLED
#  define CUB_TOPK_CLUSTER_DEVICE_LAUNCH
#else // CUB_RDC_ENABLED
#  define CUB_TOPK_CLUSTER_DEVICE_LAUNCH                                                \
    auto static_kernel = device_segmented_topk_cluster_kernel_static<                   \
      ThreadsPerBlock,                                                                  \
      HistogramItemsPerThread,                                                          \
      PipelineStages,                                                                   \
      ChunkBytes,                                                                       \
      LoadAlignBytes,                                                                   \
      BitsPerPass,                                                                      \
      TieBreakItemsPerThread,                                                           \
      SingleCtaMaxSegmentSize,                                                          \
      MinChunksPerCta,                                                                  \
      CopyItemsPerThread,                                                               \
      FirstResidentSlotSubdivision,                                                     \
      Determinism,                                                                      \
      TieBreak,                                                                         \
      KeyInputItItT,                                                                    \
      KeyOutputItItT,                                                                   \
      ValueInputItItT,                                                                  \
      ValueOutputItItT,                                                                 \
      SegmentSizeParameterT,                                                            \
      KParameterT,                                                                      \
      SelectDirectionParameterT,                                                        \
      NumSegmentsParameterT>;                                                           \
    if (const auto error = CubDebug(                                                    \
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(                        \
            static_cast<int>(grid_blocks), ThreadsPerBlock, dynamic_smem_bytes, stream) \
            .doit(static_kernel,                                                        \
                  d_key_segments_it,                                                    \
                  d_key_segments_out_it,                                                \
                  d_value_segments_it,                                                  \
                  d_value_segments_out_it,                                              \
                  segment_sizes,                                                        \
                  k_param,                                                              \
                  select_directions,                                                    \
                  num_segments,                                                         \
                  block_tile_capacity)))                                                \
    {                                                                                   \
      return error;                                                                     \
    }
#endif // CUB_RDC_ENABLED

// `Determinism`/`TieBreak` carry the requested `cuda::execution` requirements down to the kernel and agent (the public
// env-based interface is handled separately). They default to the nondeterministic, racing-atomics behavior.
template <
  ::cuda::execution::determinism::__determinism_t Determinism =
    ::cuda::execution::determinism::__determinism_t::__not_guaranteed,
  ::cuda::execution::tie_break::__tie_break_t TieBreak = ::cuda::execution::tie_break::__tie_break_t::__unspecified,
  typename KeyInputItItT,
  typename KeyOutputItItT,
  typename ValueInputItItT,
  typename ValueOutputItItT,
  typename SegmentSizeParameterT,
  typename KParameterT,
  typename SelectDirectionT,
  typename NumSegmentsParameterT,
  typename TotalNumItemsGuaranteeT,
  typename PolicySelector = policy_selector>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k_param,
  SelectDirectionT select_direction,
  NumSegmentsParameterT num_segments,
  [[maybe_unused]] TotalNumItemsGuaranteeT total_num_items_guarantee,
  cudaStream_t stream                             = nullptr,
  [[maybe_unused]] PolicySelector policy_selector = {})
{
  // The selection direction is a compile-time constant carried as `::cuda::args::constant<Dir>`. Wrap it into
  // the internal discrete param the kernel/agent expect, exactly as the baseline `batched_topk::dispatch` does.
  auto select_directions          = batched_topk::wrap_select_direction(select_direction);
  using SelectDirectionParameterT = decltype(select_directions);
  // Clusters are SM 9.0+ only and the tuning is currently identical across
  // CCs, so pin the selector query to the minimum supported CC.
  constexpr cluster_topk_policy policy       = PolicySelector{}(::cuda::compute_capability{9, 0});
  constexpr int ThreadsPerBlock              = policy.threads_per_block;
  constexpr int MinBlocksPerSm               = policy.min_blocks_per_sm;
  constexpr int HistogramItemsPerThread      = policy.histogram_items_per_thread;
  constexpr int PipelineStages               = policy.pipeline_stages;
  constexpr int ChunkBytes                   = policy.chunk_bytes;
  constexpr int LoadAlignBytes               = policy.load_align_bytes;
  constexpr int BitsPerPass                  = policy.bits_per_pass;
  constexpr int TieBreakItemsPerThread       = policy.tie_break_items_per_thread;
  constexpr int SingleCtaMaxSegmentSize      = policy.single_cta_max_segment_size;
  constexpr int MinChunksPerCta              = policy.min_chunks_per_cta;
  constexpr int CopyItemsPerThread           = policy.copy_items_per_thread;
  constexpr int FirstResidentSlotSubdivision = policy.first_resident_slot_subdivision;

  using key_it_t = it_value_t<KeyInputItItT>;
  using key_t    = it_value_t<key_it_t>;
  using layout_t = smem_block_tile_layout<key_t, ChunkBytes, LoadAlignBytes>;
  using agent_t  = agent_batched_topk_cluster<
     ThreadsPerBlock,
     HistogramItemsPerThread,
     PipelineStages,
     ChunkBytes,
     LoadAlignBytes,
     BitsPerPass,
     TieBreakItemsPerThread,
     SingleCtaMaxSegmentSize,
     MinChunksPerCta,
     CopyItemsPerThread,
     FirstResidentSlotSubdivision,
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

  static_assert(ChunkBytes % LoadAlignBytes == 0);
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

  constexpr auto dynamic_kernel = &device_segmented_topk_cluster_kernel<
    ThreadsPerBlock,
    MinBlocksPerSm,
    HistogramItemsPerThread,
    PipelineStages,
    ChunkBytes,
    LoadAlignBytes,
    BitsPerPass,
    TieBreakItemsPerThread,
    SingleCtaMaxSegmentSize,
    MinChunksPerCta,
    CopyItemsPerThread,
    FirstResidentSlotSubdivision,
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
      // necessary). At `min_chunks_per_cta == 1` this equals `c_full`; a larger knob shrinks it.
      const int desired_cluster_blocks = static_cast<int>(effective_cluster_blocks_from_chunks(
        ::cuda::ceil_div(seg, chunk_items_u64),
        MinChunksPerCta,
        static_cast<unsigned int>(max_supported_cluster_blocks)));

      int cluster_blocks   = 0;
      int dynamic_smem_sel = 0;

      if (single_cta_eligible(seg, static_cast<::cuda::std::uint64_t>(max_block_tile_capacity), SingleCtaMaxSegmentSize))
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
        // never launches a wider cluster than necessary; at `min_chunks_per_cta == 1` the cap equals `c_full`.
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

      // The cluster dimension routes the launch through `cudaLaunchKernelEx`; the sibling triple-chevron in
      // `doit_host` forces NVCC to emit `dynamic_kernel` for this TU.
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
                    block_tile_capacity)))
      {
        return error;
      }
    }),
    ({
      // CDP path: device-side launches cannot opt in to more than portable
      // total SMEM or non-portable cluster blocks. Segments that exceed the
      // portable resident coverage are still handled: the agent re-streams the
      // overflow chunks from gmem.
      constexpr int portable_total_smem_bytes = 48 * 1024;
      constexpr int dynamic_smem_bytes =
        (portable_total_smem_bytes > static_smem_bytes) ? portable_total_smem_bytes - static_smem_bytes : 0;

      // The compile-time `ChunkBytes` is reused verbatim for the device launch: the agent peels the unaligned boundary
      // edges into a tiny per-block buffer, so streaming needs only a single resident-or-streaming slot, and segments
      // exceeding the small portable-SMEM block tile are handled by re-streaming overflow from gmem. The only hard
      // requirement is that at least one load-aligned chunk fits the worst-case portable SMEM block tile.
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

// Env-based dispatch that also handles temporary-storage allocation. This is usually done by the device-layer, but
// there is no public API for cluster segmented top-k yet. Mirrors `batched_topk::dispatch_with_env`: the cluster
// algorithm is single-phase (one kernel launch over a placeholder allocation), but routing it through the shared
// env-based machinery keeps the call shape identical to the baseline backend and lets it pick up the stream, memory
// resource, and tuning carried by the environment.
//
// `Determinism`/`TieBreak` are forwarded verbatim to `dispatch` (the public env-based interface is handled separately);
// they default to the nondeterministic, unspecified-tie-break behavior.
template <
  ::cuda::execution::determinism::__determinism_t Determinism =
    ::cuda::execution::determinism::__determinism_t::__not_guaranteed,
  ::cuda::execution::tie_break::__tie_break_t TieBreak = ::cuda::execution::tie_break::__tie_break_t::__unspecified,
  typename KeyInputItItT,
  typename KeyOutputItItT,
  typename ValueInputItItT,
  typename ValueOutputItItT,
  typename SegmentSizeParameterT,
  typename KParameterT,
  typename SelectDirectionT,
  typename NumSegmentsParameterT,
  typename TotalNumItemsGuaranteeT,
  typename EnvT = ::cuda::std::execution::env<>>
[[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_with_env(
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k_param,
  SelectDirectionT select_direction,
  NumSegmentsParameterT num_segments,
  TotalNumItemsGuaranteeT total_num_items_guarantee,
  EnvT env = {})
{
  return detail::dispatch_with_env_and_tuning<policy_selector>(
    env, [&](auto policy_sel, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
      return dispatch<Determinism, TieBreak>(
        d_temp_storage,
        temp_storage_bytes,
        d_key_segments_it,
        d_key_segments_out_it,
        d_value_segments_it,
        d_value_segments_out_it,
        segment_sizes,
        k_param,
        select_direction,
        num_segments,
        total_num_items_guarantee,
        stream,
        policy_sel);
    });
}
} // namespace detail::batched_topk_cluster

CUB_NAMESPACE_END
