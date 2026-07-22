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
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

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
    return static_cast<TotalNumItemsValueType>(::cuda::ceil_div(
      params::__get_and_clamp_param_to_nonnegative(segment_sizes, segment_id), large_segment_agent_tile_size));
  }
};

// -----------------------------------------------------------------------------
// Automatic backend selector
// -----------------------------------------------------------------------------
// Stateless selector built purely from the compile-time request facts. It owns the entire backend decision, including
// computing `baseline_can_cover` from the concrete agent types -- the reason it lives here (where
// `baseline_can_cover_v` and the baseline agent are visible) rather than in the tuning header.
template <class KeyT,
          class ValueT,
          ::cuda::std::int64_t MaxK,
          ::cuda::std::int64_t StaticMaxSegSize,
          ::cuda::execution::determinism::__determinism_t Determinism,
          ::cuda::execution::tie_break::__tie_break_t TieBreak,
          class SegmentSizeParameterT,
          class KeyInputItItT,
          class KeyOutputItItT,
          class ValueInputItItT,
          class ValueOutputItItT,
          class KParameterT,
          class SelectDirectionParameterT,
          class NumSegmentsParameterT,
          class LargeSegmentTileOffsetT>
struct policy_selector_from_types
{
  // Whether a one-worker-per-segment (default baseline) policy fits the static max segment size in shared memory; feeds
  // the backend decision below.
  static constexpr bool baseline_can_cover = baseline_can_cover_v<
    baseline_policy_selector_from_types<KeyT, ValueT, ::cuda::std::int64_t, MaxK>,
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

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> topk_policy
  {
    constexpr bool deterministic = (Determinism != ::cuda::execution::determinism::__determinism_t::__not_guaranteed)
                                || (TieBreak != ::cuda::execution::tie_break::__tie_break_t::__unspecified);

    topk_algorithm backend = topk_algorithm::unsupported;
    if (deterministic || !baseline_can_cover)
    {
      // A deterministic result set / concrete tie-break preference, or a segment too large for the single-block
      // baseline, is served only by the cluster backend (SM 9.0+); otherwise the request cannot run here.
      backend = cluster_capable(cc) ? topk_algorithm::cluster : topk_algorithm::unsupported;
    }
    else
    {
      // Baseline can cover: use the cluster backend only where it is measured to win. The size crossover is a fixed
      // selector constant (not read from the tunable cluster policy), so tuning the cluster policy never shifts the
      // backend choice.
      const bool beneficial = cc >= ::cuda::compute_capability{cluster_beneficial_min_cc_major, 0}
                           && StaticMaxSegSize >= cluster_beneficial_min_segment_size;
      backend = (cluster_capable(cc) && beneficial) ? topk_algorithm::cluster : topk_algorithm::baseline;
    }
    return topk_policy{backend, make_baseline_policy(), make_cluster_policy()};
  }
};

// -----------------------------------------------------------------------------
// Dispatch (both backends behind one kernel symbol)
// -----------------------------------------------------------------------------
// The dispatch is host-only: it launches the single kernel symbol (`device_batched_topk_kernel`, in
// kernel_batched_topk.cuh) via the CUDA runtime. The algorithm does not support device-side (CDP) launch.

// Cluster arm of the dispatch (host-only): after the shared query-pass / CC-guard setup, launches the single kernel
// symbol via `cudaLaunchKernelEx` using the resolved-CC cluster policy and geometry from `policy_getter`.
// `select_directions` arrives already wrapped; the cluster tuning comes from `policy_getter` (the resolved-CC policy)
// and the requested `Determinism`/`TieBreak` from the dispatch. The kernel launch goes through `launcher_factory`; the
// cluster occupancy / shared-memory setup queries still use the CUDA runtime directly.
template <class PolicySelector,
          class LargeSegmentTileOffsetT,
          ::cuda::execution::determinism::__determinism_t Determinism,
          ::cuda::execution::tie_break::__tie_break_t TieBreak,
          class PolicyGetter,
          class KeyInputItItT,
          class KeyOutputItItT,
          class ValueInputItItT,
          class ValueOutputItItT,
          class SegmentSizeParameterT,
          class KParameterT,
          class SelectDirectionParameterT,
          class NumSegmentsParameterT,
          class KernelLauncherFactory>
_CCCL_HOST_API cudaError_t launch_cluster_arm(
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
  cudaStream_t stream,
  KernelLauncherFactory launcher_factory)
{
  // A tie-break preference is only meaningful once the result set itself is deterministic.
  static_assert(Determinism != ::cuda::execution::determinism::__determinism_t::__not_guaranteed
                  || TieBreak == ::cuda::execution::tie_break::__tie_break_t::__unspecified,
                "A tie-break preference requires a deterministic execution requirement");

  // The harness expects temp_storage_bytes > 0.
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

  // Single kernel symbol; its cluster vs baseline arm is selected device-side via `current_policy<PolicySelector>()`.
  // Taking its address here ODR-uses the `__global__` template, which is what drives its emission and registration.
  constexpr auto kernel_ptr = &device_batched_topk_kernel<
    PolicySelector,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT,
    LargeSegmentTileOffsetT,
    Determinism,
    TieBreak>;

  // Cluster sub-policy for the *resolved* architecture -- exactly what the device kernel instantiates via
  // `current_policy<PolicySelector>()`, so the host launch config (block size, shared-memory math) stays in lock-step
  // with the device policy per CC. `policy_getter()` is a constant expression, so `policy` is a non-type template arg.
  constexpr cluster_topk_policy policy = policy_getter().cluster;
  constexpr int ThreadsPerBlock        = policy.threads_per_block;
  constexpr int ChunkBytes             = policy.chunk_bytes;
  constexpr int LoadAlignBytes         = policy.load_align_bytes;
  constexpr int SingleBlockMaxSegSize  = policy.single_block_max_seg_size;
  constexpr int MinChunksPerBlock      = policy.min_chunks_per_block;
  constexpr int MaxBlocksPerCluster    = policy.max_blocks_per_cluster;
  constexpr int MaxChunkSlotsPerBlock  = policy.max_chunk_slots_per_block;
  static_assert(MaxBlocksPerCluster >= 0,
                "max_blocks_per_cluster must be 0 (unrestricted) or a positive cluster width");
  static_assert(MaxChunkSlotsPerBlock >= 0, "max_chunk_slots_per_block must be 0 (unrestricted) or a positive count");

  using key_it_t = it_value_t<KeyInputItItT>;
  using key_t    = it_value_t<key_it_t>;
  using layout_t = batched_topk_cluster::smem_block_tile_layout<key_t, ChunkBytes, LoadAlignBytes>;
  static_assert(is_valid_cluster_policy(policy));
  static_assert(LoadAlignBytes % int{sizeof(key_t)} == 0);

  // Tightest upper bound the segment-size argument carries -- for a static-bounded per-segment sequence a loose type
  // max, not the actual runtime maximum across segments.
  const auto max_seg_size  = ::cuda::args::__highest_(segment_sizes);
  using num_segments_val_t = typename ::cuda::args::__traits<NumSegmentsParameterT>::element_type;
  // `num_segments > 0` and `max_seg_size > 0` here: the generic `dispatch` returns for the empty-batch cases (no
  // segments, or a non-positive max segment size) before invoking this launch arm.
  const auto num_seg_val = detail::params::get_param(num_segments, num_segments_val_t{0});

  // Opt in to non-portable cluster blocks (>8 on Hopper).
  if (const auto error = launcher_factory.set_non_portable_cluster_allowed(kernel_ptr))
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
  // portion (`hw_dynamic_smem_bytes`) is that budget minus the static footprint (the policy slot cap may narrow it
  // further into `max_dynamic_smem_bytes`).
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
  if (const auto error = CubDebug(cudaFuncGetAttributes(&kernel_attrs, kernel_ptr)))
  {
    return error;
  }
  // `cudaDevAttrMaxSharedMemoryPerBlockOptin` already excludes the driver's per-block reserved shared memory
  // (opt-in == per-SM - reserved), so the dynamic budget is just the opt-in budget minus the static footprint;
  // reserved must not be subtracted a second time.
  const int nondynamic_smem_bytes = static_cast<int>(kernel_attrs.sharedSizeBytes);
  const int hw_dynamic_smem_bytes =
    (max_smem_optin_bytes > nondynamic_smem_bytes) ? max_smem_optin_bytes - nondynamic_smem_bytes : 0;
  // Optional policy cap on resident chunk slots per block (`max_chunk_slots_per_block == 0` -> unrestricted, i.e.
  // the full hardware budget). Expressed as the SMEM those slots need (`base_padding + slots * chunk_bytes`), then
  // clamped to the hardware budget: a cap the hardware cannot satisfy is a no-op (hardware wins). Fewer slots
  // lowers every CTA's resident dynamic shared-memory request, so a smaller segment overflows into streaming --
  // useful to leave shared memory free for a concurrent kernel (or to reach the streaming / schedule paths at a
  // small footprint in tests). A cap below one slot trips the `max_block_tile_capacity <= 0` guard below.
  const int max_dynamic_smem_bytes =
    (MaxChunkSlotsPerBlock == 0)
      ? hw_dynamic_smem_bytes
      : (::cuda::std::min) (hw_dynamic_smem_bytes,
                            layout_t::base_padding_bytes + MaxChunkSlotsPerBlock * layout_t::chunk_bytes);

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
          reinterpret_cast<const void*>(kernel_ptr), cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_bytes)))
    {
      return error;
    }
    configured_dynamic_smem_limit = dynamic_smem_bytes;
    return cudaSuccess;
  };

  // Segment/capacity basics that gate the launch shape. Computed before any occupancy query so the single-CTA fast
  // path below can be taken without one -- that driver query otherwise dominates the runtime of tiny launches.
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

  // `C_lo`: at the largest SMEM each CTA holds `max_block_tile_capacity`, the smallest resident `C`. Computed in
  // 64-bit because `max_seg_size` may be a loose bound (e.g. `numeric_limits<T>::max()` for an unbounded deferred
  // sequence); narrowing such a `C_lo` to `int` could wrap and wrongly enter the resident branch below.
  const auto c_lo = ::cuda::ceil_div(seg, static_cast<::cuda::std::uint64_t>(max_block_tile_capacity));

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
  else
  {
    // Hardware cluster-width ceiling, taken from the runtime rather than a compile-time constant so a future device
    // with wider (non-portable) clusters is not artificially capped (the same reason the dynamic-SMEM budget above
    // is queried, not hardcoded). Probed at zero dynamic SMEM to get the architectural/kernel ceiling independent
    // of the per-`C` SMEM chosen below; each candidate width is still validated against its own SMEM by the
    // resident scan's `max_active_clusters` probe, and the oversize branch re-probes at its (larger) launch SMEM.
    // Requires the non-portable opt-in already set above so the probe can report widths beyond the portable
    // ceiling.
    cluster_attr.val.clusterDim.x = 1; // ignored by `max_potential_cluster_size`
    cfg.gridDim                   = dim3(1, 1, 1);
    cfg.dynamicSmemBytes          = 0;
    int hw_cluster_ceiling        = 0;
    if (const auto error = launcher_factory.max_potential_cluster_size(hw_cluster_ceiling, kernel_ptr, &cfg))
    {
      return error;
    }
    if (hw_cluster_ceiling <= 0)
    {
      return cudaErrorInvalidValue;
    }
    // `MaxBlocksPerCluster == 0` -> the full hardware ceiling; a non-zero knob narrows it, clamped to that ceiling.
    // A cap narrower than a segment needs pushes it into the oversize/streaming fallback below.
    const int cluster_cap =
      (MaxBlocksPerCluster == 0) ? hw_cluster_ceiling : (::cuda::std::min) (MaxBlocksPerCluster, hw_cluster_ceiling);

    // Wave-aware cluster-blocks selection. The free variable is the cluster blocks `C` (one cluster per segment);
    // each `C` is paired with the smallest dynamic SMEM that keeps a segment fully resident. A smaller `C` needs
    // more SMEM (fewer clusters-per-wave, less L1); a larger `C` needs less SMEM (more clusters-per-wave, more L1).
    // We pick the `C` that minimizes waves, breaking ties toward the largest `C` (= smallest SMEM = most L1), which
    // matches the profiled fast configs. `C` is enumerated analytically rather than by discovering SMEM tiers via
    // occupancy, so a register-limited occupancy (e.g. 1 CTA/SM) cannot collapse the candidate set. `C_full`: at
    // the 1-chunk SMEM each CTA holds `chunk_items`, so full residency needs this many CTAs (capped at
    // `cluster_cap`).
    const int c_full = static_cast<int>(
      (::cuda::std::min) (static_cast<::cuda::std::uint64_t>(cluster_cap), ::cuda::ceil_div(seg, chunk_items_u64)));
    // Cluster blocks the max segment actually needs (shared with the device so the launch is never wider than
    // necessary). At `min_chunks_per_block == 1` this equals `c_full`; a larger knob shrinks it.
    const int desired_cluster_blocks = ::cuda::narrow<int>(batched_topk_cluster::effective_cluster_blocks_from_chunks(
      ::cuda::ceil_div(seg, chunk_items_u64), MinChunksPerBlock, ::cuda::narrow<unsigned int>(cluster_cap)));

    if (c_lo <= static_cast<::cuda::std::uint64_t>(cluster_cap))
    {
      // Full residency is achievable. `seg <= C_lo * max_block_tile_capacity` with `C_lo <= cluster_cap`, so every
      // per-CTA capacity (and thus its slot count and SMEM bytes) below stays well within `int` -- no overflow.
      // Scan `C` in `[max(C_lo, 2), C_end]`, minimize waves, tie-break largest `C`. The upper bound is capped at
      // the cluster blocks the max segment needs (`desired_cluster_blocks`), so the host never launches a wider
      // cluster than necessary; at `min_chunks_per_block == 1` the cap equals `c_full`. `C_end` is additionally
      // clamped to `cluster_cap`, which empties the range only at `cluster_cap == 1` (`c_begin == 2 > 1`) --
      // reachable here for a one-CTA-resident segment whose single-CTA fast path is disabled. Then `c_lo == 1` and
      // the fallback below takes the single CTA, so the width cap is never exceeded.
      const int c_begin = (::cuda::std::max) (2, static_cast<int>(c_lo));
      const int c_end =
        (::cuda::std::min) (cluster_cap,
                            (::cuda::std::max) (c_begin, (::cuda::std::min) (c_full, desired_cluster_blocks)));
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
        if (const auto error = launcher_factory.max_active_clusters(clusters_per_wave, kernel_ptr, &cfg))
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
      // Oversize (`C_lo > cluster_cap`) or nothing launchable in range: full residency is impossible, so maximize
      // residency with the largest launchable cluster at the largest SMEM and let the agent stream the overflow.
      if (const auto error = ensure_dynamic_smem_limit(max_dynamic_smem_bytes))
      {
        return error;
      }
      cluster_attr.val.clusterDim.x = 1; // ignored by `max_potential_cluster_size`
      cfg.gridDim                   = dim3(1, 1, 1);
      cfg.dynamicSmemBytes          = static_cast<unsigned int>(max_dynamic_smem_bytes);
      int hw_max_cluster_blocks     = 0;
      if (const auto error = launcher_factory.max_potential_cluster_size(hw_max_cluster_blocks, kernel_ptr, &cfg))
      {
        return error;
      }
      hw_max_cluster_blocks = (::cuda::std::min) (hw_max_cluster_blocks, cluster_cap);
      if (hw_max_cluster_blocks <= 0)
      {
        return cudaErrorInvalidValue;
      }
      cluster_blocks   = hw_max_cluster_blocks;
      dynamic_smem_sel = max_dynamic_smem_bytes;
    }
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

  // The cluster dimension routes the host launch through `cudaLaunchKernelEx`.
  if (const auto error = CubDebug(
        launcher_factory(dim3(static_cast<unsigned int>(grid_blocks), 1, 1),
                         dim3(static_cast<unsigned int>(ThreadsPerBlock), 1, 1),
                         static_cast<::cuda::std::size_t>(dynamic_smem_bytes),
                         stream,
                         /*dependent_launch=*/false,
                         dim3(static_cast<unsigned int>(cluster_blocks), 1, 1))
          .doit(kernel_ptr,
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

  // Cluster launches can fail on the device while reporting success; sync.
  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  return CubDebug(detail::DebugSyncStream(stream));
}

// Baseline host-launch arm of the dispatch. Launches the single kernel symbol
// (`device_batched_topk_kernel`, packing the large-segment bookkeeping into `baseline_kernel_args` and passing an empty
// `cluster_kernel_args`). `select_directions` arrives already wrapped and the baseline tuning is taken from the
// `PolicySelector` (via `baseline_policy_selector_adaptor`). All kernel launches, memsets and nested scans go through
// `launcher_factory`.
template <class PolicySelector,
          class LargeSegmentTileOffsetT,
          ::cuda::execution::determinism::__determinism_t Determinism,
          ::cuda::execution::tie_break::__tie_break_t TieBreak,
          class KeyInputItItT,
          class KeyOutputItItT,
          class ValueInputItItT,
          class ValueOutputItItT,
          class SegmentSizeParameterT,
          class KParameterT,
          class SelectDirectionParameterT,
          class NumSegmentsParameterT,
          class KernelLauncherFactory>
_CCCL_HOST_API cudaError_t launch_baseline_arm(
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
  cudaStream_t stream,
  KernelLauncherFactory launcher_factory)
{
  // Whether some one-worker-per-segment policy covers the static max segment size within the shared-memory limit.
  // Computed from this call's concrete agent types (works for both the default selector and a tuning override, which
  // does not expose a `baseline_can_cover` member). The selector never routes here when this is false, so the arm is
  // pruned per-arch in AOT builds; kept assert-free so it also compiles under runtime-policies mode.
  constexpr bool baseline_can_cover = baseline_can_cover_v<
    baseline_policy_selector_adaptor<PolicySelector>,
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
  if constexpr (!baseline_can_cover)
  {
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
      // TODO(topk): once this large-segment path is live, guard the `num_segments_val * sizeof(...)` byte counts
      // against size_t overflow (safe today only because the entry bounds num_segments_val to <= INT_MAX).
      allocation_sizes[0] = num_segments_val * sizeof(large_segment_tile_offset_t);
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
              stream,
              {},
              {},
              launcher_factory)))
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

    // `num_segments > 0` and the max segment size > 0 here: the generic `dispatch` returns for the empty-batch cases
    // (no segments, or a non-positive max segment size) before invoking this launch arm.

    if constexpr (any_small_segments)
    {
      if constexpr (!only_small_segments)
      {
        // Zero-initialize the counters struct read by the agent's atomics.
        if (const auto error = CubDebug(launcher_factory.MemsetAsync(allocations[1], 0, sizeof(counters_t), stream)))
        {
          return error;
        }
      }
      const int grid_dim      = static_cast<int>(params::get_param(num_segments, 0));
      constexpr int block_dim = worker_per_segment_policy.threads_per_block;
      if (const auto error = CubDebug(
            launcher_factory(grid_dim, block_dim, 0, stream, /*dependent_launch=*/false)
              .doit(
                device_batched_topk_kernel<
                  PolicySelector,
                  KeyInputItItT,
                  KeyOutputItItT,
                  ValueInputItItT,
                  ValueOutputItItT,
                  SegmentSizeParameterT,
                  KParameterT,
                  SelectDirectionParameterT,
                  NumSegmentsParameterT,
                  large_segment_tile_offset_t,
                  Determinism,
                  TieBreak>,
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
            stream,
            {},
            {},
            launcher_factory)))
      {
        return error;
      }
    }

    return CubDebug(detail::DebugSyncStream(stream));
  }
}

#if _CCCL_CUDA_COMPILATION() && !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)
// Returns true if at least one architecture this translation unit targets (the compile target list exposed as
// `::cuda::__target_compute_capabilities()`) resolves to the `unsupported` backend for `PolicySelector` -- e.g. a
// deterministic request while a pre-SM90 target is present in the list. Used to turn a would-be runtime
// `cudaErrorNotSupported` into a compile-time diagnostic (see the static_assert in `dispatch`).
template <class PolicySelector>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL bool any_target_cc_unsupported()
{
  bool any = false;
  for (const auto cc : ::cuda::__target_compute_capabilities())
  {
    any = any || (PolicySelector{}(cc).backend == topk_algorithm::unsupported);
  }
  return any;
}
#endif // _CCCL_CUDA_COMPILATION() && !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)

// Internal entry point: the single dispatch that replaces the standalone baseline / cluster dispatches. It resolves the
// runtime compute capability, then uses `dispatch_compute_cap` to pick, per architecture, the backend chosen by the
// resolved policy selector (deterministic -> cluster; otherwise the arch+size crossover). Both host arms launch the
// same kernel symbol. `Determinism`/`TieBreak` are compile-time selection inputs.
//
// `tuning_env` carries an optional `tune`d policy selector (keyed on `topk_policy`): when present it fully replaces the
// automatic selector -- its `.backend` chooses the arm and its `.baseline`/`.cluster` carry the tunings. Matching
// DeviceScan/DeviceTransform, the tuned backend choice is trusted; only the determinism/tie-break guard below still
// applies. `launcher_factory` routes the kernel launches, memsets, nested scans and the routing CC query (the cluster
// arm's occupancy / shared-memory queries still call the CUDA runtime directly).
template <
  ::cuda::execution::determinism::__determinism_t Determinism =
    ::cuda::execution::determinism::__determinism_t::__not_guaranteed,
  ::cuda::execution::tie_break::__tie_break_t TieBreak = ::cuda::execution::tie_break::__tie_break_t::__unspecified,
  class KeyInputItItT,
  class KeyOutputItItT,
  class ValueInputItItT,
  class ValueOutputItItT,
  class SegmentSizeParameterT,
  class KParameterT,
  class SelectDirectionT,
  class NumSegmentsParameterT,
  class TotalNumItemsGuaranteeT,
  class TuningEnvT            = ::cuda::std::execution::env<>,
  class KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
_CCCL_HOST_API cudaError_t dispatch(
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
  cudaStream_t stream,
  TuningEnvT tuning_env                  = {},
  KernelLauncherFactory launcher_factory = {})
{
  // Both arms resolve `num_segments` on the host via detail::params::get_param (allocation sizing, grid extent,
  // empty-batch guard), so it must be a host-known single value; device-side counts are future work (see TODOs below).
  static_assert(::cuda::args::__traits<NumSegmentsParameterT>::is_single_value
                  && !::cuda::args::__traits<NumSegmentsParameterT>::is_deferred,
                "cub::DeviceBatchedTopK requires a host-known uniform number of segments (constant, immediate, or a "
                "plain integral): a per-segment sequence is not a meaningful segment count, and a single deferred "
                "(device-resident) value is meaningful but not yet supported (resolve the count on the host).");

  // The selection direction is a compile-time constant carried as `::cuda::args::constant<Dir>`. Wrap it into the
  // internal discrete param the kernel/agent expect (both host arms take the wrapped form).
  // Type derived from the parameter type rather than `decltype(select_directions)`: GCC 7 rejects the latter ("use of
  // 'select_directions' before deduction of 'auto'") when it feeds the `constexpr baseline_can_cover` initializer
  // below. Declaring `select_directions` with the alias keeps its (const-qualified) type single-sourced.
  using SelectDirectionParameterT = const decltype(wrap_select_direction(::cuda::std::declval<SelectDirectionT>()));
  SelectDirectionParameterT select_directions = wrap_select_direction(select_direction);

  using key_t                   = it_value_t<it_value_t<KeyInputItItT>>;
  using value_t                 = it_value_t<it_value_t<ValueInputItItT>>;
  using LargeSegmentTileOffsetT = typename ::cuda::args::__traits<TotalNumItemsGuaranteeT>::element_type;

  constexpr ::cuda::std::int64_t max_k          = ::cuda::args::__traits<KParameterT>::highest;
  constexpr ::cuda::std::int64_t static_max_seg = ::cuda::args::__traits<SegmentSizeParameterT>::highest;

  // Default automatic selector from the compile-time inputs; it computes its own baseline coverage. A `tune`d selector
  // in the environment (keyed on `topk_policy`) replaces it wholesale.
  using default_selector_t = policy_selector_from_types<
    key_t,
    value_t,
    max_k,
    static_max_seg,
    Determinism,
    TieBreak,
    SegmentSizeParameterT,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT,
    LargeSegmentTileOffsetT>;

  auto policy_selector = ::cuda::std::execution::__query_or(tuning_env, topk_policy{}, default_selector_t{});
  using selector_t     = decltype(policy_selector);
#if _CCCL_HAS_CONCEPTS()
  static_assert(detail::policy_selector<selector_t, topk_policy>,
                "Invalid policy selector for cub::DeviceBatchedTopK::dispatch");
#endif // _CCCL_HAS_CONCEPTS()

  // A deterministic result set / concrete tie-break preference can only be honored by the cluster backend. A `tune`d
  // selector that forces the baseline backend for such a request cannot serve it; guard it below.
  [[maybe_unused]] constexpr bool deterministic =
    (Determinism != ::cuda::execution::determinism::__determinism_t::__not_guaranteed)
    || (TieBreak != ::cuda::execution::tie_break::__tie_break_t::__unspecified);

#if _CCCL_CUDA_COMPILATION() && !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC) \
  && !defined(CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT)
  // Strict mode (default): fail at compile time if the request cannot be served on *any* architecture this translation
  // unit targets (e.g. a deterministic / large-segment request while a pre-SM90 target is present, since the cluster
  // backend requires SM90+). This is the least-surprising UX for callers whose build targets multiple architectures.
  // Define `CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT` to defer the diagnosis to runtime instead (the dispatch then
  // returns `cudaErrorNotSupported` on unsupported devices); CUB's own tests and benchmarks do this so they can compile
  // the full configuration space across all target architectures and skip at runtime where unsupported.
  static_assert(
    !any_target_cc_unsupported<selector_t>(),
    "cub::DeviceBatchedTopK: the requested top-k configuration is not supported on at least one architecture this "
    "translation unit targets (the deterministic / large-segment cluster backend requires SM90+). Remove the "
    "unsupported architecture(s), relax the request, or define CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT to defer the "
    "diagnosis to runtime (cudaErrorNotSupported).");
#endif // _CCCL_CUDA_COMPILATION() && !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)
       // && !defined(CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT)

  // The supported maximum segment size (2^21) is enforced at compile time at the public entry; a statically negative
  // lower bound is allowed and negative runtime sizes are clamped to 0 (see
  // detail::params::__get_and_clamp_param_to_nonnegative). A per-segment value outside its declared bound is a caller
  // error (UB): the statically declared bounds are validated at compile time, while the argument values are
  // bounds-checked only by assertions active in assertion-enabled (e.g. debug) builds -- host-side for a host-known
  // immediate value and device-side for values read from a deferred / deferred_sequence handle.

  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  // `num_segments` is the grid extent of both host launch arms (the baseline arm launches one block per segment; the
  // cluster arm launches `num_segments * cluster_blocks`), so it must fit a positive 32-bit grid dimension. Reject an
  // out-of-contract count (negative, or exceeding INT_MAX) at this single host boundary: otherwise the baseline arm
  // would silently narrow it to `int` and the cluster arm's 64-bit grid-size product could overflow before its own
  // range check.
  using num_segments_val_t                  = typename ::cuda::args::__traits<NumSegmentsParameterT>::element_type;
  const num_segments_val_t num_segments_val = detail::params::get_param(num_segments, num_segments_val_t{0});
  // Unary `+` integer-promotes the count to a standard integer type so the sign-safe `cmp_*` comparators accept it:
  // they are constrained to `__cccl_is_integer_v`, which excludes the character count types the public API permits.
  if (::cuda::std::cmp_less(+num_segments_val, 0)
      || ::cuda::std::cmp_greater(+num_segments_val, ::cuda::std::numeric_limits<int>::max()))
  {
    return cudaErrorInvalidValue;
  }

  // Empty batch = no work to launch: no segments, or a non-positive tightest max segment size (every segment empty,
  // e.g. a uniform negative size clamped to 0). `== 0` not `<= 0` for `num_segments`, as a negative count is out of
  // contract. Consulted only on the launch (`d_temp_storage != nullptr`) of a *supported* arm below: the query pass
  // falls through to size `temp_storage_bytes`, and the unsupported arm ignores it so an unavailable request still
  // fails with cudaErrorNotSupported rather than being masked into success.
  const auto empty_batch_no_launch = [&] {
    return d_temp_storage != nullptr
        && (detail::params::get_param(num_segments, 0) == 0 || ::cuda::args::__highest_(segment_sizes) <= 0);
  };

  return detail::dispatch_compute_cap(policy_selector, cc, [&](auto policy_getter) -> cudaError_t {
    constexpr topk_policy active_policy = policy_getter();
    if constexpr (active_policy.backend == topk_algorithm::baseline)
    {
      if constexpr (deterministic)
      {
        // A `tune`d selector forced the baseline backend for a deterministic / tie-break request it cannot honor.
        // Report a positive temp-storage size so the two-phase protocol proceeds, then fail the launch explicitly.
        if (d_temp_storage == nullptr)
        {
          temp_storage_bytes = 1;
          return cudaSuccess;
        }
        return cudaErrorNotSupported;
      }
      else
      {
        if (empty_batch_no_launch())
        {
          return cudaSuccess;
        }
        return launch_baseline_arm<selector_t, LargeSegmentTileOffsetT, Determinism, TieBreak>(
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
          stream,
          launcher_factory);
      }
    }
    else if constexpr (active_policy.backend == topk_algorithm::cluster)
    {
      if (empty_batch_no_launch())
      {
        return cudaSuccess;
      }
      return launch_cluster_arm<selector_t, LargeSegmentTileOffsetT, Determinism, TieBreak>(
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
        stream,
        launcher_factory);
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
