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
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__host_stdlib/sstream>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>
#include <cuda/std/expected>
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
template <detail::topk::select Dir, typename _Tp>
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
template <typename SegmentSizeParameterT, typename TotalNumItemsValueType>
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
template <typename KeyT,
          typename ValueT,
          ::cuda::std::int64_t MaxK,
          ::cuda::std::int64_t StaticMaxSegSize,
          ::cuda::execution::determinism::__determinism_t Determinism,
          ::cuda::execution::tie_break::__tie_break_t TieBreak,
          typename SegmentSizeParameterT,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename LargeSegmentTileOffsetT>
struct policy_selector_from_types
{
  // TODO(bgruber): to let the baseline policy vary per CC, move this coverage check into operator() and evaluate it for
  // the passed CC. Only the check is hard: it instantiates the agent for sizeof(TempStorage), so it needs the CC as a
  // compile-time constant, whereas operator()'s `cc` is a runtime parameter (building the policy itself is just the
  // value make_baseline_policy(cc)). Recover the compile-time CC by folding over
  // ::cuda::__target_compute_capabilities() (as detail::dispatch_to_cc_list does) and evaluate baseline_can_cover_v for
  // the matching CC. That also removes the invariant below, since coverage and the returned baseline would then derive
  // from the same cc.

  // note: the baseline policy passed to baseline_can_cover_v must be the same as returned from operator(cc) below
  static constexpr baseline_topk_policy baseline_policy = make_baseline_policy();

  struct policy_getter_17 // TODO(bgruber): remove in C++20 and pass policy by value
  {
    [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()() const -> topk_policy
    {
      return topk_policy{topk_algorithm::baseline, baseline_policy, {}};
    }
  };

  // Whether a one-worker-per-segment (default baseline) policy fits the static max segment size in shared memory; feeds
  // the backend decision below.
  static constexpr bool baseline_can_cover = baseline_can_cover_v<
    policy_getter_17,
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
      // backend choice. The threshold is applied on every cluster-capable architecture, not gated to a minimum CC.
      const bool beneficial = StaticMaxSegSize >= cluster_beneficial_min_segment_size;
      backend               = (cluster_capable(cc) && beneficial) ? topk_algorithm::cluster : topk_algorithm::baseline;
    }
    return topk_policy{backend, baseline_policy, make_cluster_policy()};
  }
};

// -----------------------------------------------------------------------------
// Dispatch (both backends behind one kernel symbol)
// -----------------------------------------------------------------------------
// The dispatch is host-only: it launches the single kernel symbol (`device_batched_topk_kernel`, in
// kernel_batched_topk.cuh) via the CUDA runtime. The algorithm does not support device-side (CDP) launch.

// Corrected form of `launcher_factory.max_dynamic_smem_size_for` (host path: `cub::MaxPotentialDynamicSmemBytes`),
// returning the usable dynamic budget as `opt-in - static footprint`. That facility currently subtracts the per-block
// reserved shared memory a second time even though `cudaDevAttrMaxSharedMemoryPerBlockOptin` already excludes it,
// under-reporting the budget by ~`reserved` (~1 KiB) -- enough to drop the cluster kernel's top table tier (see the
// TODO in MaxPotentialDynamicSmemBytes). TODO: once that facility is fixed, delete this and call
// `launcher_factory.max_dynamic_smem_size_for(...)` directly.
template <typename KernelPtr>
_CCCL_HOST_API cudaError_t max_dynamic_smem_size_for_fixed(int& max_dynamic_smem_bytes, KernelPtr kernel_ptr)
{
  max_dynamic_smem_bytes = -1;
  int device_id          = 0;
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
  cudaFuncAttributes kernel_attrs{};
  if (const auto error = CubDebug(cudaFuncGetAttributes(&kernel_attrs, kernel_ptr)))
  {
    return error;
  }
  const int static_smem_bytes = static_cast<int>(kernel_attrs.sharedSizeBytes);
  max_dynamic_smem_bytes = (max_smem_optin_bytes > static_smem_bytes) ? max_smem_optin_bytes - static_smem_bytes : 0;
  return cudaSuccess;
}

// Largest number of CTA blocks per cluster the kernel/architecture admits at `dynamic_smem_bytes` of dynamic SMEM. The
// config's cluster dimension is ignored by the query (placeholder here); the non-portable opt-in must already be set
// for it to report sizes beyond the portable ceiling.
template <typename KernelPtr>
_CCCL_HOST_API ::cuda::std::expected<int, cudaError_t>
probe_max_cluster_blocks(KernelPtr kernel_ptr, cudaStream_t stream, int threads_per_block, int dynamic_smem_bytes)
{
  ::cudaLaunchAttribute cluster_attr{};
  cluster_attr.id             = ::cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim = {1, 1, 1};

  ::cudaLaunchConfig_t cfg{};
  cfg.gridDim          = dim3(1);
  cfg.blockDim         = dim3(static_cast<unsigned>(threads_per_block));
  cfg.dynamicSmemBytes = static_cast<::cuda::std::size_t>(dynamic_smem_bytes);
  cfg.stream           = stream;
  cfg.attrs            = &cluster_attr;
  cfg.numAttrs         = 1;

  int cluster_blocks = 0;
  if (const auto error = CubDebug(
        ::cudaOccupancyMaxPotentialClusterSize(&cluster_blocks, reinterpret_cast<const void*>(kernel_ptr), &cfg)))
  {
    return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(error);
  }
  return cluster_blocks;
}

// Device-wide count of `cluster_blocks`-CTA clusters that can be co-resident at `dynamic_smem_bytes` of dynamic SMEM
// (clusters per wave). `cudaOccupancyMaxActiveClusters` rejects a grid that is not a multiple of the cluster, so the
// grid is set to exactly one cluster; the returned capacity is independent of the actual grid size.
template <typename KernelPtr>
_CCCL_HOST_API ::cuda::std::expected<int, cudaError_t> probe_clusters_per_wave(
  KernelPtr kernel_ptr, cudaStream_t stream, int threads_per_block, int cluster_blocks, int dynamic_smem_bytes)
{
  ::cudaLaunchAttribute cluster_attr{};
  cluster_attr.id             = ::cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim = {static_cast<unsigned>(cluster_blocks), 1, 1};

  ::cudaLaunchConfig_t cfg{};
  cfg.gridDim          = dim3(static_cast<unsigned>(cluster_blocks));
  cfg.blockDim         = dim3(static_cast<unsigned>(threads_per_block));
  cfg.dynamicSmemBytes = static_cast<::cuda::std::size_t>(dynamic_smem_bytes);
  cfg.stream           = stream;
  cfg.attrs            = &cluster_attr;
  cfg.numAttrs         = 1;

  int clusters_per_wave = 0;
  if (const auto error =
        CubDebug(::cudaOccupancyMaxActiveClusters(&clusters_per_wave, reinterpret_cast<const void*>(kernel_ptr), &cfg)))
  {
    return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(error);
  }
  return clusters_per_wave;
}

// The cluster backend's launch shape: CTA blocks per cluster and the dynamic-SMEM bytes to launch with.
struct cluster_launch_shape
{
  int cluster_blocks     = 0;
  int dynamic_smem_bytes = 0;
};

// Chooses the cluster launch shape for the statically-bounded max segment size. Probes occupancy through the CUDA
// runtime; the caller has already set the kernel's dynamic-SMEM opt-in to the maximum, so every probed config and the
// final launch run under one consistent opt-in.
template <typename LayoutT, typename KernelPtr>
_CCCL_HOST_API ::cuda::std::expected<cluster_launch_shape, cudaError_t> select_cluster_launch_shape(
  ::cuda::std::uint64_t max_segment_size,
  ::cuda::std::uint64_t num_segments,
  int max_dynamic_smem_bytes,
  cluster_topk_policy policy,
  cudaStream_t stream,
  KernelPtr kernel_ptr)
{
  using layout_t = LayoutT;

  const int threads_per_block = policy.threads_per_block;

  // Computed before any occupancy query so the single-CTA fast path below can skip one -- that driver query
  // otherwise dominates the runtime of tiny launches.
  const int max_block_resident_items = static_cast<int>(layout_t::max_block_resident_items(max_dynamic_smem_bytes));
  if (max_block_resident_items <= 0)
  {
    // Not even one load-aligned chunk fits in the opt-in budget; the kernel cannot run.
    return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(cudaErrorInvalidValue);
  }

  // Smallest cluster block count for full residency: at the largest SMEM each CTA holds `max_block_resident_items`
  // items. 64-bit to match the launch-shape arithmetic below; the value is small (`max_segment_size <= 2^21`).
  const auto min_blocks_per_segment =
    ::cuda::ceil_div(max_segment_size, static_cast<::cuda::std::uint64_t>(max_block_resident_items));

  int cluster_blocks     = 0;
  int dynamic_smem_bytes = 0;

  if (batched_topk_cluster::is_single_cta_eligible(
        static_cast<::cuda::std::uint32_t>(max_segment_size),
        static_cast<::cuda::std::uint32_t>(max_block_resident_items),
        policy.single_block_max_seg_size))
  {
    // Single-CTA fast path: the segment fits resident in one CTA and is small enough that the agent's
    // cluster-barrier-free path beats spreading it across more CTAs. One CTA at in-budget SMEM is always launchable,
    // so the occupancy probe is skipped. Larger fully-resident segments fall through to the wave-aware search below.
    cluster_blocks     = 1;
    dynamic_smem_bytes = layout_t::min_smem_bytes_from_num_items(max_segment_size);
  }
  else
  {
    // Hardware cluster ceiling (max blocks per cluster), queried at runtime (not hardcoded) so a future device with
    // larger non-portable clusters is not capped. Probed at zero dynamic SMEM for the arch/kernel ceiling alone; each
    // candidate is re-validated against its own SMEM below.
    const auto hw_cluster_ceiling =
      probe_max_cluster_blocks(kernel_ptr, stream, threads_per_block, /*dynamic_smem_bytes=*/0);
    if (!hw_cluster_ceiling)
    {
      return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(hw_cluster_ceiling.error());
    }
    if (*hw_cluster_ceiling <= 0)
    {
      return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(cudaErrorInvalidValue);
    }
    // `max_blocks_per_cluster == 0` -> the full hardware ceiling; a non-zero knob narrows it, clamped to that ceiling.
    // A cap narrower than a segment needs pushes it into the oversize/streaming fallback below.
    const int eff_max_blocks_per_cluster =
      (policy.max_blocks_per_cluster == 0)
        ? *hw_cluster_ceiling
        : (::cuda::std::min) (policy.max_blocks_per_cluster, *hw_cluster_ceiling);

    // Wave-aware selection: the free variable is the cluster block count (one cluster per segment), paired with the
    // smallest SMEM that keeps the segment fully resident (fewer blocks = more SMEM/fewer clusters-per-wave, more =
    // the reverse). Pick the count minimizing waves, ties toward the largest (smallest SMEM, most L1 -- the profiled
    // fast configs). Enumerated analytically, so a register-limited occupancy cannot collapse the candidate set.
    if (min_blocks_per_segment <= static_cast<::cuda::std::uint64_t>(eff_max_blocks_per_cluster))
    {
      // Full residency achievable: `max_segment_size <= min_blocks_per_segment * max_block_resident_items` and
      // `min_blocks_per_segment <= eff_max_blocks_per_cluster`, so every per-CTA capacity below fits `int`.

      // Cluster blocks the max segment actually needs (shared with the device so the launch is never wider than
      // necessary). At `min_chunks_per_block == 1` this equals the segment's chunk count; a larger knob shrinks it.
      const auto desired_cluster_blocks = ::cuda::narrow<int>(batched_topk_cluster::compute_num_logical_cluster_blocks(
        static_cast<::cuda::std::uint32_t>(layout_t::num_chunks_from_num_items(max_segment_size)),
        policy.min_chunks_per_block,
        ::cuda::narrow<::cuda::std::uint32_t>(eff_max_blocks_per_cluster)));

      // Scan `[min_candidate_blocks, max_candidate_blocks]` for the min-waves block count, tie-breaking largest.
      // `max_candidate_blocks == max(desired_cluster_blocks, min(min_candidate_blocks, eff_max_blocks_per_cluster))`:
      // the segment-needed count `desired_cluster_blocks` (<= `eff_max_blocks_per_cluster`, capped in
      // `compute_num_logical_cluster_blocks`), floored at `min_candidate_blocks`. The `clamp` operands are ordered so
      // `lo <= hi` holds even when `eff_max_blocks_per_cluster == 1` forces `min_candidate_blocks (== 2) > eff_max`;
      // there `max_candidate_blocks == 1` empties the scan and the single-CTA fallback below runs (that edge is a
      // one-CTA-resident segment with the single-CTA path disabled, so `min_blocks_per_segment == 1`).
      const auto min_candidate_blocks = (::cuda::std::max) (2, static_cast<int>(min_blocks_per_segment));
      const auto max_candidate_blocks =
        ::cuda::std::clamp(min_candidate_blocks, desired_cluster_blocks, eff_max_blocks_per_cluster);
      auto best_waves = (::cuda::std::numeric_limits<::cuda::std::uint64_t>::max)();
      for (int candidate_blocks = min_candidate_blocks; candidate_blocks <= max_candidate_blocks; ++candidate_blocks)
      {
        const auto num_block_items    = ::cuda::ceil_div(max_segment_size, candidate_blocks);
        const int resident_smem_bytes = layout_t::min_smem_bytes_from_num_items(num_block_items);
        if (resident_smem_bytes > max_dynamic_smem_bytes)
        {
          // Unreachable for candidate_blocks >= min_blocks_per_segment, but guards the SMEM budget regardless.
          continue;
        }

        const auto clusters_per_wave =
          probe_clusters_per_wave(kernel_ptr, stream, threads_per_block, candidate_blocks, resident_smem_bytes);
        if (!clusters_per_wave)
        {
          return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(clusters_per_wave.error());
        }
        if (*clusters_per_wave <= 0)
        {
          continue; // cluster blocks not launchable at this SMEM.
        }

        const auto waves = ::cuda::ceil_div(num_segments, *clusters_per_wave);
        // Min waves, tie-break largest count: the loop ascends, so `<=` keeps the largest at equal waves (`best_waves`
        // starts at `UINT64_MAX`, so the first launchable count always wins).
        if (waves <= best_waves)
        {
          best_waves         = waves;
          cluster_blocks     = candidate_blocks;
          dynamic_smem_bytes = resident_smem_bytes;
        }
      }

      if (cluster_blocks == 0 && min_blocks_per_segment == 1)
      {
        // No multi-CTA config was launchable; fall back to single-CTA full residency. Slower for large segments, but
        // `min_blocks_per_segment == 1` guarantees the resident SMEM fits the budget and one CTA is always launchable.
        cluster_blocks     = 1;
        dynamic_smem_bytes = layout_t::min_smem_bytes_from_num_items(max_segment_size);
      }
    }

    if (cluster_blocks == 0)
    {
      // Oversize (`min_blocks_per_segment > eff_max_blocks_per_cluster`) or nothing launchable: full residency
      // is impossible, so maximize residency with the largest launchable cluster at the largest SMEM and stream the
      // overflow.
      const auto hw_max_cluster_blocks =
        probe_max_cluster_blocks(kernel_ptr, stream, threads_per_block, max_dynamic_smem_bytes);
      if (!hw_max_cluster_blocks)
      {
        return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(hw_max_cluster_blocks.error());
      }
      cluster_blocks = (::cuda::std::min) (*hw_max_cluster_blocks, eff_max_blocks_per_cluster);
      if (cluster_blocks <= 0)
      {
        return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(cudaErrorInvalidValue);
      }
      dynamic_smem_bytes = max_dynamic_smem_bytes;
    }
  }

  return cluster_launch_shape{cluster_blocks, dynamic_smem_bytes};
}

// Cluster arm of the dispatch (host-only): after the shared query-pass / CC-guard setup, launches the single kernel
// symbol via `cudaLaunchKernelEx` using the resolved-CC cluster policy and geometry from `policy_getter`.
// `select_directions` arrives already wrapped; the cluster tuning comes from `policy_getter` (the resolved-CC policy)
// and the requested `Determinism`/`TieBreak` from the dispatch. The kernel launch goes through `launcher_factory`; the
// cluster occupancy / shared-memory setup queries still use the CUDA runtime directly.
template <typename PolicySelector,
          typename LargeSegmentTileOffsetT,
          ::cuda::execution::determinism::__determinism_t Determinism,
          ::cuda::execution::tie_break::__tie_break_t TieBreak,
          bool UserProvidedTuning,
          typename PolicyGetter,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename KernelLauncherFactory>
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

  // The cluster arm needs no temporary storage; report a positive size so the two-phase protocol proceeds.
  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = 1;
    return cudaSuccess;
  }

  // A `tune`d override (`UserProvidedTuning`) can force the cluster backend on a device that cannot run it: return
  // cudaErrorNotSupported rather than launch a cluster kernel the device lacks (the deferred-mode runtime behavior
  // tests and benchmarks rely on). The automatic selector never routes here below SM 9.0, so its instantiation drops
  // this check. `PtxComputeCap` is the running code's capability (never above the hardware SM), so it also rejects an
  // SM 9.0+ build on older hardware.
  if constexpr (UserProvidedTuning)
  {
    ::cuda::compute_capability cc{};
    if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
    {
      return error;
    }
    if (cc < ::cuda::compute_capability{9, 0})
    {
      return cudaErrorNotSupported;
    }
  }

  // Single kernel symbol; its cluster vs baseline arm is selected device-side via `current_policy<PolicySelector>()`.
  // Taking its address here ODR-uses the `__global__` template, which is what drives its emission and registration.
  // Not `constexpr`: MSVC (C2326) rejects a `constexpr` local captured and ODR-used inside the lambdas below.
  auto kernel_ptr = &device_batched_topk_kernel<
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
  constexpr cluster_topk_policy policy    = policy_getter().cluster;
  constexpr int threads_per_block         = policy.threads_per_block;
  constexpr int chunk_bytes               = policy.chunk_bytes;
  constexpr int load_align_bytes          = policy.load_align_bytes;
  constexpr int max_chunk_slots_per_block = policy.max_chunk_slots_per_block;
  static_assert(policy.max_blocks_per_cluster >= 0,
                "max_blocks_per_cluster must be 0 (unrestricted) or a positive cluster block count");
  static_assert(max_chunk_slots_per_block >= 0,
                "max_chunk_slots_per_block must be 0 (unrestricted) or a positive count");

  using key_it_t = it_value_t<KeyInputItItT>;
  using key_t    = it_value_t<key_it_t>;
  using layout_t = batched_topk_cluster::smem_block_tile_layout<key_t, chunk_bytes, load_align_bytes>;
  static_assert(is_valid_cluster_policy(policy));
  static_assert(load_align_bytes % int{sizeof(key_t)} == 0);

  // Tightest upper bound the segment-size argument carries -- for a static-bounded per-segment sequence a loose type
  // max, not the actual runtime maximum across segments.
  const auto max_seg_size  = ::cuda::args::__highest_(segment_sizes);
  using num_segments_val_t = typename ::cuda::args::__traits<NumSegmentsParameterT>::element_type;
  // `num_segments > 0` and `max_seg_size > 0` here: the generic `dispatch` returns for the empty-batch cases (no
  // segments, or a non-positive max segment size) before invoking this launch arm.
  const auto num_seg_val = detail::params::get_param(num_segments, num_segments_val_t{0});

  // Opt in to non-portable cluster blocks (>8 on Hopper).
  if (const auto error = CubDebug(::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_ptr), cudaFuncAttributeNonPortableClusterSizeAllowed, 1)))
  {
    return error;
  }

  // Usable dynamic shared-memory budget (opt-in minus the kernel's static footprint); the policy slot cap may narrow
  // it further into `max_dynamic_smem_bytes` below.
  int hw_dynamic_smem_bytes = 0;
  if (const auto error = max_dynamic_smem_size_for_fixed(hw_dynamic_smem_bytes, kernel_ptr))
  {
    return error;
  }
  // Optional policy cap on resident chunk slots per block (`max_chunk_slots_per_block == 0` -> unrestricted, i.e.
  // the full hardware budget). Expressed as the SMEM those slots need, then clamped to the hardware budget: a cap the
  // hardware cannot satisfy is a no-op (hardware wins). Fewer slots lowers every CTA's resident dynamic shared-memory
  // request, so a smaller segment overflows into streaming -- useful to leave shared memory free for a concurrent
  // kernel (or to reach the streaming / schedule paths at a small footprint in tests). A cap below one slot trips the
  // `max_block_resident_items <= 0` guard below.
  const int max_dynamic_smem_bytes =
    (max_chunk_slots_per_block == 0)
      ? hw_dynamic_smem_bytes
      : (::cuda::std::min) (hw_dynamic_smem_bytes, layout_t::min_smem_bytes_from_num_chunks(max_chunk_slots_per_block));

  // Set the kernel's dynamic-SMEM opt-in once, to the per-symbol maximum, before any occupancy probe or launch.
  // `max_dynamic_smem_bytes` is fixed by the compile-time policy and the device, so every thread sharing this kernel
  // symbol writes the identical value: the process-global attribute cannot be raced to a lower value that would fail a
  // concurrent launch. It also covers every launch shape the search below can pick (all `<= max_dynamic_smem_bytes`)
  // and keeps the occupancy probes and the final launch on one consistent opt-in.
  if (const auto error = launcher_factory.set_max_dynamic_smem_size_for(kernel_ptr, max_dynamic_smem_bytes))
  {
    return error;
  }

  // Resolve the launch shape (cluster blocks + dynamic SMEM) for the max segment size.
  const auto shape = select_cluster_launch_shape<layout_t>(
    static_cast<::cuda::std::uint64_t>(max_seg_size),
    static_cast<::cuda::std::uint64_t>(num_seg_val),
    max_dynamic_smem_bytes,
    policy,
    stream,
    kernel_ptr);
  if (!shape)
  {
    return shape.error();
  }

  const int cluster_blocks            = shape->cluster_blocks;
  const int dynamic_smem_bytes        = shape->dynamic_smem_bytes;
  const auto max_block_resident_items = layout_t::max_block_resident_items(dynamic_smem_bytes);

  // One cluster per segment, its CTAs stacked in the grid's y-dimension so the x-extent stays `num_segments`: a
  // flattened x == num_segments * cluster_blocks would overrun the 2^31-1 grid-x limit for a multi-CTA cluster well
  // before `num_segments` reached its INT_MAX maximum (already <= INT_MAX by the entry check in `dispatch`;
  // `cluster_blocks` is far below the y-dimension limit). The device reads the segment id from clusterid.x (segments
  // stay the x-extent) and the CTA rank from cluster_ctarank (linearized across the cluster's dims), so this needs no
  // agent change.
  const dim3 grid_dim{static_cast<unsigned>(num_seg_val), static_cast<unsigned>(cluster_blocks), 1u};
  const dim3 cluster_dim{1u, static_cast<unsigned>(cluster_blocks), 1u};

  // The cluster dimension routes the host launch through `cudaLaunchKernelEx`.
  if (const auto error = CubDebug(
        launcher_factory(grid_dim,
                         dim3{static_cast<unsigned>(threads_per_block)},
                         static_cast<::cuda::std::size_t>(dynamic_smem_bytes),
                         stream,
                         /*dependent_launch=*/false,
                         cluster_dim)
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
                cluster_kernel_args{static_cast<::cuda::std::uint32_t>(max_block_resident_items)})))
  {
    return error;
  }

  return CubDebug(detail::DebugSyncStream(stream));
}

// Baseline host-launch arm of the dispatch. Launches the single kernel symbol
// (`device_batched_topk_kernel`, packing the large-segment bookkeeping into `baseline_kernel_args` and passing an empty
// `cluster_kernel_args`). `select_directions` arrives already wrapped and the baseline tuning is taken from the
// `PolicySelector`. All kernel launches, memsets and nested scans go through
// `launcher_factory`.
template <typename PolicySelector,
          typename PolicyGetter,
          typename LargeSegmentTileOffsetT,
          ::cuda::execution::determinism::__determinism_t Determinism,
          ::cuda::execution::tie_break::__tie_break_t TieBreak,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename KernelLauncherFactory>
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
  // Computed from this call's concrete agent types (a tuning override exposes no `baseline_can_cover` member). The
  // automatic selector never routes here when this is false; only a trusted `tune`d override forces the baseline
  // backend on an oversize segment. Strict mode rejects that at compile time (the static_assert below, mirroring the
  // arch-unsupported one in `dispatch`); deferred mode keeps the two-phase runtime cudaErrorNotSupported path.
  constexpr bool baseline_can_cover = baseline_can_cover_v<
    PolicyGetter,
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
#if !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC) \
  && !defined(CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT)
  static_assert(
    baseline_can_cover,
    "cub::DeviceBatchedTopK: the forced baseline backend cannot cover the static maximum segment size within the "
    "shared-memory limit. Force the cluster backend, lower the segment-size bound, or define "
    "CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT to defer the diagnosis to runtime (cudaErrorNotSupported).");
#endif // !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)
       // && !defined(CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT)
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

    // Determine which one-worker-per-segment policy covers the segment-size range and k. Resolve from the handed
    // resolved-CC `PolicyGetter` (not `PolicySelector`) so the host picks the same baseline policy the device kernel
    // instantiates for the resolved CC -- they would otherwise diverge once baseline tuning becomes CC-dependent.
    constexpr auto policy = find_smallest_covering_policy_for_getter<
      PolicyGetter,
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
      // TODO(topk): the baseline large-segment (multi-CTA) path is WIP. Completing it requires: (1) guarding the
      // `num_segments_val * sizeof(...)` byte counts below against size_t overflow (safe today only because the entry
      // bounds num_segments_val to <= INT_MAX); (2) making the baseline tunable by populating its `epilogue` and
      // `multi_worker_per_segment_policy` sub-policies and adding matching knobs to the segmented_topk benchmarks,
      // which leave them zero-initialized today so baseline sweeps are not yet meaningful.
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

#if !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)
// Returns true if at least one architecture this translation unit targets (the compile target list exposed as
// `::cuda::__target_compute_capabilities()`) resolves to the `unsupported` backend for `PolicySelector` -- e.g. a
// deterministic request while a pre-SM90 target is present in the list. Used to turn a would-be runtime
// `cudaErrorNotSupported` into a compile-time diagnostic (see the static_assert in `dispatch`).
template <typename PolicySelector>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL bool any_target_cc_unsupported()
{
  bool any = false;
  for (const auto cc : ::cuda::__target_compute_capabilities())
  {
    any = any || (PolicySelector{}(cc).backend == topk_algorithm::unsupported);
  }
  return any;
}
#endif // !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)

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
  typename KeyInputItItT,
  typename KeyOutputItItT,
  typename ValueInputItItT,
  typename ValueOutputItItT,
  typename SegmentSizeParameterT,
  typename KParameterT,
  typename SelectDirectionT,
  typename NumSegmentsParameterT,
  typename TotalNumItemsGuaranteeT,
  typename TuningEnvT            = ::cuda::std::execution::env<>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
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
  const TuningEnvT&                      = {},
  KernelLauncherFactory launcher_factory = {})
{
  // Both arms resolve `num_segments` on the host (allocation sizing, grid extent, empty-batch guard), so it must be a
  // host-known single value; device-resident counts are future work. Defensive: the public entry checks this too, but
  // `dispatch` is also called directly (tests / benchmarks).
  static_assert(::cuda::args::__traits<NumSegmentsParameterT>::is_single_value
                  && !::cuda::args::__traits<NumSegmentsParameterT>::is_deferred,
                "cub::DeviceBatchedTopK requires a host-known uniform number of segments (constant, immediate, or a "
                "plain integral value).");

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
  using default_policy_selector_t = policy_selector_from_types<
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

  // Type derived from the query-result trait rather than `decltype(policy_selector)`: GCC 7 rejects the latter ("use of
  // 'policy_selector' before deduction of 'auto'") when `policy_selector_t` is later named inside the dispatch lambda.
  using policy_selector_t =
    ::cuda::std::execution::__query_result_or_t<TuningEnvT, topk_policy, default_policy_selector_t>;
#if _CCCL_HAS_CONCEPTS()
  static_assert(topk_policy_selector<policy_selector_t>,
                "Invalid policy selector for cub::DeviceBatchedTopK::dispatch");
#endif // _CCCL_HAS_CONCEPTS()

#if !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC) \
  && !defined(CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT)
  // Strict mode (default): fail at compile time if the request cannot be served on *any* architecture this translation
  // unit targets. Two causes reach here: a deterministic / large-segment request while a pre-SM90 target is present
  // (the cluster backend requires SM90+), or _CCCL_DISABLE_DYNAMIC_CLUSTER_LAUNCH disabling the cluster backend on all
  // architectures. This is the least-surprising UX for callers whose build targets multiple architectures. Define
  // `CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT` to defer the diagnosis to runtime instead (the dispatch then returns
  // `cudaErrorNotSupported` on unsupported devices); CUB's own tests and benchmarks do this so they can compile the
  // full configuration space across all target architectures and skip at runtime where unsupported.
  static_assert(
    !any_target_cc_unsupported<policy_selector_t>(),
    "cub::DeviceBatchedTopK: the requested top-k configuration cannot be served on at least one architecture this "
    "translation unit targets. The deterministic / large-segment path requires the cluster backend (SM90+), which is "
    "unavailable either because a pre-SM90 architecture is targeted or because _CCCL_DISABLE_DYNAMIC_CLUSTER_LAUNCH is "
    "defined (which disables the cluster backend on all architectures). To fix: target only SM90+ and leave "
    "_CCCL_DISABLE_DYNAMIC_CLUSTER_LAUNCH undefined, relax the request (non-deterministic and small enough for the "
    "baseline backend), or define CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT to defer the diagnosis to runtime "
    "(cudaErrorNotSupported).");
#endif // !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)
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

  // `num_segments` maps to the grid's x-extent in both host launch arms (the baseline arm launches one block per
  // segment; the cluster arm launches one cluster per segment, stacking the cluster's CTAs in the grid's y-dimension),
  // so it must fit a positive 32-bit grid dimension. A count above INT_MAX cannot, so reject it as an out-of-contract
  // value at this single host boundary: otherwise the baseline arm would silently narrow it to `int` and the cluster
  // arm would build an out-of-range grid.x.
  using num_segments_val_t                  = typename ::cuda::args::__traits<NumSegmentsParameterT>::element_type;
  const num_segments_val_t num_segments_val = detail::params::get_param(num_segments, num_segments_val_t{0});
  // Unary `+` integer-promotes the count to a standard integer type so the sign-safe `cmp_*` comparators accept it:
  // they are constrained to `__cccl_is_integer_v`, which excludes the character count types the public API permits.
  if (::cuda::std::cmp_greater(+num_segments_val, ::cuda::std::numeric_limits<int>::max()))
  {
    return cudaErrorInvalidValue;
  }
  // A negative count is no work (like a zero count), matching DeviceSegmentedReduce. Short-circuit here, before
  // `dispatch_compute_cap`, so the query pass cannot fall into the baseline arm where `num_segments_val * sizeof(...)`
  // would cast the negative count to a huge `size_t`. (Zero is handled by `empty_batch_no_launch` below, which keeps
  // the arch-gated no-op semantics.) TODO(topk): file an issue to unify the negative-`num_segments` contract across
  // CUB device algorithms.
  if (::cuda::std::cmp_less(+num_segments_val, 0))
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
    }
    return cudaSuccess;
  }

  // Empty batch = no work to launch: no segments, or a non-positive tightest max segment size (every segment empty,
  // e.g. a uniform negative size clamped to 0). `== 0` suffices for `num_segments`: a negative count is already
  // short-circuited as no work above. Consulted only on the launch (`d_temp_storage != nullptr`) of a *supported* arm
  // below: the query pass falls through to size `temp_storage_bytes`, and the unsupported arm ignores it so an
  // unavailable request still fails with cudaErrorNotSupported rather than being masked into success.
  const auto empty_batch_no_launch = [&] {
    return d_temp_storage != nullptr
        && (detail::params::get_param(num_segments, 0) == 0 || ::cuda::args::__highest_(segment_sizes) <= 0);
  };

  return detail::dispatch_compute_cap(policy_selector_t{}, cc, [&](auto policy_getter) -> cudaError_t {
    constexpr topk_policy active_policy = policy_getter();
#if _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)
    NV_IF_TARGET(NV_IS_HOST, ({
                   ::std::stringstream ss;
                   ss << active_policy;
                   _CubLog("Dispatching DeviceBatchedTopK to compute capability %d.%d with tuning: %s\n",
                           cc.major_cap(),
                           cc.minor_cap(),
                           ss.str().c_str());
                 }))
#endif // _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)
    if constexpr (active_policy.backend == topk_algorithm::baseline)
    {
      // Computed from the template parameters, not a captured function-scope constant: MSVC rejects the latter as
      // non-constant inside this lambda's `if constexpr`.
      constexpr bool deterministic = (Determinism != ::cuda::execution::determinism::__determinism_t::__not_guaranteed)
                                  || (TieBreak != ::cuda::execution::tie_break::__tie_break_t::__unspecified);
      if constexpr (deterministic)
      {
        // A `tune`d selector forced the baseline backend for a deterministic / tie-break request it cannot serve (only
        // the SM 9.0+ cluster backend is deterministic). Mirror the arch-unsupported / oversize-baseline failure model:
        // a hard compile error by default, deferred to a runtime cudaErrorNotSupported only under the escape hatches.
#if !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC) \
  && !defined(CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT)
        static_assert(
          !deterministic,
          "cub::DeviceBatchedTopK: a tuned policy selector forced the baseline backend for a deterministic "
          "/ tie-break request it cannot serve (only the SM 9.0+ cluster backend is deterministic). Drop "
          "the override, relax the determinism / tie-break requirement, or define "
          "CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT to defer the diagnosis to runtime "
          "(cudaErrorNotSupported).");
#endif // !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)
       // && !defined(CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT)
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
        return launch_baseline_arm<policy_selector_t,
                                   decltype(policy_getter),
                                   LargeSegmentTileOffsetT,
                                   Determinism,
                                   TieBreak>(
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
#if !_CCCL_HAS_DYNAMIC_CLUSTER_LAUNCH()
      // The automatic selector never picks the cluster backend when dynamic cluster launches are disabled (see
      // cluster_capable), so reaching here means a `tune`d selector forced it. The kernel would launch without its
      // cluster extent (triple_chevron drops it), so reject the contradiction at compile time rather than run wrong.
      static_assert(active_policy.backend != topk_algorithm::cluster,
                    "cub::DeviceBatchedTopK: a tuned policy selector forced the cluster backend, but "
                    "_CCCL_DISABLE_DYNAMIC_CLUSTER_LAUNCH is defined. Drop the override or the macro.");
#endif // !_CCCL_HAS_DYNAMIC_CLUSTER_LAUNCH()
      if (empty_batch_no_launch())
      {
        return cudaSuccess;
      }
      // `UserProvidedTuning`: false for the automatic selector, which returns `cluster` solely for a
      // `cluster_capable(cc)` and so needs no runtime re-check; a `tune`d override is a different type and keeps it.
      // Inlined as a type trait rather than a function-scope constexpr, which MSVC rejects inside this lambda.
      return launch_cluster_arm<policy_selector_t,
                                LargeSegmentTileOffsetT,
                                Determinism,
                                TieBreak,
                                !::cuda::std::is_same_v<policy_selector_t, default_policy_selector_t>>(
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
