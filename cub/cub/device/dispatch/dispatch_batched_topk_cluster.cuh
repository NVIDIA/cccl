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
//!     cluster width chosen at runtime (up to 16 on Hopper).
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
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk_cluster.cuh>
#include <cub/util_device.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__argument_>
#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/round_up.h>
#include <cuda/__numeric/narrow.h>
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
// Hardware constants
// -----------------------------------------------------------------------------
// Largest cluster width guaranteed on every SM 9.0+ device.
inline constexpr int max_portable_cluster_blocks = 8;

// CUDA's hardware ceiling on cluster width (Hopper supports up to 16).
inline constexpr int max_supported_cluster_blocks = 16;

// -----------------------------------------------------------------------------
// Cluster-size / dynamic-SMEM selection
// -----------------------------------------------------------------------------
// Tightest upper bound carried by the segment-size argument. Mirrors `__argument::__traits<>::highest` semantics:
// the compile-time bound for `__constant`/bounded sequence arguments and the runtime value for a uniform
// `__immediate`. For a per-segment sequence with only a static bound this can be the loose `numeric_limits<T>::max()`.
template <typename SegmentSizeParameterT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto runtime_max_segment_size(SegmentSizeParameterT segment_sizes) noexcept
{
  return ::cuda::__argument::__highest_(segment_sizes);
}

struct launch_config
{
  int cluster_blocks;
  int dynamic_smem_bytes;
  ::cuda::std::uint32_t block_tile_capacity;
};

// -----------------------------------------------------------------------------
// Kernel entry points
// -----------------------------------------------------------------------------
// Dynamic-cluster kernel for host launches; the agent reads the active cluster
// width via cooperative groups.
template <int ThreadsPerBlock,
          int MinBlocksPerSm,
          int UnrollFactor,
          int PipelineStages,
          int ChunkBytes,
          int LoadAlignBytes,
          int BitsPerPass,
          int TieBreakItemsPerThread,
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
    UnrollFactor,
    PipelineStages,
    ChunkBytes,
    LoadAlignBytes,
    BitsPerPass,
    TieBreakItemsPerThread,
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
          int UnrollFactor,
          int PipelineStages,
          int ChunkBytes,
          int LoadAlignBytes,
          int BitsPerPass,
          int TieBreakItemsPerThread,
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
    UnrollFactor,
    PipelineStages,
    ChunkBytes,
    LoadAlignBytes,
    BitsPerPass,
    TieBreakItemsPerThread,
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
// NVCC kernel-emission workaround
// -----------------------------------------------------------------------------
// NVCC `-rdc=false` only emits the device side of a templated `__global__`
// when it sees a triple-chevron in the same TU; CUDA 13's
// `-static-global-template-stub=true` makes the host stub internally linked
// on top. Address-of and `cudaLaunchKernelEx` are not enough. The host path
// must use `cudaLaunchKernelExC` for the cluster attribute, so instantiating
// `force_emit_kernel<Kernel>::emit` parses a (dead) chevron in its place.
// `Args` are deduced from the function-pointer type to avoid repeating the
// dispatch's template parameter list.
//
// See https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/.
template <auto Kernel>
struct force_emit_kernel;

template <typename... Args, void (*Kernel)(Args...)>
struct force_emit_kernel<Kernel>
{
  [[noreturn]] _CCCL_HOST static void emit(Args... args)
  {
    _CCCL_ASSERT(false, "force_emit_kernel::emit must never be called");
    // Unreachable; present only so NVCC emits `Kernel` for this TU.
    if (false)
    {
      Kernel<<<1, 1>>>(args...);
    }
    _CCCL_UNREACHABLE();
  }
};

// -----------------------------------------------------------------------------
// Dispatch
// -----------------------------------------------------------------------------
// Keys-only; every segment must fit in one cluster_tile. Host picks
// `(cluster_blocks, dynamic_smem_bytes)` at runtime from a finite table; CDP
// uses the static kernel at `max_portable_cluster_blocks` and portable SMEM.

// CDP launch body, empty when CDP is disabled. Wrapped in a macro because
// `#ifdef` can't sit inside `NV_IF_TARGET`.
#ifndef CUB_RDC_ENABLED
#  define CUB_TOPK_CLUSTER_DEVICE_LAUNCH
#else // CUB_RDC_ENABLED
#  define CUB_TOPK_CLUSTER_DEVICE_LAUNCH                                                \
    auto static_kernel = device_segmented_topk_cluster_kernel_static<                   \
      ThreadsPerBlock,                                                                  \
      UnrollFactor,                                                                     \
      PipelineStages,                                                                   \
      ChunkBytes,                                                                       \
      LoadAlignBytes,                                                                   \
      BitsPerPass,                                                                      \
      TieBreakItemsPerThread,                                                           \
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

template <typename KeyInputItItT,
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
  // The selection direction is a compile-time constant carried as `::cuda::__argument::__constant<Dir>`. Wrap it into
  // the internal discrete param the kernel/agent expect, exactly as the baseline `batched_topk::dispatch` does.
  auto select_directions          = batched_topk::wrap_select_direction(select_direction);
  using SelectDirectionParameterT = decltype(select_directions);
  // Clusters are SM 9.0+ only and the tuning is currently identical across
  // CCs, so pin the selector query to the minimum supported CC.
  constexpr cluster_topk_policy policy = PolicySelector{}(::cuda::compute_capability{9, 0});
  constexpr int ThreadsPerBlock        = policy.threads_per_block;
  constexpr int MinBlocksPerSm         = policy.min_blocks_per_sm;
  constexpr int UnrollFactor           = policy.unroll_factor;
  constexpr int PipelineStages         = policy.pipeline_stages;
  constexpr int ChunkBytes             = policy.chunk_bytes;
  constexpr int LoadAlignBytes         = policy.load_align_bytes;
  constexpr int BitsPerPass            = policy.bits_per_pass;
  constexpr int TieBreakItemsPerThread = policy.tie_break_items_per_thread;

  using key_it_t = it_value_t<KeyInputItItT>;
  using key_t    = it_value_t<key_it_t>;
  using layout_t = smem_block_tile_layout<key_t, ChunkBytes, LoadAlignBytes>;
  using agent_t  = agent_batched_topk_cluster<
     ThreadsPerBlock,
     UnrollFactor,
     PipelineStages,
     ChunkBytes,
     LoadAlignBytes,
     BitsPerPass,
     TieBreakItemsPerThread,
     KeyInputItItT,
     KeyOutputItItT,
     ValueInputItItT,
     ValueOutputItItT,
     SegmentSizeParameterT,
     KParameterT,
     SelectDirectionParameterT,
     NumSegmentsParameterT>;

  static_assert(ChunkBytes % LoadAlignBytes == 0);
  static_assert(LoadAlignBytes % int{sizeof(key_t)} == 0);
  // Static-footprint estimate for the device-side CDP fallback, which cannot query `cudaFuncGetAttributes`.
  // The host path instead uses the driver-reported `sharedSizeBytes` (see below), which is padding-aware.
  constexpr int static_smem_bytes = static_cast<int>(sizeof(typename agent_t::TempStorage));

  const auto max_seg_size = runtime_max_segment_size(segment_sizes);
  using max_seg_size_t    = decltype(+max_seg_size);

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

  static_assert(::cuda::__argument::__traits<NumSegmentsParameterT>::is_single_value,
                "Number of segments must be resolved on the host.");

  using num_segments_val_t = typename ::cuda::__argument::__traits<NumSegmentsParameterT>::element_type;
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
    UnrollFactor,
    PipelineStages,
    ChunkBytes,
    LoadAlignBytes,
    BitsPerPass,
    TieBreakItemsPerThread,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT>;

  // Force NVCC to emit the device side of `dynamic_kernel` for this TU; see
  // `force_emit_kernel` above.
  [[maybe_unused]] constexpr auto force_emit = &force_emit_kernel<dynamic_kernel>::emit;

  NV_IF_TARGET(
    NV_IS_HOST,
    ({
      // Opt in to non-portable cluster widths (>8 on Hopper).
      if (const auto error = CubDebug(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(dynamic_kernel), cudaFuncAttributeNonPortableClusterSizeAllowed, 1)))
      {
        return error;
      }

      // Reused across the probe and the launch; `clusterDim.x` is a placeholder
      // until after `cudaOccupancyMaxPotentialClusterSize` (which ignores it).
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
      // the dynamic-SMEM math below matches exactly what the launch permits. The table entries express the
      // documented total per-block budget (`cudaDevAttrMaxSharedMemoryPerBlockOptin`); the usable dynamic portion
      // is that budget minus the static footprint.
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

      const auto runtime_policy = PolicySelector{}(::cuda::compute_capability{sm_version / 10});
      _CCCL_ASSERT(runtime_policy.launch_configs.size() <= max_launch_configs,
                   "Cluster TopK launch config table exceeds policy capacity");

      const auto total_to_dynamic_smem = [&](int total_smem_bytes) {
        return (total_smem_bytes > nondynamic_smem_bytes) ? total_smem_bytes - nondynamic_smem_bytes : 0;
      };

      constexpr int portable_total_smem_bytes = 48 * 1024;
      const int portable_dynamic_smem_bytes   = total_to_dynamic_smem(portable_total_smem_bytes);
      int configured_dynamic_smem_limit       = portable_dynamic_smem_bytes;
      const auto ensure_dynamic_smem_limit    = [&](int dynamic_smem_bytes) {
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

      launch_config selected_config{0, 0, 0};
      launch_config largest_supported_config{0, 0, 0};
      // The table is intentionally tiny; scan all entries and select the lowest coverage that satisfies the bound
      // rather than relying on the declaration order to be sorted by coverage.
      for (const auto policy_config : runtime_policy.launch_configs)
      {
        const int candidate_cluster_blocks = policy_config.cluster_blocks;
        const int candidate_dynamic_smem   = total_to_dynamic_smem(policy_config.total_smem_bytes);
        if (candidate_dynamic_smem > max_dynamic_smem_bytes)
        {
          continue;
        }

        const auto candidate_block_tile_capacity = layout_t::block_tile_capacity(candidate_dynamic_smem);
        if (candidate_block_tile_capacity == 0)
        {
          continue;
        }

        launch_config candidate{candidate_cluster_blocks, candidate_dynamic_smem, candidate_block_tile_capacity};
        const auto candidate_cluster_tile_capacity = layout_t::template cluster_tile_capacity<max_seg_size_t>(
          candidate.cluster_blocks, candidate.block_tile_capacity);

        // Every smem-fitting candidate is considered: it may either tighten the covering selection or improve the
        // largest-coverage fallback used for oversize segments. The table is tiny, so the extra driver queries below
        // are negligible.
        if (const auto error = ensure_dynamic_smem_limit(candidate_dynamic_smem))
        {
          return error;
        }

        cfg.dynamicSmemBytes                = static_cast<unsigned int>(candidate_dynamic_smem);
        int candidate_hw_max_cluster_blocks = 0;
        if (const auto error = CubDebug(cudaOccupancyMaxPotentialClusterSize(
              &candidate_hw_max_cluster_blocks, reinterpret_cast<const void*>(dynamic_kernel), &cfg)))
        {
          return error;
        }
        candidate_hw_max_cluster_blocks =
          (::cuda::std::min) (candidate_hw_max_cluster_blocks, max_supported_cluster_blocks);
        if (candidate_cluster_blocks > candidate_hw_max_cluster_blocks)
        {
          continue;
        }

        // Track the largest-coverage hardware-supported config for all parameter kinds. Segments that exceed every
        // finite candidate (always possible for per-segment sizes, and now also for static/uniform bounds) fall back
        // to this config and re-stream the overflow from gmem in the agent.
        {
          const auto largest_supported_cluster_tile_capacity = layout_t::template cluster_tile_capacity<max_seg_size_t>(
            largest_supported_config.cluster_blocks, largest_supported_config.block_tile_capacity);
          if (largest_supported_config.cluster_blocks == 0
              || candidate_cluster_tile_capacity > largest_supported_cluster_tile_capacity)
          {
            largest_supported_config = candidate;
          }
        }

        // Among the candidates that fully cover the segment(s), keep the one with the lowest coverage.
        if (candidate_cluster_tile_capacity >= max_seg_size)
        {
          const auto selected_cluster_tile_capacity = layout_t::template cluster_tile_capacity<max_seg_size_t>(
            selected_config.cluster_blocks, selected_config.block_tile_capacity);
          if (selected_config.cluster_blocks == 0 || candidate_cluster_tile_capacity < selected_cluster_tile_capacity)
          {
            selected_config = candidate;
          }
        }
      }

      if (selected_config.cluster_blocks == 0)
      {
        // No finite candidate covers the segment(s); fall back to the largest hardware-supported config and let the
        // agent stream the overflow. Applies to per-segment sizes and static/uniform bounds alike.
        selected_config = largest_supported_config;
      }
      if (selected_config.cluster_blocks == 0)
      {
        return cudaErrorInvalidValue;
      }

      const auto selected_cluster_tile_capacity = layout_t::template cluster_tile_capacity<max_seg_size_t>(
        selected_config.cluster_blocks, selected_config.block_tile_capacity);
      if (selected_cluster_tile_capacity >= max_seg_size)
      {
        const auto required_physical_cluster_tile_items =
          static_cast<::cuda::std::uint64_t>(max_seg_size) + static_cast<::cuda::std::uint64_t>(layout_t::chunk_items);
        selected_config.cluster_blocks = static_cast<int>(
          ::cuda::ceil_div(required_physical_cluster_tile_items,
                           static_cast<::cuda::std::uint64_t>(selected_config.block_tile_capacity)));

        const auto required_block_tile_capacity = ::cuda::ceil_div(
          required_physical_cluster_tile_items, static_cast<::cuda::std::uint64_t>(selected_config.cluster_blocks));
        const auto required_slots =
          ::cuda::ceil_div(required_block_tile_capacity, static_cast<::cuda::std::uint64_t>(layout_t::chunk_items));
        selected_config.dynamic_smem_bytes =
          layout_t::base_padding_bytes + static_cast<int>(required_slots) * layout_t::chunk_bytes;
        selected_config.block_tile_capacity = layout_t::block_tile_capacity(selected_config.dynamic_smem_bytes);
      }

      if (const auto error = CubDebug(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(dynamic_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            selected_config.dynamic_smem_bytes)))
      {
        return error;
      }

      const int cluster_blocks       = selected_config.cluster_blocks;
      const auto block_tile_capacity = selected_config.block_tile_capacity;
      cfg.dynamicSmemBytes           = static_cast<unsigned int>(selected_config.dynamic_smem_bytes);

      const auto grid_blocks =
        static_cast<::cuda::std::uint64_t>(num_seg_val) * static_cast<::cuda::std::uint64_t>(cluster_blocks);
      if (grid_blocks > static_cast<::cuda::std::uint64_t>(::cuda::std::numeric_limits<int>::max()))
      {
        return cudaErrorInvalidValue;
      }

      cfg.gridDim                   = dim3(static_cast<unsigned int>(grid_blocks), 1, 1);
      cluster_attr.val.clusterDim.x = static_cast<unsigned int>(cluster_blocks);

      if (const auto error = CubDebug(::cudaLaunchKernelEx(
            &cfg,
            dynamic_kernel,
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
      // total SMEM or non-portable cluster widths. Segments that exceed the
      // portable resident coverage are still handled: the agent re-streams the
      // overflow chunks from gmem.
      constexpr int portable_total_smem_bytes = 48 * 1024;
      constexpr int dynamic_smem_bytes =
        (portable_total_smem_bytes > static_smem_bytes) ? portable_total_smem_bytes - static_smem_bytes : 0;
      constexpr auto block_tile_capacity = layout_t::block_tile_capacity(dynamic_smem_bytes);

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
template <typename KeyInputItItT,
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
      return dispatch(
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
