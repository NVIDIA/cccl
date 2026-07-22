// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Kernel entry point for device-wide batched top-k.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_batched_topk.cuh>
#include <cub/agent/agent_batched_topk_cluster.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>

#include <cuda/__cmath/round_up.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/argument>
#include <cuda/std/cstdint>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// Assert-free search shared by `find_smallest_covering_policy` and the backend coverage predicate. Returns the
// index of the smallest worker policy whose tile size still covers the upper bound on segment size AND whose
// instantiated agent's shared memory usage fits within the static shared memory limit (max_smem_per_block), or -1 if
// none does. Kept separate from `find_smallest_covering_policy` so callers can query coverage as a bool without
// tripping that trait's hard `static_assert`.
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
struct find_covering_policy_index
{
private:
  struct policy_t
  {
    worker_policy worker_per_segment_policy;
    multi_worker_policy multi_worker_per_segment_policy;
  };
  static constexpr ::cuda::std::int64_t max_segment_size = ::cuda::args::__traits<SegmentSizeParameterT>::highest;
  static constexpr baseline_topk_policy active_policy    = current_policy<PolicySelector>();

  template <int Index>
  [[nodiscard]] static constexpr int find_index()
  {
    if constexpr (Index >= active_policy.worker_per_segment_policies.size())
    {
      return -1;
    }
    else
    {
      constexpr worker_policy wp = active_policy.worker_per_segment_policies[Index];
      constexpr auto tile_size   = ::cuda::std::int64_t{wp.threads_per_block} * wp.items_per_thread;

      struct policy_getter_17 // TODO(bgruber): drop this in C++17 and pass wp directly
      {
        _CCCL_HOST_DEVICE_API constexpr auto operator()() const
        {
          return policy_t{active_policy.worker_per_segment_policies[Index],
                          active_policy.multi_worker_per_segment_policy};
        }
      };
      using candidate_agent_t  = agent_batched_topk_worker_per_segment<policy_getter_17, AgentParamsT...>;
      constexpr bool covers    = tile_size >= max_segment_size;
      constexpr bool fits_smem = sizeof(typename candidate_agent_t::TempStorage) <= max_smem_per_block;
      constexpr int next       = find_index<Index + 1>();
      if constexpr (covers && fits_smem)
      {
        return next >= 0 ? next : Index;
      }
      else
      {
        return next;
      }
    }
  }

public:
  static constexpr int value = find_index<0>();
};

// True iff some one-worker-per-segment policy covers the statically-known maximum segment size within the shared-memory
// limit. Used by the backend selector to decide whether the baseline backend is viable at all (an oversize
// bound must route to the cluster backend instead of tripping the `static_assert` below).
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
inline constexpr bool baseline_can_cover_v =
  find_covering_policy_index<PolicySelector, SegmentSizeParameterT, AgentParamsT...>::value >= 0;

// Resolves the agent type the kernel instantiates via the same covering-policy search as `find_covering_policy_index`,
// adding a hard `static_assert` when no policy covers the segment size within the shared-memory limit.
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
struct find_smallest_covering_policy
{
private:
  struct policy_t
  {
    worker_policy worker_per_segment_policy;
    multi_worker_policy multi_worker_per_segment_policy;
  };
  static constexpr baseline_topk_policy active_policy = current_policy<PolicySelector>();
  static constexpr int selected_index =
    find_covering_policy_index<PolicySelector, SegmentSizeParameterT, AgentParamsT...>::value;

public:
  // TODO (elstehle): extend support for variable-size segments
  static_assert(selected_index >= 0,
                "cub::DeviceBatchedTopK: no baseline worker policy covers the statically-known maximum segment size "
                "within the shared-memory limit. Reduce the maximum segment size encoded in the segment-size argument "
                "annotation (larger segments are served by the SM 9.0+ cluster backend).");
  static constexpr policy_t policy = {
    active_policy.worker_per_segment_policies[selected_index], active_policy.multi_worker_per_segment_policy};

  struct policy_getter_17 // TODO(bgruber): drop this in C++17 and pass policy directly
  {
    _CCCL_HOST_DEVICE_API constexpr auto operator()() const
    {
      return policy;
    }
  };
  using agent_t = agent_batched_topk_worker_per_segment<policy_getter_17, AgentParamsT...>;
};

// -----------------------------------------------------------------------------
// Single kernel symbol hosting both backends
// -----------------------------------------------------------------------------
// There is exactly one kernel symbol per instantiation. Its body selects the active backend device-side via
// `current_policy<PolicySelector>()` (evaluated per `__CUDA_ARCH__` pass), so each target architecture compiles only
// the backend the selector picks for it -- honoring CUB's "one kernel per arch/problem" rule while still supporting a
// multi-architecture fatbin whose per-arch choice differs. The host still branches its launch configuration (grid,
// shared memory, cluster dimensions) per backend, but both host arms launch this same symbol.

// Backend-specific kernel arguments. The unused struct is passed default-constructed (all-null / zero) to the arm the
// selector does not pick; passing it costs nothing (a few grid-constant scalars) and keeps a single kernel signature.
template <class NumSegmentsValueT, class LargeSegmentTileOffsetT>
struct baseline_kernel_args
{
  batched_topk_counters<NumSegmentsValueT>* d_counters   = nullptr;
  NumSegmentsValueT* d_large_segments_ids                = nullptr;
  LargeSegmentTileOffsetT* d_large_segments_tile_offsets = nullptr;
};

struct cluster_kernel_args
{
  ::cuda::std::uint32_t block_tile_capacity = 0;
};

// -----------------------------------------------------------------------------
// Launch-bounds helpers
// -----------------------------------------------------------------------------
// The two backends use different `__launch_bounds__` shapes (baseline: just threads_per_block; cluster: threads plus a
// min-blocks-per-SM and an optional max-blocks-per-cluster cap). We resolve all three per architecture from the
// selected policy. `find_smallest_covering_policy` (which carries a hard `static_assert`) is only ever touched inside
// the `backend == baseline` branch, so an oversize bound routed to the cluster/unsupported backend never trips it.
_CCCL_EXEC_CHECK_DISABLE
template <class PolicySelector, class SegmentSizeParameterT, class... AgentParamsT>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL int topk_threads_per_block_helper() noexcept
{
  constexpr auto policy = current_policy<PolicySelector>();
  if constexpr (policy.backend == topk_algorithm::baseline)
  {
    return find_smallest_covering_policy<baseline_policy_selector_adaptor<PolicySelector>,
                                         SegmentSizeParameterT,
                                         AgentParamsT...>::policy.worker_per_segment_policy.threads_per_block;
  }
  else if constexpr (policy.backend == topk_algorithm::cluster)
  {
    return policy.cluster.threads_per_block;
  }
  else
  {
    // unsupported: harmless positive default; the host never launches this arm.
    return 128;
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class PolicySelector>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL int topk_min_blocks_per_sm_helper() noexcept
{
  constexpr auto policy = current_policy<PolicySelector>();
  if constexpr (policy.backend == topk_algorithm::cluster)
  {
    return policy.cluster.min_blocks_per_sm;
  }
  else
  {
    // baseline / unsupported: no minimum-blocks constraint.
    return 0;
  }
}

// Third `__launch_bounds__` argument (`maxBlocksPerCluster`): the cluster policy's `max_blocks_per_cluster` cap. The
// host arm launches a dynamic cluster width, so this is the only compile-time width hint `ptxas` sees, and
// `launch_cluster_arm` clamps the launch to `<= max_blocks_per_cluster`. `0` disables the cap.
_CCCL_EXEC_CHECK_DISABLE
template <class PolicySelector>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL int topk_max_blocks_per_cluster_helper() noexcept
{
  constexpr auto policy = current_policy<PolicySelector>();
  if constexpr (policy.backend == topk_algorithm::cluster)
  {
    return policy.cluster.max_blocks_per_cluster;
  }
  else
  {
    // baseline / unsupported: not a cluster launch, so no cluster-width cap.
    return 0;
  }
}

// Variable templates force constant evaluation of the helpers, otherwise nvcc reports a "bad attribute argument
// substitution" error on the `__launch_bounds__` below (same pattern as `transform_kernel`).
template <class PolicySelector, class SegmentSizeParameterT, class... AgentParamsT>
inline constexpr int topk_threads_per_block =
  topk_threads_per_block_helper<PolicySelector, SegmentSizeParameterT, AgentParamsT...>();

template <class PolicySelector>
inline constexpr int topk_min_blocks_per_sm = topk_min_blocks_per_sm_helper<PolicySelector>();

template <class PolicySelector>
inline constexpr int topk_max_blocks_per_cluster = topk_max_blocks_per_cluster_helper<PolicySelector>();

// -----------------------------------------------------------------------------
// Global kernel entry point (single symbol for both backends)
// -----------------------------------------------------------------------------
// Launch bounds: only `topk_threads_per_block` takes the full kernel type list (its baseline branch runs the
// covering-policy search); min/max-blocks depend on `PolicySelector` alone. The parentheses around
// `topk_threads_per_block<...>` hide its template commas from the fixed-arity `_CCCL_LAUNCH_BOUNDS_CLUSTER(a, b, c)`.
template <typename PolicySelector,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename LargeSegmentTileOffsetT,
          ::cuda::execution::determinism::__determinism_t Determinism,
          ::cuda::execution::tie_break::__tie_break_t TieBreak>
_CCCL_LAUNCH_BOUNDS_CLUSTER((topk_threads_per_block<PolicySelector,
                                                    SegmentSizeParameterT,
                                                    KeyInputItItT,
                                                    KeyOutputItItT,
                                                    ValueInputItItT,
                                                    ValueOutputItItT,
                                                    SegmentSizeParameterT,
                                                    KParameterT,
                                                    SelectDirectionParameterT,
                                                    NumSegmentsParameterT,
                                                    LargeSegmentTileOffsetT>),
                            topk_min_blocks_per_sm<PolicySelector>,
                            topk_max_blocks_per_cluster<PolicySelector>) _CCCL_KERNEL_ATTRIBUTES void
device_batched_topk_kernel(
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments,
  baseline_kernel_args<typename ::cuda::args::__traits<NumSegmentsParameterT>::element_type, LargeSegmentTileOffsetT>
    base_args,
  [[maybe_unused]] cluster_kernel_args clus_args)
{
  constexpr auto policy = current_policy<PolicySelector>();

  if constexpr (policy.backend == topk_algorithm::baseline)
  {
    using agent_t = typename find_smallest_covering_policy<
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
      LargeSegmentTileOffsetT>::agent_t;

    static_assert(agent_t::tile_size >= ::cuda::args::__traits<SegmentSizeParameterT>::highest,
                  "Block size exceeds maximum segment size supported by SegmentSizeParameterT");
    static_assert(sizeof(typename agent_t::TempStorage) <= max_smem_per_block,
                  "Static shared memory per block must not exceed 48KB limit.");

    __shared__ typename agent_t::TempStorage temp_storage;

    agent_t agent(
      temp_storage,
      d_key_segments_it,
      d_key_segments_out_it,
      d_value_segments_it,
      d_value_segments_out_it,
      segment_sizes,
      k,
      select_directions,
      num_segments,
      base_args.d_counters,
      base_args.d_large_segments_ids,
      base_args.d_large_segments_tile_offsets);

    agent.Process();
  }
  else if constexpr (policy.backend == topk_algorithm::cluster)
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_90,
      (using agent_t = batched_topk_cluster::agent_batched_topk_cluster<
         policy.cluster.threads_per_block,
         policy.cluster.histogram_items_per_thread,
         policy.cluster.pipeline_stages,
         policy.cluster.chunk_bytes,
         policy.cluster.load_align_bytes,
         policy.cluster.bits_per_pass,
         policy.cluster.tie_break_items_per_thread,
         policy.cluster.single_block_max_seg_size,
         policy.cluster.min_chunks_per_block,
         policy.cluster.copy_items_per_thread,
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
       // Align the base up to `slot_alignment` (>= load_align) so every bulk-copy destination gets the same
       // `load_align` alignment the gmem sources have (peak TMA throughput on Hopper). The layout reserves
       // `base_padding_bytes` for this.
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
         k,
         select_directions,
         num_segments,
         key_slots,
         clus_args.block_tile_capacity);

       agent.Process();),
      // Cluster-policy kernels are only ever launched on SM90+, so the sub-SM90 device pass is unreachable at runtime.
      (_CCCL_UNREACHABLE();));
  }
  else
  {
    // topk_algorithm::unsupported: the host arm returns cudaErrorNotSupported before launching, so this never
    // runs.
    return;
  }
}
} // namespace detail::batched_topk

CUB_NAMESPACE_END
