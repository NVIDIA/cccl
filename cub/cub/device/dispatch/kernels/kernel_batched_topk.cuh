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
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>

#include <cuda/__cmath/round_up.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/argument>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
template <typename KeyT, typename ValueT, int ThreadsPerBlock, int ItemsPerThread>
union sort_temp_storage
{
  typename BlockLoad<KeyT, ThreadsPerBlock, ItemsPerThread, BLOCK_LOAD_WARP_TRANSPOSE>::TempStorage load_keys;
  typename BlockLoad<ValueT, ThreadsPerBlock, ItemsPerThread, BLOCK_LOAD_WARP_TRANSPOSE>::TempStorage load_values;
  typename BlockRadixSort<KeyT, ThreadsPerBlock, ItemsPerThread, ValueT>::TempStorage sort;
};

template <typename KeyT, typename ValueT, ::cuda::std::int64_t StaticMaxOut>
struct find_sort_policy_index
{
private:
  template <int Index>
  [[nodiscard]] static constexpr int find_index()
  {
    if constexpr (Index >= sorted_output_policy_count)
    {
      return -1;
    }
    else
    {
      constexpr sort_policy candidate = sorted_output_policies[Index];
      constexpr ::cuda::std::int64_t tile_size =
        ::cuda::std::int64_t{candidate.threads_per_block} * candidate.items_per_thread;
      static_assert(tile_size <= 65535, "BlockRadixSort supports at most 65535 items per block.");
      constexpr bool covers = tile_size >= StaticMaxOut;
      constexpr bool fits_smem =
        sizeof(sort_temp_storage<KeyT, ValueT, candidate.threads_per_block, candidate.items_per_thread>)
        <= max_smem_per_block;
      constexpr int next = find_index<Index + 1>();

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

template <typename KeyT, typename ValueT, ::cuda::std::int64_t StaticMaxOut>
inline constexpr bool sort_can_cover_v = find_sort_policy_index<KeyT, ValueT, StaticMaxOut>::value >= 0;

template <typename KeyT, typename ValueT, ::cuda::std::int64_t StaticMaxOut>
struct find_smallest_sort_policy
{
  static constexpr int index = find_sort_policy_index<KeyT, ValueT, StaticMaxOut>::value;
  static_assert(index >= 0,
                "cub::DeviceBatchedTopK: no sorted-output policy covers the statically-known maximum output size "
                "within the shared-memory limit.");
  static constexpr sort_policy policy = sorted_output_policies[index];
};

template <typename KeyT, typename ValueT, int Index = 0>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::int64_t sort_covered_max()
{
  if constexpr (Index >= sorted_output_policy_count)
  {
    return 0;
  }
  else
  {
    constexpr sort_policy candidate = sorted_output_policies[Index];
    constexpr ::cuda::std::int64_t tile_size =
      ::cuda::std::int64_t{candidate.threads_per_block} * candidate.items_per_thread;
    constexpr bool fits_smem =
      sizeof(sort_temp_storage<KeyT, ValueT, candidate.threads_per_block, candidate.items_per_thread>)
      <= max_smem_per_block;
    constexpr ::cuda::std::int64_t next = sort_covered_max<KeyT, ValueT, Index + 1>();
    return fits_smem && tile_size > next ? tile_size : next;
  }
}

struct sixteen_byte_value
{
  ::cuda::std::int64_t first;
  ::cuda::std::int64_t second;
};

static_assert(sort_covered_max<::cuda::std::int32_t, ::cuda::std::int32_t>() >= 2048);
static_assert(sort_covered_max<::cuda::std::int64_t, ::cuda::std::int32_t>() >= 2048);
static_assert(sort_covered_max<::cuda::std::int64_t, ::cuda::std::int64_t>() >= 2048);
static_assert(sort_covered_max<float, ::cuda::std::int32_t>() >= 2048);
static_assert(sort_covered_max<::cuda::std::int64_t, sixteen_byte_value>() >= 2048);

// Assert-free search shared by `find_smallest_covering_policy_device` and the backend coverage predicate. Returns the
// index of the smallest worker policy whose tile size still covers the upper bound on segment size AND whose
// instantiated agent's shared memory usage fits within the static shared memory limit (max_smem_per_block), or -1 if
// none does. Kept separate from `find_smallest_covering_policy_device` so callers can query coverage as a bool without
// tripping that trait's hard `static_assert`.
template <typename PolicyGetter, typename SegmentSizeParameterT, typename... AgentParamsT>
struct find_covering_policy_index
{
private:
  struct policy_t
  {
    worker_policy worker_per_segment_policy;
    multi_worker_policy multi_worker_per_segment_policy;
  };
  static constexpr ::cuda::std::int64_t max_segment_size = ::cuda::args::__traits<SegmentSizeParameterT>::highest;
  static constexpr topk_policy active_policy             = PolicyGetter{}();

  template <int Index>
  [[nodiscard]] static constexpr int find_index()
  {
    if constexpr (Index >= active_policy.baseline.worker_per_segment_policies.size())
    {
      return -1;
    }
    else
    {
      constexpr worker_policy wp = active_policy.baseline.worker_per_segment_policies[Index];
      constexpr auto tile_size   = ::cuda::std::int64_t{wp.threads_per_block} * wp.items_per_thread;

      struct policy_getter_17 // TODO(bgruber): drop this in C++20 and pass wp directly
      {
        _CCCL_HOST_DEVICE_API constexpr auto operator()() const
        {
          return policy_t{active_policy.baseline.worker_per_segment_policies[Index],
                          active_policy.baseline.multi_worker_per_segment_policy};
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
template <typename PolicyGetter, typename SegmentSizeParameterT, typename... AgentParamsT>
inline constexpr bool baseline_can_cover_v =
  find_covering_policy_index<PolicyGetter, SegmentSizeParameterT, AgentParamsT...>::value >= 0;

// Resolves the agent type the kernel instantiates via the same covering-policy search as `find_covering_policy_index`,
// adding a hard `static_assert` when no policy covers the segment size within the shared-memory limit. `PolicyGetter`
// is a nullary constant-expression getter returning the resolved `topk_policy` (e.g. the resolved-CC policy from
// `dispatch_compute_cap`); use this form when you already have the resolved policy. Prefer the
// `find_smallest_covering_policy_device` alias below when you have a `PolicySelector`.
template <typename PolicyGetter, typename SegmentSizeParameterT, typename... AgentParamsT>
struct find_smallest_covering_policy_for_getter
{
private:
  struct policy_t
  {
    worker_policy worker_per_segment_policy;
    multi_worker_policy multi_worker_per_segment_policy;
  };
  static constexpr topk_policy active_policy = PolicyGetter{}();
  static constexpr int selected_index =
    find_covering_policy_index<PolicyGetter, SegmentSizeParameterT, AgentParamsT...>::value;

public:
  // TODO (elstehle): extend support for variable-size segments
  static_assert(selected_index >= 0,
                "cub::DeviceBatchedTopK: no baseline worker policy covers the statically-known maximum segment size "
                "within the shared-memory limit. Reduce the maximum segment size encoded in the segment-size argument "
                "annotation (larger segments are served by the SM 9.0+ cluster backend).");
  static constexpr policy_t policy = {active_policy.baseline.worker_per_segment_policies[selected_index],
                                      active_policy.baseline.multi_worker_per_segment_policy};

  struct policy_getter_17 // TODO(bgruber): drop this in C++20 and pass policy directly
  {
    _CCCL_HOST_DEVICE_API constexpr auto operator()() const
    {
      return policy;
    }
  };
  using agent_t = agent_batched_topk_worker_per_segment<policy_getter_17, AgentParamsT...>;
};

// `PolicySelector`-based form: resolves the policy for this compilation's CC via `current_policy<PolicySelector>()`
// (per-`__CUDA_ARCH__` device-side; the host default CC host-side). Device consumers (kernel body, launch-bounds
// helpers) use this. The host baseline arm instead uses `find_smallest_covering_policy_for_getter` with the
// resolved-CC getter so its policy choice matches the device kernel per CC.
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
struct find_smallest_covering_policy_device
{
private:
#if _CCCL_HAS_CONCEPTS()
  static_assert(topk_policy_selector<PolicySelector>);
#endif

  struct active_policy_getter_17 // TODO(bgruber): drop this in C++20 and pass policy directly
  {
    _CCCL_HOST_DEVICE_API constexpr auto operator()() const
    {
      return current_policy<PolicySelector>();
    }
  };
  using impl_t =
    find_smallest_covering_policy_for_getter<active_policy_getter_17, SegmentSizeParameterT, AgentParamsT...>;

public:
  static constexpr auto policy = impl_t::policy;
  using agent_t                = typename impl_t::agent_t;
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
template <typename NumSegmentsValueT, typename LargeSegmentTileOffsetT>
struct baseline_kernel_args
{
  batched_topk_counters<NumSegmentsValueT>* d_counters   = nullptr;
  NumSegmentsValueT* d_large_segments_ids                = nullptr;
  LargeSegmentTileOffsetT* d_large_segments_tile_offsets = nullptr;
};

struct cluster_kernel_args
{
  ::cuda::std::uint32_t max_block_resident_items = 0;
};

// -----------------------------------------------------------------------------
// Launch-bounds helpers
// -----------------------------------------------------------------------------
// The two backends use different `__launch_bounds__` shapes (baseline: just threads_per_block; cluster: threads plus a
// min-blocks-per-SM and an optional max-blocks-per-cluster cap). We resolve all three per architecture from the
// selected policy. `find_smallest_covering_policy_device` (which carries a hard `static_assert`) is only ever touched
// inside the `backend == baseline` branch, so an oversize bound routed to the cluster/unsupported backend never trips
// it.
_CCCL_EXEC_CHECK_DISABLE
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL int topk_threads_per_block_helper() noexcept
{
  constexpr auto policy = current_policy<PolicySelector>();
  if constexpr (policy.backend == topk_algorithm::baseline)
  {
    return find_smallest_covering_policy_device<PolicySelector, SegmentSizeParameterT, AgentParamsT...>::policy
      .worker_per_segment_policy.threads_per_block;
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
template <typename PolicySelector>
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
template <typename PolicySelector>
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
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
inline constexpr int topk_threads_per_block =
  topk_threads_per_block_helper<PolicySelector, SegmentSizeParameterT, AgentParamsT...>();

template <typename PolicySelector>
inline constexpr int topk_min_blocks_per_sm = topk_min_blocks_per_sm_helper<PolicySelector>();

template <typename PolicySelector>
inline constexpr int topk_max_blocks_per_cluster = topk_max_blocks_per_cluster_helper<PolicySelector>();

// Hands the cluster agent its resolved sub-policy as a type (C++17 has no class-type NTTP).
// TODO(bgruber): drop this in C++20 and pass `policy.cluster` by value.
template <typename PolicySelector>
struct cluster_policy_getter
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()() const
  {
    return current_policy<PolicySelector>().cluster;
  }
};

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
    using agent_t = typename find_smallest_covering_policy_device<
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
         cluster_policy_getter<PolicySelector>,
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

       // A `tune`d override with an oversized static footprint (e.g. a large `bits_per_pass` histogram) fails here
       // rather than as an opaque ptxas error. Only the static footprint is checked: the dynamic block-tile slots may
       // exceed the static shared-memory cap via opt-in.
       static_assert(sizeof(typename agent_t::TempStorage) <= max_smem_per_block,
                     "Static shared memory per block must not exceed 48KB limit.");

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
         clus_args.max_block_resident_items);

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

template <int ThreadsPerBlock,
          int ItemsPerThread,
          typename KeyT,
          typename ValueT,
          typename KeyOutputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
_CCCL_LAUNCH_BOUNDS(ThreadsPerBlock) _CCCL_KERNEL_ATTRIBUTES void device_batched_topk_sort_kernel(
  const KeyT* d_key_scratch,
  const ValueT* d_value_scratch,
  KeyOutputItItT d_key_segments_out_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k_param,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments,
  ::cuda::std::int64_t max_out)
{
  constexpr bool is_keys_only = ::cuda::std::is_same_v<ValueT, NullType>;
  using block_radix_sort_t    = BlockRadixSort<KeyT, ThreadsPerBlock, ItemsPerThread, ValueT>;
  using block_load_keys_t     = BlockLoad<KeyT, ThreadsPerBlock, ItemsPerThread, BLOCK_LOAD_WARP_TRANSPOSE>;
  using block_load_values_t   = BlockLoad<ValueT, ThreadsPerBlock, ItemsPerThread, BLOCK_LOAD_WARP_TRANSPOSE>;
  using traits                = detail::radix::traits_t<KeyT>;
  using bit_ordered_type      = typename traits::bit_ordered_type;

  const int segment_id = static_cast<int>(blockIdx.x);
  if (segment_id >= params::get_param(num_segments, 0))
  {
    return;
  }

  const auto segment_size = params::__get_and_clamp_param_to_nonnegative(segment_sizes, segment_id);
  const auto k_eff        = static_cast<decltype(segment_size)>(
    (::cuda::std::min) (static_cast<::cuda::std::uint64_t>(
                          params::__get_and_clamp_param_to_nonnegative(k_param, segment_id)),
                        static_cast<::cuda::std::uint64_t>(segment_size)));
  if (k_eff == 0)
  {
    return;
  }

  const int valid_items     = static_cast<int>(k_eff);
  const auto scratch_offset = static_cast<::cuda::std::int64_t>(segment_id) * max_out;
  const KeyT* const d_keys  = d_key_scratch + scratch_offset;

  __shared__ sort_temp_storage<KeyT, ValueT, ThreadsPerBlock, ItemsPerThread> temp_storage;

  KeyT keys[ItemsPerThread];
  ValueT values[ItemsPerThread];

  const bool is_successful_dispatch = params::dispatch_discrete(select_directions, segment_id, [&](auto direction_tag) {
    constexpr bool descending         = decltype(direction_tag)::value == detail::topk::select::max;
    bit_ordered_type default_key_bits = descending ? traits::min_raw_binary_key(detail::identity_decomposer_t{})
                                                   : traits::max_raw_binary_key(detail::identity_decomposer_t{});
    KeyT default_key                  = reinterpret_cast<KeyT&>(default_key_bits);

    // Unstable sorting permits this register-only striped load, avoiding a shared-memory transpose.
    if constexpr (is_keys_only)
    {
      LoadDirectStriped<ThreadsPerBlock>(threadIdx.x, d_keys, keys, valid_items, default_key);
    }
    else
    {
      const ValueT* const d_values = d_value_scratch + scratch_offset;
      block_load_keys_t(temp_storage.load_keys).Load(d_keys, keys, valid_items, default_key);
      __syncthreads();
      block_load_values_t(temp_storage.load_values).Load(d_values, values, valid_items, ValueT{});
      __syncthreads();
    }

    // `NullType` selects BlockRadixSort's keys-only internal path, so this works for both keys and pairs.
    if constexpr (descending)
    {
      block_radix_sort_t(temp_storage.sort).SortDescendingBlockedToStriped(keys, values);
    }
    else
    {
      block_radix_sort_t(temp_storage.sort).SortBlockedToStriped(keys, values);
    }

    StoreDirectStriped<ThreadsPerBlock>(threadIdx.x, d_key_segments_out_it[segment_id], keys, valid_items);
    if constexpr (!is_keys_only)
    {
      StoreDirectStriped<ThreadsPerBlock>(threadIdx.x, d_value_segments_out_it[segment_id], values, valid_items);
    }
  });
  _CCCL_ASSERT(is_successful_dispatch, "Error: Unsupported select direction");
}
} // namespace detail::batched_topk

CUB_NAMESPACE_END
