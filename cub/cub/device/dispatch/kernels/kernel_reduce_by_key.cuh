// SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * @brief Kernels for device-wide reduce by key.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_reduce_by_key.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce_by_key.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_vsmem.cuh>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes") // __visibility__ attribute ignored
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

namespace detail::reduce_by_key
{
/**
 * @brief Multi-block reduce-by-key sweep kernel entry point
 *
 * @tparam PolicySelector
 *   Selects the tuning policy
 *
 * @tparam KeysInputIteratorT
 *   Random-access input iterator type for keys
 *
 * @tparam UniqueOutputIteratorT
 *   Random-access output iterator type for keys
 *
 * @tparam ValuesInputIteratorT
 *   Random-access input iterator type for values
 *
 * @tparam AggregatesOutputIteratorT
 *   Random-access output iterator type for values
 *
 * @tparam NumRunsOutputIteratorT
 *   Output iterator type for recording number of segments encountered
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam EqualityOpT
 *   KeyT equality operator type
 *
 * @tparam ReductionOpT
 *   ValueT reduction operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param d_keys_in
 *   Pointer to the input sequence of keys
 *
 * @param d_unique_out
 *   Pointer to the output sequence of unique keys (one key per run)
 *
 * @param d_values_in
 *   Pointer to the input sequence of corresponding values
 *
 * @param d_aggregates_out
 *   Pointer to the output sequence of value aggregates (one aggregate per run)
 *
 * @param d_num_runs_out
 *   Pointer to total number of runs encountered
 *   (i.e., the length of d_unique_out)
 *
 * @param tile_state
 *   Tile status interface
 *
 * @param start_tile
 *   The starting tile for the current grid
 *
 * @param equality_op
 *   KeyT equality operator
 *
 * @param reduction_op
 *   ValueT reduction operator
 *
 * @param num_items
 *   Total number of items to select from
 */
template <typename PolicySelector,
          typename KeysInputIteratorT,
          typename UniqueOutputIteratorT,
          typename ValuesInputIteratorT,
          typename AggregatesOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename ScanTileStateT,
          typename EqualityOpT,
          typename ReductionOpT,
          typename OffsetT,
          typename AccumT,
          typename StreamingContextT>
#if _CCCL_HAS_CONCEPTS()
  requires reduce_by_key_policy_selector<PolicySelector>
#endif
__launch_bounds__(int(current_policy<PolicySelector>().lookback.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void DeviceReduceByKeyKernel(
    const KeysInputIteratorT d_keys_in,
    const UniqueOutputIteratorT d_unique_out,
    const ValuesInputIteratorT d_values_in,
    const AggregatesOutputIteratorT d_aggregates_out,
    const NumRunsOutputIteratorT d_num_runs_out,
    ScanTileStateT tile_state,
    const int start_tile,
    EqualityOpT equality_op,
    ReductionOpT reduction_op,
    const OffsetT num_items,
    const StreamingContextT streaming_context,
    vsmem_t vsmem)
{
  static constexpr ReduceByKeyPolicy policy = current_policy<PolicySelector>();
  using AgentReduceByKeyPolicyT             = agent_reduce_by_key_policy<
    policy.lookback.threads_per_block,
    policy.lookback.items_per_thread,
    policy.lookback.load_algorithm,
    policy.lookback.load_modifier,
    policy.lookback.scan_algorithm,
    delay_constructor_t<policy.lookback.lookback_delay.kind,
                        policy.lookback.lookback_delay.delay,
                        policy.lookback.lookback_delay.l2_write_latency>>;

  using vsmem_helper_t = vsmem_helper_default_fallback_policy_t<
    AgentReduceByKeyPolicyT,
    AgentReduceByKey,
    KeysInputIteratorT,
    UniqueOutputIteratorT,
    ValuesInputIteratorT,
    AggregatesOutputIteratorT,
    NumRunsOutputIteratorT,
    EqualityOpT,
    ReductionOpT,
    OffsetT,
    AccumT,
    StreamingContextT>;

  // Thread block type for reducing tiles of value segments
  using agent_reduce_by_key_t = typename vsmem_helper_t::agent_t;

  // Static shared memory allocation
  __shared__ typename vsmem_helper_t::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename agent_reduce_by_key_t::TempStorage& temp_storage =
    vsmem_helper_t::get_temp_storage(static_temp_storage, vsmem);

  // Process tiles
  agent_reduce_by_key_t(
    temp_storage,
    d_keys_in,
    d_unique_out,
    d_values_in,
    d_aggregates_out,
    d_num_runs_out,
    equality_op,
    reduction_op,
    streaming_context)
    .ConsumeRange(num_items, tile_state, start_tile);

  // If applicable, hints to discard modified cache lines for vsmem
  vsmem_helper_t::discard_temp_storage(temp_storage);
}
} // namespace detail::reduce_by_key

CUB_NAMESPACE_END

_CCCL_DIAG_POP
