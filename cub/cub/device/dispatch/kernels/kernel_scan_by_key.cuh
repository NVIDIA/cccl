// SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * @brief Kernels for device-wide scan by key.
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

#include <cub/agent/agent_scan_by_key.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/device/dispatch/tuning/tuning_scan_by_key.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_macro.cuh>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes") // __visibility__ attribute ignored
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

namespace detail::scan_by_key
{
/**
 * @brief Scan by key kernel entry point (multi-block)
 *
 * @tparam PolicySelector
 *   Policy selector type
 *
 * @tparam KeysInputIteratorT
 *   Random-access input iterator type
 *
 * @tparam ValuesInputIteratorT
 *   Random-access input iterator type
 *
 * @tparam ValuesOutputIteratorT
 *   Random-access output iterator type
 *
 * @tparam ScanByKeyTileStateT
 *   Tile status interface type
 *
 * @tparam EqualityOp
 *   Equality functor type
 *
 * @tparam ScanOpT
 *   Scan functor type
 *
 * @tparam InitValueT
 *   The init_value element for ScanOpT type (cub::NullType for inclusive scan)
 *
 * @tparam OffsetT
 *   Unsigned integer type for global offsets
 *
 * @param d_keys_in
 *   Input keys data
 *
 * @param d_keys_prev_in
 *   Predecessor items for each tile
 *
 * @param d_values_in
 *   Input values data
 *
 * @param d_values_out
 *   Output values data
 *
 * @param tile_state
 *   Tile status interface
 *
 * @param start_tile
 *   The starting tile for the current grid
 *
 * @param equality_op
 *   Binary equality functor
 *
 * @param scan_op
 *   Binary scan functor
 *
 * @param init_value
 *   Initial value to seed the exclusive scan
 *
 * @param num_items
 *   Total number of scan items for the entire problem
 */
template <typename PolicySelector,
          typename KeysInputIteratorT,
          typename ValuesInputIteratorT,
          typename ValuesOutputIteratorT,
          typename ScanByKeyTileStateT,
          typename EqualityOp,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT,
          typename KeyT = cub::detail::it_value_t<KeysInputIteratorT>>
__launch_bounds__(int(current_policy<PolicySelector>().lookback.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void DeviceScanByKeyKernel(
    const KeysInputIteratorT d_keys_in,
    KeyT* const d_keys_prev_in,
    const ValuesInputIteratorT d_values_in,
    const ValuesOutputIteratorT d_values_out,
    ScanByKeyTileStateT tile_state,
    const int start_tile,
    EqualityOp equality_op,
    const ScanOpT scan_op,
    const InitValueT init_value,
    const OffsetT num_items)
{
  static constexpr ScanByKeyPolicy policy = current_policy<PolicySelector>();

  using scan_by_key_policy_t = agent_scan_by_key_policy<
    policy.lookback.threads_per_block,
    policy.lookback.items_per_thread,
    policy.lookback.load_algorithm,
    policy.lookback.load_modifier,
    policy.lookback.scan_algorithm,
    policy.lookback.store_algorithm,
    delay_constructor_t<policy.lookback.lookback_delay.kind,
                        policy.lookback.lookback_delay.delay,
                        policy.lookback.lookback_delay.l2_write_latency>>;

  // Thread block type for scanning input tiles
  using AgentScanByKeyT = detail::scan_by_key::AgentScanByKey<
    scan_by_key_policy_t,
    KeysInputIteratorT,
    ValuesInputIteratorT,
    ValuesOutputIteratorT,
    EqualityOp,
    ScanOpT,
    InitValueT,
    OffsetT,
    AccumT>;

  // Shared memory for AgentScanByKey
  __shared__ typename AgentScanByKeyT::TempStorage temp_storage;

  // Process tiles
  AgentScanByKeyT(temp_storage, d_keys_in, d_keys_prev_in, d_values_in, d_values_out, equality_op, scan_op, init_value)
    .ConsumeRange(num_items, tile_state, start_tile);
}

template <typename ScanTileStateT, typename KeysInputIteratorT, typename OffsetT>
_CCCL_KERNEL_ATTRIBUTES void DeviceScanByKeyInitKernel(
  ScanTileStateT tile_state,
  const KeysInputIteratorT d_keys_in,
  cub::detail::it_value_t<KeysInputIteratorT>* d_keys_prev_in,
  const OffsetT items_per_tile,
  const int num_tiles)
{
  // Initialize tile status
  tile_state.InitializeStatus(num_tiles);

  const int tid           = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const OffsetT tile_base = static_cast<OffsetT>(tid) * items_per_tile;
  if (tid > 0 && tid < num_tiles)
  {
    d_keys_prev_in[tid] = d_keys_in[tile_base - 1];
  }
}
} // namespace detail::scan_by_key

CUB_NAMESPACE_END

_CCCL_DIAG_POP
