// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_scan_warpspeed.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/util_macro.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
template <typename AccumT>
union tile_state_kernel_arg_t
{
  tile_state_t<AccumT>* lookahead;
  ScanTileState<AccumT> lookback;

  // ScanTileState<AccumT> is not trivially [default|copy]-constructible, so because of
  // https://eel.is/c++draft/class.union#general-note-3, tile_state_kernel_arg_t's special members are deleted. We work
  // around it by explicitly defining the ones we need.
  _CCCL_HOST_DEVICE tile_state_kernel_arg_t() noexcept {}
};

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * @brief Initialization kernel for tile status initialization (multi-block)
 *
 * @param[in] tile_state
 *   Tile status interface
 *
 * @param[in] num_tiles
 *   Number of tiles
 */
template <typename ChainedPolicyT, typename AccumT>
CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(128) void DeviceScanInitKernel(tile_state_kernel_arg_t<AccumT> tile_state, int num_tiles)
{
  if constexpr (detail::scan::scan_use_warpspeed<typename ChainedPolicyT::ActivePolicy>)
  {
    device_scan_init_lookahead_body(tile_state.lookahead, num_tiles);
  }
  else
  {
    _CCCL_PDL_GRID_DEPENDENCY_SYNC();
    _CCCL_PDL_TRIGGER_NEXT_LAUNCH(); // beneficial for all problem sizes in cub.bench.scan.exclusive.sum.base
    // Initialize tile status
    tile_state.lookback.InitializeStatus(num_tiles);
  }
}

/**
 * Initialization kernel for tile status initialization (multi-block)
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @param[in] tile_state
 *   Tile status interface
 *
 * @param[in] num_tiles
 *   Number of tiles
 *
 * @param[out] d_num_selected_out
 *   Pointer to the total number of items selected
 *   (i.e., length of `d_selected_out`)
 */
template <typename ScanTileStateT, typename NumSelectedIteratorT>
CUB_DETAIL_KERNEL_ATTRIBUTES void
DeviceCompactInitKernel(ScanTileStateT tile_state, int num_tiles, NumSelectedIteratorT d_num_selected_out)
{
  // Initialize tile status
  tile_state.InitializeStatus(num_tiles);

  // Initialize d_num_selected_out
  if ((blockIdx.x == 0) && (threadIdx.x == 0))
  {
    *d_num_selected_out = 0;
  }
}

/**
 * @brief Scan kernel entry point (multi-block)
 *
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading scan inputs @iterator
 *
 * @tparam OutputIteratorT
 *   Random-access output iterator type for writing scan outputs @iterator
 *
 * @tparam ScanOpT
 *   Binary scan functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitValueT
 *   Initial value to seed the exclusive scan
 *   (cub::NullType for inclusive scans)
 *
 * @tparam OffsetT
 *   Unsigned integer type for global offsets
 *
 * @paramInput d_in
 *   data
 *
 * @paramOutput d_out
 *   data
 *
 * @paramTile tile_state
 *   status interface
 *
 * @paramThe start_tile
 *   starting tile for the current grid
 *
 * @paramBinary scan_op
 *   scan functor
 *
 * @paramInitial init_value
 *   value to seed the exclusive scan
 *
 * @paramTotal num_items
 *   number of scan items for the entire problem
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT,
          bool ForceInclusive>
__launch_bounds__(detail::scan::scan_use_warpspeed<typename ChainedPolicyT::ActivePolicy>
                    ? get_scan_block_threads<typename ChainedPolicyT::ActivePolicy>
                    : int(ChainedPolicyT::ActivePolicy::ScanPolicyT::BLOCK_THREADS),
                  1) // TODO(bgruber): is the 1 ok?
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceScanKernel(
    __grid_constant__ const InputIteratorT d_in,
    __grid_constant__ const OutputIteratorT d_out,
    tile_state_kernel_arg_t<AccumT> tile_state,
    __grid_constant__ const int start_tile,
    ScanOpT scan_op,
    __grid_constant__ const InitValueT init_value,
    __grid_constant__ const OffsetT num_items,
    __grid_constant__ const int num_stages)
{
  using ActivePolicy = typename ChainedPolicyT::ActivePolicy;
  if constexpr (detail::scan::scan_use_warpspeed<ActivePolicy>)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_100, ({
                   auto scan_params = scanKernelParams<it_value_t<InputIteratorT>, it_value_t<OutputIteratorT>, AccumT>{
                     d_in, d_out, tile_state.lookahead, num_items, num_stages};
                   device_scan_lookahead_body<typename ActivePolicy::WarpspeedPolicy, ForceInclusive>(
                     scan_params, scan_op, init_value);
                 }));
  }
  else
  {
    using RealInitValueT = typename InitValueT::value_type;

    // Thread block type for scanning input tiles
    using AgentScanT = detail::scan::AgentScan<
      typename ActivePolicy::ScanPolicyT,
      InputIteratorT,
      OutputIteratorT,
      ScanOpT,
      RealInitValueT,
      OffsetT,
      AccumT,
      ForceInclusive,
      /* UsePDL */ true>;

    // Shared memory for AgentScan
    __shared__ typename AgentScanT::TempStorage temp_storage;

    // Depending on the version of the PTX memory model the compiler and hardware implement, we could move the grid
    // dependency sync back to the first read of data, that was actually written by the previous kernel (the tile
    // state). So the BlockLoad in this kernel could even overlap with the previous tile init kernel. To be save, we
    // retain it here before the first read.
    _CCCL_PDL_GRID_DEPENDENCY_SYNC();
    RealInitValueT real_init_value = init_value;

    // Process tiles
    AgentScanT(temp_storage, d_in, d_out, scan_op, real_init_value)
      .ConsumeRange(num_items, tile_state.lookback, start_tile);
  }
}
} // namespace detail::scan

CUB_NAMESPACE_END
