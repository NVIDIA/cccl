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
#include <cub/util_macro.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * @brief Initialization kernel for tile status initialization (multi-block)
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @param[in] tile_state
 *   Tile status interface
 *
 * @param[in] num_tiles
 *   Number of tiles
 */
template <typename ScanTileStateT>
CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceScanInitKernel(ScanTileStateT tile_state, int num_tiles)
{
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH(); // beneficial for all problem sizes in cub.bench.scan.exclusive.sum.base
  // Initialize tile status
  tile_state.InitializeStatus(num_tiles);
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
 * @tparam ScanTileStateT
 *   Tile status interface type
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
          typename ScanTileStateT,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT,
          bool ForceInclusive,
          typename RealInitValueT = typename InitValueT::value_type>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ScanPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceScanKernel(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanTileStateT tile_state,
    int start_tile,
    ScanOpT scan_op,
    InitValueT init_value,
    OffsetT num_items)
{
  using ScanPolicyT = typename ChainedPolicyT::ActivePolicy::ScanPolicyT;

  // Thread block type for scanning input tiles
  using AgentScanT = detail::scan::AgentScan<
    ScanPolicyT,
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
  // dependency sync back to the first read of data, that was actually written by the previous kernel (the tile state).
  // So the BlockLoad in this kernel could even overlap with the previous tile init kernel. To be save, we retain it
  // here before the first read.
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();
  RealInitValueT real_init_value = init_value;

  // Process tiles
  AgentScanT(temp_storage, d_in, d_out, scan_op, real_init_value).ConsumeRange(num_items, tile_state, start_tile);
}
} // namespace detail::scan

CUB_NAMESPACE_END
