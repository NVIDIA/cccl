// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_three_way_partition.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::three_way_partition
{
/******************************************************************************
 * Kernel entry points
 *****************************************************************************/
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename FirstOutputIteratorT,
          typename SecondOutputIteratorT,
          typename UnselectedOutputIteratorT,
          typename NumSelectedIteratorT,
          typename ScanTileStateT,
          typename SelectFirstPartOp,
          typename SelectSecondPartOp,
          typename OffsetT,
          typename StreamingContextT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ThreeWayPartitionPolicy::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceThreeWayPartitionKernel(
    InputIteratorT d_in,
    FirstOutputIteratorT d_first_part_out,
    SecondOutputIteratorT d_second_part_out,
    UnselectedOutputIteratorT d_unselected_out,
    NumSelectedIteratorT d_num_selected_out,
    ScanTileStateT tile_status,
    SelectFirstPartOp select_first_part_op,
    SelectSecondPartOp select_second_part_op,
    OffsetT num_items,
    int num_tiles,
    _CCCL_GRID_CONSTANT const StreamingContextT streaming_context)
{
  using AgentThreeWayPartitionPolicyT = typename ChainedPolicyT::ActivePolicy::ThreeWayPartitionPolicy;

  // Thread block type for selecting data from input tiles
  using AgentThreeWayPartitionT = AgentThreeWayPartition<
    AgentThreeWayPartitionPolicyT,
    InputIteratorT,
    FirstOutputIteratorT,
    SecondOutputIteratorT,
    UnselectedOutputIteratorT,
    SelectFirstPartOp,
    SelectSecondPartOp,
    OffsetT,
    StreamingContextT>;

  // Shared memory for AgentThreeWayPartition
  __shared__ typename AgentThreeWayPartitionT::TempStorage temp_storage;

  // Process tiles
  AgentThreeWayPartitionT(
    temp_storage,
    d_in,
    d_first_part_out,
    d_second_part_out,
    d_unselected_out,
    select_first_part_op,
    select_second_part_op,
    num_items,
    streaming_context)
    .ConsumeRange(num_tiles, tile_status, d_num_selected_out);
}

/**
 * @brief Initialization kernel for tile status initialization (multi-block)
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @param[in] tile_state_1
 *   Tile status interface
 *
 * @param[in] tile_state_2
 *   Tile status interface
 *
 * @param[in] num_tiles
 *   Number of tiles
 *
 * @param[out] d_num_selected_out
 *   Pointer to the total number of items selected
 *   (i.e., length of @p d_selected_out)
 */
template <typename ScanTileStateT, typename NumSelectedIteratorT>
CUB_DETAIL_KERNEL_ATTRIBUTES void
DeviceThreeWayPartitionInitKernel(ScanTileStateT tile_state, int num_tiles, NumSelectedIteratorT d_num_selected_out)
{
  // Initialize tile status
  tile_state.InitializeStatus(num_tiles);

  // Initialize d_num_selected_out
  if (blockIdx.x == 0)
  {
    if (threadIdx.x < 2)
    {
      d_num_selected_out[threadIdx.x] = 0;
    }
  }
}
} // namespace detail::three_way_partition

CUB_NAMESPACE_END
