// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
#include <cub/detail/warpspeed/look_ahead.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/util_macro.cuh>

#if _CCCL_CUDACC_AT_LEAST(12, 8)
#  include <cub/device/dispatch/kernels/kernel_scan_warpspeed.cuh>
#endif // _CCCL_CUDACC_AT_LEAST(12, 8)

#include <thrust/type_traits/is_contiguous_iterator.h>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
template <typename ScanTileState, typename AccumT>
union tile_state_kernel_arg_t
{
  warpspeed::tile_state_t<AccumT>* lookahead;
  ScanTileState lookback;

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
template <typename PolicySelectorT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanTileState,
          typename AccumT>
CUB_DETAIL_KERNEL_ATTRIBUTES __launch_bounds__(128) void DeviceScanInitKernel(
  tile_state_kernel_arg_t<ScanTileState, AccumT> tile_state, int num_tiles)
{
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH(); // beneficial for all problem sizes in cub.bench.scan.exclusive.sum.base

#if _CCCL_CUDACC_AT_LEAST(12, 8)
  constexpr scan_policy policy = PolicySelectorT{}(::cuda::arch_id{CUB_PTX_ARCH / 10});
  if constexpr (policy.warpspeed
                && detail::scan::use_warpspeed<InputIteratorT, OutputIteratorT, AccumT>(policy.warpspeed))
  {
    device_scan_init_lookahead_body(tile_state.lookahead, num_tiles);
  }
  else
#endif // _CCCL_CUDACC_AT_LEAST(12, 8)
  {
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
template <typename PolicySelector, typename InputIteratorT, typename OutputIteratorT, typename AccumT>
_CCCL_API constexpr int get_device_scan_launch_bounds_helper() noexcept
{
  constexpr scan_policy policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10});
#if _CCCL_CUDACC_AT_LEAST(12, 8)
  if constexpr (policy.warpspeed
                && detail::scan::use_warpspeed<InputIteratorT, OutputIteratorT, AccumT>(policy.warpspeed))
  {
    return policy.warpspeed.num_total_threads;
  }
#endif // _CCCL_CUDACC_AT_LEAST(12, 8)
  return policy.block_threads;
}

// need a variable template to force constant evaluation, otherwise nvcc may emit
// "bad attribute argument substitution" errors for __launch_bounds__
template <typename PolicySelector, typename InputIteratorT, typename OutputIteratorT, typename AccumT>
inline constexpr int get_device_scan_launch_bounds =
  get_device_scan_launch_bounds_helper<PolicySelector, InputIteratorT, OutputIteratorT, AccumT>();

/**
 * @brief Scan kernel entry point (multi-block)
 *
 *
 * @tparam PolicySelector
 *   Policy selector for tuning
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
template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanTileState,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT,
          bool ForceInclusive,
          typename RealInitValueT = typename InitValueT::value_type>
__launch_bounds__(get_device_scan_launch_bounds<PolicySelector, InputIteratorT, OutputIteratorT, AccumT>)
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceScanKernel(
    _CCCL_GRID_CONSTANT const InputIteratorT d_in,
    _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
    tile_state_kernel_arg_t<ScanTileState, AccumT> tile_state,
    _CCCL_GRID_CONSTANT const int start_tile,
    ScanOpT scan_op,
// nvcc 12.0 gets stuck compiling some TUs like `cub.bench.scan.exclusive.sum.base`, so only enable for newer versions
#if _CCCL_CUDACC_AT_LEAST(12, 8)
    _CCCL_GRID_CONSTANT
#endif // _CCCL_CUDACC_AT_LEAST(12, 8)
    const InitValueT init_value,
    _CCCL_GRID_CONSTANT const OffsetT num_items,
    _CCCL_GRID_CONSTANT const int num_stages)
{
  static constexpr scan_policy policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10});
  static_assert(policy.load_modifier != CacheLoadModifier::LOAD_LDG,
                "The memory consistency model does not apply to texture "
                "accesses");

#if _CCCL_CUDACC_AT_LEAST(12, 8)
  if constexpr (policy.warpspeed
                && detail::scan::use_warpspeed<InputIteratorT, OutputIteratorT, AccumT>(policy.warpspeed))
  {
    using WarpspeedPolicyT = warpspeedKernelPolicy<
      scan_warpspeed_policy::num_squads,
      policy.warpspeed.num_reduce_warps,
      policy.warpspeed.num_scan_stor_warps,
      policy.warpspeed.num_load_warps,
      policy.warpspeed.num_sched_warps,
      policy.warpspeed.num_look_ahead_warps,
      policy.warpspeed.num_look_ahead_items,
      policy.warpspeed.num_total_threads,
      policy.warpspeed.items_per_thread,
      policy.warpspeed.tile_size>;
    NV_IF_TARGET(
      NV_PROVIDES_SM_100, ({
        auto scan_params = scanKernelParams<it_value_t<InputIteratorT>, it_value_t<OutputIteratorT>, AccumT>{
          d_in, d_out, tile_state.lookahead, num_items, num_stages};
        device_scan_lookahead_body<WarpspeedPolicyT, ForceInclusive, RealInitValueT>(scan_params, scan_op, init_value);
      }));
  }
  else
#endif // _CCCL_CUDACC_AT_LEAST(12, 8)
  {
    using ScanPolicyT = AgentScanPolicy<
      policy.block_threads,
      policy.items_per_thread,
      AccumT,
      policy.load_algorithm,
      policy.load_modifier,
      policy.store_algorithm,
      policy.scan_algorithm,
      NoScaling<policy.block_threads, policy.items_per_thread>,
      delay_constructor_t<policy.delay_constructor.kind,
                          policy.delay_constructor.delay,
                          policy.delay_constructor.l2_write_latency>>;

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
