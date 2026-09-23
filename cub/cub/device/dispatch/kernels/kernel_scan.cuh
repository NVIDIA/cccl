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
#include <cub/detail/deferred_parameter.cuh>
#include <cub/detail/warpspeed/look_ahead.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_macro.cuh>

#if _CCCL_CUDACC_AT_LEAST(12, 8)
#  include <cub/device/dispatch/kernels/kernel_scan_lookahead.cuh>
#endif // _CCCL_CUDACC_AT_LEAST(12, 8)

#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/atomic>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
// Deferred scans reuse one fixed-size tile state across batches.
inline constexpr ::cuda::std::uint32_t tiles_per_batch = 65536;

// CTAs per SM in the persistent grid.
inline constexpr int batch_ctas_per_sm = 8;

template <typename AccumT>
struct scan_batch_state
{
  ::cuda::std::uint64_t batch_idx;
  ::cuda::std::uint64_t next_tile_idx;
  ::cuda::std::uint32_t lookbacks_completed;
  Uninitialized<AccumT> batched_sum;
};

template <typename AccumT>
struct lookahead_tile_state_arg_t
{
  warpspeed::tile_state_t<AccumT>* tile_states;
  ::cuda::std::uint32_t* atomic_counter;
};

template <typename ScanTileState, typename AccumT>
union tile_state_kernel_arg_t
{
  lookahead_tile_state_arg_t<AccumT> lookahead;
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
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(128) void DeviceScanInitKernel(
  tile_state_kernel_arg_t<ScanTileState, AccumT> tile_state, const int num_tiles)
{
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH(); // beneficial for all problem sizes in cub.bench.scan.exclusive.sum.base

#if _CCCL_CUDACC_AT_LEAST(12, 8)
  constexpr ScanPolicy policy = current_policy<PolicySelectorT>();
  if constexpr (policy.algorithm == ScanAlgorithm::lookahead)
  {
    device_scan_init_lookahead_body(tile_state.lookahead.tile_states, num_tiles, tile_state.lookahead.atomic_counter);
  }
  else
#endif // _CCCL_CUDACC_AT_LEAST(12, 8)
  {
    // Initialize tile status
    tile_state.lookback.InitializeStatus(num_tiles);
  }
}

/**
 * @brief Initialization kernel for batched tile status and batch state
 *
 * @param[in] tile_state
 *   Tile status interface
 *
 * @param[out] batch_state
 *   Batched look-back state
 */
template <typename ScanTileState, typename AccumT>
_CCCL_KERNEL_ATTRIBUTES
__launch_bounds__(128) void DeviceScanBatchInitKernel(ScanTileState tile_state, scan_batch_state<AccumT>* batch_state)
{
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();

  tile_state.InitializeStatus(static_cast<int>(tiles_per_batch));
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    batch_state->batch_idx           = 0;
    batch_state->next_tile_idx       = 0;
    batch_state->lookbacks_completed = 0;
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
_CCCL_KERNEL_ATTRIBUTES void
DeviceCompactInitKernel(ScanTileStateT tile_state, const int num_tiles, NumSelectedIteratorT d_num_selected_out)
{
  // Initialize tile status
  tile_state.InitializeStatus(num_tiles);

  // Initialize d_num_selected_out
  if ((blockIdx.x == 0) && (threadIdx.x == 0))
  {
    *d_num_selected_out = 0;
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <typename PolicySelector>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL int get_device_scan_launch_bounds() noexcept
{
  constexpr ScanPolicy policy = current_policy<PolicySelector>();
#if _CCCL_CUDACC_AT_LEAST(12, 8)
  if constexpr (policy.algorithm == ScanAlgorithm::lookahead)
  {
    return num_total_threads(policy.lookahead);
  }
#endif // _CCCL_CUDACC_AT_LEAST(12, 8)
  return policy.lookback.threads_per_block;
}

// need a variable template for clang in CUDA mode to avoid:
// error: 'launch_bounds' attribute requires parameter 0 to be an integer constant
template <typename PolicySelector>
inline constexpr int device_scan_launch_bounds = get_device_scan_launch_bounds<PolicySelector>();

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
          typename KernelNumItemsT,
          typename AccumT,
          bool ForceInclusive,
          bool StableReductionOrder = false,
          typename RealInitValueT   = typename InitValueT::value_type>
__launch_bounds__(device_scan_launch_bounds<PolicySelector>, 1) _CCCL_KERNEL_ATTRIBUTES void DeviceScanKernel(
  const InputIteratorT d_in,
  const OutputIteratorT d_out,
  tile_state_kernel_arg_t<ScanTileState, AccumT> tile_state,
  const int start_tile,
  const ScanOpT scan_op,
  const InitValueT init_value,
  const KernelNumItemsT kernel_num_items,
  const int num_stages)
{
  const OffsetT num_items = CUB_NS_QUALIFIER::detail::parameter_from_device<OffsetT>(kernel_num_items);

  static constexpr ScanPolicy active_policy = current_policy<PolicySelector>();
  if constexpr (active_policy.algorithm == ScanAlgorithm::lookahead)
  {
#if _CCCL_CUDACC_AT_LEAST(12, 8)
    NV_IF_TARGET(
      NV_PROVIDES_SM_90, ({
        auto scan_params = scanKernelParams<it_value_t<InputIteratorT>, it_value_t<OutputIteratorT>, AccumT>{
          d_in, d_out, tile_state.lookahead.tile_states, tile_state.lookahead.atomic_counter, num_items, num_stages};
        device_scan_lookahead_body<PolicySelector, ForceInclusive, RealInitValueT, StableReductionOrder>(
          scan_params, scan_op, init_value);
      }));
#else
    static_assert(sizeof(d_in) == 0,
                  "Implementation bug: Tuning policy selected lookahead, but CUDA compiler does not support it");
#endif // _CCCL_CUDACC_AT_LEAST(12, 8)
  }
  else
  {
    static constexpr ScanLookbackPolicy policy = active_policy.lookback;
    static_assert(policy.load_modifier != CacheLoadModifier::LOAD_LDG,
                  "The memory consistency model does not apply to texture accesses");
    using ScanPolicyT = agent_scan_policy<
      0,
      0,
      void,
      policy.load_algorithm,
      policy.load_modifier,
      policy.store_algorithm,
      policy.scan_algorithm,
      NoScaling<policy.threads_per_block, policy.items_per_thread>,
      delay_constructor_t<policy.lookback_delay.kind, policy.lookback_delay.delay, policy.lookback_delay.l2_write_latency>>;

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
      /* UsePDL */ true,
      StableReductionOrder>;

    // Shared memory for AgentScan
    __shared__ typename AgentScanT::TempStorage temp_storage;

    // Depending on the version of the PTX memory model the compiler and hardware implement, we could move the grid
    // dependency sync back to the first read of data, that was actually written by the previous kernel (the tile
    // state). So the BlockLoad in this kernel could even overlap with the previous tile init kernel. To be save, we
    // retain it here before the first read.
    _CCCL_PDL_GRID_DEPENDENCY_SYNC();
    RealInitValueT real_init_value = init_value;

    if constexpr (!::cuda::std::is_same_v<KernelNumItemsT, OffsetT>)
    {
      constexpr auto tile_items = static_cast<OffsetT>(policy.threads_per_block * policy.items_per_thread);
      if (static_cast<OffsetT>(start_tile) + blockIdx.x >= ::cuda::ceil_div(num_items, tile_items))
      {
        return;
      }
    }

    // Process tiles
    AgentScanT(temp_storage, d_in, d_out, scan_op, real_init_value)
      .ConsumeRange(num_items, tile_state.lookback, start_tile);
  }
}

//! Claims the next tile and waits until its batch owns the tile state.
template <typename AccumT>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint64_t claim_batch_tile(
  scan_batch_state<AccumT>* batch_state, ::cuda::std::uint64_t num_tiles, ::cuda::std::uint64_t& current_batch_idx)
{
  const auto tile_idx =
    ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device>{batch_state->next_tile_idx}.fetch_add(
      1, ::cuda::std::memory_order_relaxed);
  if (tile_idx >= num_tiles)
  {
    return tile_idx;
  }

  const auto tile_batch_idx = tile_idx / tiles_per_batch;
  if (tile_batch_idx != current_batch_idx)
  {
    auto batch_idx = ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device>{batch_state->batch_idx};
    while (batch_idx.load(::cuda::std::memory_order_acquire) < tile_batch_idx)
    {
    }
    current_batch_idx = tile_batch_idx;
  }
  return tile_idx;
}

//! Completes one tile and retires the batch when its final look-back finishes.
template <typename ScanTileState, typename AccumT>
struct batch_completion_op
{
  ScanTileState tile_state;
  scan_batch_state<AccumT>* batch_state;
  ::cuda::std::uint64_t tile_idx;
  ::cuda::std::uint64_t num_tiles;
  ::cuda::std::uint32_t batch_tiles;

  [[nodiscard]] _CCCL_DEVICE_API bool operator()(const AccumT& tile_inclusive)
  {
    const auto batch_idx      = tile_idx / tiles_per_batch;
    const auto local_tile_idx = static_cast<::cuda::std::uint32_t>(tile_idx % tiles_per_batch);
    const bool has_next_batch = (batch_idx + 1) * tiles_per_batch < num_tiles;

    // Save the batch prefix before publishing completion.
    if (has_next_batch && local_tile_idx + 1 == batch_tiles)
    {
      batch_state->batched_sum.Alias() = tile_inclusive;
      __threadfence();
    }

    auto lookbacks_completed =
      ::cuda::atomic_ref<::cuda::std::uint32_t, ::cuda::thread_scope_device>{batch_state->lookbacks_completed};
    const auto completed = lookbacks_completed.fetch_add(1, ::cuda::std::memory_order_acq_rel) + 1;
    return has_next_batch && completed == batch_tiles;
  }

  //! Resets the tile state and publishes the next batch.
  _CCCL_DEVICE_API void reset_tiles_and_publish()
  {
    const auto batch_idx = tile_idx / tiles_per_batch;

    for (::cuda::std::uint32_t tile = threadIdx.x; tile < tiles_per_batch; tile += blockDim.x)
    {
      tile_state.SetInvalid(static_cast<int>(tile));
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      ::cuda::atomic_ref<::cuda::std::uint32_t, ::cuda::thread_scope_device>{batch_state->lookbacks_completed}.store(
        0, ::cuda::std::memory_order_relaxed);
      // Publish the reset state and batch prefix to waiting CTAs.
      ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device>{batch_state->batch_idx}.store(
        batch_idx + 1, ::cuda::std::memory_order_release);
    }
  }
};

template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanTileState,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename KernelNumItemsT,
          typename AccumT,
          bool ForceInclusive,
          bool StableReductionOrder = false,
          typename RealInitValueT   = typename InitValueT::value_type>
__launch_bounds__(device_scan_launch_bounds<PolicySelector>, 1) _CCCL_KERNEL_ATTRIBUTES void DeviceScanBatchKernel(
  const InputIteratorT d_in,
  const OutputIteratorT d_out,
  ScanTileState tile_state,
  scan_batch_state<AccumT>* batch_state,
  const ScanOpT scan_op,
  const InitValueT init_value,
  const KernelNumItemsT kernel_num_items)
{
  const OffsetT num_items             = CUB_NS_QUALIFIER::detail::parameter_from_device<OffsetT>(kernel_num_items);
  constexpr ScanLookbackPolicy policy = current_policy<PolicySelector>().lookback;
  using scan_policy_t                 = agent_scan_policy<
                    0,
                    0,
                    void,
                    policy.load_algorithm,
                    policy.load_modifier,
                    policy.store_algorithm,
                    policy.scan_algorithm,
                    NoScaling<policy.threads_per_block, policy.items_per_thread>,
                    delay_constructor_t<policy.lookback_delay.kind, policy.lookback_delay.delay, policy.lookback_delay.l2_write_latency>>;
  // Disable PDL because each persistent CTA consumes multiple tiles.
  using agent_t =
    AgentScan<scan_policy_t,
              InputIteratorT,
              OutputIteratorT,
              ScanOpT,
              RealInitValueT,
              OffsetT,
              AccumT,
              ForceInclusive,
              /* UsePDL */ false,
              StableReductionOrder>;

  // Keep tile indices wide while preserving the user-selected offset type.
  static_assert(::cuda::std::is_unsigned_v<OffsetT>, "the batched scan expects an unsigned offset type");

  __shared__ typename agent_t::TempStorage temp_storage;
  __shared__ ::cuda::std::uint64_t tile_idx;

  ::cuda::std::uint64_t current_batch_idx = 0;

  _CCCL_PDL_GRID_DEPENDENCY_SYNC();
  const auto num_tiles =
    static_cast<::cuda::std::uint64_t>(::cuda::ceil_div(num_items, static_cast<OffsetT>(agent_t::TILE_ITEMS)));
  RealInitValueT real_init_value = init_value;
  agent_t agent{temp_storage, d_in, d_out, scan_op, real_init_value};

  while (true)
  {
    if (threadIdx.x == 0)
    {
      tile_idx = claim_batch_tile(batch_state, num_tiles, current_batch_idx);
    }
    // Separate consecutive uses of the shared tile index.
    __syncthreads();
    if (tile_idx >= num_tiles)
    {
      return;
    }

    const auto batch_idx   = tile_idx / tiles_per_batch;
    const auto batch_begin = batch_idx * tiles_per_batch;
    const auto batch_tiles = static_cast<::cuda::std::uint32_t>(
      (::cuda::std::min) (static_cast<::cuda::std::uint64_t>(tiles_per_batch), num_tiles - batch_begin));
    const auto batch_tile_idx = static_cast<int>(tile_idx - batch_begin);
    const auto tile_offset    = static_cast<OffsetT>(tile_idx) * static_cast<OffsetT>(agent_t::TILE_ITEMS);
    const auto num_remaining  = num_items - tile_offset;
    // Seed each batch with the preceding batches' prefix.
    const AccumT* const preceding_batched_sum = batch_idx == 0 ? nullptr : &batch_state->batched_sum.Alias();
    const batch_completion_op<ScanTileState, AccumT> completion{
      tile_state, batch_state, tile_idx, num_tiles, batch_tiles};

    if (num_remaining < static_cast<OffsetT>(agent_t::TILE_ITEMS))
    {
      agent.template ConsumeBatchTile<true>(
        num_remaining, static_cast<OffsetT>(tile_idx), batch_tile_idx, preceding_batched_sum, tile_state, completion);
    }
    else
    {
      agent.template ConsumeBatchTile<false>(
        num_remaining, static_cast<OffsetT>(tile_idx), batch_tile_idx, preceding_batched_sum, tile_state, completion);
    }
  }
}
} // namespace detail::scan

CUB_NAMESPACE_END
