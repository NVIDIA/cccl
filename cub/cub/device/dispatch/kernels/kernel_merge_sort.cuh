// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_merge_sort.cuh>
#include <cub/detail/arch_dispatch.cuh>
#include <cub/device/dispatch/tuning/tuning_merge_sort.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_vsmem.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::merge_sort
{
//! @brief Helper class template for merge sort-specific virtual shared memory handling. The merge sort algorithm in
//! its current implementation relies on the fact that both the sorting as well as the merging kernels use the same tile
//! size. This circumstance needs to be respected when determining whether the fallback policy for large user types is
//! applicable: we must either use the fallback for both or for none of the two agents.
template <typename DefaultPolicyGetter,
          typename KeyInIt,
          typename ValInIt,
          typename KeyOutIt,
          typename ValOutIt,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
class merge_sort_vsmem_helper_t
{
  struct fallback_pol_getter
  {
    _CCCL_API _CCCL_FORCEINLINE constexpr auto operator()() const
    {
      merge_sort_policy policy = DefaultPolicyGetter{}();
      policy.block_threads     = 64;
      policy.items_per_thread  = 1;
      return policy;
    }
  };

  using default_block_sort_agent_t =
    AgentBlockSort<DefaultPolicyGetter, KeyInIt, ValInIt, KeyOutIt, ValOutIt, OffsetT, CompareOpT, KeyT, ValueT>;
  using fallback_block_sort_agent_t =
    AgentBlockSort<fallback_pol_getter, KeyInIt, ValInIt, KeyOutIt, ValOutIt, OffsetT, CompareOpT, KeyT, ValueT>;

  using default_merge_agent_t  = AgentMerge<DefaultPolicyGetter, KeyOutIt, ValOutIt, OffsetT, CompareOpT, KeyT, ValueT>;
  using fallback_merge_agent_t = AgentMerge<fallback_pol_getter, KeyOutIt, ValOutIt, OffsetT, CompareOpT, KeyT, ValueT>;

  // Use fallback if either (a) the default block sort or (b) the block merge agent exceed the maximum shared memory
  // available per block and both (1) the fallback block sort and (2) the fallback merge agent would not exceed the
  // available shared memory
  static constexpr auto max_default_size =
    (::cuda::std::max) (sizeof(typename default_block_sort_agent_t::TempStorage),
                        sizeof(typename default_merge_agent_t::TempStorage));
  static constexpr auto max_fallback_size =
    (::cuda::std::max) (sizeof(typename fallback_block_sort_agent_t::TempStorage),
                        sizeof(typename fallback_merge_agent_t::TempStorage));
  static constexpr bool uses_fallback_policy =
    (max_default_size > max_smem_per_block) && (max_fallback_size <= max_smem_per_block);

public:
  static constexpr merge_sort_policy policy = uses_fallback_policy ? fallback_pol_getter{}() : DefaultPolicyGetter{}();
  using block_sort_agent_t =
    ::cuda::std::_If<uses_fallback_policy, fallback_block_sort_agent_t, default_block_sort_agent_t>;
  using merge_agent_t = ::cuda::std::_If<uses_fallback_policy, fallback_merge_agent_t, default_merge_agent_t>;
};

template <typename PolicySelectorT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
__launch_bounds__(
  merge_sort_vsmem_helper_t<policy_getter<PolicySelectorT, cuda::arch_id{CUB_PTX_ARCH / 10}>,
                            KeyInputIteratorT,
                            ValueInputIteratorT,
                            KeyIteratorT,
                            ValueIteratorT,
                            OffsetT,
                            CompareOpT,
                            KeyT,
                            ValueT>::policy.block_threads)
  _CCCL_KERNEL_ATTRIBUTES void DeviceMergeSortBlockSortKernel(
    _CCCL_GRID_CONSTANT const bool ping,
    _CCCL_GRID_CONSTANT const KeyInputIteratorT keys_in,
    _CCCL_GRID_CONSTANT const ValueInputIteratorT items_in,
    _CCCL_GRID_CONSTANT const KeyIteratorT keys_out,
    _CCCL_GRID_CONSTANT const ValueIteratorT items_out,
    _CCCL_GRID_CONSTANT const OffsetT keys_count,
    _CCCL_GRID_CONSTANT KeyT* const tmp_keys_out,
    _CCCL_GRID_CONSTANT ValueT* const tmp_items_out,
    CompareOpT compare_op,
    vsmem_t vsmem)
{
  using vsmem_adapted_agents = merge_sort_vsmem_helper_t<
    policy_getter<PolicySelectorT, cuda::arch_id{CUB_PTX_ARCH / 10}>,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>;

  static constexpr merge_sort_policy active_policy = vsmem_adapted_agents::policy;
  using agent_block_sort_t                         = typename vsmem_adapted_agents::block_sort_agent_t;
  using vsmem_helper_t                             = vsmem_helper_impl<agent_block_sort_t>;

  // Static shared memory allocation
  __shared__ typename vsmem_helper_t::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename agent_block_sort_t::TempStorage& temp_storage = vsmem_helper_t::get_temp_storage(static_temp_storage, vsmem);

  agent_block_sort_t agent(
    ping,
    temp_storage,
    try_make_cache_modified_iterator<active_policy.load_modifier>(keys_in),
    try_make_cache_modified_iterator<active_policy.load_modifier>(items_in),
    keys_count,
    keys_out,
    items_out,
    tmp_keys_out,
    tmp_items_out,
    compare_op);

  agent.Process();

  // If applicable, hints to discard modified cache lines for vsmem
  vsmem_helper_t::discard_temp_storage(temp_storage);
}

template <typename KeyIteratorT, typename OffsetT, typename CompareOpT, typename KeyT>
_CCCL_KERNEL_ATTRIBUTES void DeviceMergeSortPartitionKernel(
  _CCCL_GRID_CONSTANT const bool ping,
  _CCCL_GRID_CONSTANT const KeyIteratorT keys_ping,
  _CCCL_GRID_CONSTANT KeyT* const keys_pong,
  _CCCL_GRID_CONSTANT const OffsetT keys_count,
  _CCCL_GRID_CONSTANT const OffsetT num_partitions,
  _CCCL_GRID_CONSTANT OffsetT* const merge_partitions,
  CompareOpT compare_op,
  _CCCL_GRID_CONSTANT const OffsetT target_merged_tiles_number,
  _CCCL_GRID_CONSTANT const int items_per_tile)
{
  const OffsetT partition_idx = static_cast<OffsetT>(blockDim.x * blockIdx.x + threadIdx.x);
  if (partition_idx < num_partitions)
  {
    AgentPartition<KeyIteratorT, OffsetT, CompareOpT, KeyT>{
      ping,
      keys_ping,
      keys_pong,
      keys_count,
      partition_idx,
      merge_partitions,
      compare_op,
      target_merged_tiles_number,
      items_per_tile,
      num_partitions}
      .Process();
  }
}

template <typename PolicySelectorT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
__launch_bounds__(
  merge_sort_vsmem_helper_t<policy_getter<PolicySelectorT, cuda::arch_id{CUB_PTX_ARCH / 10}>,
                            KeyInputIteratorT,
                            ValueInputIteratorT,
                            KeyIteratorT,
                            ValueIteratorT,
                            OffsetT,
                            CompareOpT,
                            KeyT,
                            ValueT>::policy.block_threads)
  _CCCL_KERNEL_ATTRIBUTES void DeviceMergeSortMergeKernel(
    _CCCL_GRID_CONSTANT const bool ping,
    _CCCL_GRID_CONSTANT const KeyIteratorT keys_ping,
    _CCCL_GRID_CONSTANT const ValueIteratorT items_ping,
    _CCCL_GRID_CONSTANT const OffsetT keys_count,
    _CCCL_GRID_CONSTANT KeyT* const keys_pong,
    _CCCL_GRID_CONSTANT ValueT* const items_pong,
    CompareOpT compare_op,
    _CCCL_GRID_CONSTANT OffsetT* const merge_partitions,
    _CCCL_GRID_CONSTANT const OffsetT target_merged_tiles_number,
    vsmem_t vsmem)
{
  using vsmem_adapted_agents = merge_sort_vsmem_helper_t<
    policy_getter<PolicySelectorT, cuda::arch_id{CUB_PTX_ARCH / 10}>,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>;

  static constexpr merge_sort_policy active_policy = vsmem_adapted_agents::policy;
  using agent_merge_t                              = typename vsmem_adapted_agents::merge_agent_t;
  using vsmem_helper_t                             = vsmem_helper_impl<agent_merge_t>;

  // Static shared memory allocation
  __shared__ typename vsmem_helper_t::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename agent_merge_t::TempStorage& temp_storage = vsmem_helper_t::get_temp_storage(static_temp_storage, vsmem);

  agent_merge_t agent(
    ping,
    temp_storage,
    try_make_cache_modified_iterator<active_policy.load_modifier>(keys_ping),
    try_make_cache_modified_iterator<active_policy.load_modifier>(items_ping),
    try_make_cache_modified_iterator<active_policy.load_modifier>(keys_pong),
    try_make_cache_modified_iterator<active_policy.load_modifier>(items_pong),
    keys_count,
    keys_pong,
    items_pong,
    keys_ping,
    items_ping,
    compare_op,
    merge_partitions,
    target_merged_tiles_number);

  agent.Process();

  // If applicable, hints to discard modified cache lines for vsmem
  vsmem_helper_t::discard_temp_storage(temp_storage);
}
} // namespace detail::merge_sort

CUB_NAMESPACE_END
