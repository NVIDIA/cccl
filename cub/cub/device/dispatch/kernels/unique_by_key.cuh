// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#include "cub/util_namespace.cuh"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_unique_by_key.cuh>
#include <cub/util_vsmem.cuh>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

namespace detail::unique_by_key
{
/**
 * @brief Unique by key kernel entry point (multi-block)
 *
 * @tparam KeyInputIteratorT
 *   Random-access input iterator type for keys
 *
 * @tparam ValueInputIteratorT
 *   Random-access input iterator type for values
 *
 * @tparam KeyOutputIteratorT
 *   Random-access output iterator type for keys
 *
 * @tparam ValueOutputIteratorT
 *   Random-access output iterator type for values
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam EqualityOpT
 *   Equality operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param[in] d_keys_in
 *   Pointer to the input sequence of keys
 *
 * @param[in] d_values_in
 *   Pointer to the input sequence of values
 *
 * @param[out] d_keys_out
 *   Pointer to the output sequence of selected data items
 *
 * @param[out] d_values_out
 *   Pointer to the output sequence of selected data items
 *
 * @param[out] d_num_selected_out
 *   Pointer to the total number of items selected
 *   (i.e., length of @p d_keys_out or @p d_values_out)
 *
 * @param[in] tile_state
 *   Tile status interface
 *
 * @param[in] equality_op
 *   Equality operator
 *
 * @param[in] num_items
 *   Total number of input items
 *   (i.e., length of @p d_keys_in or @p d_values_in)
 *
 * @param[in] num_tiles
 *   Total number of tiles for the entire problem
 *
 * @param[in] vsmem
 *   Memory to support virtual shared memory
 */
template <typename ChainedPolicyT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueOutputIteratorT,
          typename NumSelectedIteratorT,
          typename ScanTileStateT,
          typename EqualityOpT,
          typename OffsetT>
__launch_bounds__(int(
  vsmem_helper_default_fallback_policy_t<
    typename ChainedPolicyT::ActivePolicy::UniqueByKeyPolicyT,
    AgentUniqueByKey,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyOutputIteratorT,
    ValueOutputIteratorT,
    EqualityOpT,
    OffsetT>::agent_policy_t::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceUniqueByKeySweepKernel(
    KeyInputIteratorT d_keys_in,
    ValueInputIteratorT d_values_in,
    KeyOutputIteratorT d_keys_out,
    ValueOutputIteratorT d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    ScanTileStateT tile_state,
    EqualityOpT equality_op,
    OffsetT num_items,
    int num_tiles,
    vsmem_t vsmem)
{
  using VsmemHelperT = vsmem_helper_default_fallback_policy_t<
    typename ChainedPolicyT::ActivePolicy::UniqueByKeyPolicyT,
    AgentUniqueByKey,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyOutputIteratorT,
    ValueOutputIteratorT,
    EqualityOpT,
    OffsetT>;

  using AgentUniqueByKeyPolicyT = typename VsmemHelperT::agent_policy_t;

  // Thread block type for selecting data from input tiles
  using AgentUniqueByKeyT = typename VsmemHelperT::agent_t;

  // Static shared memory allocation
  __shared__ typename VsmemHelperT::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename AgentUniqueByKeyT::TempStorage& temp_storage =
    VsmemHelperT::get_temp_storage(static_temp_storage, vsmem, (blockIdx.x * gridDim.y) + blockIdx.y);

  // Process tiles
  AgentUniqueByKeyT(temp_storage, d_keys_in, d_values_in, d_keys_out, d_values_out, equality_op, num_items)
    .ConsumeRange(num_tiles, tile_state, d_num_selected_out);

  // If applicable, hints to discard modified cache lines for vsmem
  VsmemHelperT::discard_temp_storage(temp_storage);
}
} // namespace detail::unique_by_key

CUB_NAMESPACE_END
