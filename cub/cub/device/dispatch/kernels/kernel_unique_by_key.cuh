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

#include <cub/agent/agent_unique_by_key.cuh>
#include <cub/detail/arch_dispatch.cuh>
#include <cub/detail/delay_constructor.cuh>
#include <cub/device/dispatch/tuning/tuning_unique_by_key.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_vsmem.cuh>

#include <cuda/__device/compute_capability.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes") // __visibility__ attribute ignored
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

namespace detail::unique_by_key
{
template <typename PolicyGetter>
struct host_policy_provider
{
  static constexpr unique_by_key_policy selected_policy = PolicyGetter{}();

  struct fallback_pol_getter
  {
    _CCCL_API _CCCL_FORCEINLINE constexpr auto operator()() const
    {
      unique_by_key_policy policy = PolicyGetter{}();
      policy.block_threads        = 64;
      policy.items_per_thread     = 1;
      return policy;
    }
  };

  static constexpr unique_by_key_policy fallback_policy = fallback_pol_getter{}();
};

template <typename PolicyGetter>
struct device_policy_provider
{
  static constexpr unique_by_key_policy selected_policy = PolicyGetter{}();

  struct fallback_pol_getter
  {
    _CCCL_DEVICE_API _CCCL_FORCEINLINE constexpr auto operator()() const
    {
      unique_by_key_policy policy = PolicyGetter{}();
      policy.block_threads        = 64;
      policy.items_per_thread     = 1;
      return policy;
    }
  };

  static constexpr unique_by_key_policy fallback_policy = fallback_pol_getter{}();
};

template <typename PolicyProvider,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueOutputIteratorT,
          typename EqualityOpT,
          typename OffsetT>
class unique_by_key_vsmem_helper_impl
{
  static constexpr unique_by_key_policy selected_policy = PolicyProvider::selected_policy;

  using selected_policy_t = AgentUniqueByKeyPolicy<
    selected_policy.block_threads,
    selected_policy.items_per_thread,
    selected_policy.load_algorithm,
    selected_policy.load_modifier,
    selected_policy.scan_algorithm,
    delay_constructor_t<selected_policy.delay_constructor.kind,
                        selected_policy.delay_constructor.delay,
                        selected_policy.delay_constructor.l2_write_latency>>;

  static constexpr unique_by_key_policy fallback_policy = PolicyProvider::fallback_policy;

  using fallback_policy_t = AgentUniqueByKeyPolicy<
    fallback_policy.block_threads,
    fallback_policy.items_per_thread,
    fallback_policy.load_algorithm,
    fallback_policy.load_modifier,
    fallback_policy.scan_algorithm,
    delay_constructor_t<fallback_policy.delay_constructor.kind,
                        fallback_policy.delay_constructor.delay,
                        fallback_policy.delay_constructor.l2_write_latency>>;

  using default_agent_t =
    AgentUniqueByKey<selected_policy_t,
                     KeyInputIteratorT,
                     ValueInputIteratorT,
                     KeyOutputIteratorT,
                     ValueOutputIteratorT,
                     EqualityOpT,
                     OffsetT>;
  using fallback_agent_t =
    AgentUniqueByKey<fallback_policy_t,
                     KeyInputIteratorT,
                     ValueInputIteratorT,
                     KeyOutputIteratorT,
                     ValueOutputIteratorT,
                     EqualityOpT,
                     OffsetT>;

  static constexpr ::cuda::std::size_t max_default_size  = sizeof(typename default_agent_t::TempStorage);
  static constexpr ::cuda::std::size_t max_fallback_size = sizeof(typename fallback_agent_t::TempStorage);
  static constexpr bool uses_fallback_policy =
    (max_default_size > max_smem_per_block) && (max_fallback_size <= max_smem_per_block);

public:
  static constexpr unique_by_key_policy policy    = uses_fallback_policy ? fallback_policy : selected_policy;
  static constexpr bool selected_policy_fits_smem = max_default_size <= max_smem_per_block;

  using selected_agent_t = default_agent_t;
  using agent_t          = ::cuda::std::_If<uses_fallback_policy, fallback_agent_t, default_agent_t>;
};

template <typename PolicyGetter,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueOutputIteratorT,
          typename EqualityOpT,
          typename OffsetT>
using unique_by_key_vsmem_helper_t = unique_by_key_vsmem_helper_impl<
  host_policy_provider<PolicyGetter>,
  KeyInputIteratorT,
  ValueInputIteratorT,
  KeyOutputIteratorT,
  ValueOutputIteratorT,
  EqualityOpT,
  OffsetT>;

template <typename PolicyGetter,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueOutputIteratorT,
          typename EqualityOpT,
          typename OffsetT>
using device_unique_by_key_vsmem_helper_t = unique_by_key_vsmem_helper_impl<
  device_policy_provider<PolicyGetter>,
  KeyInputIteratorT,
  ValueInputIteratorT,
  KeyOutputIteratorT,
  ValueOutputIteratorT,
  EqualityOpT,
  OffsetT>;

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
template <typename PolicySelector,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueOutputIteratorT,
          typename NumSelectedIteratorT,
          typename ScanTileStateT,
          typename EqualityOpT,
          typename OffsetT>
__launch_bounds__(
  device_unique_by_key_vsmem_helper_t<
    device_policy_getter<PolicySelector, current_tuning_cc().get()>,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyOutputIteratorT,
    ValueOutputIteratorT,
    EqualityOpT,
    OffsetT>::policy.block_threads)
  _CCCL_KERNEL_ATTRIBUTES void DeviceUniqueByKeySweepKernel(
    _CCCL_GRID_CONSTANT const KeyInputIteratorT d_keys_in,
    _CCCL_GRID_CONSTANT const ValueInputIteratorT d_values_in,
    _CCCL_GRID_CONSTANT const KeyOutputIteratorT d_keys_out,
    _CCCL_GRID_CONSTANT const ValueOutputIteratorT d_values_out,
    _CCCL_GRID_CONSTANT const NumSelectedIteratorT d_num_selected_out,
    ScanTileStateT tile_state,
    EqualityOpT equality_op,
    _CCCL_GRID_CONSTANT const OffsetT num_items,
    _CCCL_GRID_CONSTANT const int num_tiles,
    vsmem_t vsmem)
{
  using vsmem_adapted_agents = device_unique_by_key_vsmem_helper_t<
    device_policy_getter<PolicySelector, current_tuning_cc().get()>,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyOutputIteratorT,
    ValueOutputIteratorT,
    EqualityOpT,
    OffsetT>;
  using agent_unique_by_key_t = typename vsmem_adapted_agents::agent_t;
  using vsmem_helper_t        = vsmem_helper_impl<agent_unique_by_key_t>;

  // Static shared memory allocation
  __shared__ typename vsmem_helper_t::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename agent_unique_by_key_t::TempStorage& temp_storage =
    vsmem_helper_t::get_temp_storage(static_temp_storage, vsmem, (blockIdx.x * gridDim.y) + blockIdx.y);

  // Process tiles
  agent_unique_by_key_t(temp_storage, d_keys_in, d_values_in, d_keys_out, d_values_out, equality_op, num_items)
    .ConsumeRange(num_tiles, tile_state, d_num_selected_out);

  // If applicable, hints to discard modified cache lines for vsmem
  vsmem_helper_t::discard_temp_storage(temp_storage);
}
} // namespace detail::unique_by_key

CUB_NAMESPACE_END

_CCCL_DIAG_POP
