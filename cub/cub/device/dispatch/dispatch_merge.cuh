// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/agent/agent_merge.cuh>
#include <cub/detail/arch_dispatch.cuh>
#include <cub/device/dispatch/tuning/tuning_merge.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>
#include <cub/util_vsmem.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/__algorithm/min.h>

#if !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)
#  include <sstream>
#endif

CUB_NAMESPACE_BEGIN
namespace detail::merge
{
inline constexpr int fallback_BLOCK_THREADS    = 64;
inline constexpr int fallback_ITEMS_PER_THREAD = 1;

// TODO(bgruber): we should choose the merge_policy rather than the agent, but before C++20 this is more verbose
template <typename PolicyGetter, class... Args>
class choose_merge_agent
{
  static constexpr merge_policy active_policy = PolicyGetter{}();

  using default_load2sh_agent_t =
    agent_t<active_policy.block_threads,
            active_policy.items_per_thread,
            active_policy.load_modifier,
            active_policy.store_algorithm,
            active_policy.use_block_load_to_shared,
            Args...>;
  using default_noload2sh_agent_t =
    agent_t<active_policy.block_threads,
            active_policy.items_per_thread,
            active_policy.load_modifier,
            active_policy.store_algorithm,
            /* UseBlockLoadToShared */ false,
            Args...>;

  using fallback_agent_t =
    agent_t<fallback_BLOCK_THREADS,
            fallback_ITEMS_PER_THREAD,
            active_policy.load_modifier,
            active_policy.store_algorithm,
            /* UseBlockLoadToShared */ false,
            Args...>;

  static constexpr bool use_default_load2sh =
    sizeof(typename default_load2sh_agent_t::TempStorage) <= max_smem_per_block;
  // Use fallback if merge agent exceeds maximum shared memory, but the fallback agent still fits, else use
  // vsmem-compatible version, so noload2sh
  static constexpr bool use_fallback = sizeof(typename fallback_agent_t::TempStorage) <= max_smem_per_block;

public:
  using type =
    ::cuda::std::conditional_t<use_default_load2sh,
                               default_load2sh_agent_t,
                               ::cuda::std::conditional_t<use_fallback, fallback_agent_t, default_noload2sh_agent_t>>;
};

// Computes the merge path intersections at equally wide intervals. The approach is outlined in the paper:
// Odeh et al, "Merge Path - Parallel Merging Made Simple" * doi : 10.1109 / IPDPSW .2012.202
// The algorithm is the same as AgentPartition for merge sort, but that agent handles a lot more.
template <typename PolicySelector,
          typename KeyIt1,
          typename ValueIt1,
          typename KeyIt2,
          typename ValueIt2,
          typename KeyIt3,
          typename ValueIt3,
          typename Offset,
          typename CompareOp>
CUB_DETAIL_KERNEL_ATTRIBUTES void device_partition_merge_path_kernel(
  KeyIt1 keys1,
  Offset keys1_count,
  KeyIt2 keys2,
  Offset keys2_count,
  Offset num_diagonals,
  Offset* key1_beg_offsets,
  CompareOp compare_op)
{
  // items_per_tile must be the same of the merge kernel later, so we have to consider whether a fallback agent will be
  // selected for the merge agent that changes the tile size
  constexpr int items_per_tile =
    choose_merge_agent<policy_getter<PolicySelector, ::cuda::arch_id{CUB_PTX_ARCH / 10}>,
                       KeyIt1,
                       ValueIt1,
                       KeyIt2,
                       ValueIt2,
                       KeyIt3,
                       ValueIt3,
                       Offset,
                       CompareOp>::type::items_per_tile;
  const Offset diagonal_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (diagonal_idx < num_diagonals)
  {
    const Offset diagonal_num      = (::cuda::std::min) (diagonal_idx * items_per_tile, keys1_count + keys2_count);
    key1_beg_offsets[diagonal_idx] = cub::MergePath(keys1, keys2, keys1_count, keys2_count, diagonal_num, compare_op);
  }
}

template <typename PolicySelector,
          typename KeyIt1,
          typename ValueIt1,
          typename KeyIt2,
          typename ValueIt2,
          typename KeyIt3,
          typename ValueIt3,
          typename Offset,
          typename CompareOp>
__launch_bounds__(
  choose_merge_agent<policy_getter<PolicySelector, ::cuda::arch_id{CUB_PTX_ARCH / 10}>,
                     KeyIt1,
                     ValueIt1,
                     KeyIt2,
                     ValueIt2,
                     KeyIt3,
                     ValueIt3,
                     Offset,
                     CompareOp>::type::block_threads)
  CUB_DETAIL_KERNEL_ATTRIBUTES void device_merge_kernel(
    KeyIt1 keys1,
    ValueIt1 items1,
    Offset num_keys1,
    KeyIt2 keys2,
    ValueIt2 items2,
    Offset num_keys2,
    KeyIt3 keys_result,
    ValueIt3 items_result,
    CompareOp compare_op,
    Offset* key1_beg_offsets,
    vsmem_t global_temp_storage)
{
  using key_t = it_value_t<KeyIt1>;
  static_assert(::cuda::std::is_invocable_v<CompareOp, key_t, key_t>, "Comparison operator cannot compare two keys");
  static_assert(::cuda::std::is_convertible_v<::cuda::std::invoke_result_t<CompareOp, key_t, key_t>, bool>,
                "Comparison operator must be convertible to bool");

  using MergeAgent = typename choose_merge_agent<
    policy_getter<PolicySelector, ::cuda::arch_id{CUB_PTX_ARCH / 10}>,
    KeyIt1,
    ValueIt1,
    KeyIt2,
    ValueIt2,
    KeyIt3,
    ValueIt3,
    Offset,
    CompareOp>::type;

  using vsmem_helper_t = vsmem_helper_impl<MergeAgent>;
  __shared__ typename vsmem_helper_t::static_temp_storage_t shared_temp_storage;
  auto& temp_storage = vsmem_helper_t::get_temp_storage(shared_temp_storage, global_temp_storage);
  MergeAgent{
    temp_storage.Alias(),
    keys1,
    items1,
    num_keys1,
    keys2,
    items2,
    num_keys2,
    keys_result,
    items_result,
    compare_op,
    key1_beg_offsets}();
  vsmem_helper_t::discard_temp_storage(temp_storage);
}

template <typename KeyIt1,
          typename ValueIt1,
          typename KeyIt2,
          typename ValueIt2,
          typename KeyIt3,
          typename ValueIt3,
          typename Offset,
          typename CompareOp,
          typename PolicySelector        = policy_selector_from_types<it_value_t<KeyIt1>, it_value_t<ValueIt1>, Offset>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
#if _CCCL_HAS_CONCEPTS()
  requires merge_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyIt1 d_keys1,
  ValueIt1 d_values1,
  Offset num_items1,
  KeyIt2 d_keys2,
  ValueIt2 d_values2,
  Offset num_items2,
  KeyIt3 d_keys_out,
  ValueIt3 d_values_out,
  CompareOp compare_op,
  cudaStream_t stream,
  PolicySelector policy_selector         = {},
  KernelLauncherFactory launcher_factory = {})
{
  ::cuda::arch_id arch_id{};
  if (const auto error = CubDebug(launcher_factory.PtxArchId(arch_id)))
  {
    return error;
  }

  return dispatch_arch(policy_selector, arch_id, [&](auto policy_getter) {
#if !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)
    NV_IF_TARGET(
      NV_IS_HOST,
      (std::stringstream ss; ss << policy_getter();
       _CubLog("Dispatching DeviceMerge to arch %d with tuning: %s\n", static_cast<int>(arch_id), ss.str().c_str());))
#endif // !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)

    static_assert(::cuda::std::is_empty_v<decltype(policy_getter)>);
    using AgentT = typename choose_merge_agent<
      decltype(policy_getter),
      KeyIt1,
      ValueIt1,
      KeyIt2,
      ValueIt2,
      KeyIt3,
      ValueIt3,
      Offset,
      CompareOp>::type;

    const auto num_tiles = ::cuda::ceil_div(num_items1 + num_items2, AgentT::items_per_tile);
    void* allocations[2] = {nullptr, nullptr};
    {
      const size_t key1_beg_offsets_size      = (1 + num_tiles) * sizeof(Offset);
      const size_t virtual_shared_memory_size = num_tiles * vsmem_helper_impl<AgentT>::vsmem_per_block;
      const size_t allocation_sizes[2]        = {key1_beg_offsets_size, virtual_shared_memory_size};
      if (const auto error =
            CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
      {
        return error;
      }
    }

    if (d_temp_storage == nullptr || num_tiles == 0)
    {
      return cudaSuccess;
    }

    auto key1_beg_offsets = static_cast<Offset*>(allocations[0]);

    // merge path kernel
    {
      const Offset num_diagonals                = num_tiles + 1;
      constexpr int threads_per_partition_block = 256;
      const int partition_grid_size = static_cast<int>(::cuda::ceil_div(num_diagonals, threads_per_partition_block));

      if (const auto error = CubDebug(
            THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
              partition_grid_size, threads_per_partition_block, 0, stream)
              .doit(device_partition_merge_path_kernel<
                      PolicySelector,
                      KeyIt1,
                      ValueIt1,
                      KeyIt2,
                      ValueIt2,
                      KeyIt3,
                      ValueIt3,
                      Offset,
                      CompareOp>,
                    d_keys1,
                    num_items1,
                    d_keys2,
                    num_items2,
                    num_diagonals,
                    key1_beg_offsets,
                    compare_op)))
      {
        return error;
      }
      if (const auto error = CubDebug(DebugSyncStream(stream)))
      {
        return error;
      }
    }

    // merge kernel
    {
      if (const auto error = CubDebug(
            THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
              static_cast<int>(num_tiles), static_cast<int>(AgentT::block_threads), 0, stream)
              .doit(
                device_merge_kernel<PolicySelector, KeyIt1, ValueIt1, KeyIt2, ValueIt2, KeyIt3, ValueIt3, Offset, CompareOp>,
                d_keys1,
                d_values1,
                num_items1,
                d_keys2,
                d_values2,
                num_items2,
                d_keys_out,
                d_values_out,
                compare_op,
                key1_beg_offsets,
                vsmem_t{allocations[1]})))
      {
        return error;
      }
      if (const auto error = CubDebug(DebugSyncStream(stream)))
      {
        return error;
      }
    }

    return cudaSuccess;
  });
}
} // namespace detail::merge
CUB_NAMESPACE_END
