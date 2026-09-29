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

#include <cub/agent/agent_set_operations.cuh>
#include <cub/detail/cc_dispatch.cuh>
#include <cub/device/dispatch/kernels/kernel_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_set_operations.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>
#include <cub/util_vsmem.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/cmath>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/pair.h>

CUB_NAMESPACE_BEGIN
namespace detail::set_ops
{
inline constexpr int init_kernel_threads      = 128;
inline constexpr int partition_kernel_threads = 256;

// Computes the duplicate-aware merge-path partition boundaries at every tile-sized diagonal. One thread per diagonal.
template <typename KeysIt1, typename KeysIt2, typename Offset, typename CompareOp>
_CCCL_KERNEL_ATTRIBUTES void device_set_op_partition_kernel(
  KeysIt1 keys1,
  KeysIt2 keys2,
  Offset num_keys1,
  Offset num_keys2,
  Offset num_partitions,
  ::cuda::std::pair<Offset, Offset>* partitions,
  CompareOp compare_op,
  int items_per_tile)
{
  const Offset partition_idx = static_cast<Offset>(blockDim.x) * blockIdx.x + threadIdx.x;
  if (partition_idx < num_partitions)
  {
    const Offset diag         = (::cuda::std::min) (partition_idx * items_per_tile, num_keys1 + num_keys2);
    partitions[partition_idx] = balanced_path(keys1, keys2, num_keys1, num_keys2, diag, 4, compare_op);
  }
}

template <typename PolicySelector,
          typename KeysIt1,
          typename KeysIt2,
          typename ValuesIt1,
          typename ValuesIt2,
          typename KeysOutputIt,
          typename ValuesOutputIt,
          typename Offset,
          typename CompareOp,
          typename SetOp,
          typename NumSelectedIteratorT>
__launch_bounds__(device_policy_getter<PolicySelector, current_tuning_cc().get()>{}().threads_per_block)
  _CCCL_KERNEL_ATTRIBUTES void device_set_op_sweep_kernel(
    KeysIt1 keys1,
    KeysIt2 keys2,
    ValuesIt1 values1,
    ValuesIt2 values2,
    Offset num_keys1,
    Offset num_keys2,
    KeysOutputIt keys_out,
    ValuesOutputIt values_out,
    CompareOp compare_op,
    SetOp set_op,
    const ::cuda::std::pair<Offset, Offset>* partitions,
    NumSelectedIteratorT d_num_selected_out,
    ScanTileState<Offset> tile_state,
    vsmem_t global_temp_storage)
{
  using agent_t =
    agent_set_op<device_policy_getter<PolicySelector, current_tuning_cc().get()>,
                 KeysIt1,
                 KeysIt2,
                 ValuesIt1,
                 ValuesIt2,
                 KeysOutputIt,
                 ValuesOutputIt,
                 Offset,
                 CompareOp,
                 SetOp,
                 NumSelectedIteratorT>;

  using vsmem_helper_t = vsmem_helper_impl<agent_t>;
  __shared__ typename vsmem_helper_t::static_temp_storage_t static_temp_storage;
  auto& storage = vsmem_helper_t::get_temp_storage(static_temp_storage, global_temp_storage);

  agent_t agent{
    storage,
    tile_state,
    keys1,
    keys2,
    values1,
    values2,
    keys_out,
    values_out,
    compare_op,
    set_op,
    partitions,
    d_num_selected_out};
  agent();

  vsmem_helper_t::discard_temp_storage(storage);
}

template <typename KeysIt1,
          typename KeysIt2,
          typename ValuesIt1,
          typename ValuesIt2,
          typename KeysOutputIt,
          typename ValuesOutputIt,
          typename Offset,
          typename CompareOp,
          typename SetOp,
          typename NumSelectedIteratorT,
          typename TuningEnvT            = ::cuda::std::execution::env<>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
[[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeysIt1 keys1,
  KeysIt2 keys2,
  ValuesIt1 values1,
  ValuesIt2 values2,
  Offset num_keys1,
  Offset num_keys2,
  KeysOutputIt keys_out,
  ValuesOutputIt values_out,
  CompareOp compare_op,
  SetOp set_op,
  NumSelectedIteratorT d_num_selected_out,
  cudaStream_t stream,
  const TuningEnvT&                      = {},
  KernelLauncherFactory launcher_factory = {})
{
  using scan_tile_state_t = ScanTileState<Offset>;

  // Resolve the tuning policy selector from the (optional) tuning environment, defaulting to the type-derived selector.
  using default_policy_selector = policy_selector_from_types<KeysIt1, ValuesIt1, KeysIt2, ValuesIt2, Offset>;
  using policy_selector_t =
    ::cuda::std::decay_t<::cuda::std::execution::__query_result_or_t<TuningEnvT, SetOpsPolicy, default_policy_selector>>;
#if _CCCL_HAS_CONCEPTS()
  static_assert(set_ops_policy_selector<policy_selector_t>, "invalid policy selector for set-ops dispatch");
#endif // _CCCL_HAS_CONCEPTS()

  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  return dispatch_compute_cap(policy_selector_t{}, cc, [&](auto policy_getter) -> cudaError_t {
    using agent_t =
      agent_set_op<decltype(policy_getter),
                   KeysIt1,
                   KeysIt2,
                   ValuesIt1,
                   ValuesIt2,
                   KeysOutputIt,
                   ValuesOutputIt,
                   Offset,
                   CompareOp,
                   SetOp,
                   NumSelectedIteratorT>;
    constexpr auto policy        = decltype(policy_getter){}();
    constexpr int block_threads  = policy.threads_per_block;
    constexpr int items_per_tile = block_threads * policy.items_per_thread - 1;

    const Offset keys_total = num_keys1 + num_keys2;
    const Offset num_tiles  = ::cuda::ceil_div(keys_total, Offset{items_per_tile});

    // Temp storage layout: [0] look-back tile state, [1] merge-path partitions (one per tile + sentinel), [2] virtual
    // shared memory (only when the agent's temp storage exceeds the static shared-memory limit).
    size_t tile_state_bytes = 0;
    if (const auto error = CubDebug(scan_tile_state_t::AllocationSize(static_cast<int>(num_tiles), tile_state_bytes)))
    {
      return error;
    }
    const size_t partitions_bytes    = static_cast<size_t>(num_tiles + 1) * sizeof(::cuda::std::pair<Offset, Offset>);
    const size_t vsmem_bytes         = static_cast<size_t>(num_tiles) * vsmem_helper_impl<agent_t>::vsmem_per_block;
    void* allocations[3]             = {nullptr, nullptr, nullptr};
    const size_t allocation_sizes[3] = {tile_state_bytes, partitions_bytes, vsmem_bytes};
    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      return cudaSuccess; // query phase: temp_storage_bytes is now populated
    }

    scan_tile_state_t tile_state;
    if (const auto error = CubDebug(tile_state.Init(static_cast<int>(num_tiles), allocations[0], allocation_sizes[0])))
    {
      return error;
    }
    auto partitions = static_cast<::cuda::std::pair<Offset, Offset>*>(allocations[1]);

    // Initialize the tile state and zero the output count.
    {
      const int init_grid_size =
        (::cuda::std::max) (1, static_cast<int>(::cuda::ceil_div(num_tiles, Offset{init_kernel_threads})));
      if (const auto error = CubDebug(
            THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(init_grid_size, init_kernel_threads, 0, stream)
              .doit(detail::scan::DeviceCompactInitKernel<scan_tile_state_t, NumSelectedIteratorT>,
                    tile_state,
                    static_cast<int>(num_tiles),
                    d_num_selected_out)))
      {
        return error;
      }
      if (const auto error = CubDebug(DebugSyncStream(stream)))
      {
        return error;
      }
    }

    if (num_tiles == 0)
    {
      return cudaSuccess; // no inputs: count already zeroed
    }

    // Compute the merge-path partitions for every tile boundary.
    {
      const Offset num_partitions = num_tiles + 1;
      const int partition_grid_size =
        static_cast<int>(::cuda::ceil_div(num_partitions, Offset{partition_kernel_threads}));
      if (const auto error = CubDebug(
            THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
              partition_grid_size, partition_kernel_threads, 0, stream)
              .doit(device_set_op_partition_kernel<KeysIt1, KeysIt2, Offset, CompareOp>,
                    keys1,
                    keys2,
                    num_keys1,
                    num_keys2,
                    num_partitions,
                    partitions,
                    compare_op,
                    items_per_tile)))
      {
        return error;
      }
      if (const auto error = CubDebug(DebugSyncStream(stream)))
      {
        return error;
      }
    }

    // Main sweep: one block per tile.
    {
      auto sweep_kernel = device_set_op_sweep_kernel<
        policy_selector_t,
        KeysIt1,
        KeysIt2,
        ValuesIt1,
        ValuesIt2,
        KeysOutputIt,
        ValuesOutputIt,
        Offset,
        CompareOp,
        SetOp,
        NumSelectedIteratorT>;
      if (const auto error = CubDebug(
            THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(static_cast<int>(num_tiles), block_threads, 0, stream)
              .doit(sweep_kernel,
                    keys1,
                    keys2,
                    values1,
                    values2,
                    num_keys1,
                    num_keys2,
                    keys_out,
                    values_out,
                    compare_op,
                    set_op,
                    partitions,
                    d_num_selected_out,
                    tile_state,
                    vsmem_t{allocations[2]})))
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
} // namespace detail::set_ops
CUB_NAMESPACE_END
