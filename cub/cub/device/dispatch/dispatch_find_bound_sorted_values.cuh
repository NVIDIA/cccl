// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

//! @file
//! Dispatch for cub::DeviceFind::LowerBoundSortedValues / UpperBoundSortedValues (merge-path partition + per-tile
//! search).

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_find_bound_sorted_values.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/detail/arch_dispatch.cuh>
#include <cub/device/dispatch/tuning/tuning_find_bound_sorted_values.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/is_empty.h>

#if !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)
#  include <sstream>
#endif // !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)

CUB_NAMESPACE_BEGIN

namespace detail::find_bound_sorted_values
{
template <typename PolicyGetter>
struct policy_traits
{
  static constexpr auto policy                     = PolicyGetter{}();
  static constexpr int block_threads               = policy.block_threads;
  static constexpr int items_per_thread            = policy.items_per_thread;
  static constexpr int tile_size                   = block_threads * items_per_thread;
  static constexpr CacheLoadModifier load_modifier = policy.load_modifier;
};

// Must be at namespace scope: NVCC host stubs reference this in a context where a lambda-local using is not visible.
template <typename Mode, typename CompareOp>
using partition_comp_op_t = typename Mode::template partition_comp_t<CompareOp>;

// Computes the merge path intersections at equally wide intervals (Odeh et al, IPDPS 2012).
template <typename PolicySelector, typename HaystackIt, typename NeedlesIt, typename Offset, typename PartitionCompOp>
_CCCL_KERNEL_ATTRIBUTES void device_partition_find_bound_sorted_values_kernel(
  _CCCL_GRID_CONSTANT const HaystackIt d_range,
  _CCCL_GRID_CONSTANT const Offset range_count,
  _CCCL_GRID_CONSTANT const NeedlesIt d_values,
  _CCCL_GRID_CONSTANT const Offset values_count,
  _CCCL_GRID_CONSTANT const Offset num_diagonals,
  _CCCL_GRID_CONSTANT Offset* const range_beg_offsets,
  PartitionCompOp partition_comp)
{
  constexpr int tile_size = policy_traits<policy_getter<PolicySelector, CUB_PTX_ARCH / 10>>::tile_size;

  const Offset diagonal_idx = static_cast<Offset>(blockDim.x) * blockIdx.x + threadIdx.x;
  if (diagonal_idx < num_diagonals)
  {
    const Offset diagonal =
      (::cuda::std::min) (diagonal_idx * static_cast<Offset>(tile_size), range_count + values_count);
    range_beg_offsets[diagonal_idx] =
      cub::MergePath(d_range, d_values, range_count, values_count, diagonal, partition_comp);
  }
}

template <typename PolicySelector,
          typename Mode,
          typename HaystackIt,
          typename NeedlesIt,
          typename OutputIt,
          typename Offset,
          typename CompareOp>
__launch_bounds__(policy_traits<policy_getter<PolicySelector, CUB_PTX_ARCH / 10>>::block_threads)
  _CCCL_KERNEL_ATTRIBUTES void device_find_bound_sorted_values_kernel(
    _CCCL_GRID_CONSTANT const HaystackIt d_range,
    _CCCL_GRID_CONSTANT const NeedlesIt d_values,
    _CCCL_GRID_CONSTANT const OutputIt d_output,
    _CCCL_GRID_CONSTANT const Offset range_count,
    _CCCL_GRID_CONSTANT const Offset values_count,
    _CCCL_GRID_CONSTANT Offset* const range_beg_offsets,
    CompareOp comp)
{
  using traits_t = policy_traits<policy_getter<PolicySelector, CUB_PTX_ARCH / 10>>;
  using AgentT =
    agent_t<traits_t::block_threads,
            traits_t::items_per_thread,
            traits_t::load_modifier,
            Mode,
            HaystackIt,
            NeedlesIt,
            OutputIt,
            Offset,
            CompareOp>;

  __shared__ typename AgentT::TempStorage temp_storage;

  AgentT{temp_storage.Alias(), d_range, d_values, d_output, range_count, values_count, range_beg_offsets, comp}();
}

template <typename Mode,
          typename HaystackIt,
          typename NeedlesIt,
          typename OutputIt,
          typename Offset,
          typename CompareOp,
          typename PolicySelector        = policy_selector_from_types<it_value_t<HaystackIt>, it_value_t<NeedlesIt>>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
#if _CCCL_HAS_CONCEPTS()
  requires find_bound_sorted_values_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  HaystackIt d_range,
  Offset range_count,
  NeedlesIt d_values,
  Offset values_count,
  OutputIt d_output,
  CompareOp comp,
  cudaStream_t stream,
  PolicySelector policy_selector         = {},
  KernelLauncherFactory launcher_factory = {})
{
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  return detail::dispatch_compute_cap(policy_selector, cc, [&](auto policy_getter) -> cudaError_t {
#if !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)
    NV_IF_TARGET(
      NV_IS_HOST,
      (::std::stringstream ss; ss << policy_getter(); _CubLog(
         "Dispatching find_bound_sorted_values (merge-path) to arch %d with tuning: %s\n", cc.get(), ss.str().c_str());))
#endif // !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)

    static_assert(::cuda::std::is_empty_v<decltype(policy_getter)>);
    using traits_t = policy_traits<decltype(policy_getter)>;

    const Offset total_items   = range_count + values_count;
    const Offset num_tiles     = ::cuda::ceil_div(total_items, Offset{traits_t::tile_size});
    const Offset num_diagonals = num_tiles + 1;

    void* allocations[1]             = {nullptr};
    const size_t allocation_sizes[1] = {static_cast<size_t>(num_diagonals) * sizeof(Offset)};
    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
    }

    if (d_temp_storage == nullptr || num_tiles == 0)
    {
      return cudaSuccess;
    }

    auto* range_beg_offsets = static_cast<Offset*>(allocations[0]);

    auto partition_comp = Mode::make_partition_comp(comp);

    {
      // Lightweight pass; not worth exposing through the tuning system.
      constexpr int threads_per_partition_block = 256;
      const int partition_grid_size =
        static_cast<int>(::cuda::ceil_div(num_diagonals, Offset{threads_per_partition_block}));

      if (const auto error = CubDebug(
            THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
              partition_grid_size, threads_per_partition_block, 0, stream)
              .doit(device_partition_find_bound_sorted_values_kernel<PolicySelector,
                                                                     HaystackIt,
                                                                     NeedlesIt,
                                                                     Offset,
                                                                     partition_comp_op_t<Mode, CompareOp>>,
                    d_range,
                    range_count,
                    d_values,
                    values_count,
                    num_diagonals,
                    range_beg_offsets,
                    partition_comp)))
      {
        return error;
      }
      if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
      {
        return error;
      }
    }

    {
      if (const auto error = CubDebug(
            THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
              static_cast<int>(num_tiles), traits_t::block_threads, 0, stream)
              .doit(
                device_find_bound_sorted_values_kernel<PolicySelector, Mode, HaystackIt, NeedlesIt, OutputIt, Offset, CompareOp>,
                d_range,
                d_values,
                d_output,
                range_count,
                values_count,
                range_beg_offsets,
                comp)))
      {
        return error;
      }
      if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
      {
        return error;
      }
    }

    return cudaSuccess;
  });
}
} // namespace detail::find_bound_sorted_values

CUB_NAMESPACE_END
