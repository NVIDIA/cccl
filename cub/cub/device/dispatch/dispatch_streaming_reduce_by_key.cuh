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

#include <cub/agent/agent_reduce_by_key.cuh>
#include <cub/detail/arch_dispatch.cuh>
#include <cub/device/dispatch/dispatch_reduce_by_key.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce_by_key.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/iterator/offset_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#if !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)
#  include <sstream>
#endif // !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)

CUB_NAMESPACE_BEGIN

namespace detail::reduce_by_key
{
template <
  typename KeysInputIteratorT,
  typename UniqueOutputIteratorT,
  typename ValuesInputIteratorT,
  typename AggregatesOutputIteratorT,
  typename NumRunsOutputIteratorT,
  typename EqualityOpT,
  typename ReductionOpT,
  typename OffsetT,
  typename AccumT =
    ::cuda::std::__accumulator_t<ReductionOpT, it_value_t<ValuesInputIteratorT>, it_value_t<ValuesInputIteratorT>>,
  typename PolicySelector =
    policy_selector_from_types<ReductionOpT,
                               AccumT,
                               non_void_value_t<UniqueOutputIteratorT, it_value_t<KeysInputIteratorT>>>>
#if _CCCL_HAS_CONCEPTS()
  requires reduce_by_key_policy_selector<PolicySelector>
#endif
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_streaming_reduce_by_key(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeysInputIteratorT d_keys_in,
  UniqueOutputIteratorT d_unique_out,
  ValuesInputIteratorT d_values_in,
  AggregatesOutputIteratorT d_aggregates_out,
  NumRunsOutputIteratorT d_num_runs_out,
  EqualityOpT equality_op,
  ReductionOpT reduction_op,
  OffsetT num_items,
  cudaStream_t stream,
  PolicySelector policy_selector = {})
{
  ::cuda::arch_id arch_id{};
  if (const auto error = CubDebug(ptx_arch_id(arch_id)))
  {
    return error;
  }

  const reduce_by_key_policy policy = policy_selector(arch_id);

#if !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)
  NV_IF_TARGET(NV_IS_HOST,
               (::std::stringstream ss; ss << policy;
                _CubLog("Dispatching streaming reduce by key to arch %d with tuning: %s\n",
                        static_cast<int>(arch_id),
                        ss.str().c_str());))
#endif

  using local_offset_t  = ::cuda::std::int32_t;
  using global_offset_t = OffsetT;
  static constexpr bool use_streaming_invocation =
    ::cuda::std::numeric_limits<OffsetT>::max() > ::cuda::std::numeric_limits<local_offset_t>::max();
  using streaming_context_t = ::cuda::std::
    conditional_t<use_streaming_invocation, streaming_context<KeysInputIteratorT, AccumT, global_offset_t>, NullType>;
  using ScanTileStateT                                      = ReduceByKeyScanTileState<AccumT, local_offset_t>;
  [[maybe_unused]] static constexpr int init_kernel_threads = 128;

  const int block_threads    = policy.block_threads;
  const int items_per_thread = policy.items_per_thread;
  const auto tile_size       = static_cast<global_offset_t>(block_threads * items_per_thread);

  auto capped_num_items_per_invocation = num_items;
  if constexpr (use_streaming_invocation)
  {
    capped_num_items_per_invocation = static_cast<global_offset_t>(::cuda::std::numeric_limits<local_offset_t>::max());
    capped_num_items_per_invocation -= (capped_num_items_per_invocation % tile_size);
  }

  const auto max_num_items_per_invocation =
    use_streaming_invocation ? ::cuda::std::min(capped_num_items_per_invocation, num_items) : num_items;
  auto const num_partitions =
    (num_items == 0) ? global_offset_t{1} : ::cuda::ceil_div(num_items, capped_num_items_per_invocation);

  const auto max_num_tiles = static_cast<int>(::cuda::ceil_div(max_num_items_per_invocation, tile_size));

  size_t allocation_sizes[3];
  if (const auto error = CubDebug(ScanTileStateT::AllocationSize(max_num_tiles, allocation_sizes[0])))
  {
    return error;
  }
  allocation_sizes[1] = num_partitions > 1 ? sizeof(global_offset_t) * 2 : size_t{0};
  allocation_sizes[2] = num_partitions > 1 ? sizeof(AccumT) * 2 : size_t{0};

  void* allocations[3] = {};
  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  {
    return cudaSuccess;
  }

  for (global_offset_t partition_idx = 0; partition_idx < num_partitions; partition_idx++)
  {
    global_offset_t current_partition_offset = partition_idx * capped_num_items_per_invocation;
    global_offset_t current_num_items =
      (partition_idx + 1 == num_partitions) ? (num_items - current_partition_offset) : capped_num_items_per_invocation;

    const auto num_current_tiles = static_cast<int>(::cuda::ceil_div(current_num_items, tile_size));
    ScanTileStateT tile_state;
    if (const auto error = CubDebug(tile_state.Init(num_current_tiles, allocations[0], allocation_sizes[0])))
    {
      return error;
    }

    const int init_grid_size = ::cuda::std::max(1, ::cuda::ceil_div(num_current_tiles, init_kernel_threads));
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, init_kernel_threads, (long long) stream);
#endif
    if (const auto error = CubDebug(
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(init_grid_size, init_kernel_threads, 0, stream)
            .doit(&detail::scan::DeviceCompactInitKernel<ScanTileStateT, NumRunsOutputIteratorT>,
                  tile_state,
                  num_current_tiles,
                  d_num_runs_out)))
    {
      return error;
    }

    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

    if (num_items == 0)
    {
      return cudaSuccess;
    }

#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking reduce_by_key_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread\n",
            num_current_tiles,
            block_threads,
            (long long) stream,
            items_per_thread);
#endif
    auto reduce_by_key_kernel = DeviceReduceByKeyKernel<
      PolicySelector,
      KeysInputIteratorT,
      UniqueOutputIteratorT,
      ValuesInputIteratorT,
      AggregatesOutputIteratorT,
      NumRunsOutputIteratorT,
      ScanTileStateT,
      EqualityOpT,
      ReductionOpT,
      local_offset_t,
      AccumT,
      streaming_context_t>;

    if constexpr (use_streaming_invocation)
    {
      auto tmp_num_uniques          = static_cast<global_offset_t*>(allocations[1]);
      auto tmp_prefix               = static_cast<AccumT*>(allocations[2]);
      const bool is_first_partition = (partition_idx == 0);
      const bool is_last_partition  = (partition_idx + 1 == num_partitions);
      const int buffer_selector     = partition_idx % 2;
      streaming_context_t streaming_context{
        is_first_partition,
        is_last_partition,
        is_first_partition ? d_keys_in : d_keys_in + current_partition_offset - 1,
        &tmp_prefix[buffer_selector],
        &tmp_prefix[buffer_selector ^ 0x01],
        &tmp_num_uniques[buffer_selector],
        &tmp_num_uniques[buffer_selector ^ 0x01]};
      if (const auto error = CubDebug(
            THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(num_current_tiles, block_threads, 0, stream)
              .doit(reduce_by_key_kernel,
                    d_keys_in + current_partition_offset,
                    d_unique_out,
                    d_values_in + current_partition_offset,
                    d_aggregates_out,
                    d_num_runs_out,
                    tile_state,
                    0,
                    equality_op,
                    reduction_op,
                    static_cast<local_offset_t>(current_num_items),
                    streaming_context,
                    detail::vsmem_t{nullptr})))
      {
        return error;
      }
    }
    else
    {
      if (const auto error = CubDebug(
            THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(num_current_tiles, block_threads, 0, stream)
              .doit(reduce_by_key_kernel,
                    d_keys_in + current_partition_offset,
                    d_unique_out,
                    d_values_in + current_partition_offset,
                    d_aggregates_out,
                    d_num_runs_out,
                    tile_state,
                    0,
                    equality_op,
                    reduction_op,
                    static_cast<local_offset_t>(current_num_items),
                    NullType{},
                    detail::vsmem_t{nullptr})))
      {
        return error;
      }
    }
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }
  }

  return cudaSuccess;
}
} // namespace detail::reduce_by_key

CUB_NAMESPACE_END
