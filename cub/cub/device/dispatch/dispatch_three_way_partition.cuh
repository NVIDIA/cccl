// SPDX-FileCopyrightText: Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/agent/agent_three_way_partition.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_three_way_partition.cuh>
#include <cub/device/dispatch/tuning/tuning_three_way_partition.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__host_stdlib/sstream>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::three_way_partition
{
template <typename PolicySelector,
          typename InputIteratorT,
          typename FirstOutputIteratorT,
          typename SecondOutputIteratorT,
          typename UnselectedOutputIteratorT,
          typename NumSelectedIteratorT,
          typename ScanTileStateT,
          typename SelectFirstPartOp,
          typename SelectSecondPartOp,
          typename per_partition_offset_t,
          typename streaming_context_t,
          typename OffsetT>
struct DeviceThreeWayPartitionKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(ThreeWayPartitionInitKernel,
                           DeviceThreeWayPartitionInitKernel<ScanTileStateT, NumSelectedIteratorT>);

  CUB_DEFINE_KERNEL_GETTER(
    ThreeWayPartitionKernel,
    DeviceThreeWayPartitionKernel<
      PolicySelector,
      InputIteratorT,
      FirstOutputIteratorT,
      SecondOutputIteratorT,
      UnselectedOutputIteratorT,
      NumSelectedIteratorT,
      ScanTileStateT,
      SelectFirstPartOp,
      SelectSecondPartOp,
      per_partition_offset_t,
      streaming_context_t>);
};

// TODO(bgruber): remove in CCCL 4.0
template <typename PolicyHub>
struct policy_selector_from_hub
{
private:
  struct extract_policy_dispatch_t
  {
    three_way_partition_policy& policy;

    template <typename ActivePolicyT>
    _CCCL_API constexpr cudaError_t Invoke()
    {
      using active_policy = typename ActivePolicyT::ThreeWayPartitionPolicy;
      policy              = three_way_partition_policy{
        active_policy::BLOCK_THREADS,
        active_policy::ITEMS_PER_THREAD,
        active_policy::LOAD_ALGORITHM,
        active_policy::LOAD_MODIFIER,
        active_policy::SCAN_ALGORITHM,
        delay_constructor_policy_from_type<typename active_policy::detail::delay_constructor_t>};
      return cudaSuccess;
    }
  };

public:
  // Because a user can also provide a custom three way partition policy hub to DispatchSegmentedSort, which does not
  // go through the Invoke mechanism, we need to support __host__ as well here.
  _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> three_way_partition_policy
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      ({
        const int ptx_version = static_cast<int>(arch) * 10;
        three_way_partition_policy policy{};
        extract_policy_dispatch_t dispatch{policy};
        PolicyHub::MaxPolicy::Invoke(ptx_version, dispatch);
        return policy;
      }),
      ({
        using active_policy = typename PolicyHub::MaxPolicy::ActivePolicy::ThreeWayPartitionPolicy;
        return three_way_partition_policy{
          active_policy::BLOCK_THREADS,
          active_policy::ITEMS_PER_THREAD,
          active_policy::LOAD_ALGORITHM,
          active_policy::LOAD_MODIFIER,
          active_policy::SCAN_ALGORITHM,
          delay_constructor_policy_from_type<typename active_policy::detail::delay_constructor_t>};
      }));
  }
};
} // namespace detail::three_way_partition

/******************************************************************************
 * Dispatch
 ******************************************************************************/

// TODO(bgruber): deprecate when we make the tuning API public and remove in CCCL 4.0
template <
  typename InputIteratorT,
  typename FirstOutputIteratorT,
  typename SecondOutputIteratorT,
  typename UnselectedOutputIteratorT,
  typename NumSelectedIteratorT,
  typename SelectFirstPartOp,
  typename SelectSecondPartOp,
  typename OffsetT,
  typename PolicyHub    = detail::three_way_partition::policy_hub<cub::detail::it_value_t<InputIteratorT>,
                                                                  detail::three_way_partition::per_partition_offset_t>,
  typename KernelSource = detail::three_way_partition::DeviceThreeWayPartitionKernelSource<
    detail::three_way_partition::policy_selector_from_hub<PolicyHub>,
    InputIteratorT,
    FirstOutputIteratorT,
    SecondOutputIteratorT,
    UnselectedOutputIteratorT,
    NumSelectedIteratorT,
    detail::three_way_partition::ScanTileStateT,
    SelectFirstPartOp,
    SelectSecondPartOp,
    detail::three_way_partition::per_partition_offset_t,
    detail::three_way_partition::streaming_context_t<OffsetT>,
    OffsetT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchThreeWayPartitionIf
{
  /*****************************************************************************
   * Types and constants
   ****************************************************************************/

  // Offset type used to instantiate the three-way partition-kernel and agent to index the items within one partition
  using per_partition_offset_t = detail::three_way_partition::per_partition_offset_t;

  // Type used to provide streaming information about each partition's context
  static constexpr per_partition_offset_t partition_size = ::cuda::std::numeric_limits<per_partition_offset_t>::max();

  using streaming_context_t = detail::three_way_partition::streaming_context_t<OffsetT>;

  using ScanTileStateT = detail::three_way_partition::ScanTileStateT;

  static constexpr int INIT_KERNEL_THREADS = 256;

  void* d_temp_storage;
  size_t& temp_storage_bytes;
  InputIteratorT d_in;
  FirstOutputIteratorT d_first_part_out;
  SecondOutputIteratorT d_second_part_out;
  UnselectedOutputIteratorT d_unselected_out;
  NumSelectedIteratorT d_num_selected_out;
  SelectFirstPartOp select_first_part_op;
  SelectSecondPartOp select_second_part_op;
  OffsetT num_items;
  cudaStream_t stream;
  KernelSource kernel_source;
  KernelLauncherFactory launcher_factory;

  /*****************************************************************************
   * Dispatch entrypoints
   ****************************************************************************/

  template <typename ScanInitKernelPtrT, typename SelectIfKernelPtrT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t __invoke(
    int block_threads,
    int items_per_thread,
    ScanInitKernelPtrT three_way_partition_init_kernel,
    SelectIfKernelPtrT three_way_partition_kernel)
  {
    const int tile_size = block_threads * items_per_thread;

    // The maximum number of items for which we will ever invoke the kernel (i.e. largest partition size)
    auto const max_partition_size = static_cast<OffsetT>(
      (::cuda::std::min) (static_cast<uint64_t>(num_items), static_cast<uint64_t>(partition_size)));

    // The number of partitions required to "iterate" over the total input
    auto const num_partitions =
      (max_partition_size == 0) ? OffsetT{1} : ::cuda::ceil_div(num_items, max_partition_size);

    // The maximum number of tiles for which we will ever invoke the kernel
    auto const max_num_tiles_per_invocation = static_cast<OffsetT>(::cuda::ceil_div(max_partition_size, tile_size));

    // For streaming invocations, we need two sets (for double-buffering) of three counters each
    constexpr ::cuda::std::size_t num_counters_per_pass  = 3;
    constexpr ::cuda::std::size_t num_streaming_counters = 2 * num_counters_per_pass;
    ::cuda::std::size_t streaming_selection_storage_bytes =
      (num_partitions > 1) ? num_streaming_counters * sizeof(OffsetT) : ::cuda::std::size_t{0};

    // Specify temporary storage allocation requirements
    size_t allocation_sizes[2] = {0ULL, streaming_selection_storage_bytes};

    if (const auto error =
          CubDebug(ScanTileStateT::AllocationSize(static_cast<int>(max_num_tiles_per_invocation), allocation_sizes[0])))
    {
      return error;
    }

    // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
    void* allocations[2] = {};

    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      // Return if the caller is simply requesting the size of the storage
      // allocation
      return cudaSuccess;
    }

    // Initialize the streaming context with the temporary storage for double-buffering the previously selected items
    // and the total number (across all partitions) of items
    OffsetT* tmp_num_selected_out = static_cast<OffsetT*>(allocations[1]);
    streaming_context_t streaming_context{
      tmp_num_selected_out, (tmp_num_selected_out + num_counters_per_pass), (num_partitions <= 1)};

    // Iterate over the partitions until all input is processed
    for (OffsetT partition_idx = 0; partition_idx < num_partitions; partition_idx++)
    {
      OffsetT current_partition_offset = partition_idx * max_partition_size;
      OffsetT current_num_items =
        (partition_idx + 1 == num_partitions) ? (num_items - current_partition_offset) : max_partition_size;

      // Construct the tile status interface
      const auto current_num_tiles = static_cast<int>(::cuda::ceil_div(current_num_items, tile_size));

      // Construct the tile status interface
      ScanTileStateT tile_status;
      if (const auto error = CubDebug(tile_status.Init(current_num_tiles, allocations[0], allocation_sizes[0])))
      {
        return error;
      }

      // Log three_way_partition_init_kernel configuration
      const int init_grid_size = ::cuda::std::max(1, ::cuda::ceil_div(current_num_tiles, INIT_KERNEL_THREADS));

#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking three_way_partition_init_kernel<<<%d, %d, 0, %lld>>>()\n",
              init_grid_size,
              INIT_KERNEL_THREADS,
              reinterpret_cast<long long>(stream));
#endif // CUB_DEBUG_LOG

      // Invoke three_way_partition_init_kernel to initialize tile descriptors
      if (const auto error = CubDebug(
            launcher_factory(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
              .doit(three_way_partition_init_kernel, tile_status, current_num_tiles, d_num_selected_out)))
      {
        return error;
      }

      // Check for failure to launch
      if (const auto error = CubDebug(cudaPeekAtLastError()))
      {
        return error;
      }

      // Sync the stream if specified to flush runtime errors
      if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
      {
        return error;
      }

      // No more items to process (note, we do not want to return early for num_items==0, because we need to make sure
      // that `three_way_partition_init_kernel` has written '0' to d_num_selected_out)
      if (current_num_items == 0)
      {
        return cudaSuccess;
      }

// Log select_if_kernel configuration
#ifdef CUB_DEBUG_LOG
      {
        // Get SM occupancy for select_if_kernel
        int range_select_sm_occupancy;
        if (const auto error = CubDebug(launcher_factory.MaxSmOccupancy(
              range_select_sm_occupancy, // out
              three_way_partition_kernel,
              block_threads)))
        {
          return error;
        }

        _CubLog("Invoking three_way_partition_kernel<<<%d, %d, 0, %lld>>>(), %d "
                "items per thread, %d SM occupancy\n",
                current_num_tiles,
                block_threads,
                reinterpret_cast<long long>(stream),
                items_per_thread,
                range_select_sm_occupancy);
      }
#endif // CUB_DEBUG_LOG

      // Invoke select_if_kernel
      if (const auto error = CubDebug(
            launcher_factory(current_num_tiles, block_threads, 0, stream)
              .doit(three_way_partition_kernel,
                    d_in,
                    d_first_part_out,
                    d_second_part_out,
                    d_unselected_out,
                    d_num_selected_out,
                    tile_status,
                    select_first_part_op,
                    select_second_part_op,
                    static_cast<per_partition_offset_t>(current_num_items),
                    current_num_tiles,
                    streaming_context)))
      {
        return error;
      }

      // Check for failure to launch
      if (const auto error = CubDebug(cudaPeekAtLastError()))
      {
        return error;
      }

      // Sync the stream if specified to flush runtime errors
      if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
      {
        return error;
      }

      // Prepare streaming context for next partition (swap double buffers, advance number of processed items, etc.)
      streaming_context.advance(current_num_items, (partition_idx + OffsetT{2} == num_partitions));
    }

    return cudaSuccess;
  }

  template <typename ActivePolicyT, typename ScanInitKernelPtrT, typename SelectIfKernelPtrT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  Invoke(ActivePolicyT policy,
         ScanInitKernelPtrT three_way_partition_init_kernel,
         SelectIfKernelPtrT three_way_partition_kernel)
  {
    const int block_threads    = policy.ThreeWayPartition().BlockThreads();
    const int items_per_thread = policy.ThreeWayPartition().ItemsPerThread();
    return __invoke(block_threads, items_per_thread, three_way_partition_init_kernel, three_way_partition_kernel);
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT active_policy = {})
  {
    const auto wrapped_policy = detail::three_way_partition::MakeThreeWayPartitionPolicyWrapper(active_policy);
    return Invoke(wrapped_policy, kernel_source.ThreeWayPartitionInitKernel(), kernel_source.ThreeWayPartitionKernel());
  }

  /**
   * Internal dispatch routine
   */
  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FirstOutputIteratorT d_first_part_out,
    SecondOutputIteratorT d_second_part_out,
    UnselectedOutputIteratorT d_unselected_out,
    NumSelectedIteratorT d_num_selected_out,
    SelectFirstPartOp select_first_part_op,
    SelectSecondPartOp select_second_part_op,
    OffsetT num_items,
    cudaStream_t stream,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    // Get PTX version
    int ptx_version = 0;
    if (cudaError error = CubDebug(launcher_factory.PtxVersion(ptx_version)); cudaSuccess != error)
    {
      return error;
    }

    DispatchThreeWayPartitionIf dispatch{
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_first_part_out,
      d_second_part_out,
      d_unselected_out,
      d_num_selected_out,
      select_first_part_op,
      select_second_part_op,
      num_items,
      stream,
      kernel_source,
      launcher_factory};

    return CubDebug(max_policy.Invoke(ptx_version, dispatch));
  }
};

namespace detail::three_way_partition
{
template <typename InputIteratorT,
          typename FirstOutputIteratorT,
          typename SecondOutputIteratorT,
          typename UnselectedOutputIteratorT,
          typename NumSelectedIteratorT,
          typename SelectFirstPartOp,
          typename SelectSecondPartOp,
          typename OffsetT,
          typename PolicySelector = policy_selector_from_types<it_value_t<InputIteratorT>, per_partition_offset_t>,
          typename KernelSource   = DeviceThreeWayPartitionKernelSource<
              PolicySelector,
              InputIteratorT,
              FirstOutputIteratorT,
              SecondOutputIteratorT,
              UnselectedOutputIteratorT,
              NumSelectedIteratorT,
              ScanTileStateT,
              SelectFirstPartOp,
              SelectSecondPartOp,
              per_partition_offset_t,
              streaming_context_t<OffsetT>,
              OffsetT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
#if _CCCL_HAS_CONCEPTS()
  requires three_way_partition_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  FirstOutputIteratorT d_first_part_out,
  SecondOutputIteratorT d_second_part_out,
  UnselectedOutputIteratorT d_unselected_out,
  NumSelectedIteratorT d_num_selected_out,
  SelectFirstPartOp select_first_part_op,
  SelectSecondPartOp select_second_part_op,
  OffsetT num_items,
  cudaStream_t stream,
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  ::cuda::arch_id arch_id{};
  if (const auto error = CubDebug(launcher_factory.PtxArchId(arch_id)))
  {
    return error;
  }

  const three_way_partition_policy active_policy = policy_selector(arch_id);

#if _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)
  NV_IF_TARGET(
    NV_IS_HOST, ({
      ::std::stringstream ss;
      ss << active_policy;
      _CubLog("Dispatching DeviceThreeWayPartition to arch %d with tuning: %s\n", (int) arch_id, ss.str().c_str());
    }))
#endif // _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)

  struct fake_hub
  {
    using MaxPolicy = void;
  };

  using dispatch_t = DispatchThreeWayPartitionIf<
    InputIteratorT,
    FirstOutputIteratorT,
    SecondOutputIteratorT,
    UnselectedOutputIteratorT,
    NumSelectedIteratorT,
    SelectFirstPartOp,
    SelectSecondPartOp,
    OffsetT,
    fake_hub,
    KernelSource,
    KernelLauncherFactory>;
  auto dispatch = dispatch_t{
    d_temp_storage,
    temp_storage_bytes,
    d_in,
    d_first_part_out,
    d_second_part_out,
    d_unselected_out,
    d_num_selected_out,
    select_first_part_op,
    select_second_part_op,
    num_items,
    stream,
    kernel_source,
    launcher_factory};
  return dispatch.__invoke(
    active_policy.block_threads,
    active_policy.items_per_thread,
    kernel_source.ThreeWayPartitionInitKernel(),
    kernel_source.ThreeWayPartitionKernel());
}
} // namespace detail::three_way_partition

CUB_NAMESPACE_END
