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

#include <cub/agent/agent_merge_sort.cuh>
#include <cub/device/dispatch/kernels/kernel_merge_sort.cuh>
#include <cub/device/dispatch/tuning/tuning_merge_sort.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_vsmem.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/ilog.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::merge_sort
{
template <typename MaxPolicyT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT>
struct DeviceMergeSortKernelSource
{
  using KeyT   = cub::detail::it_value_t<KeyIteratorT>;
  using ValueT = cub::detail::it_value_t<ValueIteratorT>;

  CUB_DEFINE_KERNEL_GETTER(
    MergeSortBlockSortKernel,
    DeviceMergeSortBlockSortKernel<
      MaxPolicyT,
      KeyInputIteratorT,
      ValueInputIteratorT,
      KeyIteratorT,
      ValueIteratorT,
      OffsetT,
      CompareOpT,
      KeyT,
      ValueT>);

  CUB_DEFINE_KERNEL_GETTER(MergeSortPartitionKernel,
                           DeviceMergeSortPartitionKernel<KeyIteratorT, OffsetT, CompareOpT, KeyT>);

  CUB_DEFINE_KERNEL_GETTER(
    MergeSortMergeKernel,
    DeviceMergeSortMergeKernel<MaxPolicyT,
                               KeyInputIteratorT,
                               ValueInputIteratorT,
                               KeyIteratorT,
                               ValueIteratorT,
                               OffsetT,
                               CompareOpT,
                               KeyT,
                               ValueT>);

  CUB_RUNTIME_FUNCTION static constexpr size_t KeySize()
  {
    return sizeof(KeyT);
  }

  CUB_RUNTIME_FUNCTION static constexpr size_t ValueSize()
  {
    return sizeof(ValueT);
  }
};
} // namespace detail::merge_sort

/*******************************************************************************
 * Policy
 ******************************************************************************/

template <typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename PolicyHub    = detail::merge_sort::policy_hub<KeyIteratorT>,
          typename KernelSource = detail::merge_sort::DeviceMergeSortKernelSource<
            typename PolicyHub::MaxPolicy,
            KeyInputIteratorT,
            ValueInputIteratorT,
            KeyIteratorT,
            ValueIteratorT,
            OffsetT,
            CompareOpT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY,
          typename VSMemHelperT          = detail::merge_sort::VSMemHelper,
          typename KeyT                  = cub::detail::it_value_t<KeyIteratorT>,
          typename ValueT                = cub::detail::it_value_t<ValueIteratorT>>
struct DispatchMergeSort
{
  /// Whether or not there are values to be trucked along with keys
  static constexpr bool KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  // Problem state

  /// Device-accessible allocation of temporary storage. When nullptr, the required
  /// allocation size is written to \p temp_storage_bytes and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of \p d_temp_storage allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of unsorted input keys
  KeyInputIteratorT d_input_keys;

  /// Pointer to the input sequence of unsorted input values
  ValueInputIteratorT d_input_items;

  /// Pointer to the output sequence of sorted input keys
  KeyIteratorT d_output_keys;

  /// Pointer to the output sequence of sorted input values
  ValueIteratorT d_output_items;

  /// Number of items to sort
  OffsetT num_items;

  /// Comparison function object which returns true if the first argument is
  /// ordered before the second
  CompareOpT compare_op;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  // Constructor
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchMergeSort(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_input_keys,
    ValueInputIteratorT d_input_items,
    KeyIteratorT d_output_keys,
    ValueIteratorT d_output_items,
    OffsetT num_items,
    CompareOpT compare_op,
    cudaStream_t stream,
    int ptx_version,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_input_keys(d_input_keys)
      , d_input_items(d_input_items)
      , d_output_keys(d_output_keys)
      , d_output_items(d_output_items)
      , num_items(num_items)
      , compare_op(compare_op)
      , stream(stream)
      , ptx_version(ptx_version)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  // Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT policy = {})
  {
    if (num_items == 0)
    {
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 0;
      }
      return cudaSuccess;
    }

    auto wrapped_policy  = detail::merge_sort::MakeMergeSortPolicyWrapper(policy);
    const auto tile_size = VSMemHelperT::template ItemsPerTile<
      typename ActivePolicyT::MergeSortPolicy,
      KeyInputIteratorT,
      ValueInputIteratorT,
      KeyIteratorT,
      ValueIteratorT,
      OffsetT,
      CompareOpT,
      KeyT,
      ValueT>(wrapped_policy.MergeSort());
    const auto num_tiles = ::cuda::ceil_div(num_items, tile_size);

    const auto merge_partitions_size         = static_cast<size_t>(1 + num_tiles) * sizeof(OffsetT);
    const auto temporary_keys_storage_size   = static_cast<size_t>(num_items * kernel_source.KeySize());
    const auto temporary_values_storage_size = static_cast<size_t>(num_items * kernel_source.ValueSize()) * !KEYS_ONLY;

    /**
     * Merge sort supports large types, which can lead to excessive shared memory size requirements. In these cases,
     * merge sort allocates virtual shared memory that resides in global memory.
     */
    const ::cuda::std::size_t block_sort_smem_size =
      num_tiles
      * VSMemHelperT::template BlockSortVSMemPerBlock<
        typename ActivePolicyT::MergeSortPolicy,
        KeyInputIteratorT,
        ValueInputIteratorT,
        KeyIteratorT,
        ValueIteratorT,
        OffsetT,
        CompareOpT,
        KeyT,
        ValueT>(wrapped_policy.MergeSort());
    const ::cuda::std::size_t merge_smem_size =
      num_tiles
      * VSMemHelperT::template MergeVSMemPerBlock<
        typename ActivePolicyT::MergeSortPolicy,
        KeyInputIteratorT,
        ValueInputIteratorT,
        KeyIteratorT,
        ValueIteratorT,
        OffsetT,
        CompareOpT,
        KeyT,
        ValueT>(wrapped_policy.MergeSort());
    const ::cuda::std::size_t virtual_shared_memory_size = (::cuda::std::max) (block_sort_smem_size, merge_smem_size);

    void* allocations[4]       = {nullptr, nullptr, nullptr, nullptr};
    size_t allocation_sizes[4] = {
      merge_partitions_size, temporary_keys_storage_size, temporary_values_storage_size, virtual_shared_memory_size};

    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      // Return if the caller is simply requesting the size of the storage allocation
      return cudaSuccess;
    }

    const int num_passes = ::cuda::ceil_ilog2(num_tiles);

    /*
     * The algorithm consists of stages. At each stage, there are input and output arrays. There are two pairs of
     * arrays allocated (keys and items). One pair is from function arguments and another from temporary storage. Ping
     * is a helper variable that controls which of these two pairs of arrays is an input and which is an output for a
     * current stage. If the ping is true - the current stage stores its result in the temporary storage. The
     * temporary storage acts as input data otherwise.
     *
     * Block sort is executed before the main loop. It stores its result in  the pair of arrays that will be an input
     * of the next stage. The initial value of the ping variable is selected so that the result of the final stage is
     * stored in the input arrays.
     */
    bool ping = num_passes % 2 == 0;

    auto merge_partitions = static_cast<OffsetT*>(allocations[0]);
    auto keys_buffer      = static_cast<KeyT*>(allocations[1]);
    auto items_buffer     = static_cast<ValueT*>(allocations[2]);

    const int block_threads = VSMemHelperT::template BlockThreads<
      typename ActivePolicyT::MergeSortPolicy,
      KeyInputIteratorT,
      ValueInputIteratorT,
      KeyIteratorT,
      ValueIteratorT,
      OffsetT,
      CompareOpT,
      KeyT,
      ValueT>(wrapped_policy.MergeSort());

    // Invoke DeviceMergeSortBlockSortKernel
    launcher_factory(static_cast<int>(num_tiles), block_threads, 0, stream, true)
      .doit(kernel_source.MergeSortBlockSortKernel(),
            ping,
            d_input_keys,
            d_input_items,
            d_output_keys,
            d_output_items,
            num_items,
            keys_buffer,
            items_buffer,
            compare_op,
            cub::detail::vsmem_t{allocations[3]});

    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    const OffsetT num_partitions              = num_tiles + 1;
    constexpr int threads_per_partition_block = 256;
    const int partition_grid_size = static_cast<int>(::cuda::ceil_div(num_partitions, threads_per_partition_block));

    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    for (int pass = 0; pass < num_passes; ++pass, ping = !ping)
    {
      const OffsetT target_merged_tiles_number = OffsetT(2) << pass;

      // Partition
      launcher_factory(partition_grid_size, threads_per_partition_block, 0, stream, true)
        .doit(kernel_source.MergeSortPartitionKernel(),
              ping,
              d_output_keys,
              keys_buffer,
              num_items,
              num_partitions,
              merge_partitions,
              compare_op,
              target_merged_tiles_number,
              tile_size);

      if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
      {
        return error;
      }

      // Check for failure to launch
      if (const auto error = CubDebug(cudaPeekAtLastError()))
      {
        return error;
      }

      // Merge
      launcher_factory(static_cast<int>(num_tiles), block_threads, 0, stream, true)
        .doit(kernel_source.MergeSortMergeKernel(),
              ping,
              d_output_keys,
              d_output_items,
              num_items,
              keys_buffer,
              items_buffer,
              compare_op,
              merge_partitions,
              target_merged_tiles_number,
              cub::detail::vsmem_t{allocations[3]});

      if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
      {
        return error;
      }

      // Check for failure to launch
      if (const auto error = CubDebug(cudaPeekAtLastError()))
      {
        return error;
      }
    }

    return cudaSuccess;
  }

  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_input_keys,
    ValueInputIteratorT d_input_items,
    KeyIteratorT d_output_keys,
    ValueIteratorT d_output_items,
    OffsetT num_items,
    CompareOpT compare_op,
    cudaStream_t stream,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    // Get PTX version
    int ptx_version = 0;
    if (const auto error = CubDebug(launcher_factory.PtxVersion(ptx_version)))
    {
      return error;
    }

    // Create dispatch functor
    DispatchMergeSort dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_input_keys,
      d_input_items,
      d_output_keys,
      d_output_items,
      num_items,
      compare_op,
      stream,
      ptx_version,
      kernel_source,
      launcher_factory);

    // Dispatch to chained policy
    if (const auto error = CubDebug(max_policy.Invoke(ptx_version, dispatch)))
    {
      return error;
    }

    return cudaSuccess;
  }
};

CUB_NAMESPACE_END
