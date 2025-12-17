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

#include <cub/detail/device_double_buffer.cuh>
#include <cub/detail/temporary_storage.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_sort.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_namespace.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__iterator/reverse_iterator.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort
{
/*
 * Continuation is called after the partitioning stage. It launches kernels
 * to sort large and small segments using the partitioning results. Separation
 * of this stage is required to eliminate device-side synchronization in
 * the CDP mode.
 */
template <typename WrappedPolicyT,
          typename LargeKernelT,
          typename SmallKernelT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN cudaError_t DeviceSegmentedSortContinuation(
  LargeKernelT large_kernel,
  SmallKernelT small_kernel,
  int num_segments,
  KeyT* d_current_keys,
  KeyT* d_final_keys,
  device_double_buffer<KeyT> d_keys_double_buffer,
  ValueT* d_current_values,
  ValueT* d_final_values,
  device_double_buffer<ValueT> d_values_double_buffer,
  BeginOffsetIteratorT d_begin_offsets,
  EndOffsetIteratorT d_end_offsets,
  local_segment_index_t* group_sizes,
  local_segment_index_t* large_and_medium_segments_indices,
  local_segment_index_t* small_segments_indices,
  cudaStream_t stream,
  KernelLauncherFactory launcher_factory,
  WrappedPolicyT wrapped_policy)
{
  using local_segment_index_t                = local_segment_index_t;
  const local_segment_index_t large_segments = group_sizes[0];

  if (large_segments > 0)
  {
    // One CTA per segment
    const local_segment_index_t blocks_in_grid = large_segments;

#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking "
            "DeviceSegmentedSortKernelLarge<<<%d, %d, 0, %lld>>>()\n",
            static_cast<int>(blocks_in_grid),
            wrapped_policy.LargeSegment().BlockThreads(),
            (long long) stream);
#endif // CUB_DEBUG_LOG

    launcher_factory(blocks_in_grid, wrapped_policy.LargeSegment().BlockThreads(), 0, stream)
      .doit(large_kernel,
            large_and_medium_segments_indices,
            d_current_keys,
            d_final_keys,
            d_keys_double_buffer,
            d_current_values,
            d_final_values,
            d_values_double_buffer,
            d_begin_offsets,
            d_end_offsets);

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    if (const auto error = CubDebug(DebugSyncStream(stream)))
    {
      return error;
    }
  }

  const local_segment_index_t small_segments = group_sizes[1];
  const local_segment_index_t medium_segments =
    static_cast<local_segment_index_t>(num_segments) - (large_segments + small_segments);

  const local_segment_index_t small_blocks = ::cuda::ceil_div(small_segments, wrapped_policy.SegmentsPerSmallBlock());

  const local_segment_index_t medium_blocks =
    ::cuda::ceil_div(medium_segments, wrapped_policy.SegmentsPerMediumBlock());

  const local_segment_index_t small_and_medium_blocks_in_grid = small_blocks + medium_blocks;

  if (small_and_medium_blocks_in_grid)
  {
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking "
            "DeviceSegmentedSortKernelSmall<<<%d, %d, 0, %lld>>>()\n",
            static_cast<int>(small_and_medium_blocks_in_grid),
            wrapped_policy.SmallSegment().BlockThreads(),
            (long long) stream);
#endif // CUB_DEBUG_LOG

    launcher_factory(small_and_medium_blocks_in_grid, wrapped_policy.SmallSegment().BlockThreads(), 0, stream)
      .doit(small_kernel,
            small_segments,
            medium_segments,
            medium_blocks,
            small_segments_indices,
            large_and_medium_segments_indices + num_segments - medium_segments,
            d_current_keys,
            d_final_keys,
            d_current_values,
            d_final_values,
            d_begin_offsets,
            d_end_offsets);

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    if (const auto error = CubDebug(DebugSyncStream(stream)))
    {
      return error;
    }
  }

  return cudaSuccess;
}

#ifdef CUB_RDC_ENABLED
/*
 * Continuation kernel is used only in the CDP mode. It's used to
 * launch DeviceSegmentedSortContinuation as a separate kernel.
 */
template <typename WrappedPolicyT,
          typename LargeKernelT,
          typename SmallKernelT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename KernelLauncherFactory>
__launch_bounds__(1) CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentedSortContinuationKernel(
  LargeKernelT large_kernel,
  SmallKernelT small_kernel,
  local_segment_index_t num_segments,
  KeyT* d_current_keys,
  KeyT* d_final_keys,
  device_double_buffer<KeyT> d_keys_double_buffer,
  ValueT* d_current_values,
  ValueT* d_final_values,
  device_double_buffer<ValueT> d_values_double_buffer,
  BeginOffsetIteratorT d_begin_offsets,
  EndOffsetIteratorT d_end_offsets,
  local_segment_index_t* group_sizes,
  local_segment_index_t* large_and_medium_segments_indices,
  local_segment_index_t* small_segments_indices,
  KernelLauncherFactory launcher_factory,
  WrappedPolicyT wrapped_policy)
{
  // In case of CDP:
  // 1. each CTA has a different main stream
  // 2. all streams are non-blocking
  // 3. child grid always completes before the parent grid
  // 4. streams can be used only from the CTA in which they were created
  // 5. streams created on the host cannot be used on the device
  //
  // Due to (4, 5), we can't pass the user-provided stream in the continuation.
  // Due to (1, 2, 3) it's safe to pass the main stream.
  [[maybe_unused]] const auto error = CubDebug(detail::segmented_sort::DeviceSegmentedSortContinuation<WrappedPolicyT>(
    large_kernel,
    small_kernel,
    num_segments,
    d_current_keys,
    d_final_keys,
    d_keys_double_buffer,
    d_current_values,
    d_final_values,
    d_values_double_buffer,
    d_begin_offsets,
    d_end_offsets,
    group_sizes,
    large_and_medium_segments_indices,
    small_segments_indices,
    0, // always launching on the main stream (see motivation above)
    launcher_factory,
    wrapped_policy));
}
#endif // CUB_RDC_ENABLED
template <typename MaxPolicyT,
          SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
struct DeviceSegmentedSortKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(
    SegmentedSortFallbackKernel,
    DeviceSegmentedSortFallbackKernel<Order, MaxPolicyT, KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT, OffsetT>);

  CUB_DEFINE_KERNEL_GETTER(
    SegmentedSortKernelSmall,
    DeviceSegmentedSortKernelSmall<Order, MaxPolicyT, KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT, OffsetT>);

  CUB_DEFINE_KERNEL_GETTER(
    SegmentedSortKernelLarge,
    DeviceSegmentedSortKernelLarge<Order, MaxPolicyT, KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT, OffsetT>);

  CUB_RUNTIME_FUNCTION static constexpr size_t KeySize()
  {
    return sizeof(KeyT);
  }

  using LargeSegmentsSelectorT =
    cub::detail::segmented_sort::LargeSegmentsSelectorT<OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>;
  using SmallSegmentsSelectorT =
    cub::detail::segmented_sort::SmallSegmentsSelectorT<OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>;

  CUB_RUNTIME_FUNCTION static constexpr auto LargeSegmentsSelector(
    OffsetT offset, BeginOffsetIteratorT begin_offset_iterator, EndOffsetIteratorT end_offset_iterator)
  {
    return LargeSegmentsSelectorT(offset, begin_offset_iterator, end_offset_iterator);
  }

  CUB_RUNTIME_FUNCTION static constexpr auto SmallSegmentsSelector(
    OffsetT offset, BeginOffsetIteratorT begin_offset_iterator, EndOffsetIteratorT end_offset_iterator)
  {
    return SmallSegmentsSelectorT(offset, begin_offset_iterator, end_offset_iterator);
  }

  template <typename SelectorT>
  CUB_RUNTIME_FUNCTION static constexpr void
  SetSegmentOffset(SelectorT& selector, global_segment_offset_t base_segment_offset)
  {
    selector.base_segment_offset = base_segment_offset;
  }
};
} // namespace detail::segmented_sort

template <
  SortOrder Order,
  typename KeyT,
  typename ValueT,
  typename OffsetT,
  typename BeginOffsetIteratorT,
  typename EndOffsetIteratorT,
  typename PolicyHub    = detail::segmented_sort::policy_hub<KeyT, ValueT>,
  typename KernelSource = detail::segmented_sort::DeviceSegmentedSortKernelSource<
    typename PolicyHub::MaxPolicy,
    Order,
    KeyT,
    ValueT,
    BeginOffsetIteratorT,
    EndOffsetIteratorT,
    OffsetT>,
  typename PartitionPolicyHub = detail::three_way_partition::policy_hub<
    cub::detail::it_value_t<THRUST_NS_QUALIFIER::counting_iterator<cub::detail::segmented_sort::local_segment_index_t>>,
    detail::three_way_partition::per_partition_offset_t>,
  typename PartitionKernelSource = detail::three_way_partition::DeviceThreeWayPartitionKernelSource<
    typename PartitionPolicyHub::MaxPolicy,
    THRUST_NS_QUALIFIER::counting_iterator<cub::detail::segmented_sort::local_segment_index_t>,
    cub::detail::segmented_sort::local_segment_index_t*,
    cub::detail::segmented_sort::local_segment_index_t*,
    ::cuda::std::reverse_iterator<cub::detail::segmented_sort::local_segment_index_t*>,
    cub::detail::segmented_sort::local_segment_index_t*,
    detail::three_way_partition::ScanTileStateT,
    cub::detail::segmented_sort::LargeSegmentsSelectorT<OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>,
    cub::detail::segmented_sort::SmallSegmentsSelectorT<OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>,
    detail::three_way_partition::per_partition_offset_t,
    detail::three_way_partition::streaming_context_t<cub::detail::segmented_sort::global_segment_offset_t>,
    detail::choose_signed_offset<cub::detail::segmented_sort::global_segment_offset_t>::type>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchSegmentedSort
{
  using local_segment_index_t   = detail::segmented_sort::local_segment_index_t;
  using global_segment_offset_t = detail::segmented_sort::global_segment_offset_t;

  static constexpr int KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  // Partition selects large and small groups. The middle group is not selected.
  static constexpr size_t num_selected_groups = 2;

  /**
   * Device-accessible allocation of temporary storage. When `nullptr`, the
   * required allocation size is written to `temp_storage_bytes` and no work
   * is done.
   */
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /**
   * Double-buffer whose current buffer contains the unsorted input keys and,
   * upon return, is updated to point to the sorted output keys
   */
  DoubleBuffer<KeyT>& d_keys;

  /**
   * Double-buffer whose current buffer contains the unsorted input values and,
   * upon return, is updated to point to the sorted output values
   */
  DoubleBuffer<ValueT>& d_values;

  /// Number of items to sort
  ::cuda::std::int64_t num_items;

  /// The number of segments that comprise the sorting data
  global_segment_offset_t num_segments;

  /**
   * Random-access input iterator to the sequence of beginning offsets of length
   * `num_segments`, such that `d_begin_offsets[i]` is the first element of the
   * <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`
   */
  BeginOffsetIteratorT d_begin_offsets;

  /**
   * Random-access input iterator to the sequence of ending offsets of length
   * `num_segments`, such that <tt>d_end_offsets[i]-1</tt> is the last element
   * of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   * `d_values_*`. If `d_end_offsets[i]-1 <= d_begin_offsets[i]`,
   * the <em>i</em><sup>th</sup> is considered empty.
   */
  EndOffsetIteratorT d_end_offsets;

  /// Whether is okay to overwrite source buffers
  bool is_overwrite_okay;

  /// CUDA stream to launch kernels within.
  cudaStream_t stream;

  KernelSource kernel_source;

  PartitionKernelSource partition_kernel_source;

  KernelLauncherFactory launcher_factory;

  typename PartitionPolicyHub::MaxPolicy partition_max_policy;

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT policy = {})
  {
    auto wrapped_policy = detail::segmented_sort::MakeSegmentedSortPolicyWrapper(policy);

    CUB_DETAIL_STATIC_ISH_ASSERT(wrapped_policy.LargeSegmentLoadModifier() != CacheLoadModifier::LOAD_LDG,
                                 "The memory consistency model does not apply to texture accesses");

    CUB_DETAIL_STATIC_ISH_ASSERT(
      KEYS_ONLY || wrapped_policy.LargeSegmentLoadAlgorithm() != BLOCK_LOAD_STRIPED
        || wrapped_policy.MediumSegmentLoadAlgorithm() != WARP_LOAD_STRIPED
        || wrapped_policy.SmallSegmentLoadAlgorithm() != WARP_LOAD_STRIPED,
      "Striped load will make this algorithm unstable");

    CUB_DETAIL_STATIC_ISH_ASSERT(wrapped_policy.MediumSegmentStoreAlgorithm() != WARP_STORE_STRIPED
                                   || wrapped_policy.SmallSegmentStoreAlgorithm() != WARP_STORE_STRIPED,
                                 "Striped stores will produce unsorted results");

    const int radix_bits = wrapped_policy.LargeSegmentRadixBits();

    //------------------------------------------------------------------------
    // Prepare temporary storage layout
    //------------------------------------------------------------------------

    const bool partition_segments = num_segments > wrapped_policy.PartitioningThreshold();

    cub::detail::temporary_storage::layout<5> temporary_storage_layout;

    auto keys_slot                          = temporary_storage_layout.get_slot(0);
    auto values_slot                        = temporary_storage_layout.get_slot(1);
    auto large_and_medium_partitioning_slot = temporary_storage_layout.get_slot(2);
    auto small_partitioning_slot            = temporary_storage_layout.get_slot(3);
    auto group_sizes_slot                   = temporary_storage_layout.get_slot(4);

    auto keys_allocation   = keys_slot->create_alias<KeyT>();
    auto values_allocation = values_slot->create_alias<ValueT>();

    if (!is_overwrite_okay)
    {
      keys_allocation.grow(num_items);

      if (!KEYS_ONLY)
      {
        values_allocation.grow(num_items);
      }
    }

    auto large_and_medium_segments_indices = large_and_medium_partitioning_slot->create_alias<local_segment_index_t>();
    auto small_segments_indices            = small_partitioning_slot->create_alias<local_segment_index_t>();
    auto group_sizes                       = group_sizes_slot->create_alias<local_segment_index_t>();

    size_t three_way_partition_temp_storage_bytes{};

    auto large_segments_selector =
      kernel_source.LargeSegmentsSelector(wrapped_policy.MediumPolicyItemsPerTile(), d_begin_offsets, d_end_offsets);
    auto small_segments_selector =
      kernel_source.SmallSegmentsSelector(wrapped_policy.SmallPolicyItemsPerTile() + 1, d_begin_offsets, d_end_offsets);

    auto device_partition_temp_storage = keys_slot->create_alias<uint8_t>();

    if (partition_segments)
    {
      constexpr auto num_segments_per_invocation_limit =
        static_cast<global_segment_offset_t>(::cuda::std::numeric_limits<int>::max());
      auto const max_num_segments_per_invocation = static_cast<global_segment_offset_t>(
        (::cuda::std::min) (static_cast<global_segment_offset_t>(num_segments), num_segments_per_invocation_limit));

      large_and_medium_segments_indices.grow(max_num_segments_per_invocation);
      small_segments_indices.grow(max_num_segments_per_invocation);
      group_sizes.grow(num_selected_groups);

      auto medium_indices_iterator = ::cuda::std::make_reverse_iterator(large_and_medium_segments_indices.get());

      // We call partition through dispatch instead of device because c.parallel needs to be able to call the kernel.
      // This approach propagates the type erasure to partition.
      using ChooseOffsetT                = detail::choose_signed_offset<global_segment_offset_t>;
      using PartitionOffsetT             = typename ChooseOffsetT::type;
      using DispatchThreeWayPartitionIfT = cub::DispatchThreeWayPartitionIf<
        THRUST_NS_QUALIFIER::counting_iterator<local_segment_index_t>,
        decltype(large_and_medium_segments_indices.get()),
        decltype(small_segments_indices.get()),
        decltype(medium_indices_iterator),
        decltype(group_sizes.get()),
        decltype(large_segments_selector),
        decltype(small_segments_selector),
        PartitionOffsetT,
        PartitionPolicyHub,
        PartitionKernelSource,
        KernelLauncherFactory>;

      // Signed integer type for global offsets
      // Check if the number of items exceeds the range covered by the selected signed offset type
      if (const auto error = ChooseOffsetT::is_exceeding_offset_type(num_items))
      {
        return error;
      }

      DispatchThreeWayPartitionIfT::Dispatch(
        nullptr,
        three_way_partition_temp_storage_bytes,
        THRUST_NS_QUALIFIER::counting_iterator<local_segment_index_t>(0),
        large_and_medium_segments_indices.get(),
        small_segments_indices.get(),
        medium_indices_iterator,
        group_sizes.get(),
        large_segments_selector,
        small_segments_selector,
        max_num_segments_per_invocation,
        stream,
        partition_kernel_source,
        launcher_factory,
        partition_max_policy);

      device_partition_temp_storage.grow(three_way_partition_temp_storage_bytes);
    }

    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = temporary_storage_layout.get_size();

      // Return if the caller is simply requesting the size of the storage allocation
      return cudaSuccess;
    }

    if (num_items == 0 || num_segments == 0)
    {
      return cudaSuccess;
    }

    if (const auto error = CubDebug(temporary_storage_layout.map_to_buffer(d_temp_storage, temp_storage_bytes)))
    {
      return error;
    }

    //------------------------------------------------------------------------
    // Sort
    //------------------------------------------------------------------------

    const bool is_num_passes_odd = GetNumPasses(radix_bits) & 1;

    /**
     * This algorithm sorts segments that don't fit into shared memory with
     * the in-global-memory radix sort. Radix sort splits key representation
     * into multiple "digits". Each digit is RADIX_BITS wide. The algorithm
     * iterates over these digits. Each of these iterations consists of a
     * couple of stages. The first stage computes a histogram for a current
     * digit in each segment key. This histogram helps to determine the
     * starting position of the keys group with a similar digit.
     * For example:
     * keys_digits  = [ 1, 0, 0, 1 ]
     * digit_prefix = [ 0, 2 ]
     * The second stage checks the keys again and increments the prefix to
     * determine the final position of the key:
     *
     *               expression            |  key  |   idx   |     result
     * ----------------------------------- | ----- | ------- | --------------
     * result[prefix[keys[0]]++] = keys[0] |   1   |    2    | [ ?, ?, 1, ? ]
     * result[prefix[keys[1]]++] = keys[0] |   0   |    0    | [ 0, ?, 1, ? ]
     * result[prefix[keys[2]]++] = keys[0] |   0   |    1    | [ 0, 0, 1, ? ]
     * result[prefix[keys[3]]++] = keys[0] |   1   |    3    | [ 0, 0, 1, 1 ]
     *
     * If the resulting memory is aliased to the input one, we'll face the
     * following issues:
     *
     *     input      |  key  |   idx   |   result/input   |      issue
     * -------------- | ----- | ------- | ---------------- | ----------------
     * [ 1, 0, 0, 1 ] |   1   |    2    | [ 1, 0, 1, 1 ]   | overwrite keys[2]
     * [ 1, 0, 1, 1 ] |   0   |    0    | [ 0, 0, 1, 1 ]   |
     * [ 0, 0, 1, 1 ] |   1   |    3    | [ 0, 0, 1, 1 ]   | extra key
     * [ 0, 0, 1, 1 ] |   1   |    4    | [ 0, 0, 1, 1 ] 1 | OOB access
     *
     * To avoid these issues, we have to use extra memory. The extra memory
     * holds temporary storage for writing intermediate results of each stage.
     * Since we iterate over digits in keys, we potentially need:
     * `sizeof(KeyT) * num_items * cuda::ceil_div(sizeof(KeyT),RADIX_BITS)`
     * auxiliary memory bytes. To reduce the auxiliary memory storage
     * requirements, the algorithm relies on a double buffer facility. The
     * idea behind it is in swapping destination and source buffers at each
     * iteration. This way, we can use only two buffers. One of these buffers
     * can be the final algorithm output destination. Therefore, only one
     * auxiliary array is needed. Depending on the number of iterations, we
     * can initialize the double buffer so that the algorithm output array
     * will match the double buffer result one at the final iteration.
     * A user can provide this algorithm with a double buffer straightaway to
     * further reduce the auxiliary memory requirements. `is_overwrite_okay`
     * indicates this use case.
     */
    detail::device_double_buffer<KeyT> d_keys_double_buffer(
      (is_overwrite_okay || is_num_passes_odd) ? d_keys.Alternate() : keys_allocation.get(),
      (is_overwrite_okay)   ? d_keys.Current()
      : (is_num_passes_odd) ? keys_allocation.get()
                            : d_keys.Alternate());

    detail::device_double_buffer<ValueT> d_values_double_buffer(
      (is_overwrite_okay || is_num_passes_odd) ? d_values.Alternate() : values_allocation.get(),
      (is_overwrite_okay)   ? d_values.Current()
      : (is_num_passes_odd) ? values_allocation.get()
                            : d_values.Alternate());

    cudaError_t error;
    if (partition_segments)
    {
      // Partition input segments into size groups and assign specialized
      // kernels for each of them.
      error = SortWithPartitioning(
        kernel_source.SegmentedSortKernelLarge(),
        kernel_source.SegmentedSortKernelSmall(),
        three_way_partition_temp_storage_bytes,
        d_keys_double_buffer,
        d_values_double_buffer,
        large_segments_selector,
        small_segments_selector,
        device_partition_temp_storage,
        large_and_medium_segments_indices,
        small_segments_indices,
        group_sizes,
        wrapped_policy);
    }
    else
    {
      // If there are not enough segments, there's no reason to spend time
      // on extra partitioning steps.

      error = SortWithoutPartitioning(
        kernel_source.SegmentedSortFallbackKernel(), d_keys_double_buffer, d_values_double_buffer, wrapped_policy);
    }

    d_keys.selector   = GetFinalSelector(d_keys.selector, radix_bits);
    d_values.selector = GetFinalSelector(d_values.selector, radix_bits);

    return error;
  }

  template <typename MaxPolicyT          = typename PolicyHub::MaxPolicy,
            typename PartitionMaxPolicyT = typename PartitionPolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    ::cuda::std::int64_t num_items,
    global_segment_offset_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    bool is_overwrite_okay,
    cudaStream_t stream,
    KernelSource kernel_source                    = {},
    PartitionKernelSource partition_kernel_source = {},
    KernelLauncherFactory launcher_factory        = {},
    MaxPolicyT max_policy                         = {},
    PartitionMaxPolicyT partition_max_policy      = {})
  {
    // Get PTX version
    int ptx_version = 0;
    if (const auto error = CubDebug(launcher_factory.PtxVersion(ptx_version)))
    {
      return error;
    }

    // Create dispatch functor
    DispatchSegmentedSort dispatch{
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      is_overwrite_okay,
      stream,
      kernel_source,
      partition_kernel_source,
      launcher_factory,
      partition_max_policy};

    // Dispatch to chained policy
    return CubDebug(max_policy.Invoke(ptx_version, dispatch));
  }

private:
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE int GetNumPasses(int radix_bits)
  {
    constexpr int byte_size = 8;
    const int num_bits      = static_cast<int>(kernel_source.KeySize()) * byte_size;
    const int num_passes    = ::cuda::ceil_div(num_bits, radix_bits);
    return num_passes;
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE int GetFinalSelector(int selector, int radix_bits)
  {
    // Sorted data always ends up in the other vector
    if (!is_overwrite_okay)
    {
      return (selector + 1) & 1;
    }

    return (selector + GetNumPasses(radix_bits)) & 1;
  }

  template <typename T>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE T* GetFinalOutput(int radix_bits, DoubleBuffer<T>& buffer)
  {
    const int final_selector = GetFinalSelector(buffer.selector, radix_bits);
    return buffer.d_buffers[final_selector];
  }

  template <typename WrappedPolicyT, typename LargeKernelT, typename SmallKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t SortWithPartitioning(
    LargeKernelT large_kernel,
    SmallKernelT small_kernel,
    size_t three_way_partition_temp_storage_bytes,
    cub::detail::device_double_buffer<KeyT>& d_keys_double_buffer,
    cub::detail::device_double_buffer<ValueT>& d_values_double_buffer,
    typename KernelSource::LargeSegmentsSelectorT& large_segments_selector,
    typename KernelSource::SmallSegmentsSelectorT& small_segments_selector,
    cub::detail::temporary_storage::alias<uint8_t>& device_partition_temp_storage,
    cub::detail::temporary_storage::alias<local_segment_index_t>& large_and_medium_segments_indices,
    cub::detail::temporary_storage::alias<local_segment_index_t>& small_segments_indices,
    cub::detail::temporary_storage::alias<local_segment_index_t>& group_sizes,
    WrappedPolicyT wrapped_policy)
  {
    constexpr global_segment_offset_t num_segments_per_invocation_limit =
      static_cast<global_segment_offset_t>(::cuda::std::numeric_limits<int>::max());

    // We repeatedly invoke the partitioning and sorting kernels until all segments are processed.
    const global_segment_offset_t num_invocations =
      ::cuda::ceil_div(static_cast<global_segment_offset_t>(num_segments), num_segments_per_invocation_limit);
    for (global_segment_offset_t invocation_index = 0; invocation_index < num_invocations; invocation_index++)
    {
      const global_segment_offset_t current_seg_offset = invocation_index * num_segments_per_invocation_limit;
      const local_segment_index_t current_num_segments =
        (invocation_index == (num_invocations - 1))
          ? static_cast<local_segment_index_t>(num_segments - current_seg_offset)
          : num_segments_per_invocation_limit;

      kernel_source.SetSegmentOffset(large_segments_selector, current_seg_offset);
      kernel_source.SetSegmentOffset(small_segments_selector, current_seg_offset);

      BeginOffsetIteratorT current_begin_offset = d_begin_offsets;
      EndOffsetIteratorT current_end_offset     = d_end_offsets;

      current_begin_offset += current_seg_offset;
      current_end_offset += current_seg_offset;

      auto medium_indices_iterator =
        ::cuda::std::make_reverse_iterator(large_and_medium_segments_indices.get() + current_num_segments);

      // We call partition through dispatch instead of device because c.parallel needs to be able to call the kernel.
      // This approach propagates the type erasure to partition.
      using ChooseOffsetT                = detail::choose_signed_offset<global_segment_offset_t>;
      using PartitionOffsetT             = typename ChooseOffsetT::type;
      using DispatchThreeWayPartitionIfT = cub::DispatchThreeWayPartitionIf<
        THRUST_NS_QUALIFIER::counting_iterator<local_segment_index_t>,
        decltype(large_and_medium_segments_indices.get()),
        decltype(small_segments_indices.get()),
        decltype(medium_indices_iterator),
        decltype(group_sizes.get()),
        decltype(large_segments_selector),
        decltype(small_segments_selector),
        PartitionOffsetT,
        PartitionPolicyHub,
        PartitionKernelSource,
        KernelLauncherFactory>;

      // Signed integer type for global offsets
      // Check if the number of items exceeds the range covered by the selected signed offset type
      if (const auto error = ChooseOffsetT::is_exceeding_offset_type(num_items))
      {
        return error;
      }

      if (const auto error = DispatchThreeWayPartitionIfT::Dispatch(
            device_partition_temp_storage.get(),
            three_way_partition_temp_storage_bytes,
            THRUST_NS_QUALIFIER::counting_iterator<local_segment_index_t>(0),
            large_and_medium_segments_indices.get(),
            small_segments_indices.get(),
            medium_indices_iterator,
            group_sizes.get(),
            large_segments_selector,
            small_segments_selector,
            current_num_segments,
            stream,
            partition_kernel_source,
            launcher_factory,
            partition_max_policy))
      {
        return error;
      }

      // The device path is only used (and only compiles) when CDP is enabled.
      // It's defined in a macro since we can't put `#ifdef`s inside of
      // `NV_IF_TARGET`.
#ifndef CUB_RDC_ENABLED
#  define CUB_TEMP_DEVICE_CODE
#else // CUB_RDC_ENABLED
#  define CUB_TEMP_DEVICE_CODE                                                          \
    if (const auto error = CubDebug(                                                    \
          launcher_factory(1, 1, 0, stream)                                             \
            .doit(                                                                      \
              detail::segmented_sort::DeviceSegmentedSortContinuationKernel<            \
                WrappedPolicyT,                                                         \
                LargeKernelT,                                                           \
                SmallKernelT,                                                           \
                KeyT,                                                                   \
                ValueT,                                                                 \
                BeginOffsetIteratorT,                                                   \
                EndOffsetIteratorT,                                                     \
                KernelLauncherFactory>,                                                 \
              large_kernel,                                                             \
              small_kernel,                                                             \
              current_num_segments,                                                     \
              d_keys.Current(),                                                         \
              GetFinalOutput<KeyT>(wrapped_policy.LargeSegmentRadixBits(), d_keys),     \
              d_keys_double_buffer,                                                     \
              d_values.Current(),                                                       \
              GetFinalOutput<ValueT>(wrapped_policy.LargeSegmentRadixBits(), d_values), \
              d_values_double_buffer,                                                   \
              current_begin_offset,                                                     \
              current_end_offset,                                                       \
              group_sizes.get(),                                                        \
              large_and_medium_segments_indices.get(),                                  \
              small_segments_indices.get(),                                             \
              launcher_factory,                                                         \
              wrapped_policy)))                                                         \
    {                                                                                   \
      return error;                                                                     \
    }                                                                                   \
                                                                                        \
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))                   \
    {                                                                                   \
      return error;                                                                     \
    }
#endif // CUB_RDC_ENABLED

      NV_IF_TARGET(
        NV_IS_HOST,
        (
          local_segment_index_t h_group_sizes[num_selected_groups];
          if (const auto error = CubDebug(launcher_factory.MemcpyAsync(
                h_group_sizes,
                group_sizes.get(),
                num_selected_groups * sizeof(local_segment_index_t),
                cudaMemcpyDeviceToHost,
                stream))) { return error; }

          if (const auto error = CubDebug(SyncStream(stream))) { return error; }

          if (const auto error = detail::segmented_sort::DeviceSegmentedSortContinuation(
                large_kernel,
                small_kernel,
                current_num_segments,
                d_keys.Current(),
                GetFinalOutput<KeyT>(wrapped_policy.LargeSegmentRadixBits(), d_keys),
                d_keys_double_buffer,
                d_values.Current(),
                GetFinalOutput<ValueT>(wrapped_policy.LargeSegmentRadixBits(), d_values),
                d_values_double_buffer,
                current_begin_offset,
                current_end_offset,
                h_group_sizes,
                large_and_medium_segments_indices.get(),
                small_segments_indices.get(),
                stream,
                launcher_factory,
                wrapped_policy)) { return error; }),
        // NV_IS_DEVICE:
        (CUB_TEMP_DEVICE_CODE));
    }
#undef CUB_TEMP_DEVICE_CODE

    return cudaSuccess;
  }

  template <typename WrappedPolicyT, typename FallbackKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t SortWithoutPartitioning(
    FallbackKernelT fallback_kernel,
    cub::detail::device_double_buffer<KeyT>& d_keys_double_buffer,
    cub::detail::device_double_buffer<ValueT>& d_values_double_buffer,
    WrappedPolicyT wrapped_policy)
  {
    const auto blocks_in_grid   = static_cast<local_segment_index_t>(num_segments);
    const auto threads_in_block = static_cast<unsigned int>(wrapped_policy.LargeSegment().BlockThreads());

// Log kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking DeviceSegmentedSortFallbackKernel<<<%d, %d, "
            "0, %lld>>>(), %d items per thread, bit_grain %d\n",
            blocks_in_grid,
            threads_in_block,
            (long long) stream,
            wrapped_policy.LargeSegment().ItemsPerThread(),
            wrapped_policy.LargeSegmentRadixBits());
#endif // CUB_DEBUG_LOG

    // Invoke fallback kernel
    launcher_factory(blocks_in_grid, threads_in_block, 0, stream)
      .doit(fallback_kernel,
            d_keys.Current(),
            GetFinalOutput(wrapped_policy.LargeSegmentRadixBits(), d_keys),
            d_keys_double_buffer,
            d_values.Current(),
            GetFinalOutput(wrapped_policy.LargeSegmentRadixBits(), d_values),
            d_values_double_buffer,
            d_begin_offsets,
            d_end_offsets);

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    return CubDebug(detail::DebugSyncStream(stream));
  }
};

CUB_NAMESPACE_END
