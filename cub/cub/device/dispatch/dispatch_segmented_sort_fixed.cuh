// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_segmented_sort.cuh>

#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__iterator/transform_iterator.h>
#include <cuda/std/__type_traits/is_signed.h>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort
{
template <typename SegmentSizeT>
struct fixed_segment_offset
{
  SegmentSizeT segment_size;

  _CCCL_HOST_DEVICE constexpr global_segment_offset_t operator()(global_segment_offset_t index) const
  {
    return index * static_cast<global_segment_offset_t>(params::get_param(segment_size, 0));
  }
};

template <SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename SegmentSizeT,
          typename PolicySelector,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_fixed_size(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  DoubleBuffer<KeyT>& d_keys,
  DoubleBuffer<ValueT>& d_values,
  SegmentSizeT segment_size,
  global_segment_offset_t num_segments,
  bool is_overwrite_okay,
  cudaStream_t stream,
  PolicySelector policy_selector,
  KernelLauncherFactory launcher_factory = {})
{
  constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, NullType>;
  const auto size          = static_cast<global_segment_offset_t>(params::get_param(segment_size, 0));
  const auto num_items     = size * num_segments;

  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  return detail::dispatch_compute_cap(policy_selector, cc, [&](auto policy_getter) -> cudaError_t {
    check_policy<keys_only>(policy_getter);
    CUB_DETAIL_CONSTEXPR_ISH const SegmentedSortPolicy policy = policy_getter();
    detail::log_dispatch("DeviceSegmentedSort (fixed size)", cc, policy);

    const bool small = size <= policy.small_segment.items_per_tile();
    const bool large = size > policy.medium_segment.items_per_tile();
    const bool multi_pass =
      large && size >= policy.large_segment.threads_per_block * policy.large_segment.items_per_thread;
    const int num_passes = ::cuda::ceil_div(static_cast<int>(sizeof(KeyT) * CHAR_BIT), policy.large_segment.radix_bits);

    temporary_storage::layout<2> storage;
    auto keys_allocation   = storage.get_slot(0)->template create_alias<KeyT>();
    auto values_allocation = storage.get_slot(1)->template create_alias<ValueT>();
    if (multi_pass && !is_overwrite_okay)
    {
      // Each allocation and the storage base may need up to 255 bytes of alignment padding.
      constexpr auto max_storage_items =
        ((::cuda::std::numeric_limits<size_t>::max)() - 3 * 255) / (sizeof(KeyT) + (keys_only ? 0 : sizeof(ValueT)));
      if (static_cast<::cuda::std::uint64_t>(num_items) > max_storage_items)
      {
        return cudaErrorInvalidValue;
      }
      keys_allocation.grow(num_items);
      if constexpr (!keys_only)
      {
        values_allocation.grow(num_items);
      }
    }
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = storage.get_size();
      return cudaSuccess;
    }
    if (num_items == 0)
    {
      return cudaSuccess;
    }
    if (const auto error = CubDebug(storage.map_to_buffer(d_temp_storage, temp_storage_bytes)))
    {
      return error;
    }

    const int selector_delta = is_overwrite_okay && multi_pass ? num_passes : 1;
    auto keys_out            = d_keys.d_buffers[(d_keys.selector + selector_delta) & 1];
    auto values_out          = d_values.d_buffers[(d_values.selector + selector_delta) & 1];
    const bool odd_passes    = num_passes & 1;
    device_double_buffer<KeyT> key_buffers(
      is_overwrite_okay || odd_passes ? d_keys.Alternate() : keys_allocation.get(),
      is_overwrite_okay ? d_keys.Current() : (odd_passes ? keys_allocation.get() : d_keys.Alternate()));
    device_double_buffer<ValueT> value_buffers(
      is_overwrite_okay || odd_passes ? d_values.Alternate() : values_allocation.get(),
      is_overwrite_okay ? d_values.Current() : (odd_passes ? values_allocation.get() : d_values.Alternate()));

    using index_iterator_t = ::cuda::counting_iterator<local_segment_index_t>;
    const index_iterator_t indices{0};
    auto offsets = ::cuda::make_transform_iterator(
      ::cuda::counting_iterator<global_segment_offset_t>{0}, fixed_segment_offset<SegmentSizeT>{segment_size});
    using offset_iterator_t = decltype(offsets);

    // A large segment uses one block, so each launch must fit in grid.x.
    constexpr global_segment_offset_t segments_per_launch = (::cuda::std::numeric_limits<int>::max)();
    for (global_segment_offset_t first = 0; first < num_segments;)
    {
      const auto count =
        static_cast<local_segment_index_t>((::cuda::std::min) (num_segments - first, segments_per_launch));
      const auto begins = offsets + first;
      const auto ends   = begins + 1;
      cudaError_t error;
      if (large)
      {
        error =
          launcher_factory(count, policy.large_segment.threads_per_block, 0, stream)
            .doit(
              DeviceSegmentedSortKernelLarge<Order,
                                             PolicySelector,
                                             KeyT,
                                             ValueT,
                                             offset_iterator_t,
                                             offset_iterator_t,
                                             global_segment_offset_t,
                                             ::cuda::counting_iterator<local_segment_index_t>>,
              indices,
              d_keys.Current(),
              keys_out,
              key_buffers,
              d_values.Current(),
              values_out,
              value_buffers,
              begins,
              ends);
      }
      else
      {
        const auto segments_per_block =
          small ? policy.small_segment.segments_per_block() : policy.medium_segment.segments_per_block();
        const auto blocks = ::cuda::ceil_div(count, segments_per_block);
        error =
          launcher_factory(blocks, policy.small_segment.threads_per_block, 0, stream)
            .doit(
              DeviceSegmentedSortKernelSmall<Order,
                                             PolicySelector,
                                             KeyT,
                                             ValueT,
                                             offset_iterator_t,
                                             offset_iterator_t,
                                             global_segment_offset_t,
                                             ::cuda::counting_iterator<local_segment_index_t>>,
              small ? count : 0,
              small ? 0 : count,
              small ? 0 : blocks,
              indices,
              indices,
              d_keys.Current(),
              keys_out,
              d_values.Current(),
              values_out,
              begins,
              ends);
      }
      if (const auto status = CubDebug(error))
      {
        return status;
      }
      if (const auto status = CubDebug(cudaPeekAtLastError()))
      {
        return status;
      }
      if (const auto status = CubDebug(detail::DebugSyncStream(stream)))
      {
        return status;
      }
      first += count;
    }
    d_keys.selector   = (d_keys.selector + selector_delta) & 1;
    d_values.selector = (d_values.selector + selector_delta) & 1;
    return cudaSuccess;
  });
}

template <SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename SegmentSizeT,
          typename NumSegmentsT,
          typename PolicySelector>
CUB_RUNTIME_FUNCTION cudaError_t dispatch_fixed(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  DoubleBuffer<KeyT>& d_keys,
  DoubleBuffer<ValueT>& d_values,
  SegmentSizeT segment_size,
  NumSegmentsT num_segments,
  bool is_overwrite_okay,
  cudaStream_t stream,
  PolicySelector policy_selector)
{
  using size_validation  = params::__validate_uniform_integral_param<SegmentSizeT>;
  using count_validation = params::__validate_uniform_integral_param<NumSegmentsT>;
  static_assert(!size_validation::args_traits::is_deferred && !count_validation::args_traits::is_deferred,
                "cub::DeviceSegmentedSort: fixed segment size and count must be host-known values.");
  if constexpr (size_validation::all_ok && count_validation::all_ok && !size_validation::args_traits::is_deferred
                && !count_validation::args_traits::is_deferred)
  {
    const auto size  = params::get_param(segment_size, 0);
    const auto count = params::get_param(num_segments, 0);
    static_assert(sizeof(size) <= sizeof(::cuda::std::uint64_t) && sizeof(count) <= sizeof(::cuda::std::uint64_t),
                  "cub::DeviceSegmentedSort: segment size and count must be at most 64 bits wide.");
    if ((::cuda::std::is_signed_v<decltype(size)> ? size < 0 : false)
        || (::cuda::std::is_signed_v<decltype(count)> ? count < 0 : false))
    {
      return cudaErrorInvalidValue;
    }
    constexpr auto max_offset =
      static_cast<::cuda::std::uint64_t>((::cuda::std::numeric_limits<global_segment_offset_t>::max)());
    const auto unsigned_size  = static_cast<::cuda::std::uint64_t>(size);
    const auto unsigned_count = static_cast<::cuda::std::uint64_t>(count);
    if (unsigned_size > max_offset || unsigned_count > max_offset
        || (unsigned_size != 0 && unsigned_count > max_offset / unsigned_size))
    {
      return cudaErrorInvalidValue;
    }
    return dispatch_fixed_size<Order>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      params::normalize_param(segment_size),
      static_cast<global_segment_offset_t>(count),
      is_overwrite_okay,
      stream,
      policy_selector);
  }
  else
  {
    return cudaErrorInvalidValue;
  }
}
} // namespace detail::segmented_sort

CUB_NAMESPACE_END
