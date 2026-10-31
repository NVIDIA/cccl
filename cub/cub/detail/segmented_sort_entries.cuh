// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sorting-kernel contract: compose capacity bounds and configurations for partitioning and sorting.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/cc_dispatch.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_capacity
{
// Ordered capacity bounds supplied by sorting kernels; partition finds the smallest fitting bin.
template <int... Bounds>
struct capacity_list
{
  static constexpr int size = sizeof...(Bounds);
  inline static constexpr ::cuda::std::array<int, size> bounds{{Bounds...}};

  // Require nonempty, positive, strictly increasing bounds for binary-search routing.
  _CCCL_HOST_DEVICE_API static constexpr bool valid()
  {
    for (int i = 0; i < size; ++i)
    {
      if (bounds[i] <= 0 || (i && bounds[i - 1] >= bounds[i]))
      {
        return false;
      }
    }
    return size > 0;
  }
  static_assert(valid(), "capacities must be positive and strictly increasing");

  // Find the smallest capacity covering length; size denotes the implicit overflow bin.
  template <int Begin = 0, int End = size, typename Offset>
  _CCCL_HOST_DEVICE_API static constexpr int first_fit([[maybe_unused]] Offset length)
  {
    if constexpr (Begin == End)
    {
      return Begin;
    }
    else
    {
      constexpr int mid = Begin + (End - Begin) / 2;
      return length <= bounds[mid] ? first_fit<Begin, mid>(length) : first_fit<mid + 1, End>(length);
    }
  }
};

// Flatten kernel capacities in launch order, preserving each kernel's bin order.
template <typename... Lists>
struct concatenate_capacities;
template <typename List>
struct concatenate_capacities<List>
{
  using type = List;
};
template <int... Left, int... Right, typename... Rest>
struct concatenate_capacities<capacity_list<Left...>, capacity_list<Right...>, Rest...>
    : concatenate_capacities<capacity_list<Left..., Right...>, Rest...>
{};

// Encode one bin's configuration in a type for compile-time mapping.
// group_merge_expansion::shape_for_bin<Bin> supplies this type to the lookup below.
template <int Threads, int ItemsPerThread>
struct bin_shape
{
  static constexpr int threads_per_segment = Threads;
  static constexpr int items_per_thread    = ItemsPerThread;
};

// Configuration values returned by lookup_bin_config to sort_kernel for work packing
// and passed to the GroupMerge worker. Also used during compile-time capacity expansion.
struct bin_config
{
  int threads_per_segment;
  int items_per_thread;
};

// An expansion enumerates a sorter's bins: capacities and a shape_for_bin<Bin> type.
// The runtime local bin selects a leaf whose configuration is known at compile time.
template <typename Expansion, int Begin = 0, int End = Expansion::capacities::size>
_CCCL_HOST_DEVICE_API constexpr bin_config lookup_bin_config(int bin_idx)
{
  static_assert(Begin < End);
  if constexpr (End - Begin == 1)
  {
    using shape = typename Expansion::template shape_for_bin<Begin>;
    return {shape::threads_per_segment, shape::items_per_thread};
  }
  else
  {
    constexpr int mid = Begin + (End - Begin) / 2;
    return bin_idx < mid ? lookup_bin_config<Expansion, Begin, mid>(bin_idx)
                         : lookup_bin_config<Expansion, mid, End>(bin_idx);
  }
}

// Describe a sorting launch without fixing its architecture-selected tuning in the type.
template <template <typename> class Implementation, typename PolicySelector>
struct sort_kernel_t
{
  using policy_selector = PolicySelector;
  template <typename ComputeCapabilityGetter>
  using implementation_t = Implementation<policy_getter<PolicySelector, ComputeCapabilityGetter{}().get()>>;
  template <typename ComputeCapabilityGetter>
  using expansion_t = typename implementation_t<ComputeCapabilityGetter>::expansion;
  template <typename ComputeCapabilityGetter>
  using capacities_t = typename expansion_t<ComputeCapabilityGetter>::capacities;
};

// Compose an ordered kernel list; resolve capacities for host setup or device compilation.
template <typename... Kernels>
struct sort_kernel_list_t
{
  static constexpr int size = sizeof...(Kernels);
  static_assert(size > 0);
  using kernels_t = ::cuda::std::tuple<Kernels...>;
  template <::cuda::std::size_t KernelIndex>
  using kernel_t = ::cuda::std::tuple_element_t<KernelIndex, kernels_t>;
  template <typename ComputeCapabilityGetter>
  using capacities_t =
    typename concatenate_capacities<typename Kernels::template capacities_t<ComputeCapabilityGetter>...>::type;

  // First global bin index for the selected kernel; local bin indices start at zero.
  template <typename ComputeCapabilityGetter>
  _CCCL_HOST_DEVICE_API static constexpr int kernel_bin_begin_idx(int kernel_idx)
  {
    constexpr int kernel_bin_counts[] = {Kernels::template capacities_t<ComputeCapabilityGetter>::size...};
    int first_bin_idx                 = 0;
    for (int i = 0; i < kernel_idx; ++i)
    {
      first_bin_idx += kernel_bin_counts[i];
    }
    return first_bin_idx;
  }
};

// Let partition-only callers supply fixed bounds through the same capacity-selector interface.
template <typename Capacities>
struct fixed_capacity_selector_t
{
  template <typename>
  using capacities_t = Capacities;
};

// Query the architecture being compiled without carrying it in a kernel template argument.
struct device_compute_capability_getter_t
{
  _CCCL_DEVICE_API constexpr auto operator()() const
  {
    return current_tuning_cc();
  }
};

// Non-owning views into temporary storage; all bins share one segment-ID queue.
struct partition_result
{
  unsigned* bin_offsets{}; // Bounded bin count + 2 offsets, including overflow's end.
  unsigned* segment_ids{};
};
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END
