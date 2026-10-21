// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/util_namespace.cuh>

#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_capacity
{
// Partition sees only these bounds, including one implicit overflow bin.
template <int... Bounds>
struct capacity_list
{
  static constexpr int size = sizeof...(Bounds);
  inline static constexpr ::cuda::std::array<int, size> bounds{{Bounds...}};

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

  template <int Begin = 0, int End = size, typename Offset>
  _CCCL_HOST_DEVICE_API static constexpr int first_fit(Offset length)
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

template <typename Algorithm, const auto* Tuning>
struct sort_entry
{
  using implementation                = typename Algorithm::template implementation<Tuning>;
  using expansion                     = typename implementation::expansion;
  using capacities                    = typename expansion::capacities;
  inline static constexpr auto tuning = *Tuning;
  static constexpr int block_threads  = Tuning->block_threads;
  static constexpr int unit_threads   = implementation::unit_threads;
  static_assert(block_threads > 0 && block_threads <= 1024 && block_threads % unit_threads == 0);
  static_assert(Tuning->blocks_per_sm > 0 && Tuning->min_blocks_per_sm > 0);
  static_assert(capacities::valid());
};

template <typename... Entries>
struct sort_entries
{
  static constexpr int size = sizeof...(Entries);
  static_assert(size > 0);
  using entries = ::cuda::std::tuple<Entries...>;
  template <::cuda::std::size_t I>
  using entry      = ::cuda::std::tuple_element_t<I, entries>;
  using capacities = typename concatenate_capacities<typename Entries::capacities...>::type;
  static_assert(capacities::valid());
  _CCCL_HOST_DEVICE_API static constexpr int bin_begin(int entry_index)
  {
    constexpr int counts[] = {Entries::capacities::size...};
    int result             = 0;
    for (int i = 0; i < entry_index; ++i)
    {
      result += counts[i];
    }
    return result;
  }
};

struct partition_result
{
  unsigned* bin_offsets{}; // size + 2 values: bounded bins followed by overflow
  unsigned* segment_ids{};
};
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END
