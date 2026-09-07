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

#include <cub/block/block_merge_sort.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__utility/forward.h>

CUB_NAMESPACE_BEGIN

namespace detail::find_bound_sorted_values
{
// lower_bound vs upper_bound: partition comparator and per-step advance differ.
struct lower_bound_mode
{
  // Wrap user comp so the merge path partitions identically to std::lower_bound.
  template <typename CompareOp>
  struct partition_comp_t
  {
    CompareOp comp;

    template <typename A, typename B>
    _CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE bool operator()(A&& a, B&& b) const
    {
      return !comp(::cuda::std::forward<B>(b), ::cuda::std::forward<A>(a));
    }
  };

  template <typename CompareOp>
  _CCCL_HOST_DEVICE_API static partition_comp_t<CompareOp> make_partition_comp(CompareOp compare_op)
  {
    return partition_comp_t<CompareOp>{compare_op};
  }

  template <typename HaystackT, typename NeedlesT, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static bool
  should_advance(const HaystackT& haystack_value, const NeedlesT& needle_value, CompareOp compare_op)
  {
    return compare_op(haystack_value, needle_value);
  }
};

struct upper_bound_mode
{
  template <typename CompareOp>
  using partition_comp_t = CompareOp;

  template <typename CompareOp>
  _CCCL_HOST_DEVICE_API static CompareOp make_partition_comp(CompareOp compare_op)
  {
    return compare_op;
  }

  template <typename HaystackT, typename NeedlesT, typename CompareOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static bool
  should_advance(const HaystackT& haystack_value, const NeedlesT& needle_value, CompareOp compare_op)
  {
    return !compare_op(needle_value, haystack_value);
  }
};

template <int ThreadsPerBlock,
          int ItemsPerThread,
          CacheLoadModifier LoadModifier,
          typename Mode,
          typename HaystackIt,
          typename NeedlesIt,
          typename OutputIt,
          typename Offset,
          typename CompareOp>
struct agent_t
{
  static constexpr int tile_size = ThreadsPerBlock * ItemsPerThread;

  using haystack_type = it_value_t<HaystackIt>;
  using needles_type  = it_value_t<NeedlesIt>;

  // Separate buffers because haystack and needles may have different value types.
  struct _TempStorage
  {
    haystack_type haystack[tile_size];
    needles_type needles[tile_size];
  };

  using TempStorage = Uninitialized<_TempStorage>;

  _TempStorage& storage;
  HaystackIt d_range;
  NeedlesIt d_values;
  OutputIt d_output;
  Offset range_count;
  Offset values_count;
  Offset* range_beg_offsets;
  CompareOp compare_op;

  template <bool IsFullTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void consume_tile(int tile_idx, Offset diag0, int total_in_tile)
  {
    const Offset range_beg = range_beg_offsets[tile_idx];
    const Offset range_end = range_beg_offsets[tile_idx + 1];
    _CCCL_ASSERT(range_end >= range_beg, "");
    _CCCL_ASSERT(diag0 >= range_beg, "");
    const Offset values_beg = diag0 - range_beg;

    const int haystack_count = static_cast<int>(range_end - range_beg);
    const int needles_count  = total_in_tile - haystack_count;

    {
      const auto d_range_cm = cub::detail::try_make_cache_modified_iterator<LoadModifier>(d_range + range_beg);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        const int idx = ThreadsPerBlock * item + threadIdx.x;
        if (idx < haystack_count)
        {
          storage.haystack[idx] = d_range_cm[idx];
        }
      }
    }

    {
      auto d_values_cm = cub::detail::try_make_cache_modified_iterator<LoadModifier>(d_values + values_beg);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        const int idx = ThreadsPerBlock * item + threadIdx.x;
        if (idx < needles_count)
        {
          storage.needles[idx] = d_values_cm[idx];
        }
      }
    }

    __syncthreads();

#ifdef CCCL_ENABLE_DEVICE_ASSERTIONS
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int idx = ThreadsPerBlock * item + threadIdx.x;
      if (idx < needles_count && (values_beg + idx) > 0)
      {
        const needles_type prev = (idx == 0) ? d_values[values_beg - 1] : storage.needles[idx - 1];
        _CCCL_ASSERT(!compare_op(storage.needles[idx], prev), "d_values must be sorted consistently with comp");
      }
    }
#endif // CCCL_ENABLE_DEVICE_ASSERTIONS

    const auto partition_comp = Mode::make_partition_comp(compare_op);

    int d0_thread = ItemsPerThread * static_cast<int>(threadIdx.x);
    if constexpr (!IsFullTile)
    {
      d0_thread = ::cuda::std::min(d0_thread, total_in_tile);
    }

    const int i0 =
      cub::MergePath(storage.haystack, storage.needles, haystack_count, needles_count, d0_thread, partition_comp);
    const int j0 = d0_thread - i0;

    int i                  = i0;
    int j                  = j0;
    int haystack_remaining = haystack_count - i0;
    int needles_remaining  = needles_count - j0;

    const int steps = IsFullTile ? ItemsPerThread : ::cuda::std::min(total_in_tile - d0_thread, ItemsPerThread);
    _CCCL_PRAGMA_UNROLL(ItemsPerThread)
    for (int step = 0; step < steps; ++step)
    {
      const bool advance_haystack =
        (needles_remaining == 0)
        || (haystack_remaining > 0 && Mode::should_advance(storage.haystack[i], storage.needles[j], compare_op));
      if (advance_haystack)
      {
        ++i;
        --haystack_remaining;
      }
      else
      {
        using output_value_t     = cub::detail::non_void_value_t<OutputIt, Offset>;
        d_output[values_beg + j] = static_cast<output_value_t>(range_beg + i);
        ++j;
        --needles_remaining;
      }
    }
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void operator()()
  {
    const int tile_idx      = static_cast<int>(blockIdx.x);
    const Offset diag0      = static_cast<Offset>(tile_size) * tile_idx;
    const Offset diag1      = ::cuda::std::min(diag0 + static_cast<Offset>(tile_size), range_count + values_count);
    const int total_in_tile = static_cast<int>(diag1 - diag0);

    if (total_in_tile == tile_size)
    {
      consume_tile</* IsFullTile = */ true>(tile_idx, diag0, tile_size);
    }
    else
    {
      consume_tile</* IsFullTile = */ false>(tile_idx, diag0, total_in_tile);
    }
  }
};
} // namespace detail::find_bound_sorted_values

CUB_NAMESPACE_END
