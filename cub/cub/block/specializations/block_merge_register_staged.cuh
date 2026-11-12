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

#include <cub/util_namespace.cuh>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>

CUB_NAMESPACE_BEGIN
namespace detail
{
template <typename MergePolicy>
class block_merge_register_staged
{
  using key_t _CCCL_NODEBUG    = typename MergePolicy::key_t;
  using layout_t _CCCL_NODEBUG = typename MergePolicy::layout_t;
  using window_t _CCCL_NODEBUG = typename MergePolicy::window_t;

  template <typename CompareOp>
  using cursor_t _CCCL_NODEBUG = typename MergePolicy::template cursor_t<CompareOp>;

  static constexpr unsigned int items_per_thread = MergePolicy::items_per_thread;
  static constexpr unsigned int pair_capacity    = items_per_thread + 2;
  const unsigned int linear_tid;

  [[nodiscard]] _CCCL_DEVICE_API int pair_fit_mask(const int* offsets, int runs, int effective, int log_runs)
  {
    if (runs <= 1 || !layout_t::use_pairs(effective))
    {
      return 0;
    }

    const int lane   = static_cast<int>(linear_tid) & 31;
    const int offset = offsets[(::cuda::std::min) (lane, runs)];
    int mask         = 0;
    // The final round always produces blocked register output instead.
    for (int round = 0; round < log_runs - 1; ++round)
    {
      const int pairs        = effective >> (round + 1);
      const int begin_lane   = lane << (round + 1);
      const int end_lane     = (lane + 1) << (round + 1);
      const int begin        = ::__shfl_sync(0xffffffffu, offset, begin_lane & 31);
      const int shuffled_end = ::__shfl_sync(0xffffffffu, offset, end_lane & 31);
      const int end          = end_lane >= 32 ? offsets[runs] : shuffled_end;
      int length             = lane < pairs ? end - begin : 0;

      for (int distance = 16; distance > 0; distance >>= 1)
      {
        length = (::cuda::std::max) (length, ::__shfl_xor_sync(0xffffffffu, length, distance));
      }

      const int threads = 1 << layout_t::pair_thread_log(log_runs, round);
      mask |= length <= (static_cast<int>(items_per_thread) + 1) * threads ? 1 << round : 0;
    }
    return mask;
  }

  template <typename CompareOp>
  [[nodiscard]] _CCCL_DEVICE_API window_t stage_pair(
    const key_t* keys,
    const int* offsets,
    int runs,
    int span,
    int round,
    int log_runs,
    key_t (&staged)[pair_capacity],
    CompareOp compare_op)
  {
    const int log_threads = layout_t::pair_thread_log(log_runs, round);
    const int pair_id     = static_cast<int>(linear_tid) >> log_threads;
    const int lane        = static_cast<int>(linear_tid) & ((1 << log_threads) - 1);
    const auto pair       = layout_t::pair_bounds(offsets, runs, pair_id, span);
    const auto window     = layout_t::pair_window(pair.end - pair.begin, log_threads, lane);
    const int count       = window.end - window.begin;

    if (count > 0)
    {
      cursor_t<CompareOp> cursor(keys, pair, window.begin, compare_op);

      _CCCL_PRAGMA_UNROLL_FULL()
      for (unsigned int item = 0; item < pair_capacity; ++item)
      {
        if (static_cast<int>(item) < count)
        {
          staged[item] = cursor.next();
        }
      }
    }
    return {pair.begin + window.begin, pair.begin + window.end};
  }

  template <typename CompareOp>
  _CCCL_DEVICE_API void merge_thread(
    const key_t* keys,
    const int* offsets,
    int runs,
    int span,
    int round,
    window_t window,
    key_t (&out)[items_per_thread],
    CompareOp compare_op)
  {
    int position = window.begin;

    while (position < window.end)
    {
      const auto pair = layout_t::find_pair(position, offsets, runs, span, round);
      const int stop  = (::cuda::std::min) (window.end, pair.end);
      cursor_t<CompareOp> cursor(keys, pair, position - pair.begin, compare_op);

      _CCCL_PRAGMA_UNROLL_FULL()
      for (unsigned int item = 0; item < items_per_thread; ++item)
      {
        const int rank = window.begin + static_cast<int>(item);
        if (rank >= position && rank < stop)
        {
          out[item] = cursor.next();
        }
      }
      position = stop;
    }
  }

public:
  using storage_t _CCCL_NODEBUG = typename MergePolicy::template storage_t<1>;

  _CCCL_DEVICE_API block_merge_register_staged(storage_t&, unsigned int tid)
      : linear_tid(tid)
  {}

  template <typename CompareOp>
  _CCCL_DEVICE_API void
  merge(key_t* keys, const int* offsets, int runs, key_t (&out)[items_per_thread], CompareOp compare_op)
  {
    const int total = offsets[runs];

    if (runs == 1)
    {
      MergePolicy::load(keys, linear_tid, total, out);
      return;
    }
    _CCCL_PRAGMA_UNROLL_FULL()
    for (unsigned int item = 0; item < items_per_thread; ++item)
    {
      out[item] = key_t{};
    }

    const int effective      = layout_t::effective_runs(runs);
    const int log_runs       = 31 - ::__clz(static_cast<unsigned int>(effective));
    const int fit            = this->pair_fit_mask(offsets, runs, effective, log_runs);
    const auto thread_window = layout_t::thread_window(linear_tid, total);

    for (int span = 1, round = 0; span < effective; span *= 2, ++round)
    {
      const bool last   = 2 * span >= effective;
      const bool paired = !last && ((fit >> round) & 1) != 0;
      // Keep paired-round staging separate from the blocked register output.
      key_t staged[pair_capacity];
      window_t pair_window{0, 0};
      if (paired)
      {
        pair_window = this->stage_pair(keys, offsets, runs, span, round, log_runs, staged, compare_op);
      }
      else
      {
        this->merge_thread(keys, offsets, runs, span, round, thread_window, out, compare_op);
      }
      if (last)
      {
        return;
      }
      ::__syncthreads(); // Finish all reads before overwriting the shared input.
      if (paired)
      {
        MergePolicy::store(keys, pair_window, staged);
      }
      else
      {
        MergePolicy::store(keys, thread_window, out);
      }
      ::__syncthreads(); // Publish this round before the next one reads it.
    }
  }

  template <typename CompareOp>
  [[nodiscard]] _CCCL_DEVICE_API const key_t*
  merge_to_shared(key_t* keys, const int* offsets, int runs, CompareOp compare_op)
  {
    key_t out[items_per_thread];
    this->merge(keys, offsets, runs, out, compare_op);
    ::__syncthreads();
    MergePolicy::store(keys, layout_t::thread_window(linear_tid, offsets[runs]), out);
    ::__syncthreads();
    return keys;
  }
};
} // namespace detail
CUB_NAMESPACE_END
