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

#include <cuda/std/__algorithm/min.h>

CUB_NAMESPACE_BEGIN
namespace detail
{
template <typename MergePolicy>
class block_merge_ping_pong
{
  using key_t _CCCL_NODEBUG    = typename MergePolicy::key_t;
  using layout_t _CCCL_NODEBUG = typename MergePolicy::layout_t;
  using window_t _CCCL_NODEBUG = typename MergePolicy::window_t;

  template <typename CompareOp>
  using cursor_t _CCCL_NODEBUG = typename MergePolicy::template cursor_t<CompareOp>;

  static constexpr unsigned int items_per_thread = MergePolicy::items_per_thread;

public:
  using storage_t _CCCL_NODEBUG = typename MergePolicy::template storage_t<layout_t::tile_size + 1>;

private:
  storage_t& temp_storage;
  const unsigned int linear_tid;

  template <typename CompareOp>
  _CCCL_DEVICE_API void merge_pair(
    const key_t* input,
    key_t* output,
    const int* offsets,
    int runs,
    int span,
    int round,
    int log_runs,
    CompareOp compare_op)
  {
    const int log_threads = layout_t::pair_thread_log(log_runs, round);
    const int pair_id     = static_cast<int>(linear_tid) >> log_threads;
    const int lane        = static_cast<int>(linear_tid) & ((1 << log_threads) - 1);
    const auto pair       = layout_t::pair_bounds(offsets, runs, pair_id, span);
    const auto window     = layout_t::pair_window(pair.end - pair.begin, log_threads, lane);
    const int count       = window.end - window.begin;

    if (count == 0)
    {
      return;
    }

    cursor_t<CompareOp> cursor(input, pair, window.begin, compare_op);
    constexpr int capacity = static_cast<int>(items_per_thread) + 2;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < capacity; ++item)
    {
      if (item < count)
      {
        output[pair.begin + window.begin + item] = cursor.next();
      }
    }
    // An oversized pair can exceed the small unrolled prefix.
    for (int item = capacity; item < count; ++item)
    {
      output[pair.begin + window.begin + item] = cursor.next();
    }
  }

  template <typename CompareOp>
  _CCCL_DEVICE_API void merge_thread(
    const key_t* input,
    key_t* output,
    const int* offsets,
    int runs,
    int span,
    int round,
    window_t window,
    CompareOp compare_op)
  {
    int position = window.begin;

    while (position < window.end)
    {
      const auto pair = layout_t::find_pair(position, offsets, runs, span, round);
      const int count = (::cuda::std::min) (window.end, pair.end) - position;
      cursor_t<CompareOp> cursor(input, pair, position - pair.begin, compare_op);

      _CCCL_PRAGMA_UNROLL_FULL()
      for (unsigned int item = 0; item < items_per_thread; ++item)
      {
        if (static_cast<int>(item) < count)
        {
          output[position + item] = cursor.next();
        }
      }
      position += count;
    }
  }

public:
  _CCCL_DEVICE_API block_merge_ping_pong(storage_t& temp_storage, unsigned int tid)
      : temp_storage(temp_storage)
      , linear_tid(tid)
  {}

  template <typename CompareOp>
  [[nodiscard]] _CCCL_DEVICE_API const key_t*
  merge_to_shared(key_t* keys, const int* offsets, int runs, CompareOp compare_op)
  {
    const int total     = offsets[runs];
    const int effective = layout_t::effective_runs(runs);
    const int log_runs  = 31 - ::__clz(static_cast<unsigned int>(effective));
    const bool paired   = layout_t::use_pairs(effective);
    const auto window   = layout_t::thread_window(linear_tid, total);
    key_t* input        = keys;
    key_t* output       = temp_storage.keys;

    if (linear_tid == 0)
    {
      output[total] = key_t{};
    }
    for (int span = 1, round = 0; span < effective; span *= 2, ++round)
    {
      if (paired)
      {
        this->merge_pair(input, output, offsets, runs, span, round, log_runs, compare_op);
      }
      else
      {
        this->merge_thread(input, output, offsets, runs, span, round, window, compare_op);
      }
      // All reads from input and writes to output finish before exchanging them.
      ::__syncthreads();
      key_t* previous = input;

      input  = output;
      output = previous;
    }
    return input;
  }

  template <typename CompareOp>
  _CCCL_DEVICE_API void
  merge(key_t* keys, const int* offsets, int runs, key_t (&out)[items_per_thread], CompareOp compare_op)
  {
    const key_t* result = this->merge_to_shared(keys, offsets, runs, compare_op);

    MergePolicy::load(result, linear_tid, offsets[runs], out);
  }
};
} // namespace detail
CUB_NAMESPACE_END
