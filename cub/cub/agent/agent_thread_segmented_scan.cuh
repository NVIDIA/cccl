// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::detail::segmented_scan::AgentSegmentedScan implements a stateful abstraction of CUDA thread
//! blocks for participating in device-wide prefix segmented scan.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/segmented_scan_helpers.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_reduce.cuh> // ThreadReduce
#include <cub/thread/thread_scan.cuh> // detail::ThreadInclusiveScan
#include <cub/util_arch.cuh> // detail::MemBoundScaling
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cub/warp/warp_store.cuh>

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/span>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
// define policy
//   Policy contains: CacheLoadModifier, block size, items per thread

template <int Nominal4ByteBlockThreads,
          int Nominal4BytesItemsPerThread,
          typename ComputeT,
          CacheLoadModifier LoadModifier,
          typename ScalingType = detail::MemBoundScaling<Nominal4ByteBlockThreads, Nominal4BytesItemsPerThread, ComputeT>>
struct agent_thread_segmented_scan_policy_t : ScalingType
{
  static constexpr CacheLoadModifier load_modifier = LoadModifier;
};

// helper

template <typename InpBeginOffsetIt, typename InpEndOffsetIt, typename OutBeginOffsetIt, typename OffsetTy, typename MapperF>
struct multi_segmented_seq_iterator
{
private:
  int segment_counter{};
  int max_segment_counter{};
  OffsetTy inp_offset{};
  OffsetTy out_offset{};
  OffsetTy segment_id{};
  OffsetTy segment_beg_offset{};
  OffsetTy segment_end_offset{};
  const InpBeginOffsetIt& inp_begin_offset_it;
  const InpEndOffsetIt& inp_end_offset_it;
  const OutBeginOffsetIt& out_begin_offset_it;
  OffsetTy n_segments;
  MapperF work_mapper_fn;
  bool is_head = false;

  _CCCL_DEVICE _CCCL_FORCEINLINE void set(bool head_flag)
  {
    if (segment_counter < max_segment_counter)
    {
      segment_id = work_mapper_fn(segment_counter);
      if (segment_id < n_segments)
      {
        segment_beg_offset = inp_begin_offset_it[segment_id];
        segment_end_offset = (::cuda::std::max<OffsetTy>) (inp_end_offset_it[segment_id], segment_beg_offset);
        inp_offset         = segment_beg_offset;
        out_offset         = out_begin_offset_it[segment_id];
        is_head            = head_flag;
      }
    }
  }

public:
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE multi_segmented_seq_iterator(
    int max_segment_counter,
    const InpBeginOffsetIt& inp_begin_offset_it,
    const InpEndOffsetIt& inp_end_offset_it,
    const OutBeginOffsetIt& out_begin_offset_it,
    OffsetTy n_segments,
    MapperF mapper_fn)
      : max_segment_counter(max_segment_counter)
      , inp_begin_offset_it(inp_begin_offset_it)
      , inp_end_offset_it(inp_end_offset_it)
      , out_begin_offset_it(out_begin_offset_it)
      , n_segments(n_segments)
      , work_mapper_fn(mapper_fn)
  {
    set(false);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void print() const
  {
    printf(
      "segment_counter = %d, max_segment_counter=%d, segment_id = %u, inp_offset=%u, out_offset=%u, n_segments=%u\n",
      segment_counter,
      max_segment_counter,
      static_cast<unsigned int>(segment_id),
      static_cast<unsigned int>(inp_offset),
      static_cast<unsigned int>(out_offset),
      static_cast<unsigned int>(n_segments));
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE multi_segmented_seq_iterator& operator++()
  {
    auto next_val = inp_offset + 1;
    if (next_val < segment_end_offset)
    {
      inp_offset = next_val;
      ++out_offset;
      is_head = false;
    }
    else
    {
      ++segment_counter;
      set(true);
    }

    return *this;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE operator bool() const
  {
    return (segment_counter < max_segment_counter) && (segment_id < n_segments);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool get_head_flag() const
  {
    return is_head;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE OffsetTy get_input_offset() const
  {
    return inp_offset;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE OffsetTy get_output_offset() const
  {
    return out_offset;
  }
};

template <typename InpBeginOffsetIt, typename InpEndOffsetIt, typename OutBeginOffsetIt, typename OffsetTy, typename MapperF>
multi_segmented_seq_iterator(int, InpBeginOffsetIt, InpEndOffsetIt, OutBeginOffsetIt, OffsetTy, MapperF)
  -> multi_segmented_seq_iterator<InpBeginOffsetIt, InpEndOffsetIt, OutBeginOffsetIt, OffsetTy, MapperF>;

// define agent code
//    agent consumes a fixed number of segments
//    for every segment, thread loads IPT elements in memory, does thread scan, writes out

template <typename AgentSegmentedScanPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorInputT,
          typename EndOffsetIteratorInputT,
          typename BeginOffsetIteratorOutputT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive = false>
struct agent_thread_segmented_scan
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  using input_t = cub::detail::it_value_t<InputIteratorT>;

  // Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator
  // or directly use the supplied input iterator type
  using wrapped_input_iterator_t =
    ::cuda::std::_If<::cuda::std::is_pointer_v<InputIteratorT>,
                     CacheModifiedInputIterator<AgentSegmentedScanPolicyT::load_modifier, input_t, OffsetT>,
                     InputIteratorT>;

  // Constants

  // Use cub::NullType means no initial value is provided
  static constexpr bool has_init = !::cuda::std::is_same_v<InitValueT, NullType>;
  // We are relying on either initial value not being `NullType`
  // or the ForceInclusive tag to be true for inclusive scan
  // to get picked up.
  static constexpr bool is_inclusive    = ForceInclusive || !has_init;
  static constexpr int block_threads    = AgentSegmentedScanPolicyT::BLOCK_THREADS;
  static constexpr int items_per_thread = AgentSegmentedScanPolicyT::ITEMS_PER_THREAD;
  static constexpr int tile_items       = items_per_thread;

  using augmented_accum_t = AccumT;

  struct _TempStorage
  {};

  // Alias wrapper allowing storage to be unioned
  using TempStorage = Uninitialized<_TempStorage>;

  _TempStorage& temp_storage; ///< Reference to temp_storage
  wrapped_input_iterator_t d_in; ///< Input data
  OutputIteratorT d_out; ///< Output data
  BeginOffsetIteratorInputT d_inp_begin_offset; ///< Offset to beginning of input segments
  EndOffsetIteratorInputT d_inp_end_offset; ///< Offsets to end of input segments
  BeginOffsetIteratorOutputT d_out_begin_offset; ///< Offset to beginning of input segments
  OffsetT n_segments;
  ScanOpT scan_op; ///< Binary associative scan operator
  InitValueT initial_value; ///< The initial value element for ScanOpT

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_thread_segmented_scan(
    TempStorage& temp_storage,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_inp_begin_offset,
    EndOffsetIteratorInputT d_inp_end_offset,
    BeginOffsetIteratorOutputT d_out_begin_offset,
    OffsetT num_segments,
    ScanOpT scan_op,
    InitValueT initial_value)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_out(d_out)
      , d_inp_begin_offset(d_inp_begin_offset)
      , d_inp_end_offset(d_inp_end_offset)
      , d_out_begin_offset(d_out_begin_offset)
      , n_segments(num_segments)
      , scan_op(scan_op)
      , initial_value(initial_value)
  {}

  //! @brief
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_range_one_segment_at_a_time(int segments_per_thread)
  {
    constexpr auto segments_per_block = static_cast<OffsetT>(block_threads * segments_per_thread);
    const OffsetT thread_work_id0 =
      static_cast<OffsetT>(blockIdx.x) * segments_per_block + static_cast<OffsetT>(threadIdx.x);

    for (int segment_id = 0; segment_id < segments_per_thread; ++segment_id)
    {
      const OffsetT work_id = thread_work_id0 + static_cast<OffsetT>(segment_id * block_threads);
      if (work_id < n_segments)
      {
        const OffsetT segment_beg_offset = d_inp_begin_offset[work_id];
        const OffsetT segment_end_offset = (::cuda::std::max<OffsetT>) (d_inp_end_offset[work_id], segment_beg_offset);
        const OffsetT out_beg_offset     = d_out_begin_offset[work_id];
        const OffsetT segment_size       = segment_end_offset - segment_beg_offset;

        constexpr auto max_chunk_size = static_cast<OffsetT>(items_per_thread);
        const OffsetT num_chunks      = ::cuda::ceil_div(segment_size, max_chunk_size);

        augmented_accum_t exclusive_prefix;
        augmented_accum_t items[items_per_thread];
        for (OffsetT chunk_id = 0; chunk_id < num_chunks; ++chunk_id)
        {
          const OffsetT inp_offset = segment_beg_offset + chunk_id * max_chunk_size;
          const OffsetT chunk_size = (::cuda::std::min) (inp_offset + max_chunk_size, segment_end_offset) - inp_offset;

          const bool entire_tile = (chunk_size == max_chunk_size);
          // load data
          if (entire_tile)
          {
            _CCCL_PRAGMA_UNROLL()
            for (int k = 0; k < items_per_thread; ++k)
            {
              items[k] = d_in[inp_offset + k];
            }
          }
          else
          {
            _CCCL_PRAGMA_UNROLL()
            for (int k = 0; k < items_per_thread; ++k)
            {
              items[k] = (k < chunk_size) ? d_in[inp_offset + k] : augmented_accum_t{};
            }
          };

          // compute scan
          if (chunk_id == 0)
          {
            scan_first_tile(items, initial_value, scan_op, exclusive_prefix);
          }
          else
          {
            scan_later_tile(items, scan_op, exclusive_prefix);
          }

          // store data
          const OffsetT out_offset = out_beg_offset + chunk_id * max_chunk_size;
          if (entire_tile)
          {
            _CCCL_PRAGMA_UNROLL()
            for (int k = 0; k < items_per_thread; ++k)
            {
              d_out[out_offset + k] = items[k];
            }
          }
          else
          {
            for (int k = 0; k < chunk_size; ++k)
            {
              d_out[out_offset + k] = items[k];
            }
          };
        }
      }
    }
  }

  //! @brief
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_range_multi_segment(int segments_per_thread)
  {
    // if (threadIdx.x == 0 && blockIdx.x == 0)
    // {
    //   printf("segments_per_thread = %d, itp: %d\n", segments_per_thread, items_per_thread);
    // }

    const auto segments_per_block = static_cast<OffsetT>(block_threads * segments_per_thread);
    const OffsetT thread_work_id0 =
      static_cast<OffsetT>(blockIdx.x) * segments_per_block + static_cast<OffsetT>(threadIdx.x);

    auto get_work_id = [&](int segment_id) {
      return thread_work_id0 + static_cast<OffsetT>(segment_id * block_threads);
    };

    multi_segmented_seq_iterator it{
      segments_per_thread, d_inp_begin_offset, d_inp_end_offset, d_out_begin_offset, n_segments, get_work_id};

    using augmented_scan_op_t = multi_segment_helpers::schwarz_scan_op<AccumT, bool, ScanOpT>;
    using hv_t                = typename augmented_scan_op_t::fv_t;

    augmented_scan_op_t augmented_scan_op{scan_op};

    hv_t exclusive_prefix;
    hv_t items[items_per_thread];
    bool flags[items_per_thread];
    OffsetT out_offsets[items_per_thread];

    OffsetT chunk_id = 0;
    while (it)
    {
      int chunk_size = 0;
#pragma unroll
      for (int k = 0; k < items_per_thread; ++k)
      {
        if (it)
        {
          const OffsetT inp_offset = it.get_input_offset();
          flags[k]                 = it.get_head_flag();
          items[k]                 = multi_segment_helpers::make_value_flag(d_in[inp_offset], flags[k]);
          out_offsets[k]           = it.get_output_offset();
          ++chunk_size;
          ++it;
        }
        else
        {
          items[k] = multi_segment_helpers::make_value_flag(AccumT{}, true);
        }
      }

      // compute scan
      if (chunk_id == 0)
      {
        using augmented_init_value_t                = multi_segment_helpers::augmented_value_t<InitValueT, bool>;
        augmented_init_value_t augmented_init_value = multi_segment_helpers::make_value_flag(initial_value, false);
        scan_first_tile(items, augmented_init_value, augmented_scan_op, exclusive_prefix);
      }
      else
      {
        scan_later_tile(items, augmented_scan_op, exclusive_prefix);
      }

      // store
#pragma unroll
      for (int k = 0; k < items_per_thread; ++k)
      {
        if (k < chunk_size)
        {
          const OffsetT oo          = out_offsets[k];
          const augmented_accum_t v = multi_segment_helpers::get_value(items[k]);
          if constexpr (is_inclusive)
          {
            d_out[oo] = v;
          }
          else
          {
            d_out[oo] = (flags[k]) ? static_cast<augmented_accum_t>(initial_value) : v;
          }
        }
      }

      ++chunk_id;
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_range(int segments_per_thread)
  {
    consume_range_multi_segment(segments_per_thread);
    // consume_range_one_segment_at_a_time();
  }

private:
  template <typename ItemTy,
            typename InitValueTy,
            typename ScanOpTy,
            bool IsInclusive = is_inclusive,
            bool HasInit     = has_init>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  scan_first_tile(ItemTy (&items)[items_per_thread], InitValueTy init_value, ScanOpTy scan_op, ItemTy& aggregate)
  {
    ItemTy init_v;
    aggregate = cub::ThreadReduce(items, scan_op);
    if constexpr (HasInit)
    {
      init_v    = static_cast<ItemTy>(init_value);
      aggregate = scan_op(init_v, aggregate);
    }
    else
    {
      init_v = ItemTy{};
    }
    if constexpr (IsInclusive)
    {
      detail::ThreadScanInclusive(items, items, scan_op, init_v, has_init);
    }
    else
    {
      detail::ThreadScanExclusive(items, items, scan_op, init_v, has_init);
    }
  }

  template <typename ItemTy, typename ScanOpTy, bool IsInclusive = is_inclusive>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  scan_later_tile(ItemTy (&items)[items_per_thread], ScanOpTy scan_op, ItemTy& exclusive_prefix)
  {
    const ItemTy thread_aggregate = cub::ThreadReduce(items, scan_op);

    const ItemTy& init_v = exclusive_prefix;
    if constexpr (IsInclusive)
    {
      detail::ThreadScanInclusive(items, items, scan_op, init_v);
    }
    else
    {
      detail::ThreadScanExclusive(items, items, scan_op, init_v);
    }
    exclusive_prefix = scan_op(exclusive_prefix, thread_aggregate);
  }
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
