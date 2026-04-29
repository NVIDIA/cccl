// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Implement kernel for DeviceSegmentedScan with threads processing individual segments.

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
#include <cub/device/dispatch/tuning/tuning_segmented_scan.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_reduce.cuh> // ThreadReduce
#include <cub/thread/thread_scan.cuh> // detail::ThreadInclusiveScan
#include <cub/util_macro.cuh>
#include <cub/util_type.cuh>

#include <cuda/iterator>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/span>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
// helper

template <typename InpBeginOffsetIt, typename InpEndOffsetIt, typename OutBeginOffsetIt, typename OffsetT, typename MapperF>
struct multi_segmented_seq_iterator
{
  static_assert(::cuda::std::is_convertible_v<cub::detail::it_reference_t<InpBeginOffsetIt>, OffsetT>,
                "Iterator values for input sequence begin offsets should be convertible to offset type");
  static_assert(::cuda::std::is_convertible_v<cub::detail::it_reference_t<InpEndOffsetIt>, OffsetT>,
                "Iterator values for input sequence end offsets should be convertible to offset type");
  static_assert(::cuda::std::is_convertible_v<cub::detail::it_reference_t<OutBeginOffsetIt>, OffsetT>,
                "Iterator values for output sequence begin offsets should be convertible to offset type");

private:
  int m_segment_counter     = 0;
  int m_max_segment_counter = 0; // read-only value
  OffsetT m_input_idx{};
  OffsetT m_output_idx{};
  OffsetT m_segment_id{};
  OffsetT m_segment_begin_idx{};
  OffsetT m_segment_end_idx{};
  const InpBeginOffsetIt& m_input_begin_idx_it; // read-only value
  const InpEndOffsetIt& m_input_end_idx_it; // read-only value
  const OutBeginOffsetIt& m_output_begin_idx_it; // read-only value
  OffsetT m_num_segments; // read-only value
  MapperF m_work_mapper_fn; // read-only value
  bool m_is_head = false;

  _CCCL_DEVICE _CCCL_FORCEINLINE void set_first_nonempty(bool head_flag)
  {
    for (; m_segment_counter < m_max_segment_counter; ++m_segment_counter)
    {
      m_segment_id = m_work_mapper_fn(m_segment_counter);
      if (m_segment_id < m_num_segments)
      {
        const OffsetT input_begin_idx = m_input_begin_idx_it[m_segment_id];
        const OffsetT input_end_idx   = m_input_end_idx_it[m_segment_id];

        if (input_end_idx > input_begin_idx)
        {
          m_segment_begin_idx = input_begin_idx;
          m_segment_end_idx   = input_end_idx;

          m_input_idx  = m_segment_begin_idx;
          m_output_idx = m_output_begin_idx_it[m_segment_id];
          m_is_head    = head_flag;
          return;
        }
      }
    }
  }

public:
  _CCCL_DEVICE _CCCL_FORCEINLINE multi_segmented_seq_iterator(
    int max_segment_counter,
    const InpBeginOffsetIt& input_begin_idx_it,
    const InpEndOffsetIt& input_end_idx_it,
    const OutBeginOffsetIt& output_begin_idx_it,
    OffsetT n_segments,
    MapperF mapper_fn)
      : m_max_segment_counter(max_segment_counter)
      , m_input_begin_idx_it(input_begin_idx_it)
      , m_input_end_idx_it(input_end_idx_it)
      , m_output_begin_idx_it(output_begin_idx_it)
      , m_num_segments(n_segments)
      , m_work_mapper_fn(mapper_fn)
  {
    set_first_nonempty(false);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE multi_segmented_seq_iterator& operator++()
  {
    auto next_val = m_input_idx + 1;
    if (next_val < m_segment_end_idx)
    {
      m_input_idx = next_val;
      ++m_output_idx;
      m_is_head = false;
    }
    else
    {
      ++m_segment_counter;
      set_first_nonempty(true);
    }

    return *this;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE operator bool() const
  {
    return (m_segment_counter < m_max_segment_counter) && (m_segment_id < m_num_segments);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE bool get_head_flag() const
  {
    return m_is_head;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE OffsetT get_input_offset() const
  {
    return m_input_idx;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE OffsetT get_output_offset() const
  {
    return m_output_idx;
  }
};

template <typename InpBeginOffsetIt, typename InpEndOffsetIt, typename OutBeginOffsetIt, typename OffsetTy, typename MapperF>
_CCCL_DEVICE multi_segmented_seq_iterator(int, InpBeginOffsetIt, InpEndOffsetIt, OutBeginOffsetIt, OffsetTy, MapperF)
  -> multi_segmented_seq_iterator<InpBeginOffsetIt, InpEndOffsetIt, OutBeginOffsetIt, OffsetTy, MapperF>;

// define agent code
//    agent consumes a fixed number of segments
//    for every segment, thread loads IPT elements in memory, does thread scan, writes out

template <typename SegmentedScanPolicyGetterT,
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
private:
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  using input_t = cub::detail::it_value_t<InputIteratorT>;

  static constexpr auto agent_policy = SegmentedScanPolicyGetterT{}().thread;

  // Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator
  // or directly use the supplied input iterator type
  using wrapped_input_iterator_t =
    ::cuda::std::conditional_t<::cuda::std::is_pointer_v<InputIteratorT>,
                               CacheModifiedInputIterator<agent_policy.load_modifier, input_t, OffsetT>,
                               InputIteratorT>;

  // Constants

  // Use cub::NullType means no initial value is provided
  static constexpr bool has_init = !::cuda::std::is_same_v<InitValueT, NullType>;
  // We are relying on either initial value not being `NullType`
  // or the ForceInclusive tag to be true for inclusive scan
  // to get picked up.
  static constexpr bool is_inclusive     = ForceInclusive || !has_init;
  static constexpr int threads_per_block = agent_policy.threads_per_block;
  static constexpr int items_per_thread  = agent_policy.items_per_thread;
  static constexpr int tile_items        = items_per_thread;

  using augmented_accum_t = AccumT;

  struct _TempStorage
  {};

  wrapped_input_iterator_t d_in; ///< Input data
  OutputIteratorT d_out; ///< Output data
  BeginOffsetIteratorInputT d_input_begin_idx; ///< Offset to beginning of input segments
  EndOffsetIteratorInputT d_input_end_idx; ///< Offsets to end of input segments
  BeginOffsetIteratorOutputT d_output_begin_idx; ///< Offset to beginning of output segments
  OffsetT n_segments;
  ScanOpT scan_op; ///< Binary associative scan operator
  InitValueT initial_value; ///< The initial value element for ScanOpT

public:
  // Alias wrapper allowing storage to be unioned
  using TempStorage = Uninitialized<_TempStorage>;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_thread_segmented_scan(
    TempStorage&,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_input_begin_idx,
    EndOffsetIteratorInputT d_input_end_idx,
    BeginOffsetIteratorOutputT d_output_begin_idx,
    OffsetT num_segments,
    ScanOpT scan_op,
    InitValueT initial_value)
      : d_in(d_in)
      , d_out(d_out)
      , d_input_begin_idx(d_input_begin_idx)
      , d_input_end_idx(d_input_end_idx)
      , d_output_begin_idx(d_output_begin_idx)
      , n_segments(num_segments)
      , scan_op(scan_op)
      , initial_value(initial_value)
  {}

private:
  //! @brief consume given number of segments per thread using double-nested loop, i.e.,
  //! iterating over segments, and then iterating within the segment in chunks of items_per_thread
  //! This approach is not efficient when segment size is smaller than items_per_thread.
  _CCCL_DEVICE _CCCL_FORCEINLINE void scan_segments_one_segment_at_a_time(int segments_per_thread)
  {
    const auto segments_per_block = static_cast<OffsetT>(threads_per_block * segments_per_thread);
    const OffsetT thread_work_id0 =
      static_cast<OffsetT>(blockIdx.x) * segments_per_block + static_cast<OffsetT>(threadIdx.x);

    for (int segment_id = 0; segment_id < segments_per_thread; ++segment_id)
    {
      const OffsetT work_id = thread_work_id0 + static_cast<OffsetT>(segment_id * threads_per_block);
      if (work_id < n_segments)
      {
        const OffsetT input_begin_idx  = d_input_begin_idx[work_id];
        const OffsetT input_end_idx    = (::cuda::std::max<OffsetT>) (d_input_end_idx[work_id], input_begin_idx);
        const OffsetT output_begin_idx = d_output_begin_idx[work_id];
        const OffsetT segment_size     = input_end_idx - input_begin_idx;

        constexpr auto max_chunk_size = static_cast<OffsetT>(items_per_thread);
        const OffsetT num_chunks      = ::cuda::ceil_div(segment_size, max_chunk_size);

        augmented_accum_t exclusive_prefix;
        augmented_accum_t items[items_per_thread];
        for (OffsetT chunk_id = 0; chunk_id < num_chunks; ++chunk_id)
        {
          const OffsetT chunk_begin = input_begin_idx + chunk_id * max_chunk_size;
          const OffsetT chunk_end   = chunk_begin + max_chunk_size;
          const OffsetT chunk_size  = (::cuda::std::min) (chunk_end, input_end_idx) - chunk_begin;

          const bool entire_tile = (chunk_size == max_chunk_size);
          // load data
          if (entire_tile)
          {
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int k = 0; k < items_per_thread; ++k)
            {
              items[k] = d_in[chunk_begin + k];
            }
          }
          else
          {
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int k = 0; k < items_per_thread; ++k)
            {
              items[k] = (k < chunk_size) ? d_in[chunk_begin + k] : augmented_accum_t{};
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
          const OffsetT output_idx = output_begin_idx + chunk_id * max_chunk_size;
          if (entire_tile)
          {
            _CCCL_PRAGMA_UNROLL()
            for (int k = 0; k < items_per_thread; ++k)
            {
              d_out[output_idx + k] = items[k];
            }
          }
          else
          {
            _CCCL_PRAGMA_UNROLL()
            for (int k = 0; k < items_per_thread; ++k)
            {
              if (k < chunk_size)
              {
                d_out[output_idx + k] = items[k];
              }
            }
          };
        }
      }
    }
  }

  //! @brief consume given number of segments per thread using multi_segment processing and augmented
  //! scan operation
  _CCCL_DEVICE _CCCL_FORCEINLINE void scan_segments_multi_segment(int segments_per_thread)
  {
    const auto segments_per_block = static_cast<OffsetT>(threads_per_block * segments_per_thread);
    const OffsetT thread_work_id0 =
      static_cast<OffsetT>(blockIdx.x) * segments_per_block + static_cast<OffsetT>(threadIdx.x);

    auto get_work_id = [&](int segment_id) {
      return thread_work_id0 + static_cast<OffsetT>(segment_id * threads_per_block);
    };

    multi_segmented_seq_iterator it{
      segments_per_thread, d_input_begin_idx, d_input_end_idx, d_output_begin_idx, n_segments, get_work_id};

    using augmented_scan_op_t = schwarz_scan_op<ScanOpT, AccumT>;
    using hv_t                = typename augmented_scan_op_t::fv_t;

    static_assert(::cuda::std::is_same_v<hv_t, augmented_value_t<AccumT>>);

    augmented_scan_op_t augmented_scan_op{scan_op};

    hv_t exclusive_prefix;
    hv_t items[items_per_thread];
    bool flags[items_per_thread];
    OffsetT out_offsets[items_per_thread];

    OffsetT chunk_id = 0;
    while (it)
    {
      int chunk_size = 0;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int k = 0; k < items_per_thread; ++k)
      {
        if (it)
        {
          const OffsetT input_idx = it.get_input_offset();
          flags[k]                = it.get_head_flag();
          const AccumT v          = d_in[input_idx];
          if constexpr (has_init)
          {
            items[k] = augmented_value_t{(flags[k]) ? scan_op(initial_value, v) : v, flags[k]};
          }
          else
          {
            items[k] = augmented_value_t{v, flags[k]};
          }
          out_offsets[k] = it.get_output_offset();
          ++chunk_size;
          ++it;
        }
        else
        {
          items[k] = augmented_value_t{AccumT{}, true};
        }
      }

      // compute scan
      if (chunk_id == 0)
      {
        const auto augmented_init_value = augmented_value_t{initial_value, false};
        // Initialize exclusive_prefix
        scan_first_tile(items, augmented_init_value, augmented_scan_op, exclusive_prefix);
      }
      else
      {
        scan_later_tile(items, augmented_scan_op, exclusive_prefix);
      }

      // store
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int k = 0; k < items_per_thread; ++k)
      {
        if (k < chunk_size)
        {
          const OffsetT output_idx  = out_offsets[k];
          const augmented_accum_t v = items[k].value;
          if constexpr (is_inclusive)
          {
            d_out[output_idx] = v;
          }
          else
          {
            static_assert(has_init);
            d_out[output_idx] = (flags[k]) ? static_cast<augmented_accum_t>(initial_value) : v;
          }
        }
      }

      ++chunk_id;
    }
  }

public:
  _CCCL_DEVICE _CCCL_FORCEINLINE void scan_segments(int segments_per_thread)
  {
    if (segments_per_thread == 1)
    {
      scan_segments_one_segment_at_a_time(1);
    }
    else
    {
      scan_segments_multi_segment(segments_per_thread);
    }
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
      init_v    = convert_initial_value<ItemTy>(init_value);
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

template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorInputT,
          typename EndOffsetIteratorInputT,
          typename BeginOffsetIteratorOutputT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive,
          typename ActualInitValueT = typename InitValueT::value_type>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_scan_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(current_policy<PolicySelector>().thread.threads_per_block)
  _CCCL_KERNEL_ATTRIBUTES void device_thread_segmented_scan_kernel(
    _CCCL_GRID_CONSTANT const InputIteratorT d_in,
    _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
    _CCCL_GRID_CONSTANT const BeginOffsetIteratorInputT begin_offset_d_in,
    _CCCL_GRID_CONSTANT const EndOffsetIteratorInputT end_offset_d_in,
    _CCCL_GRID_CONSTANT const BeginOffsetIteratorOutputT begin_offset_d_out,
    _CCCL_GRID_CONSTANT const OffsetT n_segments,
    _CCCL_GRID_CONSTANT const ScanOpT scan_op,
    _CCCL_GRID_CONSTANT const InitValueT init_value,
    _CCCL_GRID_CONSTANT const int num_segments_per_worker)
{
  static constexpr auto policy = current_policy<PolicySelector>();
  static_assert(policy.thread.load_modifier != CacheLoadModifier::LOAD_LDG,
                "The memory consistency model does not apply to texture accesses");

  struct policy_getter
  {
    constexpr auto operator()() const
    {
      return policy;
    }
  };

  using agent_t = agent_thread_segmented_scan<
    policy_getter,
    InputIteratorT,
    OutputIteratorT,
    BeginOffsetIteratorInputT,
    EndOffsetIteratorInputT,
    BeginOffsetIteratorOutputT,
    OffsetT,
    ScanOpT,
    ActualInitValueT,
    AccumT,
    ForceInclusive>;

  __shared__ typename agent_t::TempStorage temp_storage;

  const ActualInitValueT _init_value = init_value;

  _CCCL_ASSERT(num_segments_per_worker > 0, "Number of segments to be processed by thread must be positive");

  // Agent consumes interleaved segments to improve CTA' memory access locality

  agent_t agent(
    temp_storage, d_in, d_out, begin_offset_d_in, end_offset_d_in, begin_offset_d_out, n_segments, scan_op, _init_value);
  agent.scan_segments(num_segments_per_worker);
}
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
