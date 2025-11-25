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

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

//! @brief Parameterizable tuning policy type for agent_segmented_scan
//!
//! @tparam Nominal4ByteBlockThreads
//!   Threads per thread block
//!
//! @tparam Nominal4BytesItemsPerThread
//!   Items per thread (per tile of input)
//!
//! @tparam ComputeT
//!   Dominant compute type
//!
//! @tparam LoadAlgorithm
//!   The BlockLoad algorithm to use
//!
//! @tparam LoadModifier
//!   Cache load modifier for reading input elements
//!
//! @tparam StoreAlgorithm
//!   The BlockStore algorithm to use
//!
//! @tparam ScanAlgorithm
//!   The BlockScan algorithm to use
//!
//! @tparam SegmentsPerBlock
//!   The number of segments processed per block
//!
template <
  int Nominal4ByteBlockThreads,
  int Nominal4BytesItemsPerThread,
  typename ComputeT,
  BlockLoadAlgorithm LoadAlgorithm,
  CacheLoadModifier LoadModifier,
  BlockStoreAlgorithm StoreAlgorithm,
  BlockScanAlgorithm ScanAlgorithm,
  int SegmentsPerBlock = 1,
  typename ScalingType = detail::MemBoundScaling<Nominal4ByteBlockThreads, Nominal4BytesItemsPerThread, ComputeT>>
struct agent_segmented_scan_policy_t : ScalingType
{
  static_assert(SegmentsPerBlock > 0, "SegmentsPerBlock template value parameter must be positive");

  static constexpr BlockLoadAlgorithm load_algorithm   = LoadAlgorithm;
  static constexpr CacheLoadModifier load_modifier     = LoadModifier;
  static constexpr BlockStoreAlgorithm store_algorithm = StoreAlgorithm;
  static constexpr BlockScanAlgorithm scan_algorithm   = ScanAlgorithm;
  static constexpr int segments_per_block              = SegmentsPerBlock;
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

//! @brief agent_segmented_scan implements a stateful abstraction of CUDA thread blocks for
//!        participating in device-wide segmented prefix scan.
//!
//! @tparam AgentSegmentedScanPolicyT
//!   Parameterized AgentSegmentedScanPolicyT tuning policy type
//!
//! @tparam InputIteratorT
//!   Random-access input iterator type
//!
//! @tparam OutputIteratorT
//!   Random-access output iterator type
//!
//! @tparam OffsetT
//!   Signed integer type for global offsets
//!
//! @tparam ScanOpT
//!   Scan functor type
//!
//! @tparam InitValueT
//!   The init_value element for ScanOpT type (cub::NullType for inclusive scan)
//!
//! @tparam AccumT
//!   The type of intermediate accumulator (according to P2322R6)
//!
template <typename AgentSegmentedScanPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive = false>
struct agent_segmented_scan
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
  static constexpr bool is_inclusive      = ForceInclusive || !has_init;
  static constexpr int block_threads      = AgentSegmentedScanPolicyT::BLOCK_THREADS;
  static constexpr int items_per_thread   = AgentSegmentedScanPolicyT::ITEMS_PER_THREAD;
  static constexpr int tile_items         = block_threads * items_per_thread;
  static constexpr int segments_per_block = AgentSegmentedScanPolicyT::segments_per_block;

  using augmented_accum_t =
    ::cuda::std::conditional_t<segments_per_block == 1, AccumT, ::cuda::std::tuple<bool, AccumT>>;

  using block_load_t  = BlockLoad<AccumT, block_threads, items_per_thread, AgentSegmentedScanPolicyT::load_algorithm>;
  using block_store_t = BlockStore<AccumT, block_threads, items_per_thread, AgentSegmentedScanPolicyT::store_algorithm>;
  using block_scan_t  = BlockScan<augmented_accum_t, block_threads, AgentSegmentedScanPolicyT::scan_algorithm>;

  union _TempStorage
  {
    typename block_load_t::TempStorage load;
    typename block_store_t::TempStorage store;
    typename block_scan_t::TempStorage scan;
  };

  // Alias wrapper allowing storage to be unioned
  using TempStorage = Uninitialized<_TempStorage>;

  _TempStorage& temp_storage; ///< Reference to temp_storage
  wrapped_input_iterator_t d_in; ///< Input data
  OutputIteratorT d_out; ///< Output data
  ScanOpT scan_op; ///< Binary associative scan operator
  InitValueT initial_value; ///< The initial value element for ScanOpT

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  //! @param temp_storage
  //!   Reference to temp_storage
  //!
  //! @param d_in
  //!   Input data
  //!
  //! @param d_out
  //!   Output data
  //!
  //! @param scan_op
  //!   Binary scan operator
  //!
  //! @param init_value
  //!   Initial value to seed the exclusive scan
  //!
  _CCCL_DEVICE _CCCL_FORCEINLINE agent_segmented_scan(
    TempStorage& temp_storage, InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op, InitValueT initial_value)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , initial_value(initial_value)
  {}

  //! @brief Scan one segment of values
  //!
  //! @param inp_idx_begin
  //!   Index of start of the segment in input array
  //!
  //! @param inp_idx_end
  //!  Index of end of the segment in input array
  //!
  //! @param out_idx_begin
  //!  Index of start of the segment's prefix scan result in the output array
  //!
  template <int NumSegments = segments_per_block, class = ::cuda::std::enable_if_t<(NumSegments == 1)>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_range(OffsetT inp_idx_begin, OffsetT inp_idx_end, OffsetT out_idx_begin)
  {
    const OffsetT segment_items = ::cuda::std::max(inp_idx_end, inp_idx_begin) - inp_idx_begin;
    const OffsetT n_chunks      = ::cuda::ceil_div(segment_items, tile_items);

    AccumT thread_values[items_per_thread] = {};

    AccumT exclusive_prefix{};
    block_prefix_callback_t<AccumT, ScanOpT> prefix_op{exclusive_prefix, scan_op};

    for (OffsetT chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
    {
      const OffsetT chunk_begin = inp_idx_begin + chunk_id * tile_items;
      const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, inp_idx_end);

      // chunk_size <= TILE_ITEMS, casting to int is safe
      const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

      block_load_t(temp_storage.load).Load(d_in + chunk_begin, thread_values, chunk_size);
      __syncthreads();

      if (chunk_id == 0)
      {
        // Initialize exlusive_prefix, referenced from prefix_op
        scan_first_tile(thread_values, initial_value, scan_op, exclusive_prefix);
      }
      else
      {
        scan_later_tile(thread_values, scan_op, prefix_op);
      }
      __syncthreads();

      block_store_t(temp_storage.store).Store(d_out + out_idx_begin + chunk_id * tile_items, thread_values, chunk_size);
      __syncthreads();
    }
  };

  //! @brief Scan one segment of values
  //!
  //! @param inp_idx_begin
  //!   Index of start of the segment in input array
  //!
  //! @param inp_idx_end
  //!  Index of end of the segment in input array
  //!
  //! @param out_idx_begin
  //!  Index of start of the segment's prefix scan result in the output array
  //!
  template <typename InputBeginOffsetIteratorT,
            typename OutputBeginOffsetIteratorT,
            int NumSegments = segments_per_block,
            class           = ::cuda::std::enable_if_t<(NumSegments > 1)>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_ranges(
    InputBeginOffsetIteratorT inp_idx_begin_it,
    OffsetT (&inp_idx_end)[NumSegments],
    OutputBeginOffsetIteratorT out_idx_begin_it)
  {
    OffsetT items_per_block{0};
    OffsetT(&cum_sizes)[NumSegments] = inp_idx_end;

    for (int i = 0; i < NumSegments; ++i)
    {
      const OffsetT input_segment_begin = inp_idx_begin_it[i];
      const OffsetT segment_items       = ::cuda::std::max(inp_idx_end[i], input_segment_begin) - input_segment_begin;
      items_per_block += segment_items;
      cum_sizes[i] = items_per_block;
    }
    const OffsetT n_chunks = ::cuda::ceil_div(items_per_block, tile_items);

    using augmented_init_value_t = ::cuda::std::tuple<bool, InitValueT>;
    using augmented_scan_op_t    = schwarz_scan_op<bool, AccumT, ScanOpT>;

    augmented_accum_t thread_flag_values[items_per_thread] = {};

    augmented_scan_op_t augmented_scan_op{scan_op};

    augmented_accum_t exclusive_prefix{};
    block_prefix_callback_t<augmented_accum_t, augmented_scan_op_t> prefix_op{exclusive_prefix, augmented_scan_op};

    for (OffsetT chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
    {
      const OffsetT chunk_begin = chunk_id * tile_items;
      const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, items_per_block);

      // chunk_size <= TILE_ITEMS, casting to int is safe
      const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

      // load values, and pack them into head_flag-value pairs
      {
        AccumT thread_values[items_per_thread] = {};
        multi_segmented_iterator it_in{d_in, chunk_begin, cum_sizes, inp_idx_begin_it};
        block_load_t(temp_storage.load).Load(it_in, thread_values, chunk_size);

        // reconstruct flags
        for (int i = 0; i < items_per_thread; ++i)
        {
          const OffsetT value_id = chunk_begin + items_per_thread * threadIdx.x + static_cast<OffsetT>(i);
          bool is_segment_head   = is_head_of_segment<OffsetT, segments_per_block>(cum_sizes, value_id);

          thread_flag_values[i] = augmented_accum_t{is_segment_head, thread_values[i]};
        }
      }
      __syncthreads();

      if (chunk_id == 0)
      {
        // Initialize exlusive_prefix, referenced from prefix_op
        augmented_init_value_t augmented_init_value{false, initial_value};
        scan_first_tile(thread_flag_values, augmented_init_value, augmented_scan_op, exclusive_prefix);
      }
      else
      {
        scan_later_tile(thread_flag_values, augmented_scan_op, prefix_op);
      }
      __syncthreads();

      // store prefix-scan values, discarding head flags
      {
        AccumT thread_values[items_per_thread] = {};

#pragma unroll
        for (int i = 0; i < items_per_thread; ++i)
        {
          if constexpr (is_inclusive)
          {
            thread_values[i] = get_value(thread_flag_values[i]);
          }
          else
          {
            const OffsetT value_id = chunk_begin + items_per_thread * threadIdx.x + static_cast<OffsetT>(i);
            bool is_segment_head   = is_head_of_segment<OffsetT, segments_per_block>(cum_sizes, value_id);
            thread_values[i]       = (is_segment_head) ? initial_value : get_value(thread_flag_values[i]);
          }
        }

        const OffsetT out_offset = chunk_id * tile_items;
        multi_segmented_iterator it_out{d_out, out_offset, cum_sizes, out_idx_begin_it};
        block_store_t(temp_storage.store).Store(it_out, thread_values, chunk_size);
      }
      __syncthreads();
    }
  }

  template <typename InputBeginOffsetIteratorT,
            typename OutputBeginOffsetIteratorT,
            int NumSegments = segments_per_block,
            class           = ::cuda::std::enable_if_t<(NumSegments > 1)>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_ranges(
    InputBeginOffsetIteratorT inp_idx_begin_it,
    OffsetT (&inp_idx_end)[NumSegments],
    OutputBeginOffsetIteratorT out_idx_begin_it,
    int n_segments)
  {
    OffsetT items_per_block{0};
    OffsetT(&cum_sizes)[NumSegments] = inp_idx_end;

    n_segments = ::cuda::std::min(n_segments, NumSegments);

    for (int i = 0; i < n_segments; ++i)
    {
      const OffsetT input_segment_begin = inp_idx_begin_it[i];
      const OffsetT segment_items       = ::cuda::std::max(inp_idx_end[i], input_segment_begin) - input_segment_begin;
      items_per_block += segment_items;
      cum_sizes[i] = items_per_block;
    }
    for (int i = ::cuda::std::max(n_segments, 0); i < NumSegments; ++i)
    {
      cum_sizes[i] = items_per_block + 1;
    }
    const OffsetT n_chunks = ::cuda::ceil_div(items_per_block, tile_items);

    using augmented_init_value_t =
      ::cuda::std::conditional_t<has_init, augmented_accum_t, ::cuda::std::tuple<bool, InitValueT>>;
    using augmented_scan_op_t = schwarz_scan_op<bool, AccumT, ScanOpT>;

    augmented_accum_t thread_flag_values[items_per_thread] = {};

    augmented_scan_op_t augmented_scan_op{scan_op};

    augmented_accum_t exclusive_prefix{};
    block_prefix_callback_t<augmented_accum_t, augmented_scan_op_t> prefix_op{exclusive_prefix, augmented_scan_op};

    for (OffsetT chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
    {
      const OffsetT chunk_begin = chunk_id * tile_items;
      const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, items_per_block);

      // chunk_size <= TILE_ITEMS, casting to int is safe
      const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

      // load values, and pack them into head_flag-value pairs
      {
        AccumT thread_values[items_per_thread] = {};
        multi_segmented_iterator it_in{d_in, chunk_begin, cum_sizes, inp_idx_begin_it};
        block_load_t(temp_storage.load).Load(it_in, thread_values, chunk_size);

        // reconstruct flags
        for (int i = 0; i < items_per_thread; ++i)
        {
          const OffsetT value_id = chunk_begin + items_per_thread * threadIdx.x + static_cast<OffsetT>(i);
          bool is_segment_head   = is_head_of_segment<OffsetT, segments_per_block>(cum_sizes, value_id);

          thread_flag_values[i] = augmented_accum_t{is_segment_head, thread_values[i]};
        }
      }
      __syncthreads();

      if (chunk_id == 0)
      {
        // Initialize exlusive_prefix, referenced from prefix_op
        augmented_init_value_t augmented_init_value{false, initial_value};
        scan_first_tile(thread_flag_values, augmented_init_value, augmented_scan_op, exclusive_prefix);
      }
      else
      {
        scan_later_tile(thread_flag_values, augmented_scan_op, prefix_op);
      }
      __syncthreads();

      // store prefix-scan values, discarding head flags
      {
        AccumT thread_values[items_per_thread] = {};

#pragma unroll
        for (int i = 0; i < items_per_thread; ++i)
        {
          if constexpr (is_inclusive)
          {
            thread_values[i] = get_value(thread_flag_values[i]);
          }
          else
          {
            const OffsetT value_id = chunk_begin + items_per_thread * threadIdx.x + static_cast<OffsetT>(i);
            bool is_segment_head   = is_head_of_segment<OffsetT, segments_per_block>(cum_sizes, value_id);
            thread_values[i]       = (is_segment_head) ? initial_value : get_value(thread_flag_values[i]);
          }
        }

        const OffsetT out_offset = chunk_id * tile_items;
        multi_segmented_iterator it_out{d_out, out_offset, cum_sizes, out_idx_begin_it};
        block_store_t(temp_storage.store).Store(it_out, thread_values, chunk_size);
      }
      __syncthreads();
    }
  }

private:
  template <typename FlagTy, typename ValueTy>
  _CCCL_DEVICE _CCCL_FORCEINLINE static FlagTy get_flag(::cuda::std::tuple<FlagTy, ValueTy> fv)
  {
    return ::cuda::std::get<0>(fv);
  }

  template <typename FlagTy, typename ValueTy>
  _CCCL_DEVICE _CCCL_FORCEINLINE static ValueTy get_value(::cuda::std::tuple<FlagTy, ValueTy> fv)
  {
    return ::cuda::std::get<1>(fv);
  }

  template <typename OffsetTy, int N>
  _CCCL_DEVICE _CCCL_FORCEINLINE static bool
  is_head_of_segment(const OffsetTy (&cumulative_sizes)[N], const OffsetTy item_id)
  {
    static_assert(N > 1, "Array size should be greater than one");

    bool is_segment_head{item_id == cumulative_sizes[0]};
#pragma unroll
    for (int j = 1; j < N; ++j)
    {
      is_segment_head = is_segment_head || (item_id == cumulative_sizes[j]);
    }
    return is_segment_head;
  }

  template <typename FlagTy, typename ValueTy, typename BinaryOpTy>
  struct schwarz_scan_op
  {
    using fv_t = ::cuda::std::tuple<FlagTy, ValueTy>;
    BinaryOpTy scan_op;

    _CCCL_DEVICE _CCCL_FORCEINLINE fv_t operator()(fv_t o1, fv_t o2)
    {
      const auto& [o1_flag, o1_value] = o1;
      const auto& [o2_flag, o2_value] = o2;
      const FlagTy res_flag           = (o1_flag || o2_flag);
      const ValueTy res_value         = (o2_flag) ? o2_value : scan_op(o1_value, o2_value);
      return {res_flag, res_value};
    }
  };

  template <typename IterTy, typename OffsetTy, typename Ty, int N, typename BeginOffsetIterTy>
  struct multi_segmented_iterator
  {
    IterTy m_it;
    OffsetTy m_start;
    Ty (&m_offsets)[N];
    BeginOffsetIterTy m_it_idx_begin;

    using iterator_concept  = ::cuda::std::random_access_iterator_tag;
    using iterator_category = ::cuda::std::random_access_iterator_tag;
    using value_type        = ::cuda::std::iter_value_t<IterTy>;
    using difference_type   = ::cuda::std::remove_cv_t<OffsetTy>;
    using reference         = ::cuda::std::iter_reference_t<IterTy>;
    using pointer           = void;

    static_assert(::cuda::std::is_same_v<difference_type, ::cuda::std::decay_t<Ty>>, "types are inconsistent");

    _CCCL_DEVICE _CCCL_FORCEINLINE
    multi_segmented_iterator(IterTy it, OffsetTy start, Ty (&cum_sizes)[N], BeginOffsetIterTy it_idx_begin)
        : m_it{it}
        , m_start{start}
        , m_offsets{cum_sizes}
        , m_it_idx_begin{it_idx_begin}
    {}

    _CCCL_DEVICE _CCCL_FORCEINLINE decltype(auto) operator[](difference_type n) const
    {
      const difference_type offset = m_start + n;
      const auto begin_it          = m_offsets;
      const auto end_it            = m_offsets + N;

      difference_type shifted_offset = offset;
      int segment_id                 = 0;
#pragma unroll
      for (int i = 0; i + 1 < N; ++i)
      {
        if ((m_offsets[i] <= offset) && (offset < m_offsets[i + 1]))
        {
          segment_id     = i + 1;
          shifted_offset = offset - m_offsets[i];
        }
      }
      return m_it[m_it_idx_begin[segment_id] + shifted_offset];
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE friend multi_segmented_iterator
    operator+(const multi_segmented_iterator& iter, OffsetTy n)
    {
      return {iter.m_it, iter.m_start + n, iter.m_offsets, iter.m_it_idx_begin};
    }
  };

  template <typename PrefixTy, typename BinaryOpTy>
  struct block_prefix_callback_t
  {
    PrefixTy& m_exclusive_prefix;
    BinaryOpTy m_scan_op;

    _CCCL_DEVICE _CCCL_FORCEINLINE PrefixTy operator()(PrefixTy block_aggregate)
    {
      PrefixTy previous_prefix = m_exclusive_prefix;
      m_exclusive_prefix       = m_scan_op(m_exclusive_prefix, block_aggregate);
      return previous_prefix;
    }
  };

  template <typename ItemTy, typename InitValueTy, typename ScanOpTy, bool IsInclusive = is_inclusive>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  scan_first_tile(ItemTy (&items)[items_per_thread], InitValueTy init_value, ScanOpTy scan_op, ItemTy& block_aggregate)
  {
    block_scan_t block_scan_algo(temp_storage.scan);
    if constexpr (has_init)
    {
      if constexpr (IsInclusive)
      {
        block_scan_algo.InclusiveScan(items, items, init_value, scan_op, block_aggregate);
      }
      else
      {
        block_scan_algo.ExclusiveScan(items, items, init_value, scan_op, block_aggregate);
      }
      block_aggregate = scan_op(init_value, block_aggregate);
    }
    else
    {
      static_assert(IsInclusive, "Unexpected ExclusiveScan without initial value call");
      block_scan_algo.InclusiveScan(items, items, scan_op, block_aggregate);
    }
  }

  template <typename ItemTy, typename ScanOpTy, typename PrefixCallback, bool IsInclusive = is_inclusive>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  scan_later_tile(ItemTy (&items)[items_per_thread], ScanOpTy scan_op, PrefixCallback& prefix_op)
  {
    block_scan_t block_scan_algo(temp_storage.scan);
    if constexpr (IsInclusive)
    {
      block_scan_algo.InclusiveScan(items, items, scan_op, prefix_op);
    }
    else
    {
      block_scan_algo.ExclusiveScan(items, items, scan_op, prefix_op);
    }
  }
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
