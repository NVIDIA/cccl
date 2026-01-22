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
#include <cub/detail/segmented_scan_helpers.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_arch.cuh> // MemBoundScaling

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/span>
#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

template <typename ComputeT, int NumSegmentsPerBlock>
using agent_segmented_scan_compute_t =
  ::cuda::std::conditional_t<NumSegmentsPerBlock == 1, ComputeT, ::cuda::std::tuple<ComputeT, bool>>;

template <typename V, typename F>
struct packer
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
  {
    return make_value_flag(v, f);
  }
};

template <typename V, typename F, typename ScanOp>
struct packer_iv
{
  V init_v;
  ScanOp& op;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
  {
    return make_value_flag((f) ? op(init_v, v) : v, f);
  }
};

template <typename V, typename F>
struct projector
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F) const
  {
    return v;
  }
};

template <typename V, typename F>
struct projector_iv
{
  V init_v;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
  {
    return (f) ? init_v : v;
  }
};

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
template <int Nominal4ByteBlockThreads,
          int Nominal4BytesItemsPerThread,
          typename ComputeT,
          BlockLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadModifier,
          BlockStoreAlgorithm StoreAlgorithm,
          BlockScanAlgorithm ScanAlgorithm,
          int SegmentsPerBlock = 1,
          typename ScalingType = detail::MemBoundScaling<Nominal4ByteBlockThreads,
                                                         Nominal4BytesItemsPerThread,
                                                         agent_segmented_scan_compute_t<ComputeT, SegmentsPerBlock>>>
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

  using augmented_accum_t = agent_segmented_scan_compute_t<AccumT, segments_per_block>;

  using block_load_t =
    BlockLoad<augmented_accum_t, block_threads, items_per_thread, AgentSegmentedScanPolicyT::load_algorithm>;
  using block_store_t =
    BlockStore<augmented_accum_t, block_threads, items_per_thread, AgentSegmentedScanPolicyT::store_algorithm>;
  using old_block_load_t =
    BlockLoad<AccumT, block_threads, items_per_thread, AgentSegmentedScanPolicyT::load_algorithm>;
  using old_block_store_t =
    BlockStore<AccumT, block_threads, items_per_thread, AgentSegmentedScanPolicyT::store_algorithm>;
  using block_scan_t        = BlockScan<augmented_accum_t, block_threads, AgentSegmentedScanPolicyT::scan_algorithm>;
  using block_offset_scan_t = BlockScan<OffsetT, block_threads, AgentSegmentedScanPolicyT::scan_algorithm>;

  struct _TempStorage
  {
    OffsetT logical_segment_offsets[segments_per_block];
    union AlgorithmsStorage
    {
      typename block_load_t::TempStorage load;
      typename block_store_t::TempStorage store;
      typename old_block_load_t::TempStorage old_load;
      typename old_block_store_t::TempStorage old_store;
      typename block_scan_t::TempStorage scan;
      typename block_offset_scan_t::TempStorage offset_scan;
    } reused;
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

    AccumT exclusive_prefix{};
    block_prefix_callback_t prefix_op{exclusive_prefix, scan_op};

    for (OffsetT chunk_id = 0; chunk_id < n_chunks;)
    {
      const OffsetT chunk_begin = inp_idx_begin + chunk_id * tile_items;
      const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, inp_idx_end);

      // chunk_size <= TILE_ITEMS, casting to int is safe
      const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

      AccumT thread_values[items_per_thread];
      if (chunk_size == tile_items)
      {
        block_load_t(temp_storage.reused.load).Load(d_in + chunk_begin, thread_values);
      }
      else
      {
        block_load_t(temp_storage.reused.load).Load(d_in + chunk_begin, thread_values, chunk_size, AccumT{});
      }
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

      if (chunk_size == tile_items)
      {
        block_store_t(temp_storage.reused.store).Store(d_out + out_idx_begin + chunk_id * tile_items, thread_values);
      }
      else
      {
        block_store_t(temp_storage.reused.store)
          .Store(d_out + out_idx_begin + chunk_id * tile_items, thread_values, chunk_size);
      }
      if (++chunk_id < n_chunks)
      {
        __syncthreads();
      }
    }
  };

#if 0
  //! @brief Scan statically given number of segment of values
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
            typename InputEndOffsetIteratorT,
            typename OutputBeginOffsetIteratorT,
            ::cuda::std::size_t NumSegments = segments_per_block,
            class = ::cuda::std::enable_if_t<(NumSegments > 1) && (NumSegments != ::cuda::std::dynamic_extent)>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_ranges(
    InputBeginOffsetIteratorT inp_idx_begin_it,
    InputEndOffsetIteratorT inp_idx_end_it,
    OutputBeginOffsetIteratorT out_idx_begin_it)
  {
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_value_t<InputBeginOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_value_t<InputEndOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_value_t<OutputBeginOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");
    static_assert(NumSegments >= segments_per_block, "Span's extent is not sufficient");

    // cooperatively compute inclusive scan of sizes of segments to be processed by this block

    {
      unsigned n_chunks        = ::cuda::ceil_div<unsigned>(NumSegments, block_threads);
      OffsetT exclusive_prefix = 0;
      using plus_t             = ::cuda::std::plus<>;
      block_prefix_callback_t<OffsetT, plus_t> prefix_callback_op{exclusive_prefix, plus_t{}};

      for (unsigned chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
      {
        const unsigned work_id = chunk_id * block_threads + threadIdx.x;

        const OffsetT input_segment_begin = (work_id < NumSegments) ? inp_idx_begin_it[work_id] : 0;
        const OffsetT segment_size =
          (work_id < NumSegments)
            ? ::cuda::std::max(inp_idx_end_it[work_id], input_segment_begin) - input_segment_begin
            : 0;

        OffsetT prefix;
        block_offset_scan_t(temp_storage.reused.offset_scan).InclusiveSum(segment_size, prefix, prefix_callback_op);
        __syncthreads();

        temp_storage.logical_segment_offsets[work_id] = prefix;
      }
    }

    __syncthreads();

    ::cuda::std::span<OffsetT, NumSegments> cum_sizes{temp_storage.logical_segment_offsets};
    OffsetT items_per_block = temp_storage.logical_segment_offsets[NumSegments - 1];
    const OffsetT n_chunks  = ::cuda::ceil_div(items_per_block, tile_items);

    using augmented_init_value_t = ::cuda::std::tuple<InitValueT, bool>;
    using augmented_scan_op_t    = schwarz_scan_op<AccumT, bool, ScanOpT>;

    augmented_scan_op_t augmented_scan_op{scan_op};

    augmented_accum_t exclusive_prefix{};
    block_prefix_callback_t<augmented_accum_t, augmented_scan_op_t> prefix_op{exclusive_prefix, augmented_scan_op};

    for (OffsetT chunk_id = 0; chunk_id < n_chunks;)
    {
      const OffsetT chunk_begin = chunk_id * tile_items;
      const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, items_per_block);

      // chunk_size <= TILE_ITEMS, casting to int is safe
      const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

      // load values, and pack them into head_flag-value pairs
      augmented_accum_t thread_flag_values[items_per_thread];
      {
        multi_segmented_iterator it_in{d_in, chunk_begin, cum_sizes, inp_idx_begin_it};
        AccumT thread_values[items_per_thread];
        block_load_t(temp_storage.reused.load).Load(it_in, thread_values, chunk_size, AccumT{});

        // reconstruct flags
#  pragma unroll
        for (int i = 0; i < items_per_thread; ++i)
        {
          const OffsetT value_id     = chunk_begin + items_per_thread * threadIdx.x + static_cast<OffsetT>(i);
          const bool is_segment_head = is_head_of_segment<OffsetT, segments_per_block>(cum_sizes, value_id);

          thread_flag_values[i] = make_value_flag(thread_values[i], is_segment_head);
        }
      }
      __syncthreads();

      if (chunk_id == 0)
      {
        // Initialize exlusive_prefix, referenced from prefix_op
        augmented_init_value_t augmented_init_value = make_value_flag(initial_value, false);
        scan_first_tile(thread_flag_values, augmented_init_value, augmented_scan_op, exclusive_prefix);
      }
      else
      {
        scan_later_tile(thread_flag_values, augmented_scan_op, prefix_op);
      }
      __syncthreads();

      // store prefix-scan values, discarding head flags
      {
        AccumT thread_values[items_per_thread];

#  pragma unroll
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
        block_store_t(temp_storage.reused.store).Store(it_out, thread_values, chunk_size);
      }
      if (++chunk_id < n_chunks)
      {
        __syncthreads();
      }
    }
  }
#endif

  //! @brief Scan dynamically given number of segment of values
  template <typename InputBeginOffsetIteratorT,
            typename InputEndOffsetIteratorT,
            typename OutputBeginOffsetIteratorT,
            ::cuda::std::size_t NumSegments = segments_per_block,
            class = ::cuda::std::enable_if_t<(NumSegments > 1) && (NumSegments != ::cuda::std::dynamic_extent)>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void old_consume_ranges(
    InputBeginOffsetIteratorT inp_idx_begin_it,
    InputEndOffsetIteratorT inp_idx_end_it,
    OutputBeginOffsetIteratorT out_idx_begin_it,
    int n_segments)
  {
    // if (threadIdx.x == 0 && blockIdx.x == 0)
    // {
    //   printf("old\n");
    // }
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_value_t<InputBeginOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_value_t<InputEndOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_value_t<OutputBeginOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");

    // cooperatively compute inclusive scan of sizes of segments to be processed by this block
    {
      n_segments               = ::cuda::std::min(n_segments, static_cast<int>(NumSegments));
      unsigned n_chunks        = ::cuda::ceil_div<unsigned>(n_segments, block_threads);
      OffsetT exclusive_prefix = 0;
      using plus_t             = ::cuda::std::plus<>;
      const plus_t offsets_scan_op{};
      block_prefix_callback_t prefix_callback_op{exclusive_prefix, offsets_scan_op};

      for (unsigned chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
      {
        const unsigned work_id = chunk_id * block_threads + threadIdx.x;

        const OffsetT input_segment_begin = (work_id < n_segments) ? inp_idx_begin_it[work_id] : 0;
        const OffsetT segment_size =
          (work_id < n_segments)
            ? ::cuda::std::max(inp_idx_end_it[work_id], input_segment_begin) - input_segment_begin
            : 0;

        OffsetT prefix;
        block_offset_scan_t(temp_storage.reused.offset_scan).InclusiveSum(segment_size, prefix, prefix_callback_op);
        __syncthreads();

        temp_storage.logical_segment_offsets[work_id] = prefix;
      }
    }

    __syncthreads();

    ::cuda::std::span<OffsetT, NumSegments> cum_sizes{temp_storage.logical_segment_offsets};
    OffsetT items_per_block = temp_storage.logical_segment_offsets[n_segments - 1];

    const OffsetT n_chunks = ::cuda::ceil_div(items_per_block, tile_items);

    using augmented_scan_op_t = schwarz_scan_op<AccumT, bool, ScanOpT>;
    using augmented_init_value_t =
      ::cuda::std::conditional_t<has_init, augmented_accum_t, ::cuda::std::tuple<InitValueT, bool>>;

    augmented_scan_op_t augmented_scan_op{scan_op};

    augmented_accum_t exclusive_prefix{};
    block_prefix_callback_t prefix_op{exclusive_prefix, augmented_scan_op};

    for (OffsetT chunk_id = 0; chunk_id < n_chunks;)
    {
      const OffsetT chunk_begin = chunk_id * tile_items;
      const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, items_per_block);

      // chunk_size <= TILE_ITEMS, casting to int is safe
      const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

      // load values, and pack them into head_flag-value pairs
      augmented_accum_t thread_flag_values[items_per_thread];
      {
        constexpr auto oob_default = AccumT{};
        multi_segmented_iterator it_in{d_in, chunk_begin, cum_sizes, inp_idx_begin_it};
        AccumT thread_values[items_per_thread];
        old_block_load_t(temp_storage.reused.old_load).Load(it_in, thread_values, chunk_size, oob_default);

        // reconstruct flags
#pragma unroll
        for (int i = 0; i < items_per_thread; ++i)
        {
          const OffsetT value_id     = chunk_begin + items_per_thread * threadIdx.x + static_cast<OffsetT>(i);
          const bool is_segment_head = is_head_of_segment<OffsetT, segments_per_block>(cum_sizes, value_id);

          if constexpr (has_init)
          {
            if (is_segment_head)
            {
              thread_values[i] = scan_op(static_cast<AccumT>(initial_value), thread_values[i]);
            }
          }

          thread_flag_values[i] = make_value_flag(thread_values[i], is_segment_head);
        }
      }

      __syncthreads();

      if (chunk_id == 0)
      {
        // Initialize exlusive_prefix, referenced from prefix_op
        augmented_init_value_t augmented_init_value = make_value_flag(initial_value, false);
        // if constexpr (::cuda::std::is_same_v<InitValueT, ::cuda::std::pair<unsigned, unsigned>>)
        // {
        //   if (blockIdx.x == 0 && threadIdx.x == 0)
        //   {
        //     printf("Initial Value = (%u, %u)\n", initial_value.first, initial_value.second);
        //   }
        // }
        // if constexpr (::cuda::std::is_same_v<InitValueT, ::cuda::std::pair<unsigned, unsigned>> && items_per_thread >
        // 2)
        // {
        //   if (blockIdx.x == 0 && threadIdx.x == 0)
        //   {
        //     printf(
        //       "Before: thread_flag_values[0] = ((%u, %u), %u), thread_flag_values[1] = ((%u, %u), %u), "
        //       "thread_flag_values[2] = "
        //       "((%u, %u), %u), thread_flag_values[3] = ((%u, %u), %u)\n",
        //       get_value(thread_flag_values[0]).first,
        //       get_value(thread_flag_values[0]).second,
        //       get_flag(thread_flag_values[0]),
        //       get_value(thread_flag_values[1]).first,
        //       get_value(thread_flag_values[1]).second,
        //       get_flag(thread_flag_values[1]),
        //       get_value(thread_flag_values[2]).first,
        //       get_value(thread_flag_values[2]).second,
        //       get_flag(thread_flag_values[2]),
        //       get_value(thread_flag_values[3]).first,
        //       get_value(thread_flag_values[3]).second,
        //       get_flag(thread_flag_values[3]));
        //   }
        // }
        scan_first_tile(thread_flag_values, augmented_init_value, augmented_scan_op, exclusive_prefix);
        // if constexpr (::cuda::std::is_same_v<InitValueT, ::cuda::std::pair<unsigned, unsigned>> && items_per_thread >
        // 2)
        // {
        //   if (blockIdx.x == 0 && threadIdx.x == 0)
        //   {
        //     printf(
        //       "After: thread_flag_values[0] = ((%u, %u), %u), thread_flag_values[1] = ((%u, %u), %u), "
        //       "thread_flag_values[2] = "
        //       "((%u, %u), %u), thread_flag_values[3] = ((%u, %u), %u)\n",
        //       get_value(thread_flag_values[0]).first,
        //       get_value(thread_flag_values[0]).second,
        //       get_flag(thread_flag_values[0]),
        //       get_value(thread_flag_values[1]).first,
        //       get_value(thread_flag_values[1]).second,
        //       get_flag(thread_flag_values[1]),
        //       get_value(thread_flag_values[2]).first,
        //       get_value(thread_flag_values[2]).second,
        //       get_flag(thread_flag_values[2]),
        //       get_value(thread_flag_values[3]).first,
        //       get_value(thread_flag_values[3]).second,
        //       get_flag(thread_flag_values[3]));
        //   }
        // }
      }
      else
      {
        scan_later_tile(thread_flag_values, augmented_scan_op, prefix_op);
      }
      __syncthreads();

      // store prefix-scan values, discarding head flags
      {
        AccumT thread_values[items_per_thread];

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
        old_block_store_t(temp_storage.reused.old_store).Store(it_out, thread_values, chunk_size);
      }
      if (++chunk_id < n_chunks)
      {
        __syncthreads();
      }
    }
  }

  //! @brief Scan dynamically given number of segment of values
  template <typename InputBeginOffsetIteratorT,
            typename InputEndOffsetIteratorT,
            typename OutputBeginOffsetIteratorT,
            ::cuda::std::size_t NumSegments = segments_per_block,
            class = ::cuda::std::enable_if_t<(NumSegments > 1) && (NumSegments != ::cuda::std::dynamic_extent)>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void new_consume_ranges(
    InputBeginOffsetIteratorT inp_idx_begin_it,
    InputEndOffsetIteratorT inp_idx_end_it,
    OutputBeginOffsetIteratorT out_idx_begin_it,
    int n_segments)
  {
    // if (threadIdx.x == 0 && blockIdx.x == 0)
    // {
    //   printf("new\n");
    // }
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_value_t<InputBeginOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_value_t<InputEndOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_value_t<OutputBeginOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");

    // cooperatively compute inclusive scan of sizes of segments to be processed by this block
    {
      n_segments               = ::cuda::std::min(n_segments, static_cast<int>(NumSegments));
      unsigned n_chunks        = ::cuda::ceil_div<unsigned>(n_segments, block_threads);
      OffsetT exclusive_prefix = 0;
      using plus_t             = ::cuda::std::plus<>;
      const plus_t offsets_scan_op{};
      block_prefix_callback_t prefix_callback_op{exclusive_prefix, offsets_scan_op};

      for (unsigned chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
      {
        const unsigned work_id = chunk_id * block_threads + threadIdx.x;

        // TODO: use BlockLoad to load
        const OffsetT input_segment_begin = (work_id < n_segments) ? inp_idx_begin_it[work_id] : 0;
        const OffsetT input_segment_end = (work_id < n_segments) ? inp_idx_end_it[work_id] : 0;
        const OffsetT segment_size = input_segment_end - input_segment_begin;

        OffsetT prefix;
        block_offset_scan_t(temp_storage.reused.offset_scan).InclusiveSum(segment_size, prefix, prefix_callback_op);
        __syncthreads();

        temp_storage.logical_segment_offsets[work_id] = prefix;
      }
    }

    __syncthreads();

    ::cuda::std::span<OffsetT, NumSegments> cum_sizes{temp_storage.logical_segment_offsets};
    const OffsetT items_per_block = temp_storage.logical_segment_offsets[n_segments - 1];

    const OffsetT n_chunks = ::cuda::ceil_div(items_per_block, tile_items);

    using augmented_scan_op_t = schwarz_scan_op<AccumT, bool, ScanOpT>;
    using augmented_init_value_t =
      ::cuda::std::conditional_t<has_init, augmented_accum_t, ::cuda::std::tuple<InitValueT, bool>>;

    augmented_scan_op_t augmented_scan_op{scan_op};

    augmented_accum_t exclusive_prefix{};
    block_prefix_callback_t prefix_op{exclusive_prefix, augmented_scan_op};

    for (OffsetT chunk_id = 0; chunk_id < n_chunks;)
    {
      const OffsetT chunk_begin = chunk_id * tile_items;
      const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, items_per_block);

      // chunk_size <= TILE_ITEMS, casting to int is safe
      const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

      // load values, and pack them into head_flag-value pairs
      augmented_accum_t thread_flag_values[items_per_thread];
      {
        constexpr auto oob_default = make_value_flag(AccumT{}, false);
        constexpr projector<AccumT, bool> projection_op{};
        if constexpr (has_init)
        {
          const packer_iv<AccumT, bool, ScanOpT> packer_op{static_cast<AccumT>(initial_value), scan_op};
          new_multi_segmented_iterator it_in{
            d_in,
            chunk_begin,
            cum_sizes,
            inp_idx_begin_it,
            packer_op,
            projection_op};

          // AccumT thread_values[items_per_thread];
          block_load_t(temp_storage.reused.load).Load(it_in, thread_flag_values, chunk_size, oob_default);
        }
        else
        {
          constexpr packer<AccumT, bool> packer_op{};
          new_multi_segmented_iterator it_in{
            d_in, chunk_begin, cum_sizes, inp_idx_begin_it, packer_op, projection_op};

          // AccumT thread_values[items_per_thread];
          block_load_t(temp_storage.reused.load).Load(it_in, thread_flag_values, chunk_size, oob_default);
        }
      }
      __syncthreads();

      if (chunk_id == 0)
      {
        // Initialize exlusive_prefix, referenced from prefix_op
        augmented_init_value_t augmented_init_value = make_value_flag(initial_value, false);
        scan_first_tile(thread_flag_values, augmented_init_value, augmented_scan_op, exclusive_prefix);
      }
      else
      {
        scan_later_tile(thread_flag_values, augmented_scan_op, prefix_op);
      }
      __syncthreads();

      // store prefix-scan values, discarding head flags
      {
        constexpr packer<AccumT, bool> packer_op{};
        const OffsetT out_offset = chunk_id * tile_items;
        if constexpr (is_inclusive)
        {
          constexpr projector<AccumT, bool> projector_op{};
          new_multi_segmented_iterator it_out{
            d_out, out_offset, cum_sizes, out_idx_begin_it, packer_op, projector_op};
          block_store_t(temp_storage.reused.store).Store(it_out, thread_flag_values, chunk_size);
        }
        else
        {
          const projector_iv<AccumT, bool> projector_op{static_cast<AccumT>(initial_value)}; 
          new_multi_segmented_iterator it_out{
            d_out,
            out_offset,
            cum_sizes,
            out_idx_begin_it,
            packer_op,
            projector_op};
          block_store_t(temp_storage.reused.store).Store(it_out, thread_flag_values, chunk_size);
        }
      }
      if (++chunk_id < n_chunks)
      {
        __syncthreads();
      }
    }
  }

  template <typename InputBeginOffsetIteratorT,
            typename InputEndOffsetIteratorT,
            typename OutputBeginOffsetIteratorT,
            ::cuda::std::size_t NumSegments = segments_per_block,
            class = ::cuda::std::enable_if_t<(NumSegments > 1) && (NumSegments != ::cuda::std::dynamic_extent)>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_ranges(
    InputBeginOffsetIteratorT inp_idx_begin_it,
    InputEndOffsetIteratorT inp_idx_end_it,
    OutputBeginOffsetIteratorT out_idx_begin_it,
    int n_segments)
  {
    return new_consume_ranges(inp_idx_begin_it, inp_idx_end_it, out_idx_begin_it, n_segments);
  }

private:
  template <typename OffsetTy, ::cuda::std::size_t N>
  _CCCL_DEVICE _CCCL_FORCEINLINE static bool
  is_head_of_segment(::cuda::std::span<OffsetTy, N> cumulative_sizes, const OffsetTy item_id) noexcept
  {
    static_assert(N > 1, "Array size should be greater than one");
    static constexpr int j_max = static_cast<int>(N);

    bool is_segment_head{item_id == cumulative_sizes[0]};
#pragma unroll
    for (int j = 1; j < j_max; ++j)
    {
      is_segment_head = is_segment_head || (item_id == cumulative_sizes[j]);
    }

    return is_segment_head;
  }

  template <typename IterTy, typename OffsetTy, typename SpanTy, typename BeginOffsetIterTy>
  struct multi_segmented_iterator
  {
    IterTy m_it;
    OffsetTy m_start;
    SpanTy m_offsets;
    BeginOffsetIterTy m_it_idx_begin;

    using iterator_concept  = ::cuda::std::random_access_iterator_tag;
    using iterator_category = ::cuda::std::random_access_iterator_tag;
    using value_type        = ::cuda::std::iter_value_t<IterTy>;
    using difference_type   = ::cuda::std::remove_cv_t<OffsetTy>;
    using reference         = ::cuda::std::iter_reference_t<IterTy>;
    using pointer           = void;

    // workaround for CTK 12.0 where span::extent is not constexpr
    template <typename T>
    struct extract_extend
    {};

    template <typename T, ::cuda::std::size_t N>
    struct extract_extend<::cuda::std::span<T, N>>
    {
      static constexpr int extend = static_cast<int>(N);
    };

    static_assert(::cuda::std::is_same_v<difference_type, typename SpanTy::value_type>, "types are inconsistent");

    _CCCL_DEVICE _CCCL_FORCEINLINE
    multi_segmented_iterator(IterTy it, OffsetTy start, SpanTy cum_sizes, BeginOffsetIterTy it_idx_begin)
        : m_it{it}
        , m_start{start}
        , m_offsets{cum_sizes}
        , m_it_idx_begin{it_idx_begin}
    {}

    _CCCL_DEVICE _CCCL_FORCEINLINE decltype(auto) operator[](difference_type n)
    {
      static constexpr int offset_size = extract_extend<SpanTy>::extend;
      const difference_type offset     = m_start + n;

      difference_type shifted_offset = offset;
      int segment_id                 = 0;
#pragma unroll
      for (int i = 0; i + 1 < offset_size; ++i)
      {
        const bool cond = ((m_offsets[i] <= offset) && (offset < m_offsets[i + 1]));
        segment_id      = (cond) ? i + 1 : segment_id;
        shifted_offset  = (cond) ? offset - m_offsets[i] : shifted_offset;
      }
      return m_it[m_it_idx_begin[segment_id] + shifted_offset];
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE friend multi_segmented_iterator
    operator+(const multi_segmented_iterator& iter, OffsetTy n)
    {
      return {iter.m_it, iter.m_start + n, iter.m_offsets, iter.m_it_idx_begin};
    }
  };

  template <typename IterTy,
            typename OffsetTy,
            typename SpanTy,
            typename BeginOffsetIterTy,
            typename ReadTransformT,
            typename WriteTransformT>
  struct new_multi_segmented_iterator
  {
    IterTy m_it;
    OffsetTy m_start;
    SpanTy m_offsets;
    BeginOffsetIterTy m_it_idx_begin;
    ReadTransformT m_read_transform_fn;
    WriteTransformT m_write_transform_fn;

    using iterator_concept      = ::cuda::std::random_access_iterator_tag;
    using iterator_category     = ::cuda::std::random_access_iterator_tag;
    using underlying_value_type = ::cuda::std::iter_value_t<IterTy>;
    using value_type            = ::cuda::std::invoke_result_t<ReadTransformT, underlying_value_type, bool>;
    using difference_type       = ::cuda::std::remove_cv_t<OffsetTy>;
    using reference             = void;
    using pointer               = void;

    static_assert(::cuda::std::is_same_v<difference_type, typename SpanTy::value_type>, "types are inconsistent");

    struct __mapping_proxy
    {
      IterTy m_it;
      OffsetTy m_offset;
      bool m_head_flag;
      ReadTransformT m_read_fn;
      WriteTransformT m_write_fn;

      _CCCL_DEVICE _CCCL_FORCEINLINE explicit __mapping_proxy(
        IterTy it, OffsetTy offset, bool head_flag, ReadTransformT read_fn, WriteTransformT write_fn)
          : m_it(it)
          , m_offset(offset)
          , m_head_flag(head_flag)
          , m_read_fn(::cuda::std::move(read_fn))
          , m_write_fn(::cuda::std::move(write_fn))
      {}

      _CCCL_DEVICE _CCCL_FORCEINLINE operator value_type() const
      {
        return m_read_fn(m_it[m_offset], m_head_flag);
      }

      template <typename V, typename F>
      _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy& operator=(::cuda::std::tuple<V, F> new_value)
      {
        m_it[m_offset] = m_write_fn(get_value(::cuda::std::move(new_value)), m_head_flag);
        return *this;
      }

      _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy& operator=(const __mapping_proxy& other)
      {
        return (*this = static_cast<value_type>(other));
      }
    };

    _CCCL_DEVICE _CCCL_FORCEINLINE new_multi_segmented_iterator(
      IterTy it,
      OffsetTy start,
      SpanTy cum_sizes,
      BeginOffsetIterTy it_idx_begin,
      ReadTransformT read_fn,
      WriteTransformT write_fn)
        : m_it{it}
        , m_start{start}
        , m_offsets{cum_sizes}
        , m_it_idx_begin{it_idx_begin}
        , m_read_transform_fn{::cuda::std::move(read_fn)}
        , m_write_transform_fn{::cuda::std::move(write_fn)}
    {}

    _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy operator*() const
    {
      const auto& [segment_id, rel_offset] = locate(0);
      const auto offset                    = m_it_idx_begin[segment_id] + rel_offset;
      const bool head_flag                 = rel_offset == 0;
      return __mapping_proxy(m_it, offset, head_flag, m_read_transform_fn, m_write_transform_fn);
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE __mapping_proxy operator[](difference_type n) const
    {
      const auto& [segment_id, rel_offset] = locate(n);
      const auto offset                    = m_it_idx_begin[segment_id] + rel_offset;
      const bool head_flag                 = (rel_offset == 0);
      return __mapping_proxy(m_it, offset, head_flag, m_read_transform_fn, m_write_transform_fn);
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE friend new_multi_segmented_iterator
    operator+(const new_multi_segmented_iterator& iter, OffsetTy n)
    {
      return {iter.m_it,
              iter.m_start + n,
              iter.m_offsets,
              iter.m_it_idx_begin,
              iter.m_read_transform_fn,
              iter.m_write_transform_fn};
    }

  private:
    _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::tuple<int, difference_type> locate(difference_type n) const
    {
      int offset_size = static_cast<int>(m_offsets.size());
      const difference_type offset     = m_start + n;

      difference_type shifted_offset = offset;
      int segment_id                 = 0;
      auto offset_c = m_offsets[0];
      for (int i = 1; i < offset_size; ++i)
      {
        const auto offset_n = m_offsets[i];
        const bool cond = ((offset_c <= offset) && (offset < offset_n));
        segment_id      = (cond) ? i: segment_id;
        shifted_offset  = (cond) ? offset - offset_c : shifted_offset;
        offset_c = offset_n;
      }
      return {segment_id, shifted_offset};
    }
  };

  template <typename PrefixTy, typename BinaryOpTy>
  struct block_prefix_callback_t
  {
    PrefixTy& m_exclusive_prefix;
    BinaryOpTy& m_scan_op;

    _CCCL_DEVICE _CCCL_FORCEINLINE block_prefix_callback_t(PrefixTy &prefix, BinaryOpTy& op) : m_exclusive_prefix(prefix), m_scan_op(op) {}

    _CCCL_DEVICE _CCCL_FORCEINLINE PrefixTy operator()(PrefixTy block_aggregate)
    {
      const PrefixTy previous_prefix = m_exclusive_prefix;
      m_exclusive_prefix       = m_scan_op(m_exclusive_prefix, block_aggregate);
      return previous_prefix;
    }
  };

  template <typename ItemTy, typename InitValueTy, typename ScanOpTy, bool IsInclusive = is_inclusive, bool HasInit = has_init>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  scan_first_tile(ItemTy (&items)[items_per_thread], InitValueTy init_value, ScanOpTy scan_op, ItemTy& block_aggregate)
  {
    block_scan_t block_scan_algo(temp_storage.reused.scan);
    if constexpr (HasInit)
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
    block_scan_t block_scan_algo(temp_storage.reused.scan);
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
