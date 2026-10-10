// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! \file
//! cub::AgentHistogram implements a stateful abstraction of CUDA thread blocks for participating in device-wide
//! histogram.

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
#include <cub/grid/grid_queue.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__fwd/format.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

enum BlockHistogramMemoryPreference // NOLINT(cppcoreguidelines-use-enum-class)
{
  GMEM,
  SMEM,
  BLEND
};

#if _CCCL_HOSTED()
namespace detail
{
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const char* to_string(BlockHistogramMemoryPreference mempref) noexcept
{
  switch (mempref)
  {
    case GMEM:
      return "GMEM";
    case SMEM:
      return "SMEM";
    case BLEND:
      return "BLEND";
  }
  return "<unknown BlockHistogramMemoryPreference>";
}
} // namespace detail

inline ::std::ostream& operator<<(::std::ostream& os, BlockHistogramMemoryPreference mempref)
{
  return os << CUB_NS_QUALIFIER::detail::to_string(mempref);
}
#endif // _CCCL_HOSTED()

CUB_NAMESPACE_END

#if __cpp_lib_format >= 201907L && !defined(_CCCL_DOXYGEN_INVOKED)
template <::cuda::std::same_as<char> CharT>
struct std::formatter<CUB_NS_QUALIFIER::BlockHistogramMemoryPreference, CharT> : formatter<const CharT*, CharT>
{
  template <class FmtCtx>
  auto format(const CUB_NS_QUALIFIER::BlockHistogramMemoryPreference& mempref, FmtCtx& ctx) const
  {
    return formatter<const CharT*, CharT>::format(CUB_NS_QUALIFIER::detail::to_string(mempref), ctx);
  }
};
#endif // __cpp_lib_format >= 201907L && !defined(_CCCL_DOXYGEN_INVOKED)

CUB_NAMESPACE_BEGIN

namespace detail
{
//! Parameterizable tuning policy type for AgentHistogram
template <int ThreadsPerBlock,
          int ItemsPerThread,
          BlockLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadModifier,
          bool RleCompress,
          BlockHistogramMemoryPreference MemoryPreference,
          bool WorkStealing,
          int VecSize = 4>
struct agent_histogram_policy
{
  /// Threads per thread block
  static constexpr int BLOCK_THREADS = ThreadsPerBlock;
  /// Items per thread (per tile of input)
  static constexpr int ITEMS_PER_THREAD = ItemsPerThread;
  // TODO(bgruber): remove this compatibility alias in CCCL 4.0.
  static constexpr int PIXELS_PER_THREAD = ItemsPerThread;

  /// Whether to perform localized RLE to compress samples before histogramming
  static constexpr bool IS_RLE_COMPRESS = RleCompress;

  /// Whether to prefer privatized shared-memory bins (versus privatized global-memory bins)
  static constexpr BlockHistogramMemoryPreference MEM_PREFERENCE = MemoryPreference;

  /// Whether to dequeue tiles from a global work queue
  static constexpr bool IS_WORK_STEALING = WorkStealing;

  /// Vector size for samples loading (1, 2, 4)
  static constexpr int VEC_SIZE = VecSize;
  static_assert(VEC_SIZE == 1 || VEC_SIZE == 2 || VEC_SIZE == 4);

  ///< The BlockLoad algorithm to use
  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM = LoadAlgorithm;

  ///< Cache load modifier for reading input elements
  static constexpr CacheLoadModifier LOAD_MODIFIER = LoadModifier;
};
} // namespace detail

//! Deprecated [Since 3.5]
template <int ThreadsPerBlock,
          int ItemsPerThread,
          BlockLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadModifier,
          bool RleCompress,
          BlockHistogramMemoryPreference MemoryPreference,
          bool WorkStealing,
          int VecSize = 4>
using AgentHistogramPolicy
  CCCL_DEPRECATED_BECAUSE("Use the tuning API for DeviceHistogram") = detail::agent_histogram_policy<
    ThreadsPerBlock,
    ItemsPerThread,
    LoadAlgorithm,
    LoadModifier,
    RleCompress,
    MemoryPreference,
    WorkStealing,
    VecSize>;

namespace detail::histogram
{
// Return a native item pointer (specialized for CacheModifiedInputIterator types)
template <CacheLoadModifier Modifier, typename ValueT, typename OffsetT>
_CCCL_DEVICE _CCCL_FORCEINLINE auto NativePointer(CacheModifiedInputIterator<Modifier, ValueT, OffsetT> itr)
{
  return itr.ptr;
}

// Return a native item pointer (specialized for other types)
template <typename IteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE auto NativePointer(IteratorT itr)
{
  return nullptr;
}

//! @brief AgentHistogram implements a stateful abstraction of CUDA thread blocks for participating
//! in device-wide histogram .
//!
//! @tparam AgentHistogramPolicyT
//!   Parameterized AgentHistogramPolicy tuning policy type
//!
//! @tparam PrivatizedSmemBins
//!   Number of privatized shared-memory histogram bins of any channel.  Zero indicates privatized
//! counters to be maintained in device-accessible memory.
//!
//! @tparam NumChannels
//!   Number of channels interleaved in the input data.  Supports up to four channels.
//!
//! @tparam NumActiveChannels
//!   Number of channels actively being histogrammed
//!
//! @tparam SampleIteratorT
//!   Random-access input iterator type for reading samples
//!
//! @tparam CounterT
//!   Integer type for counting sample occurrences per histogram bin
//!
//! @tparam PrivatizedDecodeOpT
//!   The transform operator type for determining privatized counter indices from samples, one for
//! each channel
//!
//! @tparam OutputDecodeOpT
//!   The transform operator type for determining output bin-ids from privatized counter indices, one
//! for each channel
//!
//! @tparam OffsetT
//!   Signed integer type for global offsets
template <typename AgentHistogramPolicyT,
          int PrivatizedSmemBins,
          int NumChannels,
          int NumActiveChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT>
struct AgentHistogram
{
  static constexpr int vec_size                    = AgentHistogramPolicyT::VEC_SIZE;
  static constexpr int threads_per_block           = AgentHistogramPolicyT::BLOCK_THREADS;
  static constexpr int items_per_thread            = AgentHistogramPolicyT::ITEMS_PER_THREAD;
  static constexpr int samples_per_thread          = items_per_thread * NumChannels;
  static constexpr int vecs_per_thread             = samples_per_thread / vec_size;
  static constexpr int tile_items                  = items_per_thread * threads_per_block;
  static constexpr int tile_samples                = samples_per_thread * threads_per_block;
  static constexpr bool is_rle_compress            = AgentHistogramPolicyT::IS_RLE_COMPRESS;
  static constexpr bool is_work_stealing           = AgentHistogramPolicyT::IS_WORK_STEALING;
  static constexpr CacheLoadModifier load_modifier = AgentHistogramPolicyT::LOAD_MODIFIER;
  static constexpr auto mem_preference =
    (PrivatizedSmemBins > 0) ? BlockHistogramMemoryPreference{AgentHistogramPolicyT::MEM_PREFERENCE} : GMEM;

  using SampleT = it_value_t<SampleIteratorT>;
  using ItemT   = typename CubVector<SampleT, NumChannels>::Type;
  using VecT    = typename CubVector<SampleT, vec_size>::Type;

  /// Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator or directly use the supplied input iterator type
  // TODO(bgruber): we can wrap all contiguous iterators, not just pointers
  using WrappedSampleIteratorT =
    ::cuda::std::_If<::cuda::std::is_pointer_v<SampleIteratorT>,
                     CacheModifiedInputIterator<load_modifier, SampleT, OffsetT>,
                     SampleIteratorT>;
  using WrappedItemIteratorT = CacheModifiedInputIterator<load_modifier, ItemT, OffsetT>;
  using WrappedVecsIteratorT = CacheModifiedInputIterator<load_modifier, VecT, OffsetT>;
  using BlockLoadSampleT =
    BlockLoad<SampleT, threads_per_block, samples_per_thread, AgentHistogramPolicyT::LOAD_ALGORITHM>;
  using BlockLoadItemT = BlockLoad<ItemT, threads_per_block, items_per_thread, AgentHistogramPolicyT::LOAD_ALGORITHM>;
  using BlockLoadVecT  = BlockLoad<VecT, threads_per_block, vecs_per_thread, AgentHistogramPolicyT::LOAD_ALGORITHM>;

  struct _TempStorage
  {
    // Smem needed for block-privatized smem histogram (with 1 word of padding)
    CounterT histograms[NumActiveChannels][PrivatizedSmemBins + 1];
    int tile_idx;

    union
    {
      typename BlockLoadSampleT::TempStorage sample_load;
      typename BlockLoadItemT::TempStorage item_load;
      typename BlockLoadVecT::TempStorage vec_load;
    };
  };

  using TempStorage = Uninitialized<_TempStorage>;

  _TempStorage& temp_storage;
  WrappedSampleIteratorT d_wrapped_samples; // with cache modifier applied, if possible
  SampleT* d_native_samples; // possibly nullptr if unavailable
  const int* num_output_bins; // one for each channel
  const int* num_privatized_bins; // one for each channel
  CounterT* d_privatized_histograms[NumActiveChannels]; // one for each channel
  CounterT** d_output_histograms; // in global memory
  const OutputDecodeOpT* output_decode_op; // determines output bin-id from privatized counter index, one for each
                                           // channel
  const PrivatizedDecodeOpT* privatized_decode_op; // determines privatized counter index from sample, one for each
                                                   // channel
  bool prefer_smem; // for privatized counterss

  template <typename TwoDimSubscriptableCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ZeroBinCounters(TwoDimSubscriptableCounterT& privatized_histograms)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      for (int bin = static_cast<int>(threadIdx.x); bin < num_privatized_bins[ch]; bin += threads_per_block)
      {
        privatized_histograms[ch][bin] = 0;
      }
    }

    // TODO(bgruber): do we also need the __syncthreads() when prefer_smem is false?
    // Barrier to make sure all threads are done updating counters
    __syncthreads();
  }

  // Update final output histograms from privatized histograms
  template <typename TwoDimSubscriptableCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void StoreOutput(TwoDimSubscriptableCounterT& privatized_histograms)
  {
    // Barrier to make sure all threads are done updating counters
    __syncthreads();

    // Apply privatized bin counts to output bin counts
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      const int channel_bins = num_privatized_bins[ch];
      for (int bin = static_cast<int>(threadIdx.x); bin < channel_bins; bin += threads_per_block)
      {
        int output_bin       = -1;
        const CounterT count = privatized_histograms[ch][bin];
        const bool is_valid  = count > 0;
        output_decode_op[ch].template BinSelect<load_modifier>(bin, output_bin, is_valid);

        if (output_bin >= 0)
        {
          atomicAdd(&d_output_histograms[ch][output_bin], count);
        }
      }
    }
  }

  // Accumulate items.  Specialized for RLE compression.
  template <typename TwoDimSubscriptableCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void AccumulateItems(
    SampleT samples[items_per_thread][NumChannels],
    bool is_valid[items_per_thread],
    TwoDimSubscriptableCounterT& privatized_histograms,
    ::cuda::std::true_type is_rle_compress)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      // Bin items
      int bins[items_per_thread];

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < items_per_thread; ++item)
      {
        bins[item] = -1;
        privatized_decode_op[ch].template BinSelect<load_modifier>(samples[item][ch], bins[item], is_valid[item]);
      }

      CounterT accumulator = 1;

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < items_per_thread - 1; ++item)
      {
        if (bins[item] != bins[item + 1])
        {
          if (bins[item] >= 0)
          {
            NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
                              (atomicAdd_block(privatized_histograms[ch] + bins[item], accumulator);),
                              (atomicAdd(privatized_histograms[ch] + bins[item], accumulator);));
          }

          accumulator = 0;
        }
        accumulator++;
      }

      // Last item
      if (bins[items_per_thread - 1] >= 0)
      {
        NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
                          (atomicAdd_block(privatized_histograms[ch] + bins[items_per_thread - 1], accumulator);),
                          (atomicAdd(privatized_histograms[ch] + bins[items_per_thread - 1], accumulator);));
      }
    }
  }

  // Accumulate items.  Specialized for individual accumulation of each item.
  template <typename TwoDimSubscriptableCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void AccumulateItems(
    SampleT samples[items_per_thread][NumChannels],
    bool is_valid[items_per_thread],
    TwoDimSubscriptableCounterT& privatized_histograms,
    ::cuda::std::false_type is_rle_compress)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < items_per_thread; ++item)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        int bin = -1;
        privatized_decode_op[ch].template BinSelect<load_modifier>(samples[item][ch], bin, is_valid[item]);
        if (bin >= 0)
        {
          NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
                            (atomicAdd_block(privatized_histograms[ch] + bin, 1);),
                            (atomicAdd(privatized_histograms[ch] + bin, 1);));
        }
      }
    }
  }

  // Load full, aligned tile using item iterator
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  LoadFullAlignedTile(OffsetT block_offset, SampleT (&samples)[items_per_thread][NumChannels])
  {
    if constexpr (NumActiveChannels == 1)
    {
      using AliasedVecs = VecT[vecs_per_thread];
      const WrappedVecsIteratorT d_wrapped_vecs(reinterpret_cast<VecT*>(d_native_samples + block_offset));
      // Load using a wrapped vec iterator
      BlockLoadVecT{temp_storage.vec_load}.Load(d_wrapped_vecs, reinterpret_cast<AliasedVecs&>(samples));
    }
    else
    {
      using AliasedItems = ItemT[items_per_thread];
      const WrappedItemIteratorT d_wrapped_items(reinterpret_cast<ItemT*>(d_native_samples + block_offset));
      // Load using a wrapped item iterator
      BlockLoadItemT{temp_storage.item_load}.Load(d_wrapped_items, reinterpret_cast<AliasedItems&>(samples));
    }
  }

  template <bool IsFullTile, bool IsAligned>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  LoadTile(OffsetT block_offset, int valid_samples, SampleT (&samples)[items_per_thread][NumChannels])
  {
    if constexpr (IsFullTile)
    {
      if constexpr (IsAligned)
      {
        LoadFullAlignedTile(block_offset, samples);
      }
      else
      {
        // Load using sample iterator
        using AliasedSamples = SampleT[samples_per_thread];
        BlockLoadSampleT{temp_storage.sample_load}.Load(
          d_wrapped_samples + block_offset, reinterpret_cast<AliasedSamples&>(samples));
      }
    }
    else
    {
      if constexpr (IsAligned)
      {
        // Load partially-full, aligned tile using the item iterator
        using AliasedItems = ItemT[items_per_thread];
        const WrappedItemIteratorT d_wrapped_items((ItemT*) (d_native_samples + block_offset));
        const int valid_items = valid_samples / NumChannels;

        // Load using a wrapped item iterator
        BlockLoadItemT{temp_storage.item_load}.Load(
          d_wrapped_items, reinterpret_cast<AliasedItems&>(samples), valid_items);
      }
      else
      {
        using AliasedSamples = SampleT[samples_per_thread];
        BlockLoadSampleT{temp_storage.sample_load}.Load(
          d_wrapped_samples + block_offset, reinterpret_cast<AliasedSamples&>(samples), valid_samples);
      }
    }
  }

  template <bool IsFullTile, bool IsStriped>
  _CCCL_DEVICE _CCCL_FORCEINLINE void MarkValid(bool (&is_valid)[items_per_thread], int valid_samples)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < items_per_thread; ++item)
    {
      if constexpr (IsStriped)
      {
        is_valid[item] = IsFullTile || (((threadIdx.x + threads_per_block * item) * NumChannels) < valid_samples);
      }
      else
      {
        is_valid[item] = IsFullTile || (((threadIdx.x * items_per_thread + item) * NumChannels) < valid_samples);
      }
    }
  }

  //! @brief Consume a tile of data samples
  //!
  //! @tparam IsAligned
  //!   Whether the tile offset is aligned (vec-aligned for single-channel, item-aligned for multi-channel)
  //!
  //! @tparam IsFullTile
  //!  Whether the tile is full
  template <bool IsAligned, bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTile(OffsetT block_offset, int valid_samples)
  {
    SampleT samples[items_per_thread][NumChannels];
    bool is_valid[items_per_thread];

    LoadTile<IsFullTile, IsAligned>(block_offset, valid_samples, samples);
    MarkValid<IsFullTile, AgentHistogramPolicyT::LOAD_ALGORITHM == BLOCK_LOAD_STRIPED>(is_valid, valid_samples);

    if (prefer_smem)
    {
      AccumulateItems(samples, is_valid, temp_storage.histograms, ::cuda::std::bool_constant<is_rle_compress>{});
    }
    else
    {
      AccumulateItems(samples, is_valid, d_privatized_histograms, ::cuda::std::bool_constant<is_rle_compress>{});
    }
  }

  //! @brief Consume row tiles. Specialized for work-stealing from queue
  //!
  //! @param num_row_items
  //!   The number of multi-channel items per row in the region of interest
  //!
  //! @param num_rows
  //!   The number of rows in the region of interest
  //!
  //! @param row_stride_samples
  //!   The number of samples between starts of consecutive rows in the region of interest
  //!
  //! @param tiles_per_row
  //!   Number of image tiles per row
  template <bool IsAligned>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTiles(
    OffsetT num_row_items,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    int tiles_per_row,
    GridQueue<int> tile_queue,
    ::cuda::std::true_type is_work_stealing)
  {
    int num_tiles                = num_rows * tiles_per_row;
    int tile_idx                 = static_cast<int>((blockIdx.y * gridDim.x) + blockIdx.x);
    OffsetT num_even_share_tiles = gridDim.x * gridDim.y;

    while (tile_idx < num_tiles)
    {
      int row             = tile_idx / tiles_per_row;
      const int col       = tile_idx - (row * tiles_per_row);
      OffsetT row_offset  = row * row_stride_samples;
      OffsetT col_offset  = (col * tile_samples);
      OffsetT tile_offset = row_offset + col_offset;

      if (col == tiles_per_row - 1)
      {
        // Consume a partially-full tile at the end of the row
        OffsetT num_remaining = (num_row_items * NumChannels) - col_offset;
        ConsumeTile<IsAligned, false>(tile_offset, num_remaining);
      }
      else
      {
        // Consume full tile
        ConsumeTile<IsAligned, true>(tile_offset, tile_samples);
      }

      __syncthreads();

      // Get next tile
      if (threadIdx.x == 0)
      {
        temp_storage.tile_idx = tile_queue.Drain(1) + num_even_share_tiles;
      }

      __syncthreads();

      tile_idx = temp_storage.tile_idx;
    }
  }

  //! @brief Consume row tiles.  Specialized for even-share (striped across thread blocks)
  //!
  //! @param num_row_items
  //!   The number of multi-channel items per row in the region of interest
  //!
  //! @param num_rows
  //!   The number of rows in the region of interest
  //!
  //! @param row_stride_samples
  //!   The number of samples between starts of consecutive rows in the region of interest
  template <bool IsAligned>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTiles(
    OffsetT num_row_items, OffsetT num_rows, OffsetT row_stride_samples, int, GridQueue<int>, ::cuda::std::false_type)
  {
    for (int row = static_cast<int>(blockIdx.y); row < num_rows; row += static_cast<int>(gridDim.y))
    {
      OffsetT row_begin   = row * row_stride_samples;
      OffsetT row_end     = row_begin + (num_row_items * NumChannels);
      OffsetT tile_offset = row_begin + (blockIdx.x * tile_samples);

      while (tile_offset < row_end)
      {
        OffsetT num_remaining = row_end - tile_offset;

        if (num_remaining < tile_samples)
        {
          // Consume partial tile
          ConsumeTile<IsAligned, false>(tile_offset, num_remaining);
          break;
        }

        // Consume full tile
        ConsumeTile<IsAligned, true>(tile_offset, tile_samples);
        tile_offset += gridDim.x * tile_samples;
      }
    }
  }

  //---------------------------------------------------------------------
  // Parameter extraction
  //---------------------------------------------------------------------

  //! @brief Constructor
  //!
  //! @param temp_storage
  //!   Reference to temp_storage
  //!
  //! @param d_samples
  //!   Input data to reduce
  //!
  //! @param num_output_bins
  //!   The number bins per final output histogram
  //!
  //! @param num_privatized_bins
  //!   The number bins per privatized histogram
  //!
  //! @param d_output_histograms
  //!   Reference to final output histograms
  //!
  //! @param d_privatized_histograms
  //!   Reference to privatized histograms
  //!
  //! @param output_decode_op
  //!   The transform operator for determining output bin-ids from privatized counter indices, one for each channel
  //!
  //! @param privatized_decode_op
  //!   The transform operator for determining privatized counter indices from samples, one for each channel
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentHistogram(
    TempStorage& temp_storage,
    SampleIteratorT d_samples,
    const int* num_output_bins,
    const int* num_privatized_bins,
    CounterT** d_output_histograms,
    CounterT** d_privatized_histograms,
    const OutputDecodeOpT* output_decode_op,
    const PrivatizedDecodeOpT* privatized_decode_op)
      : temp_storage(temp_storage.Alias())
      , d_wrapped_samples(d_samples)
      , d_native_samples(NativePointer(d_wrapped_samples))
      , num_output_bins(num_output_bins)
      , num_privatized_bins(num_privatized_bins)
      , d_output_histograms(d_output_histograms)
      , output_decode_op(output_decode_op)
      , privatized_decode_op(privatized_decode_op)
      , prefer_smem((mem_preference == SMEM) ? true : // prefer smem privatized histograms
                      (mem_preference == GMEM) ? false
                                               : // prefer gmem privatized histograms
                      blockIdx.x & 1) // prefer blended privatized histograms
  {
    const int blockId = static_cast<int>((blockIdx.y * gridDim.x) + blockIdx.x);

    // TODO(bgruber): d_privatized_histograms seems only used when !prefer_smem, can we skip it if prefer_smem?
    // Initialize the locations of this block's privatized histograms
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      const auto offset                 = static_cast<::cuda::std::int64_t>(blockId) * num_privatized_bins[ch];
      this->d_privatized_histograms[ch] = d_privatized_histograms[ch] + offset;
    }
  }

  //! @brief Consume image
  //!
  //! @param num_row_items
  //!   The number of multi-channel items per row in the region of interest
  //!
  //! @param num_rows
  //!   The number of rows in the region of interest
  //!
  //! @param row_stride_samples
  //!   The number of samples between starts of consecutive rows in the region of interest
  //!
  //! @param tiles_per_row
  //!   Number of image tiles per row
  //!
  //! @param tile_queue
  //!   Queue descriptor for assigning tiles of work to thread blocks
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTiles(
    OffsetT num_row_items, OffsetT num_rows, OffsetT row_stride_samples, int tiles_per_row, GridQueue<int> tile_queue)
  {
    // Check whether all row starting offsets are vec-aligned (in single-channel) or item-aligned (in multi-channel)
    constexpr int vec_mask  = alignof(VecT) - 1;
    constexpr int item_mask = alignof(ItemT) - 1;
    const size_t row_bytes  = sizeof(SampleT) * row_stride_samples;

    const bool vec_aligned_rows =
      (NumChannels == 1) && (samples_per_thread % vec_size == 0) && // Single channel
      ((size_t(d_native_samples) & vec_mask) == 0) && // ptr is quad-aligned
      ((num_rows == 1) || ((row_bytes & vec_mask) == 0)); // number of row-samples is a multiple of the alignment of the
                                                          // quad

    const bool item_aligned_rows =
      (NumChannels > 1) && // Multi channel
      ((size_t(d_native_samples) & item_mask) == 0) && // ptr is item-aligned
      ((row_bytes & item_mask) == 0); // number of row-samples is a multiple of the alignment of the item

    _CCCL_PDL_GRID_DEPENDENCY_SYNC();

    // Whether rows are aligned and can be vectorized
    if ((d_native_samples != nullptr) && (vec_aligned_rows || item_aligned_rows))
    {
      ConsumeTiles<true>(
        num_row_items, num_rows, row_stride_samples, tiles_per_row, tile_queue, bool_constant_v<is_work_stealing>);
    }
    else
    {
      ConsumeTiles<false>(
        num_row_items, num_rows, row_stride_samples, tiles_per_row, tile_queue, bool_constant_v<is_work_stealing>);
    }

    _CCCL_PDL_TRIGGER_NEXT_LAUNCH(); // omitting makes no difference in cub.bench.histogram.even.base
  }

  //! Initialize privatized bin counters.  Specialized for privatized shared-memory counters
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitBinCounters()
  {
    if (prefer_smem)
    {
      ZeroBinCounters(temp_storage.histograms);
    }
    else
    {
      ZeroBinCounters(d_privatized_histograms);
    }
  }

  //! Store privatized histogram to device-accessible memory.  Specialized for privatized shared-memory counters
  _CCCL_DEVICE _CCCL_FORCEINLINE void StoreOutput()
  {
    if (prefer_smem)
    {
      StoreOutput(temp_storage.histograms);
    }
    else
    {
      StoreOutput(d_privatized_histograms);
    }
  }
};
} // namespace detail::histogram

CUB_NAMESPACE_END
