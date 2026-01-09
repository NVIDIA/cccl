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

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_pointer.h>

CUB_NAMESPACE_BEGIN

enum BlockHistogramMemoryPreference
{
  GMEM,
  SMEM,
  BLEND
};

//! Parameterizable tuning policy type for AgentHistogram
//!
//! @tparam BlockThreads
//!   Threads per thread block
//!
//! @tparam PixelsPerThread
//!   Pixels per thread (per tile of input)
//!
//! @tparam LoadAlgorithm
//!   The BlockLoad algorithm to use
//!
//! @tparam LoadModifier
//!   Cache load modifier for reading input elements
//!
//! @tparam RleCompress
//!   Whether to perform localized RLE to compress samples before histogramming
//!
//! @tparam MemoryPreference
//!   Whether to prefer privatized shared-memory bins (versus privatized global-memory bins)
//!
//! @tparam WorkStealing
//!   Whether to dequeue tiles from a global work queue
//!
//! @tparam VecSize
//!   Vector size for samples loading (1, 2, 4)
template <int BlockThreads,
          int PixelsPerThread,
          BlockLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadModifier,
          bool RleCompress,
          BlockHistogramMemoryPreference MemoryPreference,
          bool WorkStealing,
          int VecSize = 4>
struct AgentHistogramPolicy
{
  /// Threads per thread block
  static constexpr int BLOCK_THREADS = BlockThreads;
  /// Pixels per thread (per tile of input)
  static constexpr int PIXELS_PER_THREAD = PixelsPerThread;

  /// Whether to perform localized RLE to compress samples before histogramming
  static constexpr bool IS_RLE_COMPRESS = RleCompress;

  /// Whether to prefer privatized shared-memory bins (versus privatized global-memory bins)
  static constexpr BlockHistogramMemoryPreference MEM_PREFERENCE = MemoryPreference;

  /// Whether to dequeue tiles from a global work queue
  static constexpr bool IS_WORK_STEALING = WorkStealing;

  /// Vector size for samples loading (1, 2, 4)
  static constexpr int VEC_SIZE = VecSize;

  ///< The BlockLoad algorithm to use
  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM = LoadAlgorithm;

  ///< Cache load modifier for reading input elements
  static constexpr CacheLoadModifier LOAD_MODIFIER = LoadModifier;
};

#if defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)
namespace detail
{
// Only define this when needed.
// Because of overload woes, this depends on C++20 concepts. util_device.h checks that concepts are available when
// either runtime policies or PTX JSON information are enabled, so if they are, this is always valid. The generic
// version is always defined, and that's the only one needed for regular CUB operations.
//
// TODO: enable this unconditionally once concepts are always available
CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  HistogramAgentPolicy,
  (always_true),
  (BLOCK_THREADS, BlockThreads, int),
  (PIXELS_PER_THREAD, PixelsPerThread, int),
  (IS_RLE_COMPRESS, IsRleCompress, bool),
  (MEM_PREFERENCE, MemPreference, BlockHistogramMemoryPreference),
  (IS_WORK_STEALING, IsWorkStealing, bool),
  (VEC_SIZE, VecSize, int),
  (LOAD_ALGORITHM, LoadAlgorithm, cub::BlockLoadAlgorithm),
  (LOAD_MODIFIER, LoadModifier, cub::CacheLoadModifier))
} // namespace detail
#endif

namespace detail::histogram
{
// Return a native pixel pointer (specialized for CacheModifiedInputIterator types)
template <CacheLoadModifier Modifier, typename ValueT, typename OffsetT>
_CCCL_DEVICE _CCCL_FORCEINLINE auto NativePointer(CacheModifiedInputIterator<Modifier, ValueT, OffsetT> itr)
{
  return itr.ptr;
}

// Return a native pixel pointer (specialized for other types)
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
  static constexpr int block_threads               = AgentHistogramPolicyT::BLOCK_THREADS;
  static constexpr int pixels_per_thread           = AgentHistogramPolicyT::PIXELS_PER_THREAD;
  static constexpr int samples_per_thread          = pixels_per_thread * NumChannels;
  static constexpr int vecs_per_thread             = samples_per_thread / vec_size;
  static constexpr int tile_pixels                 = pixels_per_thread * block_threads;
  static constexpr int tile_samples                = samples_per_thread * block_threads;
  static constexpr bool is_rle_compress            = AgentHistogramPolicyT::IS_RLE_COMPRESS;
  static constexpr bool is_work_stealing           = AgentHistogramPolicyT::IS_WORK_STEALING;
  static constexpr CacheLoadModifier load_modifier = AgentHistogramPolicyT::LOAD_MODIFIER;
  static constexpr auto mem_preference =
    (PrivatizedSmemBins > 0) ? BlockHistogramMemoryPreference{AgentHistogramPolicyT::MEM_PREFERENCE} : GMEM;

  using SampleT = it_value_t<SampleIteratorT>;
  using PixelT  = typename CubVector<SampleT, NumChannels>::Type;
  using VecT    = typename CubVector<SampleT, vec_size>::Type;

  /// Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator or directly use the supplied input iterator type
  // TODO(bgruber): we can wrap all contiguous iterators, not just pointers
  using WrappedSampleIteratorT =
    ::cuda::std::_If<::cuda::std::is_pointer_v<SampleIteratorT>,
                     CacheModifiedInputIterator<load_modifier, SampleT, OffsetT>,
                     SampleIteratorT>;
  using WrappedPixelIteratorT = CacheModifiedInputIterator<load_modifier, PixelT, OffsetT>;
  using WrappedVecsIteratorT  = CacheModifiedInputIterator<load_modifier, VecT, OffsetT>;
  using BlockLoadSampleT = BlockLoad<SampleT, block_threads, samples_per_thread, AgentHistogramPolicyT::LOAD_ALGORITHM>;
  using BlockLoadPixelT  = BlockLoad<PixelT, block_threads, pixels_per_thread, AgentHistogramPolicyT::LOAD_ALGORITHM>;
  using BlockLoadVecT    = BlockLoad<VecT, block_threads, vecs_per_thread, AgentHistogramPolicyT::LOAD_ALGORITHM>;

  struct _TempStorage
  {
    // Smem needed for block-privatized smem histogram (with 1 word of padding)
    CounterT histograms[NumActiveChannels][PrivatizedSmemBins + 1];
    int tile_idx;

    union
    {
      typename BlockLoadSampleT::TempStorage sample_load;
      typename BlockLoadPixelT::TempStorage pixel_load;
      typename BlockLoadVecT::TempStorage vec_load;
    };
  };

  using TempStorage = Uninitialized<_TempStorage>;

  _TempStorage& temp_storage;
  WrappedSampleIteratorT d_wrapped_samples; // with cache modifier applied, if possible
  SampleT* d_native_samples; // possibly nullptr if unavailable
  int* num_output_bins; // one for each channel
  int* num_privatized_bins; // one for each channel
  CounterT* d_privatized_histograms[NumActiveChannels]; // one for each channel
  CounterT** d_output_histograms; // in global memory
  OutputDecodeOpT* output_decode_op; // determines output bin-id from privatized counter index, one for each channel
  PrivatizedDecodeOpT* privatized_decode_op; // determines privatized counter index from sample, one for each channel
  bool prefer_smem; // for privatized counterss

  template <typename TwoDimSubscriptableCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ZeroBinCounters(TwoDimSubscriptableCounterT& privatized_histograms)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      for (int bin = threadIdx.x; bin < num_privatized_bins[ch]; bin += block_threads)
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
      for (int bin = threadIdx.x; bin < channel_bins; bin += block_threads)
      {
        int output_bin       = -1;
        const CounterT count = privatized_histograms[ch][bin];
        const bool is_valid  = count > 0;
        output_decode_op[ch].template BinSelect<load_modifier>(static_cast<SampleT>(bin), output_bin, is_valid);

        if (output_bin >= 0)
        {
          atomicAdd(&d_output_histograms[ch][output_bin], count);
        }
      }
    }
  }

  // Accumulate pixels.  Specialized for RLE compression.
  template <typename TwoDimSubscriptableCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void AccumulatePixels(
    SampleT samples[pixels_per_thread][NumChannels],
    bool is_valid[pixels_per_thread],
    TwoDimSubscriptableCounterT& privatized_histograms,
    ::cuda::std::true_type is_rle_compress)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      // Bin pixels
      int bins[pixels_per_thread];

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int pixel = 0; pixel < pixels_per_thread; ++pixel)
      {
        bins[pixel] = -1;
        privatized_decode_op[ch].template BinSelect<load_modifier>(samples[pixel][ch], bins[pixel], is_valid[pixel]);
      }

      CounterT accumulator = 1;

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int pixel = 0; pixel < pixels_per_thread - 1; ++pixel)
      {
        if (bins[pixel] != bins[pixel + 1])
        {
          if (bins[pixel] >= 0)
          {
            NV_IF_TARGET(NV_PROVIDES_SM_60,
                         (atomicAdd_block(privatized_histograms[ch] + bins[pixel], accumulator);),
                         (atomicAdd(privatized_histograms[ch] + bins[pixel], accumulator);));
          }

          accumulator = 0;
        }
        accumulator++;
      }

      // Last pixel
      if (bins[pixels_per_thread - 1] >= 0)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_60,
                     (atomicAdd_block(privatized_histograms[ch] + bins[pixels_per_thread - 1], accumulator);),
                     (atomicAdd(privatized_histograms[ch] + bins[pixels_per_thread - 1], accumulator);));
      }
    }
  }

  // Accumulate pixels.  Specialized for individual accumulation of each pixel.
  template <typename TwoDimSubscriptableCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void AccumulatePixels(
    SampleT samples[pixels_per_thread][NumChannels],
    bool is_valid[pixels_per_thread],
    TwoDimSubscriptableCounterT& privatized_histograms,
    ::cuda::std::false_type is_rle_compress)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int pixel = 0; pixel < pixels_per_thread; ++pixel)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        int bin = -1;
        privatized_decode_op[ch].template BinSelect<load_modifier>(samples[pixel][ch], bin, is_valid[pixel]);
        if (bin >= 0)
        {
          NV_IF_TARGET(NV_PROVIDES_SM_60,
                       (atomicAdd_block(privatized_histograms[ch] + bin, 1);),
                       (atomicAdd(privatized_histograms[ch] + bin, 1);));
        }
      }
    }
  }

  // Load full, aligned tile using pixel iterator
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  LoadFullAlignedTile(OffsetT block_offset, SampleT (&samples)[pixels_per_thread][NumChannels])
  {
    if constexpr (NumActiveChannels == 1)
    {
      using AliasedVecs = VecT[vecs_per_thread];
      WrappedVecsIteratorT d_wrapped_vecs(reinterpret_cast<VecT*>(d_native_samples + block_offset));
      // Load using a wrapped vec iterator
      BlockLoadVecT{temp_storage.vec_load}.Load(d_wrapped_vecs, reinterpret_cast<AliasedVecs&>(samples));
    }
    else
    {
      using AliasedPixels = PixelT[pixels_per_thread];
      WrappedPixelIteratorT d_wrapped_pixels(reinterpret_cast<PixelT*>(d_native_samples + block_offset));
      // Load using a wrapped pixel iterator
      BlockLoadPixelT{temp_storage.pixel_load}.Load(d_wrapped_pixels, reinterpret_cast<AliasedPixels&>(samples));
    }
  }

  template <bool IsFullTile, bool IsAligned>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  LoadTile(OffsetT block_offset, int valid_samples, SampleT (&samples)[pixels_per_thread][NumChannels])
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
        // Load partially-full, aligned tile using the pixel iterator
        using AliasedPixels = PixelT[pixels_per_thread];
        WrappedPixelIteratorT d_wrapped_pixels((PixelT*) (d_native_samples + block_offset));
        int valid_pixels = valid_samples / NumChannels;

        // Load using a wrapped pixel iterator
        BlockLoadPixelT{temp_storage.pixel_load}.Load(
          d_wrapped_pixels, reinterpret_cast<AliasedPixels&>(samples), valid_pixels);
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
  _CCCL_DEVICE _CCCL_FORCEINLINE void MarkValid(bool (&is_valid)[pixels_per_thread], int valid_samples)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int pixel = 0; pixel < pixels_per_thread; ++pixel)
    {
      if constexpr (IsStriped)
      {
        is_valid[pixel] = IsFullTile || (((threadIdx.x + block_threads * pixel) * NumChannels) < valid_samples);
      }
      else
      {
        is_valid[pixel] = IsFullTile || (((threadIdx.x * pixels_per_thread + pixel) * NumChannels) < valid_samples);
      }
    }
  }

  //! @brief Consume a tile of data samples
  //!
  //! @tparam IsAligned
  //!   Whether the tile offset is aligned (vec-aligned for single-channel, pixel-aligned for multi-channel)
  //!
  //! @tparam IsFullTile
  //!  Whether the tile is full
  template <bool IsAligned, bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTile(OffsetT block_offset, int valid_samples)
  {
    SampleT samples[pixels_per_thread][NumChannels];
    bool is_valid[pixels_per_thread];

    LoadTile<IsFullTile, IsAligned>(block_offset, valid_samples, samples);
    MarkValid<IsFullTile, AgentHistogramPolicyT::LOAD_ALGORITHM == BLOCK_LOAD_STRIPED>(is_valid, valid_samples);

    if (prefer_smem)
    {
      AccumulatePixels(samples, is_valid, temp_storage.histograms, ::cuda::std::bool_constant<is_rle_compress>{});
    }
    else
    {
      AccumulatePixels(samples, is_valid, d_privatized_histograms, ::cuda::std::bool_constant<is_rle_compress>{});
    }
  }

  //! @brief Consume row tiles. Specialized for work-stealing from queue
  //!
  //! @param num_row_pixels
  //!   The number of multi-channel pixels per row in the region of interest
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
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    int tiles_per_row,
    GridQueue<int> tile_queue,
    ::cuda::std::true_type is_work_stealing)
  {
    int num_tiles                = num_rows * tiles_per_row;
    int tile_idx                 = (blockIdx.y * gridDim.x) + blockIdx.x;
    OffsetT num_even_share_tiles = gridDim.x * gridDim.y;

    while (tile_idx < num_tiles)
    {
      int row             = tile_idx / tiles_per_row;
      int col             = tile_idx - (row * tiles_per_row);
      OffsetT row_offset  = row * row_stride_samples;
      OffsetT col_offset  = (col * tile_samples);
      OffsetT tile_offset = row_offset + col_offset;

      if (col == tiles_per_row - 1)
      {
        // Consume a partially-full tile at the end of the row
        OffsetT num_remaining = (num_row_pixels * NumChannels) - col_offset;
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
  //! @param num_row_pixels
  //!   The number of multi-channel pixels per row in the region of interest
  //!
  //! @param num_rows
  //!   The number of rows in the region of interest
  //!
  //! @param row_stride_samples
  //!   The number of samples between starts of consecutive rows in the region of interest
  template <bool IsAligned>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTiles(
    OffsetT num_row_pixels, OffsetT num_rows, OffsetT row_stride_samples, int, GridQueue<int>, ::cuda::std::false_type)
  {
    for (int row = blockIdx.y; row < num_rows; row += gridDim.y)
    {
      OffsetT row_begin   = row * row_stride_samples;
      OffsetT row_end     = row_begin + (num_row_pixels * NumChannels);
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
    int* num_output_bins,
    int* num_privatized_bins,
    CounterT** d_output_histograms,
    CounterT** d_privatized_histograms,
    OutputDecodeOpT* output_decode_op,
    PrivatizedDecodeOpT* privatized_decode_op)
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
    const int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;

    // TODO(bgruber): d_privatized_histograms seems only used when !prefer_smem, can we skip it if prefer_smem?
    // Initialize the locations of this block's privatized histograms
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      this->d_privatized_histograms[ch] = d_privatized_histograms[ch] + (blockId * num_privatized_bins[ch]);
    }
  }

  //! @brief Consume image
  //!
  //! @param num_row_pixels
  //!   The number of multi-channel pixels per row in the region of interest
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
    OffsetT num_row_pixels, OffsetT num_rows, OffsetT row_stride_samples, int tiles_per_row, GridQueue<int> tile_queue)
  {
    // Check whether all row starting offsets are vec-aligned (in single-channel) or pixel-aligned (in multi-channel)
    constexpr int vec_mask   = AlignBytes<VecT>::ALIGN_BYTES - 1;
    constexpr int pixel_mask = AlignBytes<PixelT>::ALIGN_BYTES - 1;
    const size_t row_bytes   = sizeof(SampleT) * row_stride_samples;

    const bool vec_aligned_rows =
      (NumChannels == 1) && (samples_per_thread % vec_size == 0) && // Single channel
      ((size_t(d_native_samples) & vec_mask) == 0) && // ptr is quad-aligned
      ((num_rows == 1) || ((row_bytes & vec_mask) == 0)); // number of row-samples is a multiple of the alignment of the
                                                          // quad

    const bool pixel_aligned_rows =
      (NumChannels > 1) && // Multi channel
      ((size_t(d_native_samples) & pixel_mask) == 0) && // ptr is pixel-aligned
      ((row_bytes & pixel_mask) == 0); // number of row-samples is a multiple of the alignment of the pixel

    _CCCL_PDL_GRID_DEPENDENCY_SYNC();

    // Whether rows are aligned and can be vectorized
    if ((d_native_samples != nullptr) && (vec_aligned_rows || pixel_aligned_rows))
    {
      ConsumeTiles<true>(
        num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue, bool_constant_v<is_work_stealing>);
    }
    else
    {
      ConsumeTiles<false>(
        num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue, bool_constant_v<is_work_stealing>);
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
