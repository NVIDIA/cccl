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
#include <cub/util_arch.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

//! Deprecated [Since 3.5]
template <int ThreadsPerBlock,
          int PixelsPerThread,
          BlockLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadModifier,
          bool RleCompress,
          bool WorkStealing,
          int VecSize = 4>
struct CCCL_DEPRECATED_BECAUSE("Use the tuning API for DeviceHistogram") AgentHistogramPolicy
{
  /// Threads per thread block
  static constexpr int BLOCK_THREADS = ThreadsPerBlock;
  /// Pixels per thread (per tile of input)
  static constexpr int PIXELS_PER_THREAD = PixelsPerThread;

  /// Whether to perform localized RLE to compress samples before histogramming
  static constexpr bool IS_RLE_COMPRESS = RleCompress;

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

namespace detail::histogram
{
struct HistogramPrivatizedStaticSmem
{};

struct HistogramPrivatizedDynamicSmem
{};

struct HistogramPrivatizedGmem
{};

inline constexpr auto histogram_privatized_static_smem  = HistogramPrivatizedStaticSmem{};
inline constexpr auto histogram_privatized_dynamic_smem = HistogramPrivatizedDynamicSmem{};
inline constexpr auto histogram_privatized_gmem         = HistogramPrivatizedGmem{};

template <class PrivatizationMode>
inline constexpr bool is_privatized_static_smem_v =
  ::cuda::std::is_same_v<PrivatizationMode, HistogramPrivatizedStaticSmem>;

template <class PrivatizationMode>
inline constexpr bool is_privatized_dynamic_smem_v =
  ::cuda::std::is_same_v<PrivatizationMode, HistogramPrivatizedDynamicSmem>;

template <class PrivatizationMode>
inline constexpr bool is_privatized_gmem_v = ::cuda::std::is_same_v<PrivatizationMode, HistogramPrivatizedGmem>;

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
//! @tparam PolicySelector
//!   Selector that returns the active HistogramPolicy.
//!
//! @tparam PrivatizationMode
//!   Storage mode for the privatized histogram.
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
//!   Integer type for per-block privatized histogram bins
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
//!
//! @tparam OutputCounterT
//!   Integer type for final output histogram bins. May be wider than `CounterT`.
template <typename PolicySelector,
          typename PrivatizationMode,
          int NumChannels,
          int NumActiveChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT,
          typename OutputCounterT = CounterT>
struct AgentHistogram
{
  static_assert(sizeof(CounterT) <= sizeof(OutputCounterT),
                "The output histogram counter must be at least as wide as the local counter");
  static_assert(is_privatized_static_smem_v<PrivatizationMode> || is_privatized_dynamic_smem_v<PrivatizationMode>
                || is_privatized_gmem_v<PrivatizationMode>);
  static constexpr bool uses_static_smem  = is_privatized_static_smem_v<PrivatizationMode>;
  static constexpr bool uses_dynamic_smem = is_privatized_dynamic_smem_v<PrivatizationMode>;
  static constexpr bool uses_gmem         = is_privatized_gmem_v<PrivatizationMode>;
  static constexpr auto policy            = current_policy<PolicySelector>();
  static constexpr auto sweep =
    uses_static_smem ? policy.static_smem
    : uses_dynamic_smem
      ? policy.dynamic_smem
      : policy.gmem;
  static constexpr int privatized_static_smem_bins =
    uses_static_smem ? policy.max_privatized_static_smem_bytes / int{sizeof(CounterT)} / NumActiveChannels : 0;
  static_assert(!uses_static_smem || privatized_static_smem_bins > 0,
                "Static-SMEM privatization requires room for at least one bin");
  static constexpr int vec_size                    = sweep.vec_size;
  static constexpr int threads_per_block           = sweep.threads_per_block;
  static constexpr int pixels_per_thread           = sweep.items_per_thread;
  static constexpr int samples_per_thread          = pixels_per_thread * NumChannels;
  static constexpr int vecs_per_thread             = samples_per_thread / vec_size;
  static constexpr int tile_pixels                 = pixels_per_thread * threads_per_block;
  static constexpr int tile_samples                = samples_per_thread * threads_per_block;
  static constexpr bool is_rle_compress            = sweep.rle_compress;
  static constexpr bool is_work_stealing           = sweep.work_stealing;
  static constexpr CacheLoadModifier load_modifier = sweep.load_modifier;

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
  using BlockLoadSampleT      = BlockLoad<SampleT, threads_per_block, samples_per_thread, sweep.load_algorithm>;
  using BlockLoadPixelT       = BlockLoad<PixelT, threads_per_block, pixels_per_thread, sweep.load_algorithm>;
  using BlockLoadVecT         = BlockLoad<VecT, threads_per_block, vecs_per_thread, sweep.load_algorithm>;

  struct _TempStorage
  {
    // The one-element fallback keeps this type well-formed for modes that do not
    // use the compile-time-sized histogram. Static-SMEM mode uses exactly the
    // configured number of bins; out-of-range samples are rejected before the
    // atomic update and therefore require no padding bin.
    CounterT privatized_histogram[NumActiveChannels][privatized_static_smem_bins > 0 ? privatized_static_smem_bins : 1];
    int tile_idx;

    union
    {
      typename BlockLoadSampleT::TempStorage sample_load;
      typename BlockLoadPixelT::TempStorage pixel_load;
      typename BlockLoadVecT::TempStorage vec_load;
    };
  };

  using TempStorage = Uninitialized<_TempStorage>;

  _TempStorage& static_smem_storage;
  WrappedSampleIteratorT d_wrapped_samples; // with cache modifier applied, if possible
  SampleT* d_native_samples; // possibly nullptr if unavailable
  const int* num_output_bins; // one for each channel
  const int* num_privatized_bins; // one for each channel
  CounterT* gmem_privatized_histograms[NumActiveChannels]; // one for each channel
  CounterT* dyn_smem_privatized_histograms[NumActiveChannels]; // dynamic shared-memory channel bases, when enabled
  OutputCounterT** output_histogram; // final output, in global memory
  const OutputDecodeOpT* output_decode_op; // determines output bin-id from privatized counter index, one for each
                                           // channel
  PrivatizedDecodeOpT* privatized_decode_op; // determines privatized counter index from sample, one for each channel
  _CCCL_DEVICE _CCCL_FORCEINLINE CounterT* PrivatizedHistogram(int channel)
  {
    if constexpr (uses_dynamic_smem)
    {
      return dyn_smem_privatized_histograms[channel];
    }
    else if constexpr (uses_static_smem)
    {
      return static_smem_storage.privatized_histogram[channel];
    }
    else
    {
      return gmem_privatized_histograms[channel];
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void ZeroBinCounters()
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      CounterT* privatized_histogram = PrivatizedHistogram(ch);
      for (int bin = static_cast<int>(threadIdx.x); bin < num_privatized_bins[ch]; bin += threads_per_block)
      {
        privatized_histogram[bin] = 0;
      }
    }

    // Barrier to make sure all threads are done updating counters
    __syncthreads();
  }

  // Accumulate pixels.  Specialized for RLE compression.
  _CCCL_DEVICE _CCCL_FORCEINLINE void AccumulatePixels(
    SampleT samples[pixels_per_thread][NumChannels],
    bool is_valid[pixels_per_thread],
    ::cuda::std::true_type is_rle_compress)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      CounterT* privatized_histogram = PrivatizedHistogram(ch);
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
            atomicAdd_block(privatized_histogram + bins[pixel], accumulator);
          }

          accumulator = 0;
        }
        accumulator++;
      }

      // Last pixel
      if (bins[pixels_per_thread - 1] >= 0)
      {
        atomicAdd_block(privatized_histogram + bins[pixels_per_thread - 1], accumulator);
      }
    }
  }

  // Accumulate pixels.  Specialized for individual accumulation of each pixel.
  _CCCL_DEVICE _CCCL_FORCEINLINE void AccumulatePixels(
    SampleT samples[pixels_per_thread][NumChannels],
    bool is_valid[pixels_per_thread],
    ::cuda::std::false_type is_rle_compress)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int pixel = 0; pixel < pixels_per_thread; ++pixel)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        CounterT* privatized_histogram = PrivatizedHistogram(ch);
        int bin                        = -1;
        privatized_decode_op[ch].template BinSelect<load_modifier>(samples[pixel][ch], bin, is_valid[pixel]);
        if (bin >= 0)
        {
          atomicAdd_block(privatized_histogram + bin, 1);
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
      BlockLoadVecT{static_smem_storage.vec_load}.Load(d_wrapped_vecs, reinterpret_cast<AliasedVecs&>(samples));
    }
    else
    {
      using AliasedPixels = PixelT[pixels_per_thread];
      WrappedPixelIteratorT d_wrapped_pixels(reinterpret_cast<PixelT*>(d_native_samples + block_offset));
      // Load using a wrapped pixel iterator
      BlockLoadPixelT{static_smem_storage.pixel_load}.Load(d_wrapped_pixels, reinterpret_cast<AliasedPixels&>(samples));
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
        BlockLoadSampleT{static_smem_storage.sample_load}.Load(
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
        BlockLoadPixelT{static_smem_storage.pixel_load}.Load(
          d_wrapped_pixels, reinterpret_cast<AliasedPixels&>(samples), valid_pixels);
      }
      else
      {
        using AliasedSamples = SampleT[samples_per_thread];
        BlockLoadSampleT{static_smem_storage.sample_load}.Load(
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
        is_valid[pixel] = IsFullTile || (((threadIdx.x + threads_per_block * pixel) * NumChannels) < valid_samples);
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
    MarkValid<IsFullTile, sweep.load_algorithm == BLOCK_LOAD_STRIPED>(is_valid, valid_samples);

    AccumulatePixels(samples, is_valid, ::cuda::std::bool_constant<is_rle_compress>{});
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
    int tile_idx                 = static_cast<int>((blockIdx.y * gridDim.x) + blockIdx.x);
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
        static_smem_storage.tile_idx = tile_queue.Drain(1) + num_even_share_tiles;
      }

      __syncthreads();

      tile_idx = static_smem_storage.tile_idx;
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
    for (int row = static_cast<int>(blockIdx.y); row < num_rows; row += static_cast<int>(gridDim.y))
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
  //! @param static_smem_storage
  //!   Shared storage used by the agent
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
  //! @param output_histogram
  //!   Reference to final output histograms
  //!
  //! @param gmem_privatized_histograms
  //!   Global-memory privatized histograms, or `nullptr` entries for a shared-memory mode
  //!
  //! @param output_decode_op
  //!   The transform operator for determining output bin-ids from privatized counter indices, one for each channel
  //!
  //! @param privatized_decode_op
  //!   The transform operator for determining privatized counter indices from samples, one for each channel
  //!
  //! @param dyn_smem_privatized_histograms
  //!   Base of the runtime-sized shared-memory histogram, or `nullptr` for a static-SMEM or global-memory mode
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentHistogram(
    TempStorage& static_smem_storage,
    SampleIteratorT d_samples,
    const int* num_output_bins,
    const int* num_privatized_bins,
    OutputCounterT** output_histogram,
    CounterT** gmem_privatized_histograms,
    const OutputDecodeOpT* output_decode_op,
    PrivatizedDecodeOpT* privatized_decode_op,
    CounterT* dyn_smem_privatized_histograms)
      : static_smem_storage(static_smem_storage.Alias())
      , d_wrapped_samples(d_samples)
      , d_native_samples(NativePointer(d_wrapped_samples))
      , num_output_bins(num_output_bins)
      , num_privatized_bins(num_privatized_bins)
      , output_histogram(output_histogram)
      , output_decode_op(output_decode_op)
      , privatized_decode_op(privatized_decode_op)
  {
    if constexpr (uses_dynamic_smem)
    {
      _CCCL_ASSERT(dyn_smem_privatized_histograms != nullptr, "Dynamic-SMEM mode requires a shared-memory base");
      CounterT* channel_histogram = dyn_smem_privatized_histograms;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        this->dyn_smem_privatized_histograms[ch] = channel_histogram;
        channel_histogram += num_privatized_bins[ch];
      }
    }
    else if constexpr (uses_gmem)
    {
      const int block_id = static_cast<int>((blockIdx.y * gridDim.x) + blockIdx.x);
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        const auto offset                    = static_cast<::cuda::std::int64_t>(block_id) * num_privatized_bins[ch];
        this->gmem_privatized_histograms[ch] = gmem_privatized_histograms[ch] + offset;
      }
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
    constexpr int vec_mask   = alignof(VecT) - 1;
    constexpr int pixel_mask = alignof(PixelT) - 1;
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
    ZeroBinCounters();
  }

  //! Store privatized histogram to device-accessible memory.  Specialized for privatized shared-memory counters
  _CCCL_DEVICE _CCCL_FORCEINLINE void StoreOutput()
  {
    // Barrier to make sure all threads are done updating counters
    __syncthreads();

    // Apply privatized bin counts to output bin counts
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      CounterT* privatized_histogram = PrivatizedHistogram(ch);
      const int channel_bins         = num_privatized_bins[ch];
      for (int bin = static_cast<int>(threadIdx.x); bin < channel_bins; bin += threads_per_block)
      {
        int output_bin       = -1;
        const CounterT count = privatized_histogram[bin];
        const bool is_valid  = count > 0;
        output_decode_op[ch].template BinSelect<load_modifier>(static_cast<SampleT>(bin), output_bin, is_valid);

        if (output_bin >= 0)
        {
          atomicAdd(&output_histogram[ch][output_bin], static_cast<OutputCounterT>(count));
        }
      }
    }
  }
};
} // namespace detail::histogram

CUB_NAMESPACE_END
