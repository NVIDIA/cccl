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

enum BlockHistogramMemoryPreference
{
  GMEM,
  SMEM,
  BLEND
};

namespace detail::histogram
{
// Warp-level same-bin coalescing for atomic accumulation.
//
// When several lanes of a warp classify a sample into the same bin, folding
// their increments into one atomic (issued by a single leader lane, with the
// peer count as the increment) collapses up to 32 contended atomics on a hot
// bin into one. This is the dual of intra-thread RLE (`rle_compress`): RLE
// coalesces a thread's consecutive same-bin samples, warp-coalescing coalesces
// a warp's same-bin lanes. It is the win on the high-latency GMEM-privatized /
// direct-atomic paths, where contention on hot output bins dominates.
//
// `WarpCoalesce` gates the mechanism so it is controlled by one policy flag
// rather than hard-coded at each call site (mirroring `rle_compress`):
//   - true  (SM70+): leader-elect via __match_any_sync and apply once.
//   - false, or pre-SM70: every valid lane applies its own increment of 1.
// `apply(bin, count)` is invoked on exactly the lane(s) that should issue the
// atomic, with `count` summed appropriately. Bins < 0 are skipped.
template <bool WarpCoalesce, typename ApplyFn>
_CCCL_DEVICE _CCCL_FORCEINLINE void warp_coalesce_atomic(int lane_id, int bin, ApplyFn apply)
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_70,
    (if constexpr (WarpCoalesce) {
       const unsigned int peers = __match_any_sync(0xffffffffu, static_cast<unsigned int>(bin));
       const int leader         = __ffs(static_cast<int>(peers)) - 1;
       if (bin >= 0 && lane_id == leader)
       {
         apply(bin, __popc(peers));
       }
     } else {
       if (bin >= 0)
       {
         apply(bin, 1);
       }
     }),
    (// Pre-SM70: no warp-coalesce primitive; each valid lane applies its own.
     (void) lane_id;
     if (bin >= 0) { apply(bin, 1); }));
}
} // namespace detail::histogram

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
//!
//! @tparam ThreadsPerBlock
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
//!
//! @tparam WarpCoalesce
//!   Whether to coalesce a warp's same-bin lanes into one atomic on the
//!   GMEM-privatized / direct-atomic paths (the dual of RLE; see
//!   `warp_coalesce_atomic`). Defaults on; only meaningfully disabled for study.
template <int ThreadsPerBlock,
          int PixelsPerThread,
          BlockLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadModifier,
          bool RleCompress,
          BlockHistogramMemoryPreference MemoryPreference,
          bool WorkStealing,
          int VecSize        = 4,
          bool WarpCoalesce  = true>
struct agent_histogram_policy
{
  /// Threads per thread block
  static constexpr int BLOCK_THREADS = ThreadsPerBlock;
  /// Pixels per thread (per tile of input)
  static constexpr int PIXELS_PER_THREAD = PixelsPerThread;

  /// Whether to perform localized RLE to compress samples before histogramming
  static constexpr bool IS_RLE_COMPRESS = RleCompress;

  /// Whether to coalesce a warp's same-bin lanes into one atomic (GMEM-priv path)
  static constexpr bool IS_WARP_COALESCE = WarpCoalesce;

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
          int PixelsPerThread,
          BlockLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadModifier,
          bool RleCompress,
          BlockHistogramMemoryPreference MemoryPreference,
          bool WorkStealing,
          int VecSize = 4>
using AgentHistogramPolicy
  CCCL_DEPRECATED_BECAUSE("Use the tuning API for DeviceHistogram") = detail::agent_histogram_policy<
    ThreadsPerBlock,
    PixelsPerThread,
    LoadAlgorithm,
    LoadModifier,
    RleCompress,
    MemoryPreference,
    WorkStealing,
    VecSize>;

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
//!
//! @tparam UseDynamicSmemHistogram
//!   When true, the privatized histogram storage lives in dynamic shared memory (passed in
//!   via an external pointer) rather than in the agent's static `_TempStorage`. This is used
//!   to lift the per-block bin count above the ptxas default 48 KB static-SMEM cap on
//!   architectures that support large dynamic SMEM (e.g. SM100 supports up to ~228 KiB
//!   per CTA via cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)).
//!   When true, `_TempStorage::histograms` becomes a per-channel pointer array initialized
//!   from a caller-supplied extern shared-memory base pointer; all accumulate / init / store
//!   paths still index `histograms[ch][bin]` and remain unchanged.
template <typename AgentHistogramPolicyT,
          int PrivatizedSmemBins,
          int NumChannels,
          int NumActiveChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT,
          bool UseDynamicSmemHistogram = false>
struct AgentHistogram
{
  static constexpr int vec_size                    = AgentHistogramPolicyT::VEC_SIZE;
  static constexpr int threads_per_block           = AgentHistogramPolicyT::BLOCK_THREADS;
  static constexpr int pixels_per_thread           = AgentHistogramPolicyT::PIXELS_PER_THREAD;
  static constexpr int samples_per_thread          = pixels_per_thread * NumChannels;
  static constexpr int vecs_per_thread             = samples_per_thread / vec_size;
  static constexpr int tile_pixels                 = pixels_per_thread * threads_per_block;
  static constexpr int tile_samples                = samples_per_thread * threads_per_block;
  static constexpr bool is_rle_compress            = AgentHistogramPolicyT::IS_RLE_COMPRESS;
  static constexpr bool is_warp_coalesce           = AgentHistogramPolicyT::IS_WARP_COALESCE;
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
  using BlockLoadSampleT =
    BlockLoad<SampleT, threads_per_block, samples_per_thread, AgentHistogramPolicyT::LOAD_ALGORITHM>;
  using BlockLoadPixelT =
    BlockLoad<PixelT, threads_per_block, pixels_per_thread, AgentHistogramPolicyT::LOAD_ALGORITHM>;
  using BlockLoadVecT = BlockLoad<VecT, threads_per_block, vecs_per_thread, AgentHistogramPolicyT::LOAD_ALGORITHM>;

  // Histogram storage type. With static SMEM, we store the histogram inline in the
  // agent's _TempStorage. With dynamic SMEM (UseDynamicSmemHistogram == true), we
  // store only a per-channel pointer array; the actual bin storage lives in extern
  // __shared__ memory allocated by the caller's kernel launch (via the third
  // triple-chevron parameter, with `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
  // set so the launch is permitted to use more than the 48 KB ptxas default).
  using HistogramsStorageT =
    ::cuda::std::_If<UseDynamicSmemHistogram, CounterT* [NumActiveChannels], CounterT[NumActiveChannels][PrivatizedSmemBins + 1]>;

  struct _TempStorage
  {
    // SMEM holding (or pointing at) per-block-privatized histogram.
    // - Static path: a CounterT[NumActiveChannels][PrivatizedSmemBins+1] inline array (with 1 word of padding).
    // - Dynamic path: a CounterT*[NumActiveChannels] pointer array; bins live in extern __shared__.
    HistogramsStorageT histograms;
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
  const int* num_output_bins; // one for each channel
  const int* num_privatized_bins; // one for each channel
  CounterT* d_privatized_histograms[NumActiveChannels]; // one for each channel
  CounterT** d_output_histograms; // in global memory
  const OutputDecodeOpT* output_decode_op; // determines output bin-id from privatized counter index, one for each
                                           // channel
  const PrivatizedDecodeOpT* privatized_decode_op; // determines privatized counter index from sample, one for each
                                                   // channel
  bool prefer_smem; // for privatized counterss

  // Hybrid SMEM+GMEM single-pass mode (used by the hybrid kernel for Bins=60000
  // single-channel). Each block has a per-block GMEM staging slab containing the
  // "secondary" bin range [hybrid_split_bin, hybrid_split_bin + hybrid_secondary_size).
  // The primary range [0, hybrid_split_bin) lives in dyn-SMEM (the existing
  // `temp_storage.histograms[ch]` pointer set up by the dynamic-SMEM constructor).
  // When `hybrid_split_bin > 0`, AccumulatePixelsHybrid routes each pixel's bin
  // (from the un-chunked decode op) to either SMEM or per-block GMEM, eliminating
  // the second sample-read pass that the dual-chunk kernel pays.
  CounterT* d_hybrid_secondary_histograms[NumActiveChannels]; // per-block GMEM slab base, offset to this block
  int hybrid_split_bin; // bins < this go to SMEM; bins in [split, split+secondary_size) go to GMEM
  int hybrid_secondary_size; // size of the secondary (GMEM) range

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
        output_decode_op[ch].template BinSelect<load_modifier>(static_cast<SampleT>(bin), output_bin, is_valid);

        if (output_bin >= 0)
        {
          atomicAdd(&d_output_histograms[ch][output_bin], count);
        }
      }
    }
  }

  // Accumulate pixels.  Specialized for RLE compression.
  //
  // For GMEM-privatized paths on SM_70+ we drop the intra-thread RLE
  // compression in favour of warp-coalesced atomics: at each pixel
  // every warp lane participates in
  // `__match_any_sync(0xffffffffu, bin)` and the leader of each peer
  // group issues one `atomicAdd_block` with `__popc(peers)` as the
  // increment. This collapses up to 32 contended same-bin atomics on
  // a hot bin into 1, which is a much larger win than intra-thread
  // RLE compression of pixels_per_thread <= 16 neighbours when bin
  // counts are >= a few thousand and atomics target high-latency
  // GMEM-priv slabs.
  //
  // For SMEM-privatized paths (`PrivatizedSmemBins > 0` and runtime
  // `prefer_smem`) we keep the legacy intra-thread RLE because SMEM
  // atomics are cheap (~5 cycle latency) so the warp-coalesce
  // overhead dominates the saved-atomic count for low-bin configs
  // (e.g. Bins == 32).
  //
  // The 3-active-channel iteration order (channel-outer, pixel-inner)
  // is preserved for both paths.
  template <typename TwoDimSubscriptableCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void AccumulatePixels(
    SampleT samples[pixels_per_thread][NumChannels],
    bool is_valid[pixels_per_thread],
    TwoDimSubscriptableCounterT& privatized_histograms,
    ::cuda::std::true_type is_rle_compress)
  {
    // On the GMEM-privatized path (PrivatizedSmemBins == 0) the atomics target
    // high-latency global-memory bins, so coalescing a warp's same-bin lanes
    // into one atomic (the dual of this RLE overload) is the win. The SMEM path
    // (handled in the else branch) keeps intra-thread RLE instead: shared atomics
    // are cheap enough that the coalesce overhead does not pay off at low bins.
    if constexpr (PrivatizedSmemBins == 0)
    {
      const int lane_id = static_cast<int>(threadIdx.x & 0x1f);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int pixel = 0; pixel < pixels_per_thread; ++pixel)
        {
          int bin = -1;
          privatized_decode_op[ch].template BinSelect<load_modifier>(samples[pixel][ch], bin, is_valid[pixel]);
          detail::histogram::warp_coalesce_atomic<is_warp_coalesce>(lane_id, bin, [&](int b, int count) {
            NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
                              (atomicAdd_block(privatized_histograms[ch] + b, static_cast<CounterT>(count));),
                              (atomicAdd(privatized_histograms[ch] + b, static_cast<CounterT>(count));));
          });
        }
      }
    }
    else
    {
      // SMEM-privatized: keep legacy intra-thread RLE compression with
      // per-lane atomics on cheap shared-memory bins.
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        // Bin pixels
        int bins[pixels_per_thread];

        // The per-channel MRU bracket cache (4-arg BinSelect) accelerates the RANGE
        // SearchTransform's interpolate+clamp+verify ladder on a cache hit, and it
        // pays off on the DYNAMIC-SMEM tier (bins >= 512, classify-dominated). But on
        // the STATIC <=256-bin tier the kernel is occupancy/register-bound and the
        // classify is already cheap (tiny level array), so the cache's live state
        // (BracketCacheT = 2 LevelT + int, held across the unrolled pixels loop)
        // lowers occupancy and adds a per-sample compare with ~zero amortization --
        // measured ~10% SLOWER than upstream's plain UpperBound at bins 16/32/64. So
        // gate the cache on the tier: dynamic SMEM keeps the MRU; the static tier uses
        // the plain 3-arg BinSelect (byte-identical to upstream main's accumulate).
        // EVEN is unaffected either way (its ScaleTransform 4-arg BinSelect is a no-op
        // forwarder), and the GMEM-privatized path (PrivatizedSmemBins == 0) uses a
        // different accumulate function entirely.
        if constexpr (UseDynamicSmemHistogram)
        {
          // Per-channel MRU bracket cache, scoped to this channel iteration so only
          // ONE bracket is live at a time. It threads across the `pixels_per_thread`
          // consecutive same-channel classifies; low-entropy inputs have high
          // consecutive-sample bracket locality.
          typename PrivatizedDecodeOpT::BracketCacheT mru;

          _CCCL_PRAGMA_UNROLL_FULL()
          for (int pixel = 0; pixel < pixels_per_thread; ++pixel)
          {
            bins[pixel] = -1;
            privatized_decode_op[ch].template BinSelect<load_modifier>(
              samples[pixel][ch], bins[pixel], is_valid[pixel], mru);
          }
        }
        else
        {
          // Static <=256-bin tier: plain 3-arg BinSelect, no MRU register state.
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int pixel = 0; pixel < pixels_per_thread; ++pixel)
          {
            bins[pixel] = -1;
            privatized_decode_op[ch].template BinSelect<load_modifier>(
              samples[pixel][ch], bins[pixel], is_valid[pixel]);
          }
        }

        CounterT accumulator = 1;

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int pixel = 0; pixel < pixels_per_thread - 1; ++pixel)
        {
          if (bins[pixel] != bins[pixel + 1])
          {
            if (bins[pixel] >= 0)
            {
              NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
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
          NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
                            (atomicAdd_block(privatized_histograms[ch] + bins[pixels_per_thread - 1], accumulator);),
                            (atomicAdd(privatized_histograms[ch] + bins[pixels_per_thread - 1], accumulator);));
        }
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
          NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
                            (atomicAdd_block(privatized_histograms[ch] + bin, 1);),
                            (atomicAdd(privatized_histograms[ch] + bin, 1);));
        }
      }
    }
  }

  //! Hybrid SMEM+GMEM accumulation. Routes each pixel's bin (computed from the
  //! un-chunked decode op) to either the per-channel SMEM histogram for the
  //! primary range `[0, hybrid_split_bin)` or to the per-block per-channel GMEM
  //! staging slab for the secondary range `[hybrid_split_bin, hybrid_split_bin
  //! + hybrid_secondary_size)`. Both atomics are CTA-scoped via `atomicAdd_block`,
  //! which is cheap for SMEM and reasonably cheap for per-block GMEM (no cross-CTA
  //! coherence required). RLE-compressed variant: applies the existing intra-thread
  //! RLE compression to consecutive same-bin pixels, but the flush splits between
  //! SMEM and GMEM based on the bin value.
  _CCCL_DEVICE _CCCL_FORCEINLINE void AccumulatePixelsHybrid(
    SampleT samples[pixels_per_thread][NumChannels],
    bool is_valid[pixels_per_thread],
    ::cuda::std::true_type is_rle_compress)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
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
          const int b = bins[pixel];
          if (b >= 0)
          {
            if (b < hybrid_split_bin)
            {
              NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
                                (atomicAdd_block(temp_storage.histograms[ch] + b, accumulator);),
                                (atomicAdd(temp_storage.histograms[ch] + b, accumulator);));
            }
            else
            {
              const int gbin = b - hybrid_split_bin;
              NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
                                (atomicAdd_block(d_hybrid_secondary_histograms[ch] + gbin, accumulator);),
                                (atomicAdd(d_hybrid_secondary_histograms[ch] + gbin, accumulator);));
            }
          }
          accumulator = 0;
        }
        accumulator++;
      }

      // Last pixel
      const int b = bins[pixels_per_thread - 1];
      if (b >= 0)
      {
        if (b < hybrid_split_bin)
        {
          NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
                            (atomicAdd_block(temp_storage.histograms[ch] + b, accumulator);),
                            (atomicAdd(temp_storage.histograms[ch] + b, accumulator);));
        }
        else
        {
          const int gbin = b - hybrid_split_bin;
          NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
                            (atomicAdd_block(d_hybrid_secondary_histograms[ch] + gbin, accumulator);),
                            (atomicAdd(d_hybrid_secondary_histograms[ch] + gbin, accumulator);));
        }
      }
    }
  }

  //! Hybrid SMEM+GMEM accumulation, non-RLE variant.
  _CCCL_DEVICE _CCCL_FORCEINLINE void AccumulatePixelsHybrid(
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
        int bin = -1;
        privatized_decode_op[ch].template BinSelect<load_modifier>(samples[pixel][ch], bin, is_valid[pixel]);
        if (bin >= 0)
        {
          if (bin < hybrid_split_bin)
          {
            NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
                              (atomicAdd_block(temp_storage.histograms[ch] + bin, 1);),
                              (atomicAdd(temp_storage.histograms[ch] + bin, 1);));
          }
          else
          {
            const int gbin = bin - hybrid_split_bin;
            NV_IF_ELSE_TARGET(NV_PROVIDES_SM_60,
                              (atomicAdd_block(d_hybrid_secondary_histograms[ch] + gbin, 1);),
                              (atomicAdd(d_hybrid_secondary_histograms[ch] + gbin, 1);));
          }
        }
      }
    }
  }

  //! Hybrid mode tile consumer.
  template <bool IsAligned, bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTileHybrid(OffsetT block_offset, int valid_samples)
  {
    SampleT samples[pixels_per_thread][NumChannels];
    bool is_valid[pixels_per_thread];

    LoadTile<IsFullTile, IsAligned>(block_offset, valid_samples, samples);
    MarkValid<IsFullTile, AgentHistogramPolicyT::LOAD_ALGORITHM == BLOCK_LOAD_STRIPED>(is_valid, valid_samples);

    AccumulatePixelsHybrid(samples, is_valid, ::cuda::std::bool_constant<is_rle_compress>{});
  }

  //! Hybrid mode ConsumeTiles - work-stealing variant.
  template <bool IsAligned>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTilesHybrid(
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
        OffsetT num_remaining = (num_row_pixels * NumChannels) - col_offset;
        ConsumeTileHybrid<IsAligned, false>(tile_offset, num_remaining);
      }
      else
      {
        ConsumeTileHybrid<IsAligned, true>(tile_offset, tile_samples);
      }

      __syncthreads();

      if (threadIdx.x == 0)
      {
        temp_storage.tile_idx = tile_queue.Drain(1) + num_even_share_tiles;
      }

      __syncthreads();

      tile_idx = temp_storage.tile_idx;
    }
  }

  //! Hybrid mode ConsumeTiles - even-share variant.
  template <bool IsAligned>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTilesHybrid(
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    int,
    GridQueue<int>,
    ::cuda::std::false_type)
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
          ConsumeTileHybrid<IsAligned, false>(tile_offset, num_remaining);
          break;
        }

        ConsumeTileHybrid<IsAligned, true>(tile_offset, tile_samples);
        tile_offset += gridDim.x * tile_samples;
      }
    }
  }

  //! Hybrid mode entry point.
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTilesHybrid(
    OffsetT num_row_pixels, OffsetT num_rows, OffsetT row_stride_samples, int tiles_per_row, GridQueue<int> tile_queue)
  {
    constexpr int vec_mask   = alignof(VecT) - 1;
    constexpr int pixel_mask = alignof(PixelT) - 1;
    const size_t row_bytes   = sizeof(SampleT) * row_stride_samples;

    const bool vec_aligned_rows =
      (NumChannels == 1) && (samples_per_thread % vec_size == 0)
      && ((size_t(d_native_samples) & vec_mask) == 0)
      && ((num_rows == 1) || ((row_bytes & vec_mask) == 0));

    const bool pixel_aligned_rows =
      (NumChannels > 1)
      && ((size_t(d_native_samples) & pixel_mask) == 0)
      && ((row_bytes & pixel_mask) == 0);

    _CCCL_PDL_GRID_DEPENDENCY_SYNC();

    if ((d_native_samples != nullptr) && (vec_aligned_rows || pixel_aligned_rows))
    {
      ConsumeTilesHybrid<true>(
        num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue, bool_constant_v<is_work_stealing>);
    }
    else
    {
      ConsumeTilesHybrid<false>(
        num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue, bool_constant_v<is_work_stealing>);
    }
  }

  //! Initialize hybrid mode counters: SMEM range [0, hybrid_split_bin) and per-block GMEM
  //! range [0, hybrid_secondary_size). Single sync at the end.
  //!
  //! Vectorized SMEM init: writes 4 counters per store (when CounterT is 4 bytes
  //! and the histogram base is 16-byte aligned). This roughly quarters the number
  //! of SMEM store instructions for the SMEM init pass on B200, where each SM has
  //! 256B-wide SMEM access.
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitBinCountersHybrid()
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      // Zero the SMEM primary range. If `hybrid_split_bin` is a multiple of 4 and
      // the histogram base pointer is 16-byte aligned (CounterT == int 4-byte),
      // use vectorized 4-wide writes. Otherwise fall back to scalar writes.
      if constexpr (sizeof(CounterT) == 4 && alignof(CounterT) == 4)
      {
        const int vec_count = hybrid_split_bin >> 2; // hybrid_split_bin / 4
        // hybrid_split_bin == 56000 is a multiple of 4 (56000 = 4 * 14000); the
        // dyn-SMEM extern base is 16-byte aligned by the CUDA runtime.
        uint4* const ptr4 = reinterpret_cast<uint4*>(temp_storage.histograms[ch]);
        const uint4 zero4 = {0u, 0u, 0u, 0u};
        for (int i = threadIdx.x; i < vec_count; i += threads_per_block)
        {
          ptr4[i] = zero4;
        }
        // Tail: write any leftover scalar bins (hybrid_split_bin & 3).
        for (int bin = (vec_count << 2) + threadIdx.x; bin < hybrid_split_bin; bin += threads_per_block)
        {
          temp_storage.histograms[ch][bin] = 0;
        }
      }
      else
      {
        for (int bin = threadIdx.x; bin < hybrid_split_bin; bin += threads_per_block)
        {
          temp_storage.histograms[ch][bin] = 0;
        }
      }

      // Zero the per-block GMEM secondary slab. Vectorize the same way; the slab
      // base is 16-byte aligned by alias_temporaries.
      if constexpr (sizeof(CounterT) == 4 && alignof(CounterT) == 4)
      {
        const int vec_count = hybrid_secondary_size >> 2;
        uint4* const ptr4 = reinterpret_cast<uint4*>(d_hybrid_secondary_histograms[ch]);
        const uint4 zero4 = {0u, 0u, 0u, 0u};
        for (int i = threadIdx.x; i < vec_count; i += threads_per_block)
        {
          ptr4[i] = zero4;
        }
        for (int bin = (vec_count << 2) + threadIdx.x; bin < hybrid_secondary_size; bin += threads_per_block)
        {
          d_hybrid_secondary_histograms[ch][bin] = 0;
        }
      }
      else
      {
        for (int bin = threadIdx.x; bin < hybrid_secondary_size; bin += threads_per_block)
        {
          d_hybrid_secondary_histograms[ch][bin] = 0;
        }
      }
    }

    __syncthreads();
  }

  //! Flush the SMEM primary histogram for hybrid mode to the per-block staging slab.
  //! After this call, both the primary slab (`d_privatized_histograms[ch][0..split)`)
  //! and the secondary slab (`d_hybrid_secondary_histograms[ch][0..secondary)`) hold
  //! this block's contributions for chunk0 and chunk1 respectively.
  //!
  //! Vectorized SMEM->GMEM flush: reads 4 counters per SMEM load and writes 4 per
  //! GMEM store (when CounterT is 4 bytes). This roughly quarters the load+store
  //! instruction count for the flush phase.
  _CCCL_DEVICE _CCCL_FORCEINLINE void StoreHybridSmemToStagingSlab()
  {
    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      if constexpr (sizeof(CounterT) == 4 && alignof(CounterT) == 4)
      {
        const int vec_count = hybrid_split_bin >> 2;
        const uint4* const src4 = reinterpret_cast<const uint4*>(temp_storage.histograms[ch]);
        uint4* const dst4 = reinterpret_cast<uint4*>(d_privatized_histograms[ch]);
        for (int i = threadIdx.x; i < vec_count; i += threads_per_block)
        {
          dst4[i] = src4[i];
        }
        // Tail
        for (int bin = (vec_count << 2) + threadIdx.x; bin < hybrid_split_bin; bin += threads_per_block)
        {
          d_privatized_histograms[ch][bin] = temp_storage.histograms[ch][bin];
        }
      }
      else
      {
        for (int bin = threadIdx.x; bin < hybrid_split_bin; bin += threads_per_block)
        {
          d_privatized_histograms[ch][bin] = temp_storage.histograms[ch][bin];
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
    static_assert(!UseDynamicSmemHistogram,
                  "AgentHistogram with UseDynamicSmemHistogram=true requires the dynamic-SMEM "
                  "constructor that takes an extern __shared__ base pointer.");

    const int blockId = static_cast<int>((blockIdx.y * gridDim.x) + blockIdx.x);

    // TODO(bgruber): d_privatized_histograms seems only used when !prefer_smem, can we skip it if prefer_smem?
    // Initialize the locations of this block's privatized histograms
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      const auto offset                 = static_cast<::cuda::std::int64_t>(blockId) * num_privatized_bins[ch];
      this->d_privatized_histograms[ch] = d_privatized_histograms[ch] + offset;
    }
  }

  //! Dynamic-SMEM constructor.
  //!
  //! Used when `UseDynamicSmemHistogram == true`. The caller's kernel allocates a contiguous
  //! `extern __shared__ CounterT[]` block of size sum(num_privatized_bins[ch]) entries and
  //! passes its base pointer here. We initialize per-channel pointers in `_TempStorage::histograms`
  //! so the existing accumulate / init / store paths can index `histograms[ch][bin]` unchanged.
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentHistogram(
    TempStorage& temp_storage,
    SampleIteratorT d_samples,
    const int* num_output_bins,
    const int* num_privatized_bins,
    CounterT** d_output_histograms,
    CounterT** d_privatized_histograms,
    const OutputDecodeOpT* output_decode_op,
    const PrivatizedDecodeOpT* privatized_decode_op,
    CounterT* dyn_smem_histogram_base)
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
    static_assert(UseDynamicSmemHistogram,
                  "Dynamic-SMEM AgentHistogram constructor requires UseDynamicSmemHistogram=true.");

    const int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;

    // Initialize the locations of this block's privatized GMEM histograms.
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      this->d_privatized_histograms[ch] = d_privatized_histograms[ch] + (blockId * num_privatized_bins[ch]);
    }

    // Initialize per-channel SMEM pointers from the extern __shared__ base. Channels are laid
    // out contiguously: ch=0 starts at base, ch=1 starts at base + num_privatized_bins[0], etc.
    // num_privatized_bins is the per-channel bin count and is a small int array passed in
    // grid-constant memory, so this loop is cheap.
    CounterT* p = dyn_smem_histogram_base;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      this->temp_storage.histograms[ch] = p;
      p += num_privatized_bins[ch];
    }
  }

  //! Hybrid SMEM+GMEM constructor (for the hybrid single-pass kernel).
  //!
  //! Used when `UseDynamicSmemHistogram == true` and the caller wants a hybrid
  //! split: bins `[0, hybrid_split_bin)` accumulate in dyn-SMEM (sized for `hybrid_split_bin`
  //! per channel), and bins `[hybrid_split_bin, hybrid_split_bin + hybrid_secondary_size)`
  //! accumulate in a per-block per-channel GMEM staging slab.
  //!
  //! `d_secondary_histograms[ch]` is the all-blocks staging-slab base for channel ch,
  //! the same shape as `d_privatized_histograms[ch]` but sized for `hybrid_secondary_size`
  //! bins per block. Both `d_privatized_histograms[ch]` and `d_secondary_histograms[ch]`
  //! are offset to this block's slab inside the constructor.
  //!
  //! `num_privatized_bins[ch]` here is the SMEM (primary) bin count per channel, which
  //! equals `hybrid_split_bin` for the simple equal-channels case used by the hybrid kernel.
  //! `hybrid_secondary_size` is the GMEM (secondary) bin count per channel.
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentHistogram(
    TempStorage& temp_storage,
    SampleIteratorT d_samples,
    const int* num_output_bins,
    const int* num_privatized_bins,
    CounterT** d_output_histograms,
    CounterT** d_privatized_histograms,
    CounterT** d_secondary_histograms,
    const OutputDecodeOpT* output_decode_op,
    const PrivatizedDecodeOpT* privatized_decode_op,
    CounterT* dyn_smem_histogram_base,
    int hybrid_split_bin_arg,
    int hybrid_secondary_size_arg)
      : temp_storage(temp_storage.Alias())
      , d_wrapped_samples(d_samples)
      , d_native_samples(NativePointer(d_wrapped_samples))
      , num_output_bins(num_output_bins)
      , num_privatized_bins(num_privatized_bins)
      , d_output_histograms(d_output_histograms)
      , output_decode_op(output_decode_op)
      , privatized_decode_op(privatized_decode_op)
      , prefer_smem(true)
      , hybrid_split_bin(hybrid_split_bin_arg)
      , hybrid_secondary_size(hybrid_secondary_size_arg)
  {
    static_assert(UseDynamicSmemHistogram,
                  "Hybrid AgentHistogram constructor requires UseDynamicSmemHistogram=true.");

    const int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;

    // Initialize per-channel per-block primary GMEM staging slab pointers (sized for hybrid_split_bin).
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      this->d_privatized_histograms[ch] =
        d_privatized_histograms[ch] + (blockId * hybrid_split_bin_arg);
    }

    // Initialize per-channel per-block secondary GMEM slab pointers (sized for hybrid_secondary_size).
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      this->d_hybrid_secondary_histograms[ch] =
        d_secondary_histograms[ch] + (blockId * hybrid_secondary_size_arg);
    }

    // Initialize per-channel SMEM pointers from the extern __shared__ base. Channels are laid
    // out contiguously: ch=0 starts at base, ch=1 starts at base + hybrid_split_bin, etc.
    CounterT* p = dyn_smem_histogram_base;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      this->temp_storage.histograms[ch] = p;
      p += hybrid_split_bin_arg;
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

    // NOTE: `_CCCL_PDL_TRIGGER_NEXT_LAUNCH` was previously called here, but
    // for the staging dispatch paths the kernel calls `StoreSmemToStagingSlab`
    // (per-block SMEM->GMEM flush) AFTER `ConsumeTiles`, and the follow-on
    // combine kernel reads from those staging slabs. Triggering the next
    // launch here releases the combine kernel before the staging slabs are
    // written, causing intermittent multi-channel test failures (~30%) in
    // cub.test.device.histogram.lid_0. The trigger is now emitted by each
    // kernel call site in kernel_histogram.cuh after all of its work that
    // the next kernel depends on has completed.
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

  //! Copy the privatized SMEM histogram to this block's per-block GMEM staging slab.
  //!
  //! Used by the staging dispatch path: instead of doing per-block atomicAdd into the
  //! global output histogram, each block leaves its privatized histogram in GMEM as a
  //! per-block staging slab. A follow-on combine kernel reduces across blocks.
  //!
  //! Only meaningful when prefer_smem is true (SMEM-privatized path); for the
  //! GMEM-privatized path the per-block histograms are already in GMEM.
  _CCCL_DEVICE _CCCL_FORCEINLINE void StoreSmemToStagingSlab()
  {
    // Barrier to make sure all SMEM updates have completed
    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      const int channel_bins = num_privatized_bins[ch];
      for (int bin = threadIdx.x; bin < channel_bins; bin += threads_per_block)
      {
        d_privatized_histograms[ch][bin] = temp_storage.histograms[ch][bin];
      }
    }
  }
};
} // namespace detail::histogram

CUB_NAMESPACE_END
