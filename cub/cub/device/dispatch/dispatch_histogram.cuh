// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 *   cub::DeviceHistogram provides device-wide parallel operations for constructing histogram(s)
 *   from a sequence of samples data residing within device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>

#include <cuda/std/__type_traits/is_void.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_histogram.cuh>
#include <cub/device/dispatch/kernels/kernel_histogram.cuh>
#include <cub/device/dispatch/tuning/tuning_histogram.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__functional/proclaim_return_type.h>
#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/transform.h>
#include <cuda/std/__tuple_dir/apply.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/array>
#include <cuda/std/limits>
#include <cuda/std/tuple>

#include <cstdio>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::histogram
{
// Maximum number of bins per channel for which we will use a privatized smem strategy
static constexpr int max_privatized_smem_bins = 256;

// Extended SMEM-privatized tiers for single-channel non-byte sample histograms. These tiers
// scale the per-block SMEM histogram allocation up so the agent can keep larger histograms
// on chip (avoiding the slow GMEM atomicAdd_block of the GMEM-privatized path) at the cost
// of extra per-block SMEM and reduced occupancy.
//
// We carry three tiers:
//   - 2048 bins x 4 bytes = 8 KB SMEM/block (static SMEM), plenty of occupancy headroom.
//   - 8192 bins x 4 bytes = 32 KB SMEM/block (static SMEM), fits within ptxas default
//     static-SMEM cap (48 KB).
//   - 16384 bins x 4 bytes = 64 KB SMEM/block (dynamic SMEM). Static SMEM exceeds the
//     ptxas default 48 KB cap, so this tier uses extern __shared__ + cudaFuncSetAttribute
//     (cudaFuncAttributeMaxDynamicSharedMemorySize). On SM90/SM100 the per-CTA SMEM
//     budget is large enough (B200 supports ~228 KiB per CTA) for this tier to launch
//     with reasonable occupancy.
static constexpr int max_extended_smem_bins_single_channel        = 2048;
static constexpr int max_extended_smem_bins_single_channel_large  = 8192;
static constexpr int max_extended_smem_bins_single_channel_xlarge = 16384;

// Chunked dyn-SMEM staging-fused tier for histograms with bins exceeding the dyn-SMEM
// xlarge tier (16384). Each chunk runs the existing dyn-SMEM staging-combine path over
// a `chunk_size`-bin slice of the privatized-bin space, with the per-chunk decode op
// remapping samples outside the slice to bin -1. Trades `num_chunks` x sample reads
// for `num_chunks` x SMEM atomicAdd_block (instead of 1x GMEM atomicAdd_block on the
// legacy persistent-kernel GMEM-priv path).
//
// Single-channel chunk_size choice (per worker-1 brief-6 empirical sweep): 2 chunks
// of 30000 (2-chunk) wins for the 60000-bin EVEN axis. 3-chunk of 20000 (-4.7% on
// even.base vs 2-chunk) and 4-chunk of 15000 (-6.3% vs 2-chunk) lose: more launches
// and more wasted classify+sample-read passes dominate the SMEM atomicAdd_block savings.
// Per-block dyn-SMEM at 32768 bins is 131 KB single-channel, well within B200's
// ~228 KiB per-CTA cap.
//
// Iteration 2: bumped from 30000 to 32768 (a clean power-of-2). Only the first chunk
// changes effective size when bins=60000 (32768 vs 30000); the second chunk drops from
// 30000 used to 27232 used. Slightly larger first-chunk SMEM atomic surface should
// reduce per-bin contention probability.
static constexpr int chunked_smem_chunk_size_single_channel = 32768;
static constexpr int chunked_smem_num_chunks_single_channel = 2;
static constexpr int chunked_smem_bins_max_single_channel =
  chunked_smem_chunk_size_single_channel * chunked_smem_num_chunks_single_channel;

// Hybrid SMEM+GMEM split point: bins [0, hybrid_split) live in per-block dyn-SMEM;
// bins [hybrid_split, max_total) live in per-block GMEM staging. Larger split fits
// more bins in fast SMEM at the cost of larger SMEM zeroing/flush overhead.
// 49152 = 192 KB/CTA dyn-SMEM (48 * 1024, power-of-2 aligned). For Bins=60000:
// SMEM = 49152 bins (82%), GMEM secondary = 10848 bins (18%). Trades smaller
// init/flush windows for more GMEM atomic traffic; whether that wins depends on
// the per-bin atomic distribution.
static constexpr int hybrid_smem_split_bin_single_channel = 49152;

// ---------------------------------------------------------------------------
// Histogram algorithm catalog and selector.
//
// Every dispatch decision in this file goes through the `algorithm` enum and
// the `select_algorithm` function below. The per-algorithm dispatch helpers
// (`dispatch<>`, `dispatch_hybrid_single_pass_staging_smem`) stay; they are
// launchers, not pickers. To add or retire a kernel, add an enumerator here,
// teach `select_algorithm` when to pick it, and add a `case` to
// `dispatch_by_algorithm`.
//
// Centralising the choice keeps the dispatch trade-off surface (single vs
// multi channel, EVEN vs RANGE classify cost, bin count, element count,
// sample width) in one place rather than scattered across cascades in two
// dispatch entry points.
enum class algorithm : unsigned char
{
  // Tier-0: bins <= 256, fits in the legacy fixed-size privatized SMEM.
  // The deeper `dispatch<>` chooses between the cooperative SweepPersistent
  // kernel (single launch) and the legacy Init+Sweep pair.
  smem_priv_256,

  // Tier-1/2/3: extended SMEM-priv tiers selected per channel count.
  smem_priv_2k, // 2048 bins (multi-channel medium, single-channel medium)
  smem_priv_8k, // 8192 bins (single-channel large)
  smem_priv_16k, // 16384 bins (single-channel xlarge, multi-channel up to 3 active)

  // Hybrid SMEM+GMEM single-pass kernel. One cooperative launch; bins in
  // (xlarge, chunked_max] for sizeof(SampleT)<=4 single-channel paths.
  hybrid_single_pass,

  // High-bin GMEM-priv path with the persistent direct-atomic-to-output
  // kernel + per-block SMEM cuckoo cache. Cache absorbs cross-block
  // contention for low-entropy hot-bin workloads.
  gmem_priv_cuckoo,

  // High-bin GMEM-priv path with the cooperative SweepPersistent kernel
  // (per-block privatised histograms + atomic-free gather merge after
  // grid.sync). Wins on bandwidth-bound large-input workloads where the
  // cuckoo cache's lookup chains stall.
  gmem_priv_sweep,

  // High-bin GMEM-priv path with the persistent direct-atomic-to-output
  // kernel + per-block SINGLE-PROBE direct-mapped SMEM cache. Identical to
  // gmem_priv_cuckoo except the cache probes exactly one slot (miss -> GMEM
  // atomic) instead of a 2-hash cuckoo chain. The shorter, less-divergent
  // critical section wins at HUGE element counts with very high bin counts
  // (single channel), where the cooperative sweep's per-block privatized
  // intermediate is DRAM-bound and privatization buys ~nothing (only a
  // handful of counts per bin), while the cuckoo chain's extra CAS/probe
  // work is wasted because the bins vastly outnumber the cache slots.
  gmem_priv_single_probe,
};

// Inputs to the selector. Every value used to make a dispatch decision must
// come through here.
struct selector_features
{
  int num_active_channels; // 1, 2, 3, 4
  int sample_bytes; // sizeof(SampleT)
  bool is_byte_sample; // sample_bytes == 1
  bool is_even; // EVEN entry point (true) or RANGE (false)
  int num_bins; // max_num_output_bins across active channels
  long long num_pixels; // total pixels per active channel
};

// Pick a single algorithm for one cell. Rules are first-match.
//
// The selector is split into two regions:
//
//   1. Low-bin region (num_bins <= 16384): SMEM-privatised tiers handle these.
//      Tiers cascade by bin count: smem_priv_256 -> 2K -> 8K -> 16K. Multi-
//      channel non-EVEN cuts the cascade short at 2K because the per-block
//      privatisation storage scales with NUM_ACTIVE_CHANNELS and the
//      SearchTransform classify cost dominates beyond that point.
//
//   2. High-bin region (num_bins > 16384): one of three algorithms runs.
//      Per the panel-based ablation analysis (see autocuda report on
//      2026-05-28):
//
//      Single-channel:
//        * num_pixels >= ~256M  -> sweep    (hybrid collapses at large input)
//        * num_bins  >  ~524K   -> cuckoo   (hybrid's tile gets too big)
//        * RANGE && sizeof==8   -> cuckoo   (binary-search classify hurts hybrid)
//        * else                 -> hybrid   (single-channel default)
//
//      Multi-channel:
//        * num_pixels >= ~256M  -> sweep    (large-input default)
//        * else                 -> cuckoo   (multi-channel default)
template <bool IsByteSample>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE algorithm select_algorithm(selector_features const& f)
{
  // Byte samples: 256-entry pass-thru privatized histograms, then a final
  // scale-transform combine. Bin counts above that are not legal for byte
  // samples (LevelT == SampleT and num_bins fits in [0, 255]).
  if constexpr (IsByteSample)
  {
    return algorithm::smem_priv_256;
  }

  // -----------------------------------------------------------------------
  // Low-bin region: SMEM-priv tier cascade.
  // -----------------------------------------------------------------------
  if (f.num_bins <= max_privatized_smem_bins)
  {
    return algorithm::smem_priv_256;
  }
  if (f.num_active_channels == 1)
  {
    if (f.num_bins <= max_extended_smem_bins_single_channel)
    {
      return algorithm::smem_priv_2k;
    }
    if (f.num_bins <= max_extended_smem_bins_single_channel_large)
    {
      return algorithm::smem_priv_8k;
    }
    if (f.num_bins <= max_extended_smem_bins_single_channel_xlarge)
    {
      return algorithm::smem_priv_16k;
    }
  }
  else
  {
    // Multi-channel SMEM-priv tiers: only the medium tier (2K) is universally
    // safe; the large/xlarge tiers stay in play for EVEN (cheap classify)
    // and only with at most 3 active channels (dyn-SMEM headroom).
    if (f.num_bins <= max_extended_smem_bins_single_channel)
    {
      return algorithm::smem_priv_2k;
    }
    if (f.is_even && f.num_bins <= max_extended_smem_bins_single_channel_large)
    {
      return algorithm::smem_priv_8k;
    }
    if (f.is_even && f.num_active_channels <= 3
        && f.num_bins <= max_extended_smem_bins_single_channel_xlarge)
    {
      return algorithm::smem_priv_16k;
    }
  }

  // -----------------------------------------------------------------------
  // High-bin region (num_bins > 16384).
  // -----------------------------------------------------------------------
  constexpr long long kSweepPixelThreshold = 1LL << 28; // ~256M pixels

  // Bin threshold above which the cooperative sweep's per-block privatized
  // intermediate becomes catastrophic at large input. The intermediate is
  // `num_blocks * num_bins * sizeof(CounterT)` bytes; on B200 num_blocks is
  // grid-capped at ~444, so at 1M bins it is ~1.78 GB -- written scattered,
  // read strided at a tiny L2 hit rate (DRAM-bound). At <=262144 bins the
  // intermediate is <=~445 MB and the privatization still nets a win across
  // the entropy mix (it absorbs the heavy hot-bin contention of constant /
  // skewed inputs, which the single-probe cache cannot -- a single hot bin
  // serialises through one cache slot, and a skewed hot set thrashes the
  // direct-mapped cache). Only at the 1M-bin tier does the single-probe path
  // win net: there the uniform case is ~5-6x faster (the catastrophic sweep
  // is the baseline) and the constant/skewed losses are bounded because the
  // 1.78 GB intermediate hurts the sweep too. Measured (same-conditions,
  // single-channel EVEN, 180-cell geomean): routing only >= this threshold is
  // +1.6%; routing all huge-N high-bin cells regresses -1.6% because the
  // 65536/262144-bin constant/skewed cells lose more than the uniform cells
  // gain. 524288 sits between the 262144 and 1048576 axis points.
  constexpr int kSingleProbeBinThreshold = 524288;

  // Multi-channel single-probe threshold. The cooperative sweep's per-channel
  // privatized intermediate is `num_blocks * num_bins * NUM_ACTIVE_CHANNELS *
  // sizeof(CounterT)` bytes -- a factor of NUM_ACTIVE_CHANNELS (3 here) LARGER
  // than the single-channel intermediate at the same bin count. So at the
  // 1M-bin tier the multi-channel intermediate is ~5.3 GB (444 blocks * 1M *
  // 3 * 4 B), even more catastrophic for sweep than single-channel's 1.78 GB
  // (nsys on the parent confirms: the 262144- and 1048576-bin RANGE cells at
  // 256M pixels run DeviceHistogramSweepKernel at ~50/61 ms, ~5x the cuckoo
  // cells, and dominate ~65% of high-bin multi_range GPU time). The
  // single-probe direct-mapped cache sidesteps the intermediate entirely, so
  // the same route that won for single-channel wins at least as much for
  // multi-channel. The 262144-bin multi cell's intermediate is ~1.3 GB (3x the
  // single-channel 262144 ~445 MB, which was ~neutral for single-channel), so
  // single-probe nets positive there for multi too. We route the 65536-bin
  // multi cells too (threshold 65536): nsys after the 131072 route showed the
  // 65536 multi RANGE cell at 256M uniform still runs DeviceHistogramSweepKernel
  // at ~28 ms -- its per-channel intermediate is ~334 MB but the privatized
  // gather merge is still DRAM-bound on the strided read at this scale, so
  // single-probe (which atomic-adds directly to the output and caches the few
  // hot bins) should win on the uniform/skewed cells. (worker-4 brief-3 left
  // 65536 on sweep based on a four-metric geomean dominated by single-channel,
  // where 65536 single-probe was a net loser; this brief is multi_range-scoped
  // and measures the 65536-multi route directly.)
  constexpr int kSingleProbeBinThresholdMulti = 65536;

  // Large-input cells.
  //
  // Single channel: route only the 1M-bin tier (>= kSingleProbeBinThreshold)
  // to the single-probe direct-mapped cache, which sidesteps the catastrophic
  // privatized intermediate entirely (it atomic-adds directly to the output,
  // absorbing the few genuinely hot bins in SMEM and streaming the rest to
  // GMEM). Lower bin counts keep sweep, whose privatization still wins net.
  //
  // Multi channel: route the 262144- and 1M-bin tiers (>=
  // kSingleProbeBinThresholdMulti) to single-probe as well -- the per-channel
  // intermediate is NUM_ACTIVE_CHANNELS-times larger, so sweep turns
  // catastrophic at a lower bin count than single-channel. The single-probe
  // kernel already loops over all active channels (1024 cache slots/channel
  // for multi vs 4096 for single, so the static-SMEM footprint and occupancy
  // are unchanged); dispatch verifies the kernel's cooperative co-residence
  // and falls back to sweep if it does not fit. Lower bin counts keep sweep.
  if (f.num_pixels >= kSweepPixelThreshold)
  {
    const int single_probe_threshold =
      (f.num_active_channels == 1) ? kSingleProbeBinThreshold : kSingleProbeBinThresholdMulti;
    if (f.num_bins >= single_probe_threshold)
    {
      return algorithm::gmem_priv_single_probe;
    }
    return algorithm::gmem_priv_sweep;
  }

  if (f.num_active_channels > 1)
  {
    // Multi-channel: hybrid is single-channel-only. Cuckoo is the small/
    // medium-input default.
    return algorithm::gmem_priv_cuckoo;
  }

  // Single-channel high-bin region. Hybrid is capped at
  // `chunked_smem_bins_max_single_channel` (the dispatch helper's
  // `kMaxTotalBins` parameter); above that, bins outside the
  // primary+secondary tile go uncounted. Also skip hybrid when the RANGE
  // classify cost would dominate (F64 SearchTransform).
  if (f.num_bins > chunked_smem_bins_max_single_channel)
  {
    return algorithm::gmem_priv_cuckoo;
  }
  if (!f.is_even && f.sample_bytes >= 8)
  {
    return algorithm::gmem_priv_cuckoo;
  }
  return algorithm::hybrid_single_pass;
}

template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          typename SampleIteratorT,
          typename CounterT,
          typename LevelT,
          typename OffsetT,
          typename SampleT>
struct DeviceHistogramKernelSource
{
  using TransformsT = detail::histogram::Transforms<LevelT, OffsetT, SampleT>;

  template <typename PolicyT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramInitKernel()
  {
    return &DeviceHistogramInitKernel<PolicyT, NUM_ACTIVE_CHANNELS, CounterT, OffsetT>;
  }

  /// Returns the default histogram sweep kernel that receives pre-initialized decode operators from the host.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepKernel()
  {
    return &DeviceHistogramSweepKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT>;
  }

  /// Returns the device-init histogram sweep kernel that initializes decode operators from level arrays in the kernel.
  template <typename PolicyT,
            int PRIVATIZED_SMEM_BINS,
            typename FirstLevelArrayT,
            typename SecondLevelArrayT,
            bool IsEven,
            bool IsByteSample>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepKernelDeviceInit()
  {
    // For DispatchEven, we use the scale transform to convert samples to
    // privatized bins and pass-thru transform to convert privatized bins to
    // output bins, vice verse for byte samples.

    // For DispatchRange, we use the search transform to convert samples to
    // privatized bins and scale transform to convert privatized bins to output bins,
    // vice verse for byte samples.

    using DecodeOpT = ::cuda::std::conditional_t<IsEven,
                                                 typename TransformsT::ScaleTransform,
                                                 typename TransformsT::template SearchTransform<const LevelT*>>;

    using PrivatizedDecodeOpT =
      ::cuda::std::conditional_t<IsByteSample, typename TransformsT::PassThruTransform, DecodeOpT>;
    using OutputDecodeOpT =
      ::cuda::std::conditional_t<IsByteSample, DecodeOpT, typename TransformsT::PassThruTransform>;

    return &DeviceHistogramSweepDeviceInitKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      FirstLevelArrayT,
      SecondLevelArrayT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT,
      IsEven>;
  }

  /// Returns the persistent grid-resident histogram sweep kernel that fuses
  /// output-histogram initialization with the sweep+store phase via a
  /// `cooperative_groups::this_grid().sync()` between them. The returned
  /// kernel must be launched cooperatively (`cudaLaunchCooperativeKernel`)
  /// so that all blocks are co-resident, which is a precondition of the
  /// grid sync. This is the host-init variant: it accepts pre-initialized
  /// decode operators, mirroring `HistogramSweepKernel`.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepKernelPersistent()
  {
    return &DeviceHistogramSweepPersistentKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT>;
  }

  /// Host-init variant of the staging histogram sweep kernel. Skips StoreOutput so
  /// a follow-on combine kernel handles cross-block reduction.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepStagingHostInitKernel()
  {
    return &DeviceHistogramSweepStagingHostInitKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT>;
  }

  /// Device-init staging variant of the histogram sweep kernel.
  template <typename PolicyT,
            int PRIVATIZED_SMEM_BINS,
            typename FirstLevelArrayT,
            typename SecondLevelArrayT,
            bool IsEven,
            bool IsByteSample>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepStagingKernelDeviceInit()
  {
    using DecodeOpT = ::cuda::std::conditional_t<IsEven,
                                                 typename TransformsT::ScaleTransform,
                                                 typename TransformsT::template SearchTransform<const LevelT*>>;

    using PrivatizedDecodeOpT =
      ::cuda::std::conditional_t<IsByteSample, typename TransformsT::PassThruTransform, DecodeOpT>;
    using OutputDecodeOpT =
      ::cuda::std::conditional_t<IsByteSample, DecodeOpT, typename TransformsT::PassThruTransform>;

    return &DeviceHistogramSweepStagingKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      FirstLevelArrayT,
      SecondLevelArrayT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT,
      IsEven>;
  }

  /// Combine kernel: reduces per-block privatized histograms across all blocks.
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramCombineKernel()
  {
    return &DeviceHistogramCombineKernel<NUM_ACTIVE_CHANNELS, CounterT>;
  }

  /// Host-init dynamic-SMEM variant of the staging histogram sweep kernel. Used for the
  /// xlarge tier (>48 KB SMEM/block) on architectures supporting larger dyn-SMEM via
  /// cudaFuncSetAttribute.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepStagingHostInitDynSmemKernel()
  {
    return &DeviceHistogramSweepStagingHostInitDynSmemKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT>;
  }

  /// Host-init dynamic-SMEM, NON-staging variant: merges each block's dyn-SMEM
  /// privatized histogram directly into the global output via atomicAdd
  /// (no staging slabs, no combine kernel). Host must launch the init kernel first.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepNonStagingDynSmemKernel()
  {
    return &DeviceHistogramSweepNonStagingDynSmemKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT>;
  }

  /// Host-init FUSED dynamic-SMEM staging+combine sweep kernel. Used for cooperative
  /// launch that fuses sweep+combine into one kernel via grid_group::sync().
  template <typename PolicyT,
            int PRIVATIZED_SMEM_BINS,
            typename PrivatizedDecodeOpT,
            typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepStagingFusedHostInitDynSmemKernel()
  {
    return &DeviceHistogramSweepStagingFusedHostInitDynSmemKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT>;
  }

  /// Host-init FUSED HYBRID single-pass dynamic-SMEM staging+combine sweep kernel. Eliminates
  /// the 2x sample re-read of the dual-chunk kernel by handling both bin "chunks" in a single
  /// sweep: bins in the primary range live in dyn-SMEM, bins in the secondary range live in
  /// per-block GMEM staging slabs. The decode op is the un-chunked privatized op, classifying
  /// each sample once into the full bin space.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto
  HistogramSweepStagingFusedHybridSinglePassHostInitDynSmemKernel()
  {
    return &DeviceHistogramSweepStagingFusedHybridSinglePassHostInitDynSmemKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT>;
  }

  /// Device-init dynamic-SMEM variant of the staging histogram sweep kernel.
  template <typename PolicyT,
            int PRIVATIZED_SMEM_BINS,
            typename FirstLevelArrayT,
            typename SecondLevelArrayT,
            bool IsEven,
            bool IsByteSample>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepStagingDynSmemKernelDeviceInit()
  {
    using DecodeOpT = ::cuda::std::conditional_t<IsEven,
                                                 typename TransformsT::ScaleTransform,
                                                 typename TransformsT::template SearchTransform<const LevelT*>>;

    using PrivatizedDecodeOpT =
      ::cuda::std::conditional_t<IsByteSample, typename TransformsT::PassThruTransform, DecodeOpT>;
    using OutputDecodeOpT =
      ::cuda::std::conditional_t<IsByteSample, DecodeOpT, typename TransformsT::PassThruTransform>;

    return &DeviceHistogramSweepStagingDynSmemKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      FirstLevelArrayT,
      SecondLevelArrayT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT,
      IsEven>;
  }

  CUB_RUNTIME_FUNCTION static constexpr size_t CounterSize()
  {
    return sizeof(CounterT);
  }

  template <typename NumBinsT, typename UpperLevelArrayT, typename LowerLevelArrayT>
  CUB_RUNTIME_FUNCTION static constexpr bool MayOverflow(
    [[maybe_unused]] NumBinsT num_bins,
    [[maybe_unused]] const UpperLevelArrayT& upper_level,
    [[maybe_unused]] const LowerLevelArrayT& lower_level,
    [[maybe_unused]] int channel)
  {
    using CommonT = typename TransformsT::ScaleTransform::CommonT;

    if constexpr (::cuda::std::is_integral_v<CommonT>)
    {
      using IntArithmeticT = typename TransformsT::ScaleTransform::IntArithmeticT;
      // Compute `upper - lower` without overflowing the level type. For
      // signed integer level types with full range (e.g. int32_t with
      // [INT_MIN, INT_MAX/4]), the signed difference overflows.
      // Reinterpret-cast each operand through its unsigned counterpart and
      // subtract in unsigned arithmetic; modular wrap-around produces the
      // correct (non-negative) difference.
      // Compute `upper - lower` without overflow. For signed integer
      // level types, the signed difference can overflow (e.g. int32_t
      // with [INT_MIN, INT_MAX]) and even narrow types are subject to
      // C++ integer promotion to `int`, which then converts back to
      // unsigned through sign-extension. We:
      //   1. Cast each operand to its unsigned counterpart of the same
      //      width (`make_unsigned_t<LevelT>`). This reinterprets the
      //      bit pattern: int8_t(-128..127) -> uint8_t(128..255, 0..127).
      //   2. Subtract through an unsigned-wraparound assignment. C++
      //      integer promotion forces the operands up to `int` for the
      //      subtraction, but assigning the result back to ULevelT
      //      truncates via unsigned modular wrap-around, producing the
      //      correct difference in [0, 2^N - 1]. Pre-promoting to the
      //      destination integer arithmetic type before subtraction would
      //      sign-extend the int promotion's negative result into the
      //      wider type and yield a giant garbage value.
      //   3. Widen the truncated unsigned difference to `IntArithmeticT`,
      //      which holds it without overflow.
      using ArrayLevelT = typename UpperLevelArrayT::value_type;
      using ULevelT     = ::cuda::std::make_unsigned_t<ArrayLevelT>;
      const ULevelT diff = static_cast<ULevelT>(static_cast<ULevelT>(upper_level[channel])
                                                - static_cast<ULevelT>(lower_level[channel]));
      const IntArithmeticT range = static_cast<IntArithmeticT>(diff);
      return range > (::cuda::std::numeric_limits<IntArithmeticT>::max() / static_cast<IntArithmeticT>(num_bins));
    }
    else
    {
      return false;
    }
  }
};

template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          int PRIVATIZED_SMEM_BINS,
          bool IsDeviceInit,
          bool IsEven,
          bool IsByteSample,
          typename SampleIteratorT,
          typename CounterT,
          typename FirstLevelArrayT,
          typename SecondLevelArrayT,
          typename OffsetT,
          typename PolicySelector,
          typename KernelSource,
          typename KernelLauncherFactory>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE auto dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_levels,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
  FirstLevelArrayT first_level_array,
  SecondLevelArrayT second_level_array,
  int max_num_output_bins,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_samples,
  cudaStream_t stream,
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {},
  // When the caller already picked between the direct-atomic-to-output
  // (cuckoo) path and the cooperative SweepPersistent (gather-merge) path
  // via the unified algorithm selector, this overrides the legacy
  // `direct_atomic_bin_threshold` heuristic. Set true to force the
  // SweepPersistent / Init+Sweep path regardless of bin count.
  bool disable_direct_atomic = false,
  // Cache policy for the direct-atomic-to-output kernel (only consulted when
  // the direct-atomic path is taken, i.e. !disable_direct_atomic):
  //   0 -> 2-hash cuckoo cache (DeviceHistogramSweepDirectAtomicPersistentKernel)
  //   1 -> single-probe direct-mapped cache
  //        (DeviceHistogramSweepDirectAtomicSingleProbePersistentKernel)
  // Both kernels share the same dynamic-SMEM cache layout and the
  // dispatch-chosen `cache_slots_per_channel`; only the probe policy differs.
  int direct_atomic_cache_mode = 0)
{
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  const HistogramPolicy active_policy = policy_selector(cc);

#if _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::stringstream ss;
                 ss << active_policy;
                 _CubLog("Dispatching DeviceHistogram to compute capability %d.%d with tuning: %s\n",
                         cc.major_cap(),
                         cc.minor_cap(),
                         ss.str().c_str());
               }))
#endif // _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)

  const auto init_kernel    = kernel_source.template HistogramInitKernel<PolicySelector>();
  const auto combine_kernel = kernel_source.HistogramCombineKernel();

  // Whether this dispatch uses the dynamic-SMEM staging tier. The xlarge tier
  // (16384 bins, 64 KB SMEM/block) MUST use dyn-SMEM because static SMEM
  // exceeds the ptxas 48 KB cap. The large (8192 bins, 32 KB) tier also uses
  // dyn-SMEM here so it shares the staging+fused-launch code path with xlarge.
  //
  // The medium (2048 bins, 8 KB) tier instead uses the NON-staging path:
  // static-SMEM privatization with a per-block atomicAdd StoreOutput merge to
  // the global histogram. For only 2048 bins the staging path's overhead --
  // a full grid.sync plus a GMEM round-trip (write each block's SMEM histogram
  // to a per-block staging slab, then read it back in the cross-block gather) --
  // is not amortised; the direct atomic merge touches half the GMEM traffic and
  // avoids the cooperative-launch grid.sync. (The brief's "which tiers earn
  // their place" question: at 2K bins, staging does not.)
  static constexpr bool kStagingUsesDynSmem =
    (PRIVATIZED_SMEM_BINS == max_extended_smem_bins_single_channel_xlarge)
    || (PRIVATIZED_SMEM_BINS == max_extended_smem_bins_single_channel_large);
  static constexpr bool kStagingChannelOk = (NUM_ACTIVE_CHANNELS >= 1 && NUM_ACTIVE_CHANNELS <= 4);
  static constexpr bool kStagingPrivOk    = kStagingUsesDynSmem;

  // Non-staging dyn-SMEM merge for the xlarge (16384-bin) tier: keep the
  // privatized histogram in dyn-SMEM (it exceeds the 48 KB static cap) but merge
  // each block directly into the global output via atomicAdd in StoreOutput,
  // skipping the per-block GMEM staging slab + grid.sync + cross-block gather.
  // At 16384 bins cross-block atomic contention on the output is spread over
  // 16384 distinct bins, so the direct merge avoids the staging GMEM round-trip
  // without paying heavy contention. The 8192-bin tier keeps staging.
  // (Extends the 2K-tier finding to the dyn-SMEM tier.)
  static constexpr bool kUseNonStagingDynSmem =
    kStagingUsesDynSmem && (PRIVATIZED_SMEM_BINS == max_extended_smem_bins_single_channel_xlarge);

  // Use the staging path for dyn-SMEM tiers EXCEPT the ones routed to the
  // non-staging dyn-SMEM merge above.
  static constexpr bool kUseStagingPath = kStagingChannelOk && kStagingPrivOk && !kUseNonStagingDynSmem;

  // Build the staging sweep kernel pointer for the dyn-SMEM xlarge tier. For other tiers this is unused.
  auto staging_sweep_kernel = [&] {
    if constexpr (kStagingUsesDynSmem)
    {
      if constexpr (IsDeviceInit)
      {
        return kernel_source.template HistogramSweepStagingDynSmemKernelDeviceInit<
               PolicySelector,
               PRIVATIZED_SMEM_BINS,
               FirstLevelArrayT,
               SecondLevelArrayT,
               IsEven,
               IsByteSample>();
      }
      else
      {
        using output_decode_op_t     = typename FirstLevelArrayT::value_type;
        using privatized_decode_op_t = typename SecondLevelArrayT::value_type;
        return kernel_source.template HistogramSweepStagingHostInitDynSmemKernel<
               PolicySelector,
               PRIVATIZED_SMEM_BINS,
               privatized_decode_op_t,
               output_decode_op_t>();
      }
    }
    else if constexpr (IsDeviceInit)
    {
      // Returned but unused for non-dyn-SMEM tiers; pick a kernel pointer with the same shape
      // so `decltype(staging_sweep_kernel)` is well-defined.
      return kernel_source.template HistogramSweepKernelDeviceInit<
             PolicySelector,
             PRIVATIZED_SMEM_BINS,
             FirstLevelArrayT,
             SecondLevelArrayT,
             IsEven,
             IsByteSample>();
    }
    else
    {
      using output_decode_op_t     = typename FirstLevelArrayT::value_type;
      using privatized_decode_op_t = typename SecondLevelArrayT::value_type;
      return kernel_source
        .template HistogramSweepKernel<PolicySelector, PRIVATIZED_SMEM_BINS, privatized_decode_op_t, output_decode_op_t>();
    }
  }();

  // For the dyn-SMEM xlarge tier, alias `sweep_kernel` to `staging_sweep_kernel` to avoid instantiating
  // the static-SMEM AgentHistogram (which would exceed the ptxas 48 KB cap for 16384 bins).
  // For the non-staging dyn-SMEM tier, use the dedicated dyn-SMEM kernel that
  // merges directly to the output via atomicAdd (StoreOutput) instead of staging.
  auto sweep_kernel = [&] {
    if constexpr (kUseNonStagingDynSmem)
    {
      if constexpr (IsDeviceInit)
      {
        // Device-init non-staging dyn-SMEM is not used by the active paths; alias
        // to the staging device-init kernel so decltype is well-defined. (The
        // non-staging dyn-SMEM tier is only selected on the host-init path.)
        return kernel_source.template HistogramSweepStagingDynSmemKernelDeviceInit<
               PolicySelector,
               PRIVATIZED_SMEM_BINS,
               FirstLevelArrayT,
               SecondLevelArrayT,
               IsEven,
               IsByteSample>();
      }
      else
      {
        using output_decode_op_t     = typename FirstLevelArrayT::value_type;
        using privatized_decode_op_t = typename SecondLevelArrayT::value_type;
        return kernel_source.template HistogramSweepNonStagingDynSmemKernel<
               PolicySelector,
               PRIVATIZED_SMEM_BINS,
               privatized_decode_op_t,
               output_decode_op_t>();
      }
    }
    else if constexpr (kStagingUsesDynSmem)
    {
      return staging_sweep_kernel;
    }
    else if constexpr (IsDeviceInit)
    {
      return kernel_source.template HistogramSweepKernelDeviceInit<
             PolicySelector,
             PRIVATIZED_SMEM_BINS,
             FirstLevelArrayT,
             SecondLevelArrayT,
             IsEven,
             IsByteSample>();
    }
    else
    {
      using output_decode_op_t     = typename FirstLevelArrayT::value_type;
      using privatized_decode_op_t = typename SecondLevelArrayT::value_type;
      return kernel_source
        .template HistogramSweepKernel<PolicySelector, PRIVATIZED_SMEM_BINS, privatized_decode_op_t, output_decode_op_t>();
    }
  }();

  const int threads_per_block = active_policy.threads_per_block;
  const int pixels_per_thread = active_policy.pixels_per_thread;

  // Get SM count
  int sm_count;
  if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
  {
    return error;
  }

  // Compute the dynamic-SMEM size for the staging-dyn-smem path. When staging is enabled and
  // PRIVATIZED_SMEM_BINS exceeds the ptxas static-SMEM cap, the per-block histogram lives in
  // extern __shared__; the launch must reserve enough dynamic SMEM and the kernel must have
  // its cudaFuncAttributeMaxDynamicSharedMemorySize attribute raised accordingly.
  // Layout: per-channel contiguous, sum_ch num_privatized_bins[ch] entries.
  int dyn_smem_bytes_for_staging = 0;
  if constexpr (kStagingUsesDynSmem)
  {
    int total_bins = 0;
    for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ++ch)
    {
      total_bins += (num_privatized_levels[ch] - 1);
    }
    dyn_smem_bytes_for_staging = total_bins * static_cast<int>(kernel_source.CounterSize());
  }

  // Get SM occupancy for sweep_kernel. For the staging-dyn-smem path, occupancy must be queried
  // with the dynamic-SMEM byte budget set so the driver accounts for the per-CTA SMEM footprint.
  int histogram_sweep_sm_occupancy;
  if constexpr (kStagingUsesDynSmem)
  {
    // Raise the kernel's max-dynamic-SMEM cap so the occupancy query accounts for the dyn-SMEM
    // CTA footprint. (The cap also has to be raised before the actual launch below.)
    if (const auto error =
          CubDebug(launcher_factory.set_max_dynamic_smem_size_for(sweep_kernel, dyn_smem_bytes_for_staging)))
    {
      return error;
    }
    if (const auto error = CubDebug(launcher_factory.MaxSmOccupancy(
          histogram_sweep_sm_occupancy, sweep_kernel, threads_per_block, dyn_smem_bytes_for_staging)))
    {
      return error;
    }
  }
  else
  {
    if (const auto error =
          CubDebug(launcher_factory.MaxSmOccupancy(histogram_sweep_sm_occupancy, sweep_kernel, threads_per_block)))
    {
      return error;
    }
  }

  // Get device occupancy for sweep_kernel
  int histogram_sweep_occupancy = histogram_sweep_sm_occupancy * sm_count;

  if (num_row_pixels * NUM_CHANNELS == row_stride_samples)
  {
    // Treat as a single linear array of samples
    num_row_pixels *= num_rows;
    num_rows           = 1;
    row_stride_samples = num_row_pixels * NUM_CHANNELS;
  }

  // Get grid dimensions, trying to keep total blocks ~histogram_sweep_occupancy
  int pixels_per_tile = threads_per_block * pixels_per_thread;
  int tiles_per_row   = static_cast<int>(::cuda::ceil_div(num_row_pixels, pixels_per_tile));
  int blocks_per_row  = ::cuda::std::min(histogram_sweep_occupancy, tiles_per_row);
  int blocks_per_col =
    (blocks_per_row > 0)
      ? int(::cuda::std::min(static_cast<OffsetT>(histogram_sweep_occupancy / blocks_per_row), num_rows))
      : 0;
  int num_thread_blocks = blocks_per_row * blocks_per_col;

  dim3 sweep_grid_dims;
  sweep_grid_dims.x = (unsigned int) blocks_per_row;
  sweep_grid_dims.y = (unsigned int) blocks_per_col;
  sweep_grid_dims.z = 1;

  // For the GMEM-privatized host-init path the dispatch uses a
  // direct-atomic-to-output kernel instead of per-block privatization +
  // gather merge. The direct-atomic kernel uses warp-level coalesced
  // atomics (`__match_any_sync`) so cross-block contention on hot bins
  // is largely amortised; the main remaining concern is mid-bin paths
  // where contention per bin is moderate but per-block privatization is
  // also small.
  //
  // Multi-active-channel paths benefit from direct-atomic at lower bin
  // counts than single-channel paths because the per-block privatization
  // storage scales with NUM_ACTIVE_CHANNELS while contention per channel
  // does not, and because warp-level coalescing is more effective when
  // the same warp scans more samples (i.e. more chances for matching
  // bins per warp scan).
  // Caller may have already picked sweep over direct-atomic via the unified
  // algorithm selector; honour that. Otherwise fall back to the bin-count-
  // based heuristic for backwards compatibility.
  //
  // Single-channel threshold lowered from 1<<20 to 1<<16 (65536): now that
  // the cuckoo cache lives in dynamic SMEM and grows to use all free shared
  // memory, the direct-atomic + per-block SMEM cache path is competitive
  // with the gather-merge persistent kernel down to 65536 bins, and it
  // avoids the gather-merge's O(num_blocks * num_bins) cross-block reduction.
  // This routes the 262144-bin single-channel cells (the weakest high-bin
  // cells, gather-merge-bound at ~110 GiB/s on uniform input) through the
  // larger cache. Verified by measurement (see iteration log).
  constexpr int direct_atomic_bin_threshold_single = 1 << 16;
  constexpr int direct_atomic_bin_threshold_multi  = 16384;
  const int direct_atomic_bin_threshold =
    (NUM_ACTIVE_CHANNELS > 1) ? direct_atomic_bin_threshold_multi : direct_atomic_bin_threshold_single;
  // When the unified selector explicitly requested the single-probe
  // direct-atomic cache (`direct_atomic_cache_mode == 1`), it has already
  // decided the direct-atomic path is wanted; the legacy bin-count threshold
  // (used by callers that don't route through the selector) must not veto it.
  // The IsDeviceInit / PRIVATIZED_SMEM_BINS / disable_direct_atomic guards
  // still apply (the single-probe kernel is a host-init, PRIVATIZED_SMEM_BINS==0
  // cooperative kernel exactly like the cuckoo one).
  const bool selector_forces_direct_atomic = (direct_atomic_cache_mode == 1);
  const bool use_direct_atomic_to_output =
#if _CCCL_HOSTED()
    (!IsDeviceInit && PRIVATIZED_SMEM_BINS == 0 && !disable_direct_atomic
     && (selector_forces_direct_atomic || max_num_output_bins >= direct_atomic_bin_threshold));
#else
    false;
#endif

  // Temporary storage allocation requirements. Even when the direct-atomic
  // path is selected, we still report the full privatization storage: this
  // keeps a safe legacy fallback (the non-cooperative two-kernel sequence
  // below) usable if `cudaLaunchCooperativeKernel` fails on this device,
  // and lets `temp_storage_bytes` remain a valid upper bound for callers
  // that decide between paths at run-time.
  constexpr int NUM_ALLOCATIONS      = NUM_ACTIVE_CHANNELS + 1;
  void* allocations[NUM_ALLOCATIONS] = {};
  size_t allocation_sizes[NUM_ALLOCATIONS];

  for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
  {
    allocation_sizes[CHANNEL] =
      size_t(num_thread_blocks) * (num_privatized_levels[CHANNEL] - 1) * kernel_source.CounterSize();
  }

  allocation_sizes[NUM_ALLOCATIONS - 1] = GridQueue<int>::AllocationSize();

  // Alias the temporary allocations from the single storage blob (or compute the
  // necessary size of the blob)
  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  {
    // Return if the caller is simply requesting the size of the storage allocation
    return cudaSuccess;
  }

  // Construct the grid queue descriptor
  GridQueue<int> tile_queue(allocations[NUM_ALLOCATIONS - 1]);

  // Wrap arrays so we can pass them by-value to the kernel
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_privatized_histograms_wrapper;
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_bins_wrapper;
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_bins_wrapper;

  auto* typed_allocations = reinterpret_cast<CounterT**>(allocations);
  ::cuda::std::copy(typed_allocations, typed_allocations + NUM_ACTIVE_CHANNELS, d_privatized_histograms_wrapper.begin());

  auto minus_one = ::cuda::proclaim_return_type<int>([](int levels) {
    return levels - 1;
  });
  ::cuda::std::transform(
    num_privatized_levels.begin(), num_privatized_levels.end(), num_privatized_bins_wrapper.begin(), minus_one);
  ::cuda::std::transform(num_output_levels.begin(), num_output_levels.end(), num_output_bins_wrapper.begin(), minus_one);

  // For the host-init path we can fuse the init and sweep kernels into a
  // single persistent grid-resident kernel that uses
  // `cooperative_groups::this_grid().sync()` between the two phases. This
  // eliminates the separate init-kernel launch and the associated launch
  // overhead. Cooperative launch requires all blocks to be co-resident on
  // the device; since `num_thread_blocks <= histogram_sweep_occupancy =
  // sm_count * sm_occupancy` the grid is already sized to fit.
  // The persistent kernel is only worth using for the GMEM-privatized path
  // (`PRIVATIZED_SMEM_BINS == 0`), which corresponds to high bin counts
  // (`max_num_output_bins > 256`). For that path we replace the
  // O(num_blocks * num_bins) atomic merge in `StoreOutput` with an
  // atomic-free gather merge after a `grid.sync()`. For the SMEM-privatized
  // path the persistent kernel only adds cooperative-launch overhead with
  // no benefit, so we keep the legacy two-kernel sequence there.
  bool launched_persistent = false;
#if _CCCL_HOSTED()
  if constexpr (!IsDeviceInit && PRIVATIZED_SMEM_BINS == 0)
  {
    if (blocks_per_row > 0 && blocks_per_col > 0)
    {
      using output_decode_op_t     = typename FirstLevelArrayT::value_type;
      using privatized_decode_op_t = typename SecondLevelArrayT::value_type;

      // Use a 1-D grid for the cooperative launch; the kernel's
      // `block_id = blockIdx.y * gridDim.x + blockIdx.x` evaluates
      // identically for the 2-D `(blocks_per_row, blocks_per_col, 1)`
      // grid and a 1-D `(num_thread_blocks, 1, 1)` grid.
      dim3 persistent_grid_dims{static_cast<unsigned int>(num_thread_blocks), 1u, 1u};

      // Reference the kernel template by name in a chevron call to ensure
      // nvcc emits the device-side kernel during device compilation. The
      // launch itself runs only once via `cudaLaunchCooperativeKernel`
      // below so `grid.sync()` works.
      auto persistent_kernel_ptr = &DeviceHistogramSweepPersistentKernel<PolicySelector,
                                                                         PRIVATIZED_SMEM_BINS,
                                                                         NUM_CHANNELS,
                                                                         NUM_ACTIVE_CHANNELS,
                                                                         SampleIteratorT,
                                                                         CounterT,
                                                                         privatized_decode_op_t,
                                                                         output_decode_op_t,
                                                                         OffsetT>;

      // Force device-side instantiation of the kernel template by referencing
      // it via a dead `<<<>>>` syntax. nvcc emits device code for kernels
      // whose templates are referenced by a chevron call, regardless of
      // whether the call is reachable at runtime. Without this, just taking
      // `&kernel` produces only the host shadow function, and the runtime's
      // kernel-registration table has no device-side entry to match it,
      // causing `cudaLaunchCooperativeKernel` to fail with
      // `cudaErrorInvalidResourceHandle`.
      if (false)
      {
        DeviceHistogramSweepPersistentKernel<PolicySelector,
                                             PRIVATIZED_SMEM_BINS,
                                             NUM_CHANNELS,
                                             NUM_ACTIVE_CHANNELS,
                                             SampleIteratorT,
                                             CounterT,
                                             privatized_decode_op_t,
                                             output_decode_op_t,
                                             OffsetT>
          <<<persistent_grid_dims, dim3{static_cast<unsigned int>(threads_per_block)}, 0, stream>>>(
            d_samples,
            num_output_bins_wrapper,
            num_privatized_bins_wrapper,
            d_output_histograms,
            d_privatized_histograms_wrapper,
            first_level_array,
            second_level_array,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            tiles_per_row,
            tile_queue,
            max_num_output_bins);
      }

      // The direct-atomic kernel skips per-block privatization entirely
      // and writes atomically to the output histograms. Used only when
      // `use_direct_atomic_to_output` is true (see threshold above).
      auto direct_atomic_kernel_ptr =
        &DeviceHistogramSweepDirectAtomicPersistentKernel<PolicySelector,
                                                          PRIVATIZED_SMEM_BINS,
                                                          NUM_CHANNELS,
                                                          NUM_ACTIVE_CHANNELS,
                                                          SampleIteratorT,
                                                          CounterT,
                                                          privatized_decode_op_t,
                                                          output_decode_op_t,
                                                          OffsetT>;
      const void* direct_atomic_kernel_ptr_void =
        reinterpret_cast<const void*>(direct_atomic_kernel_ptr);

      // The single-probe direct-mapped variant of the direct-atomic kernel.
      // It shares the cuckoo kernel's signature (including the runtime
      // `cache_slots_per_channel` dynamic-SMEM cache), so the occupancy-
      // preserving cache sizing, the dynamic-SMEM cap, and the cooperative
      // launch args below are all common to both; only the leader's probe
      // policy differs. Selected when the caller requests
      // `direct_atomic_cache_mode == 1` (the huge-N single-channel high-bin
      // route picked by the unified selector as gmem_priv_single_probe).
      auto direct_atomic_single_probe_kernel_ptr =
        &DeviceHistogramSweepDirectAtomicSingleProbePersistentKernel<PolicySelector,
                                                                     PRIVATIZED_SMEM_BINS,
                                                                     NUM_CHANNELS,
                                                                     NUM_ACTIVE_CHANNELS,
                                                                     SampleIteratorT,
                                                                     CounterT,
                                                                     privatized_decode_op_t,
                                                                     output_decode_op_t,
                                                                     OffsetT>;
      const void* direct_atomic_single_probe_kernel_ptr_void =
        reinterpret_cast<const void*>(direct_atomic_single_probe_kernel_ptr);

      // Pick the active direct-atomic kernel per the cache mode. Both kernels
      // are PRIVATIZED_SMEM_BINS==0 host-init cooperative kernels with the same
      // dynamic-SMEM cache layout; the rest of the launch path treats them
      // uniformly through `active_direct_atomic_kernel_ptr(_void)`.
      const bool use_single_probe_cache = (direct_atomic_cache_mode == 1);
      const void* active_direct_atomic_kernel_ptr_void =
        use_single_probe_cache ? direct_atomic_single_probe_kernel_ptr_void : direct_atomic_kernel_ptr_void;

      int device_ordinal = 0;
      if (cudaGetDevice(&device_ordinal) != cudaSuccess)
      {
        (void) cudaGetLastError();
        device_ordinal = 0;
      }

      // Size the per-block SMEM cuckoo cache (now in DYNAMIC shared memory).
      // We pick the largest power-of-two slot count per channel whose
      // dynamic-SMEM footprint still lets the direct-atomic kernel hold the
      // cooperative grid (occupancy * sm_count >= num_thread_blocks), capped
      // so a single block never exceeds the device's opt-in dynamic-SMEM
      // limit. More slots => higher cache hit rate => fewer scattered
      // GMEM-atomic spills on the high-bin path (the measured bottleneck).
      // The floor is the legacy static size (4096 single-channel / 1024
      // multi-channel) so we never regress below the previous behaviour.
      const int cache_bytes_per_slot = static_cast<int>(sizeof(int)) + static_cast<int>(kernel_source.CounterSize());
      const int cache_slots_floor    = (NUM_ACTIVE_CHANNELS == 1) ? 4096 : 1024;
      // Cap the per-CTA dynamic SMEM for the cache. B200/SM100 supports ~228
      // KiB opt-in dynamic SMEM per CTA; query the device max and stay under
      // it (leaving headroom for driver reserve). Fall back to 96 KiB if the
      // query fails.
      int max_optin_smem = 0;
      if (cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_ordinal) != cudaSuccess
          || max_optin_smem <= 0)
      {
        (void) cudaGetLastError();
        max_optin_smem = 96 * 1024;
      }
      // Reserve ~4 KiB for static/driver shared use; the rest is for the cache.
      const int cache_smem_budget = (max_optin_smem > 4096) ? (max_optin_smem - 4096) : max_optin_smem;
      const int max_slots_by_smem =
        cache_smem_budget / (NUM_ACTIVE_CHANNELS * cache_bytes_per_slot);

      // Occupancy-preserving sizing. We measured that simply "fitting the
      // cooperative grid" lets the cache grow until occupancy collapses to
      // 1 block/SM, which slows the latency-bound multi-channel cells. So we
      // only spend SMEM that is FREE: pick the largest power-of-two slot
      // count whose per-SM occupancy is no lower than the occupancy at the
      // floor size. This keeps the single-channel 1M-bin gains (where the
      // extra slots are free) without trading away occupancy on the
      // multi-channel paths.
      //
      // The query / attribute target is the ACTIVE direct-atomic kernel
      // (cuckoo or single-probe per `direct_atomic_cache_mode`): they have the
      // same dynamic-SMEM layout but can differ in register usage, so the
      // free-SMEM occupancy budget must be measured against the kernel that
      // will actually run.
      auto cache_occupancy_for = [&](auto kernel_ptr, int slots) -> int {
        const int bytes = NUM_ACTIVE_CHANNELS * slots * cache_bytes_per_slot;
        if (launcher_factory.set_max_dynamic_smem_size_for(kernel_ptr, bytes) != cudaSuccess)
        {
          (void) cudaGetLastError();
          return 0;
        }
        int occ = 0;
        if (launcher_factory.MaxSmOccupancy(occ, kernel_ptr, threads_per_block, bytes) != cudaSuccess)
        {
          (void) cudaGetLastError();
          return 0;
        }
        return occ;
      };

      auto size_cache_for = [&](auto kernel_ptr) -> int {
        const int floor_occ_local = cache_occupancy_for(kernel_ptr, cache_slots_floor);
        int slots                 = cache_slots_floor;
        // Grow while occupancy stays at the floor occupancy (free SMEM).
        for (int cand = cache_slots_floor << 1; cand <= max_slots_by_smem; cand <<= 1)
        {
          const int occ = cache_occupancy_for(kernel_ptr, cand);
          if (floor_occ_local > 0 && occ >= floor_occ_local)
          {
            slots = cand;
          }
          else
          {
            break; // growth would cost occupancy; stop.
          }
        }
        return slots;
      };

      const int cache_slots_per_channel =
        use_single_probe_cache ? size_cache_for(direct_atomic_single_probe_kernel_ptr)
                               : size_cache_for(direct_atomic_kernel_ptr);
      const int cuckoo_cache_smem_bytes = NUM_ACTIVE_CHANNELS * cache_slots_per_channel * cache_bytes_per_slot;
      // Make sure the active kernel's attribute matches the final chosen size
      // (the last probe in the loop may have set a larger size that we
      // rejected).
      if (use_single_probe_cache)
      {
        if (launcher_factory.set_max_dynamic_smem_size_for(direct_atomic_single_probe_kernel_ptr, cuckoo_cache_smem_bytes)
            != cudaSuccess)
        {
          (void) cudaGetLastError();
        }
      }
      else
      {
        if (launcher_factory.set_max_dynamic_smem_size_for(direct_atomic_kernel_ptr, cuckoo_cache_smem_bytes)
            != cudaSuccess)
        {
          (void) cudaGetLastError();
        }
      }

      if (false)
      {
        DeviceHistogramSweepDirectAtomicPersistentKernel<PolicySelector,
                                                         PRIVATIZED_SMEM_BINS,
                                                         NUM_CHANNELS,
                                                         NUM_ACTIVE_CHANNELS,
                                                         SampleIteratorT,
                                                         CounterT,
                                                         privatized_decode_op_t,
                                                         output_decode_op_t,
                                                         OffsetT>
          <<<persistent_grid_dims, dim3{static_cast<unsigned int>(threads_per_block)}, cuckoo_cache_smem_bytes, stream>>>(
            d_samples,
            num_output_bins_wrapper,
            d_output_histograms,
            second_level_array,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            cache_slots_per_channel);
        DeviceHistogramSweepDirectAtomicSingleProbePersistentKernel<PolicySelector,
                                                                    PRIVATIZED_SMEM_BINS,
                                                                    NUM_CHANNELS,
                                                                    NUM_ACTIVE_CHANNELS,
                                                                    SampleIteratorT,
                                                                    CounterT,
                                                                    privatized_decode_op_t,
                                                                    output_decode_op_t,
                                                                    OffsetT>
          <<<persistent_grid_dims, dim3{static_cast<unsigned int>(threads_per_block)}, cuckoo_cache_smem_bytes, stream>>>(
            d_samples,
            num_output_bins_wrapper,
            d_output_histograms,
            second_level_array,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            cache_slots_per_channel);
      }

      int cooperative_supported = 0;
      const bool coop_query_ok =
        (cudaDeviceGetAttribute(&cooperative_supported, cudaDevAttrCooperativeLaunch, device_ordinal) == cudaSuccess
         && cooperative_supported != 0);
      // The persistent / direct-atomic kernels may have lower per-SM occupancy
      // than the staging sweep kernel that was used to size `num_thread_blocks`,
      // so we must verify the chosen kernel's occupancy fits the requested
      // cooperative grid; otherwise `cudaLaunchCooperativeKernel` will fail
      // with cudaErrorCooperativeLaunchTooLarge and we fall back to the legacy
      // two-kernel path (which is much slower for high-bin GMEM-priv configs).
      int persistent_sm_occupancy = 0;
      int direct_atomic_sm_occupancy = 0;
      const auto persist_occ_err = launcher_factory.MaxSmOccupancy(
        persistent_sm_occupancy, persistent_kernel_ptr, threads_per_block);
      if (persist_occ_err != cudaSuccess)
      {
        (void) cudaGetLastError();
        persistent_sm_occupancy = 0;
      }
      const auto direct_occ_err =
        use_single_probe_cache
          ? launcher_factory.MaxSmOccupancy(
              direct_atomic_sm_occupancy, direct_atomic_single_probe_kernel_ptr, threads_per_block, cuckoo_cache_smem_bytes)
          : launcher_factory.MaxSmOccupancy(
              direct_atomic_sm_occupancy, direct_atomic_kernel_ptr, threads_per_block, cuckoo_cache_smem_bytes);
      if (direct_occ_err != cudaSuccess)
      {
        (void) cudaGetLastError();
        direct_atomic_sm_occupancy = 0;
      }
      const int persistent_capacity = persistent_sm_occupancy * sm_count;
      const int direct_atomic_capacity = direct_atomic_sm_occupancy * sm_count;
      const bool persistent_fits = (persistent_sm_occupancy > 0)
                                   && (num_thread_blocks <= persistent_capacity);
      const bool direct_atomic_fits = (direct_atomic_sm_occupancy > 0)
                                      && (num_thread_blocks <= direct_atomic_capacity);
      const bool selected_fits = use_direct_atomic_to_output ? direct_atomic_fits : persistent_fits;

      // Grid size for the direct-atomic kernels. Unlike the gather-merge
      // SweepPersistent kernel, the direct-atomic cuckoo / single-probe kernels
      // distribute work via a pure grid-stride loop over `total_pixels` and use
      // neither `tile_queue` nor `tiles_per_row`, so ANY block count produces
      // correct counts -- more blocks simply means more resident warps. The
      // shared `num_thread_blocks` above is sized off the SweepPersistent
      // kernel's per-SM occupancy, which is lower (the gather-merge kernel is
      // register-heavy). Profiling the 262144-bin single-channel cuckoo cell
      // showed it launches only ~3 blocks/SM (0.6 waves) and is MIO-stall bound
      // (SMEM-atomic scoreboard latency) at ~55% achieved occupancy while the
      // cuckoo kernel itself admits 5 blocks/SM. Grow the grid to the
      // direct-atomic kernel's OWN co-resident capacity so the extra warps hide
      // that latency, capped by the available work (no point launching blocks
      // that would process zero pixels) and never shrinking below the sweep
      // grid.
      dim3 direct_atomic_grid_dims = persistent_grid_dims;
      if (use_direct_atomic_to_output && direct_atomic_capacity > num_thread_blocks)
      {
        // Upper bound on useful blocks: one block per pixel-tile across all
        // rows (the same tile granularity the sweep grid uses). Beyond this,
        // additional blocks would have no pixels to process.
        const long long work_tiles = static_cast<long long>(tiles_per_row) * static_cast<long long>(num_rows);
        long long target = static_cast<long long>(direct_atomic_capacity);
        if (target > work_tiles)
        {
          target = work_tiles;
        }
        if (target < static_cast<long long>(num_thread_blocks))
        {
          target = static_cast<long long>(num_thread_blocks);
        }
        direct_atomic_grid_dims = dim3{static_cast<unsigned int>(target), 1u, 1u};
      }

      if (coop_query_ok && selected_fits)
      {
        cudaError_t coop_status = cudaSuccess;
        if (use_direct_atomic_to_output)
        {
          // For the very-high-bin GMEM-privatized path, dispatch the
          // direct-atomic kernel instead of the gather-merge persistent
          // kernel. It needs only the output histograms, the privatized
          // decode op, and the input geometry.
          void* direct_kernel_args[] = {
            const_cast<void*>(static_cast<const void*>(&d_samples)),
            const_cast<void*>(static_cast<const void*>(&num_output_bins_wrapper)),
            const_cast<void*>(static_cast<const void*>(&d_output_histograms)),
            const_cast<void*>(static_cast<const void*>(&second_level_array)),
            const_cast<void*>(static_cast<const void*>(&num_row_pixels)),
            const_cast<void*>(static_cast<const void*>(&num_rows)),
            const_cast<void*>(static_cast<const void*>(&row_stride_samples)),
            const_cast<void*>(static_cast<const void*>(&cache_slots_per_channel))};
          coop_status = cudaLaunchCooperativeKernel(
            active_direct_atomic_kernel_ptr_void,
            direct_atomic_grid_dims,
            dim3{static_cast<unsigned int>(threads_per_block)},
            direct_kernel_args,
            /*sharedMem=*/static_cast<size_t>(cuckoo_cache_smem_bytes),
            stream);
        }
        else
        {
          void* kernel_args[] = {
            const_cast<void*>(static_cast<const void*>(&d_samples)),
            const_cast<void*>(static_cast<const void*>(&num_output_bins_wrapper)),
            const_cast<void*>(static_cast<const void*>(&num_privatized_bins_wrapper)),
            const_cast<void*>(static_cast<const void*>(&d_output_histograms)),
            const_cast<void*>(static_cast<const void*>(&d_privatized_histograms_wrapper)),
            const_cast<void*>(static_cast<const void*>(&first_level_array)),
            const_cast<void*>(static_cast<const void*>(&second_level_array)),
            const_cast<void*>(static_cast<const void*>(&num_row_pixels)),
            const_cast<void*>(static_cast<const void*>(&num_rows)),
            const_cast<void*>(static_cast<const void*>(&row_stride_samples)),
            const_cast<void*>(static_cast<const void*>(&tiles_per_row)),
            const_cast<void*>(static_cast<const void*>(&tile_queue)),
            const_cast<void*>(static_cast<const void*>(&max_num_output_bins))};
          coop_status = cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(persistent_kernel_ptr),
            persistent_grid_dims,
            dim3{static_cast<unsigned int>(threads_per_block)},
            kernel_args,
            /*sharedMem=*/0,
            stream);
        }
        if (coop_status == cudaSuccess)
        {
          launched_persistent = true;
        }
        else
        {
          // Clear the sticky error so the legacy two-kernel fallback below
          // does not see a stale error from cudaLaunchCooperativeKernel.
          (void) cudaGetLastError();
        }
      }
    }
  }
#endif // _CCCL_HOSTED()

  // For the dyn-SMEM staging tier (xlarge, 16384 bins, single-channel, host-init,
  // non-byte) we can fuse the staging-sweep + cross-block combine pair into a
  // single cooperative launch using `grid_group::sync()`. This saves one
  // `cudaLaunch*` round-trip + the standalone combine kernel's ~18us launch
  // overhead, which is a meaningful fraction of total runtime on the
  // small-Elements (1048576) configurations of the xlarge tier.
  bool launched_fused_staging = false;
#if _CCCL_HOSTED()
  if constexpr (kUseStagingPath && !IsDeviceInit)
  {
    if (!launched_persistent && blocks_per_row > 0 && blocks_per_col > 0)
    {
      using output_decode_op_t     = typename FirstLevelArrayT::value_type;
      using privatized_decode_op_t = typename SecondLevelArrayT::value_type;

      // Use a 1-D grid for the cooperative launch; the staging kernel computes
      // `block_id = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x`,
      // which evaluates identically for a 1-D `(num_thread_blocks, 1, 1)` grid
      // and the 2-D `(blocks_per_row, blocks_per_col, 1)` grid; AgentHistogram
      // also uses this convention.
      dim3 fused_grid_dims{static_cast<unsigned int>(num_thread_blocks), 1u, 1u};

      auto fused_kernel_ptr =
        kernel_source.template HistogramSweepStagingFusedHostInitDynSmemKernel<
            PolicySelector,
            PRIVATIZED_SMEM_BINS,
            privatized_decode_op_t,
            output_decode_op_t>();

      // Force device-side instantiation of the fused kernel template via a dead
      // `<<<>>>` call. Without this, just taking `&kernel` produces only the
      // host shadow function and the runtime kernel-registration table has no
      // device-side entry to match it, causing `cudaLaunchCooperativeKernel`
      // to fail with `cudaErrorInvalidResourceHandle`.
      if (false)
      {
        DeviceHistogramSweepStagingFusedHostInitDynSmemKernel<PolicySelector,
                                                              PRIVATIZED_SMEM_BINS,
                                                              NUM_CHANNELS,
                                                              NUM_ACTIVE_CHANNELS,
                                                              SampleIteratorT,
                                                              CounterT,
                                                              privatized_decode_op_t,
                                                              output_decode_op_t,
                                                              OffsetT>
          <<<fused_grid_dims, dim3{static_cast<unsigned int>(threads_per_block)}, dyn_smem_bytes_for_staging, stream>>>(
            d_samples,
            num_output_bins_wrapper,
            num_privatized_bins_wrapper,
            d_output_histograms,
            d_privatized_histograms_wrapper,
            first_level_array,
            second_level_array,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            tiles_per_row,
            tile_queue);
      }

      const void* fused_kernel_ptr_void = reinterpret_cast<const void*>(fused_kernel_ptr);

      // Raise the dyn-SMEM cap on the fused kernel before launch.
      cudaError_t cap_err =
        launcher_factory.set_max_dynamic_smem_size_for(fused_kernel_ptr, dyn_smem_bytes_for_staging);
      if (cap_err != cudaSuccess)
      {
        // Don't propagate; just clear and fall through to the two-launch path.
        (void) cudaGetLastError();
      }
      else
      {
        // Query the FUSED kernel's per-SM occupancy with the actual dyn-SMEM
        // budget. The fused kernel may have higher register / shared-memory
        // pressure than the staging-only kernel (extra grid.sync code paths,
        // the inline reduce-and-write loop), so its occupancy can be lower.
        // Cooperative launch requires `num_thread_blocks <= sm_count *
        // sm_occupancy_of_the_kernel_we_are_launching`. If the fused kernel
        // doesn't fit at the staging-grid size, fall back gracefully.
        int fused_sm_occupancy = 0;
        const auto occ_err = launcher_factory.MaxSmOccupancy(
          fused_sm_occupancy, fused_kernel_ptr, threads_per_block, dyn_smem_bytes_for_staging);
        if (occ_err != cudaSuccess)
        {
          (void) cudaGetLastError();
          fused_sm_occupancy = 0;
        }
        const bool fused_fits = (fused_sm_occupancy > 0)
                                && (num_thread_blocks <= fused_sm_occupancy * sm_count);
        int device_ordinal        = 0;
        int cooperative_supported = 0;
        const bool coop_query_ok =
          (cudaGetDevice(&device_ordinal) == cudaSuccess
           && cudaDeviceGetAttribute(&cooperative_supported, cudaDevAttrCooperativeLaunch, device_ordinal) == cudaSuccess
           && cooperative_supported != 0);
        if (coop_query_ok && fused_fits)
        {
          // The fused kernel takes the same arguments as the staging-only
          // sweep kernel: (d_samples, num_output_bins, num_privatized_bins,
          // d_output_histograms, d_privatized_histograms, output_decode_op,
          // privatized_decode_op, num_row_pixels, num_rows, row_stride_samples,
          // tiles_per_row, tile_queue).
          //
          // For host-init non-byte, the dispatch caller passes the decode-op
          // arrays via `first_level_array` (output decode op) and
          // `second_level_array` (privatized decode op). They were originally
          // defined here as ::cuda::std::array<DecodeOpT, NUM_ACTIVE_CHANNELS>.
          void* kernel_args[] = {
            const_cast<void*>(static_cast<const void*>(&d_samples)),
            const_cast<void*>(static_cast<const void*>(&num_output_bins_wrapper)),
            const_cast<void*>(static_cast<const void*>(&num_privatized_bins_wrapper)),
            const_cast<void*>(static_cast<const void*>(&d_output_histograms)),
            const_cast<void*>(static_cast<const void*>(&d_privatized_histograms_wrapper)),
            const_cast<void*>(static_cast<const void*>(&first_level_array)),
            const_cast<void*>(static_cast<const void*>(&second_level_array)),
            const_cast<void*>(static_cast<const void*>(&num_row_pixels)),
            const_cast<void*>(static_cast<const void*>(&num_rows)),
            const_cast<void*>(static_cast<const void*>(&row_stride_samples)),
            const_cast<void*>(static_cast<const void*>(&tiles_per_row)),
            const_cast<void*>(static_cast<const void*>(&tile_queue))};
          cudaError_t coop_status = cudaLaunchCooperativeKernel(
            fused_kernel_ptr_void,
            fused_grid_dims,
            dim3{static_cast<unsigned int>(threads_per_block)},
            kernel_args,
            /*sharedMem=*/static_cast<size_t>(dyn_smem_bytes_for_staging),
            stream);
          if (coop_status == cudaSuccess)
          {
            launched_fused_staging = true;
          }
          else
          {
            // Clear sticky error so legacy two-launch fallback can run cleanly.
            (void) cudaGetLastError();
          }
        }
      }
    }
  }
#endif // _CCCL_HOSTED()

  if (!launched_persistent && !launched_fused_staging)
  {
    constexpr int histogram_init_threads_per_block = 256;
    int histogram_init_grid_dims =
      (max_num_output_bins + histogram_init_threads_per_block - 1) / histogram_init_threads_per_block;

// Log DeviceHistogramInitKernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking DeviceHistogramInitKernel<<<%d, %d, 0, %lld>>>()\n",
            histogram_init_grid_dims,
            histogram_init_threads_per_block,
            (long long) stream);
#endif // CUB_DEBUG_LOG

    // Invoke histogram_init_kernel
    if (const auto error = CubDebug(
          launcher_factory(histogram_init_grid_dims,
                           histogram_init_threads_per_block,
                           0,
                           stream,
                           /* dependent_launch */ cc >= ::cuda::compute_capability{9, 0})
            .doit(init_kernel, num_output_bins_wrapper, d_output_histograms, tile_queue)))
    {
      return error;
    }

    // Return if empty problem
    if (blocks_per_row == 0 || blocks_per_col == 0)
    {
      return cudaSuccess;
    }

// Log histogram_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking histogram_sweep_kernel<<<{%d, %d, %d}, %d, 0, %lld>>>(), %d pixels "
            "per thread, %d SM occupancy\n",
            sweep_grid_dims.x,
            sweep_grid_dims.y,
            sweep_grid_dims.z,
            threads_per_block,
            (long long) stream,
            pixels_per_thread,
            histogram_sweep_sm_occupancy);
#endif // CUB_DEBUG_LOG

    if constexpr (kUseStagingPath)
    {
      // Dynamic-SMEM staging path: launch the staging sweep kernel (which skips per-block
      // atomicAdd-to-global), then run the combine kernel to reduce per-block staging slabs
      // into the final output histogram.
      // The dynamic-SMEM cap was already raised above as part of the occupancy query
      // (set_max_dynamic_smem_size_for in the kStagingUsesDynSmem branch).
      if (const auto error = CubDebug(
            launcher_factory(sweep_grid_dims,
                             threads_per_block,
                             dyn_smem_bytes_for_staging,
                             stream,
                             /* dependent_launch */ cc >= ::cuda::compute_capability{9, 0})
              .doit(staging_sweep_kernel,
                    d_samples,
                    num_output_bins_wrapper,
                    num_privatized_bins_wrapper,
                    d_output_histograms,
                    d_privatized_histograms_wrapper,
                    first_level_array,
                    second_level_array,
                    num_row_pixels,
                    num_rows,
                    row_stride_samples,
                    tiles_per_row,
                    tile_queue)))
      {
        return error;
      }

      // Combine kernel: 256 threads x ceil(num_privatized_bins / 256) blocks per channel.
      // For non-byte samples, num_privatized_bins == num_output_bins (PassThru output decode).
      constexpr int combine_threads = 256;
      int combine_blocks_x          = (max_num_output_bins + combine_threads - 1) / combine_threads;

      // Cast d_privatized_histograms to const for combine kernel.
      ::cuda::std::array<const CounterT*, NUM_ACTIVE_CHANNELS> d_privatized_const_wrapper;
      for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ++ch)
      {
        d_privatized_const_wrapper[ch] = d_privatized_histograms_wrapper[ch];
      }

      dim3 combine_grid_dims;
      combine_grid_dims.x = (unsigned int) combine_blocks_x;
      combine_grid_dims.y = (unsigned int) NUM_ACTIVE_CHANNELS;
      combine_grid_dims.z = 1;

      if (const auto error = CubDebug(
            launcher_factory(combine_grid_dims,
                             combine_threads,
                             0,
                             stream,
                             /* dependent_launch */ cc >= ::cuda::compute_capability{9, 0})
              .doit(combine_kernel,
                    d_output_histograms,
                    d_privatized_const_wrapper,
                    num_privatized_bins_wrapper,
                    num_thread_blocks)))
      {
        return error;
      }
    }
    else
    {
      // Non-staging launch. For the non-staging dyn-SMEM tier the per-block
      // histogram lives in extern __shared__, so we must pass the dyn-SMEM byte
      // budget (its cap was already raised in the occupancy-query branch above);
      // the static-SMEM non-staging tiers pass 0.
      const int non_staging_smem_bytes = kUseNonStagingDynSmem ? dyn_smem_bytes_for_staging : 0;
      if (const auto error = CubDebug(
            launcher_factory(sweep_grid_dims,
                             threads_per_block,
                             non_staging_smem_bytes,
                             stream,
                             /* dependent_launch */ cc >= ::cuda::compute_capability{9, 0})
              .doit(sweep_kernel,
                    d_samples,
                    num_output_bins_wrapper,
                    num_privatized_bins_wrapper,
                    d_output_histograms,
                    d_privatized_histograms_wrapper,
                    first_level_array,
                    second_level_array,
                    num_row_pixels,
                    num_rows,
                    row_stride_samples,
                    tiles_per_row,
                    tile_queue)))
      {
        return error;
      }
    }
  }

  // Check for failure to launch
  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  // Sync the stream if specified to flush runtime errors
  if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
  {
    return error;
  }

  return cudaSuccess;
}

// Dispatch routines for device-side decode operator initialization. These differ from the default dispatch routines in
// that they initialize the decode operators inside the kernel from level arrays, instead of initializing them on the
// host, but they are otherwise the same. This is needed for c.parallel, since we cannot instantiate the Transforms
// class on the host, as SampleT and LevelT are type erased. Another change needed is that the level arrays are now
// templates instead of concrete ::cuda::std::array types, since we are passing indirect_args from c.parallel.
//
// Initializing the decode operators inside the kernel results in some regressions (and some performance improvements)
// in the benchmark, which indicates that we need to re-tune the algorithm. This is why we kept the two dispatch paths
// (host init and device init) separate. We should think about merging them back together later on.

/**
 * Dispatch routine for HistogramEven with device-side decode operator initialization,
 * specialized for sample types larger than 8-bit.
 * This variant initializes the decode operators inside the kernel from level bounds.
 *
 * @param d_temp_storage
 *   Device-accessible allocation of temporary storage.
 *   When nullptr, the required allocation size is written to
 *   `temp_storage_bytes` and no work is done.
 *
 * @param temp_storage_bytes
 *   Reference to size in bytes of `d_temp_storage` allocation
 *
 * @param d_samples
 *   The pointer to the input sequence of sample items.
 *   The samples from different channels are assumed to be interleaved
 *   (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
 *
 * @param d_output_histograms
 *   The pointers to the histogram counter output arrays, one for each active channel.
 *   For channel<sub><em>i</em></sub>, the allocation length of `d_histograms[i]` should be
 *   `num_output_levels[i] - 1`.
 *
 * @param num_output_levels
 *   The number of bin level boundaries for delineating histogram samples in each active channel.
 *   Implies that the number of bins for channel<sub><em>i</em></sub> is
 *   `num_output_levels[i] - 1`.
 *
 * @param lower_level
 *   The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
 *
 * @param upper_level
 *   The upper sample value bound (exclusive) for the highest histogram bin in each active
 * channel.
 *
 * @param num_row_pixels
 *   The number of multi-channel pixels per row in the region of interest
 *
 * @param num_rows
 *   The number of rows in the region of interest
 *
 * @param row_stride_samples
 *   The number of samples between starts of consecutive rows in the region of interest
 *
 * @param stream
 *   CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
 *
 */
template <
  int NUM_CHANNELS,
  int NUM_ACTIVE_CHANNELS,
  typename SampleIteratorT,
  typename CounterT,
  typename LevelT,
  typename OffsetT,
  typename PolicySelector,
  typename SampleT = it_value_t<SampleIteratorT>, /// The sample value type of the input iterator
  typename KernelSource =
    DeviceHistogramKernelSource<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, LevelT, OffsetT, SampleT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY,
  typename LowerLevelArrayT      = ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS>,
  typename UpperLevelArrayT      = ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS>>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t __dispatch_even_device_init(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
  LowerLevelArrayT lower_level,
  UpperLevelArrayT upper_level,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_samples,
  cudaStream_t stream,
  ::cuda::std::false_type /*is_byte_sample*/,
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  int max_levels = num_output_levels[0];

  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    int num_levels = num_output_levels[channel];
    if (kernel_source.MayOverflow(num_levels - 1, upper_level, lower_level, channel))
    {
      // Make sure to also return a reasonable value for `temp_storage_bytes` in case of
      // an overflow of the bin computation, in which case a subsequent algorithm
      // invocation will also fail
      if (!d_temp_storage)
      {
        temp_storage_bytes = 1U;
      }
      return cudaErrorInvalidValue;
    }

    if (num_levels > max_levels)
    {
      max_levels = num_levels;
    }
  }
  int max_num_output_bins = max_levels - 1;

  if (max_num_output_bins > detail::histogram::max_privatized_smem_bins)
  {
    // Dispatch shared-privatized approach
    constexpr int PRIVATIZED_SMEM_BINS = 0;

    if (const auto error = CubDebug(
          (detail::histogram::dispatch<NUM_CHANNELS,
                                       NUM_ACTIVE_CHANNELS,
                                       PRIVATIZED_SMEM_BINS,
                                       /* IsDeviceInit = */ true,
                                       /* IsEven = */ true,
                                       /* IsByteSample = */ false>(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_output_histograms,
            num_output_levels,
            num_output_levels,
            upper_level,
            lower_level,
            max_num_output_bins,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            stream,
            policy_selector,
            kernel_source,
            launcher_factory))))
    {
      return error;
    }
  }
  else
  {
    // Dispatch shared-privatized approach
    constexpr int PRIVATIZED_SMEM_BINS = detail::histogram::max_privatized_smem_bins;

    if (const auto error = CubDebug(
          (detail::histogram::dispatch<NUM_CHANNELS,
                                       NUM_ACTIVE_CHANNELS,
                                       PRIVATIZED_SMEM_BINS,
                                       /* IsDeviceInit = */ true,
                                       /* IsEven = */ true,
                                       /* IsByteSample = */ false>(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_output_histograms,
            num_output_levels,
            num_output_levels,
            upper_level,
            lower_level,
            max_num_output_bins,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            stream,
            policy_selector,
            kernel_source,
            launcher_factory))))
    {
      return error;
    }
  }

  return cudaSuccess;
}

/**
 * Dispatch routine for HistogramEven with device-side decode operator initialization,
 * specialized for 8-bit sample types
 * (computes 256-bin privatized histograms and then reduces to user-specified levels).
 * This variant initializes the decode operators inside the kernel from level bounds.
 *
 * @param d_temp_storage
 *   Device-accessible allocation of temporary storage.
 *   When nullptr, the required allocation size is written to `temp_storage_bytes` and
 *   no work is done.
 *
 * @param temp_storage_bytes
 *   Reference to size in bytes of `d_temp_storage` allocation
 *
 * @param d_samples
 *   The pointer to the input sequence of sample items. The samples from different channels are
 *   assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of
 *   four RGBA 8-bit samples).
 *
 * @param d_output_histograms
 *   The pointers to the histogram counter output arrays, one for each active channel.
 *   For channel<sub><em>i</em></sub>, the allocation length of `d_histograms[i]` should be
 *   `num_output_levels[i] - 1`.
 *
 * @param num_output_levels
 *   The number of bin level boundaries for delineating histogram samples in each active channel.
 *   Implies that the number of bins for channel<sub><em>i</em></sub> is
 *   `num_output_levels[i] - 1`.
 *
 * @param lower_level
 *   The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
 *
 * @param upper_level
 *   The upper sample value bound (exclusive) for the highest histogram bin in each active
 * channel.
 *
 * @param num_row_pixels
 *   The number of multi-channel pixels per row in the region of interest
 *
 * @param num_rows
 *   The number of rows in the region of interest
 *
 * @param row_stride_samples
 *   The number of samples between starts of consecutive rows in the region of interest
 *
 * @param stream
 *   CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
 *
 */
template <
  int NUM_CHANNELS,
  int NUM_ACTIVE_CHANNELS,
  typename SampleIteratorT,
  typename CounterT,
  typename LevelT,
  typename OffsetT,
  typename PolicySelector,
  typename SampleT = it_value_t<SampleIteratorT>, /// The sample value type of the input iterator
  typename KernelSource =
    DeviceHistogramKernelSource<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, LevelT, OffsetT, SampleT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY,
  typename LowerLevelArrayT      = ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS>,
  typename UpperLevelArrayT      = ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS>>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t __dispatch_even_device_init(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
  LowerLevelArrayT lower_level,
  UpperLevelArrayT upper_level,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_samples,
  cudaStream_t stream,
  ::cuda::std::true_type /*is_byte_sample*/,
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_levels;
  int max_levels = num_output_levels[0];

  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    num_privatized_levels[channel] = 257;

    int num_levels = num_output_levels[channel];
    if (kernel_source.MayOverflow(num_levels - 1, upper_level, lower_level, channel))
    {
      // Make sure to also return a reasonable value for `temp_storage_bytes` in case of
      // an overflow of the bin computation, in which case a subsequent algorithm
      // invocation will also fail
      if (!d_temp_storage)
      {
        temp_storage_bytes = 1U;
      }
      return cudaErrorInvalidValue;
    }

    if (num_levels > max_levels)
    {
      max_levels = num_levels;
    }
  }
  int max_num_output_bins = max_levels - 1;

  constexpr int PRIVATIZED_SMEM_BINS = 256;

  if (const auto error = CubDebug(
        (detail::histogram::dispatch<NUM_CHANNELS,
                                     NUM_ACTIVE_CHANNELS,
                                     PRIVATIZED_SMEM_BINS,
                                     /* IsDeviceInit = */ true,
                                     /* IsEven = */ true,
                                     /* IsByteSample = */ true>(
          d_temp_storage,
          temp_storage_bytes,
          d_samples,
          d_output_histograms,
          num_privatized_levels,
          num_output_levels,
          upper_level,
          lower_level,
          max_num_output_bins,
          num_row_pixels,
          num_rows,
          row_stride_samples,
          stream,
          policy_selector,
          kernel_source,
          launcher_factory))))
  {
    return error;
  }

  return cudaSuccess;
}

// Hybrid SMEM+GMEM single-pass dispatch helper.
// Issues a SINGLE cooperative kernel launch that handles both bin chunks of the
// privatized-bin space in ONE sweep through the input. Bins in the primary range
// `[0, kSplitBin)` accumulate in dyn-SMEM (sized for kSplitBin per channel), bins
// in the secondary range `[kSplitBin, max_num_output_bins)` accumulate in a
// per-block per-channel GMEM staging slab. Eliminates the 2x sample re-read of
// the dual-chunk kernel by classifying each sample once with the un-chunked
// privatized decode op and routing the resulting bin to either SMEM or per-block
// GMEM based on the bin value.
// Falls back to the chunked dispatch on any setup or launch failure (the chunked
// path covers the same axis range and is verified-correct).
template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          int kSplitBin,
          int kMaxTotalBins,
          typename InnerPrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename SampleIteratorT,
          typename CounterT,
          typename OffsetT,
          typename PolicySelector,
          typename KernelSource,
          typename KernelLauncherFactory>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_hybrid_single_pass_staging_smem(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
  ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op,
  ::cuda::std::array<InnerPrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> inner_privatized_decode_op,
  int max_num_output_bins,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_samples,
  cudaStream_t stream,
  PolicySelector policy_selector,
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  constexpr int kPrivatizedSmemBins = max_extended_smem_bins_single_channel_xlarge;

  // The hybrid kernel is single-channel-focused and assumes a chunked split with
  // primary in SMEM and secondary in per-block GMEM. We require kSplitBin to be at
  // most kMaxTotalBins / 2 (well, just at most max_num_output_bins).
  if (max_num_output_bins <= kSplitBin)
  {
    // Fallback to the chunked dispatch if the bin count fits entirely in the SMEM
    // primary range (no secondary GMEM region needed). Should not occur in normal
    // operation since callers gate this dispatch on max_num_output_bins > xlarge.
    return cudaErrorNotSupported;
  }

  const int hybrid_split_bin      = kSplitBin;
  const int hybrid_secondary_size = max_num_output_bins - kSplitBin;

#if _CCCL_HOSTED()
  // Step 1: Replicate the inner-dispatch setup so we can launch the hybrid kernel.
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    (void) error;
    return cudaErrorNotSupported;
  }

  const histogram_policy active_policy = policy_selector(cc);
  const int threads_per_block          = active_policy.threads_per_block;
  const int pixels_per_thread          = active_policy.pixels_per_thread;

  int sm_count = 0;
  if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
  {
    (void) error;
    return cudaErrorNotSupported;
  }

  // dyn-SMEM bytes per block: per-channel kSplitBin counters.
  const int dyn_smem_bytes_for_staging =
    int(sizeof(CounterT)) * hybrid_split_bin * NUM_ACTIVE_CHANNELS;

  // Calculate occupancy and grid size.
  int fused_sm_occupancy = 0;
  auto fused_hybrid_kernel_ptr =
    kernel_source.template HistogramSweepStagingFusedHybridSinglePassHostInitDynSmemKernel<
      PolicySelector,
      kPrivatizedSmemBins,
      InnerPrivatizedDecodeOpT,
      OutputDecodeOpT>();

  // Force device-side instantiation of the hybrid kernel template via a dead `<<<>>>` call,
  // mirroring the pattern used by the existing fused-staging-kernel dispatch. Without this,
  // just taking `&kernel` produces only the host shadow function and the runtime
  // kernel-registration table has no device-side entry to match it.
  if (false)
  {
    DeviceHistogramSweepStagingFusedHybridSinglePassHostInitDynSmemKernel<PolicySelector,
                                                                          kPrivatizedSmemBins,
                                                                          NUM_CHANNELS,
                                                                          NUM_ACTIVE_CHANNELS,
                                                                          SampleIteratorT,
                                                                          CounterT,
                                                                          InnerPrivatizedDecodeOpT,
                                                                          OutputDecodeOpT,
                                                                          OffsetT>
      <<<1, 1, 0, stream>>>(d_samples,
                            ::cuda::std::array<int, NUM_ACTIVE_CHANNELS>{},
                            ::cuda::std::array<int, NUM_ACTIVE_CHANNELS>{},
                            ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS>{},
                            ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS>{},
                            ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS>{},
                            ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS>{},
                            ::cuda::std::array<InnerPrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS>{},
                            int{},
                            int{},
                            num_row_pixels,
                            num_rows,
                            row_stride_samples,
                            int{},
                            GridQueue<int>{nullptr});
  }

  if (const auto error = launcher_factory.set_max_dynamic_smem_size_for(
        fused_hybrid_kernel_ptr, dyn_smem_bytes_for_staging))
  {
    (void) error;
    return cudaErrorNotSupported;
  }

  if (const auto error =
        CubDebug(launcher_factory.MaxSmOccupancy(
          fused_sm_occupancy, fused_hybrid_kernel_ptr, threads_per_block, dyn_smem_bytes_for_staging)))
  {
    (void) error;
    return cudaErrorNotSupported;
  }

  if (fused_sm_occupancy <= 0)
  {
    return cudaErrorNotSupported;
  }

  // Calculate launch geometry: pixels per block and tile counts.
  const int pixels_per_block = threads_per_block * pixels_per_thread;
  const int total_pixels     = static_cast<int>(num_row_pixels);
  const int tiles_per_row    = (total_pixels + pixels_per_block - 1) / pixels_per_block;

  // Number of blocks: max grid that fits both occupancy and tiles_per_row * num_rows
  // (matches the persistent-grid sizing in the existing fused kernel).
  const int max_blocks_per_grid_by_occupancy = sm_count * fused_sm_occupancy;
  const int max_blocks_for_work              = ::cuda::std::min(
    static_cast<int>(num_rows) * tiles_per_row, ::cuda::std::numeric_limits<int>::max() / 2);
  const int num_thread_blocks =
    ::cuda::std::min(max_blocks_per_grid_by_occupancy, max_blocks_for_work);

  if (num_thread_blocks <= 0)
  {
    return cudaErrorNotSupported;
  }

  // Allocate per-block staging slabs:
  //   primary slab:   num_thread_blocks * NUM_ACTIVE_CHANNELS * hybrid_split_bin * sizeof(CounterT)
  //   secondary slab: num_thread_blocks * NUM_ACTIVE_CHANNELS * hybrid_secondary_size * sizeof(CounterT)
  //   queue counter:  GridQueue<int>::AllocationSize()
  const size_t primary_slab_bytes_per_channel =
    size_t(num_thread_blocks) * size_t(hybrid_split_bin) * size_t(kernel_source.CounterSize());
  const size_t secondary_slab_bytes_per_channel =
    size_t(num_thread_blocks) * size_t(hybrid_secondary_size) * size_t(kernel_source.CounterSize());

  void* allocations[NUM_ACTIVE_CHANNELS * 2 + 1] = {};
  size_t allocation_sizes[NUM_ACTIVE_CHANNELS * 2 + 1];
  for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ++ch)
  {
    allocation_sizes[ch]                       = primary_slab_bytes_per_channel;
    allocation_sizes[ch + NUM_ACTIVE_CHANNELS] = secondary_slab_bytes_per_channel;
  }
  allocation_sizes[NUM_ACTIVE_CHANNELS * 2] = GridQueue<int>::AllocationSize();

  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  {
    return cudaSuccess;
  }

  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_primary_staging_array{};
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_secondary_staging_array{};
  for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ++ch)
  {
    d_primary_staging_array[ch]   = static_cast<CounterT*>(allocations[ch]);
    d_secondary_staging_array[ch] = static_cast<CounterT*>(allocations[ch + NUM_ACTIVE_CHANNELS]);
  }
  GridQueue<int> tile_queue(allocations[NUM_ACTIVE_CHANNELS * 2]);

  // Initialize the smem/gmem bin counts wrappers.
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_smem_bins_wrapper{};
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_gmem_bins_wrapper{};
  for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ++ch)
  {
    num_smem_bins_wrapper[ch] = hybrid_split_bin;
    num_gmem_bins_wrapper[ch] = hybrid_secondary_size;
  }

  // Build the launch.
  dim3 grid_dims(num_thread_blocks, 1, 1);
  dim3 block_dims(threads_per_block, 1, 1);

  void* kernel_args[] = {
    const_cast<void*>(static_cast<const void*>(&d_samples)),
    const_cast<void*>(static_cast<const void*>(&num_smem_bins_wrapper)),
    const_cast<void*>(static_cast<const void*>(&num_gmem_bins_wrapper)),
    const_cast<void*>(static_cast<const void*>(&d_output_histograms)),
    const_cast<void*>(static_cast<const void*>(&d_primary_staging_array)),
    const_cast<void*>(static_cast<const void*>(&d_secondary_staging_array)),
    const_cast<void*>(static_cast<const void*>(&output_decode_op)),
    const_cast<void*>(static_cast<const void*>(&inner_privatized_decode_op)),
    const_cast<void*>(static_cast<const void*>(&hybrid_split_bin)),
    const_cast<void*>(static_cast<const void*>(&hybrid_secondary_size)),
    const_cast<void*>(static_cast<const void*>(&num_row_pixels)),
    const_cast<void*>(static_cast<const void*>(&num_rows)),
    const_cast<void*>(static_cast<const void*>(&row_stride_samples)),
    const_cast<void*>(static_cast<const void*>(&tiles_per_row)),
    const_cast<void*>(static_cast<const void*>(&tile_queue))};

  cudaError_t launch_error = cudaLaunchCooperativeKernel(
    reinterpret_cast<const void*>(fused_hybrid_kernel_ptr),
    grid_dims,
    block_dims,
    kernel_args,
    static_cast<size_t>(dyn_smem_bytes_for_staging),
    stream);

  if (launch_error != cudaSuccess)
  {
    // Fallback: re-issue via the chunked dispatch.
    (void) cudaGetLastError();
    return cudaErrorNotSupported;
  }

  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  return cudaSuccess;
#else
  // Device-side dispatch is not supported for the cooperative hybrid path.
  return cudaErrorNotSupported;
#endif
}

// TODO(bgruber): drop in CCCL 4.0
template <typename ActivePolicy>
_CCCL_HOST_DEVICE_API constexpr auto convert_pdl_trigger(int)
  -> decltype(ActivePolicy::init_kernel_pdl_trigger_max_bins)
{
  return ActivePolicy::init_kernel_pdl_trigger_max_bins;
}

// TODO(bgruber): drop in CCCL 4.0
template <typename ActivePolicy>
_CCCL_HOST_DEVICE_API constexpr auto convert_pdl_trigger(long)
{
  return 0;
}

// TODO(bgruber): drop in CCCL 4.0
template <typename ActivePolicy>
_CCCL_HOST_DEVICE_API constexpr auto convert_policy() -> HistogramPolicy
{
  using ap = typename ActivePolicy::AgentHistogramPolicyT;
  return HistogramPolicy{
    ap::BLOCK_THREADS,
    ap::PIXELS_PER_THREAD,
    ap::VEC_SIZE,
    ap::LOAD_ALGORITHM,
    ap::LOAD_MODIFIER,
    ap::IS_RLE_COMPRESS,
    ap::MEM_PREFERENCE,
    ap::IS_WORK_STEALING,
    convert_pdl_trigger<ActivePolicy>(0)};
}

// TODO(bgruber): drop in CCCL 4.0
template <typename MaxPolicy>
struct policy_selector_from_max_policy
{
private:
  struct extract_policy_dispatch_t
  {
    HistogramPolicy& policy;

    template <typename ActivePolicyT>
    _CCCL_HOST_DEVICE_API constexpr cudaError_t Invoke()
    {
      policy = convert_policy<ActivePolicyT>();
      return cudaSuccess;
    }
  };

public:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> HistogramPolicy
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      ({
                        HistogramPolicy policy{};
                        extract_policy_dispatch_t dispatch{policy};
                        _CCCL_VERIFY(MaxPolicy::Invoke(cc.get() * 10, dispatch) == cudaSuccess, "");
                        return policy;
                      }),
                      ({ return convert_policy<typename MaxPolicy::ActivePolicy>(); }));
  }
};

// Single dispatch entry point used by both `dispatch_even` and `dispatch_range`.
// All algorithm choices flow through this helper: every histogram launch picks
// a member of `algorithm` via `select_algorithm` and then this switch maps
// that to the launcher.
//
// Arguments mirror what the per-algorithm launchers need: privatized + output
// level counts, decode-op arrays (output + privatized), max bin count, and
// the launch geometry. The hybrid launcher ignores `num_privatized_levels`.
//
// On hybrid setup failure (`cudaErrorNotSupported` from
// `dispatch_hybrid_single_pass_staging_smem`), this helper falls through to
// the GMEM-priv path so users do not see a hard error on devices/conditions
// where hybrid's cooperative launch cannot be set up.
template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT,
          typename PolicySelector,
          typename KernelSource,
          typename KernelLauncherFactory>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_by_algorithm(
  algorithm algo,
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_levels,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
  ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op,
  ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op,
  int max_num_output_bins,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_samples,
  cudaStream_t stream,
  PolicySelector policy_selector,
  KernelSource kernel_source,
  KernelLauncherFactory launcher_factory)
{
  switch (algo)
  {
    case algorithm::smem_priv_256: {
      constexpr int PRIVATIZED_SMEM_BINS = max_privatized_smem_bins;
      return dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
        d_temp_storage,
        temp_storage_bytes,
        d_samples,
        d_output_histograms,
        num_privatized_levels,
        num_output_levels,
        output_decode_op,
        privatized_decode_op,
        max_num_output_bins,
        num_row_pixels,
        num_rows,
        row_stride_samples,
        stream,
        policy_selector,
        kernel_source,
        launcher_factory);
    }
    case algorithm::smem_priv_2k: {
      constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel;
      return dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
        d_temp_storage,
        temp_storage_bytes,
        d_samples,
        d_output_histograms,
        num_privatized_levels,
        num_output_levels,
        output_decode_op,
        privatized_decode_op,
        max_num_output_bins,
        num_row_pixels,
        num_rows,
        row_stride_samples,
        stream,
        policy_selector,
        kernel_source,
        launcher_factory);
    }
    case algorithm::smem_priv_8k: {
      constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel_large;
      return dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
        d_temp_storage,
        temp_storage_bytes,
        d_samples,
        d_output_histograms,
        num_privatized_levels,
        num_output_levels,
        output_decode_op,
        privatized_decode_op,
        max_num_output_bins,
        num_row_pixels,
        num_rows,
        row_stride_samples,
        stream,
        policy_selector,
        kernel_source,
        launcher_factory);
    }
    case algorithm::smem_priv_16k: {
      constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel_xlarge;
      return dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
        d_temp_storage,
        temp_storage_bytes,
        d_samples,
        d_output_histograms,
        num_privatized_levels,
        num_output_levels,
        output_decode_op,
        privatized_decode_op,
        max_num_output_bins,
        num_row_pixels,
        num_rows,
        row_stride_samples,
        stream,
        policy_selector,
        kernel_source,
        launcher_factory);
    }
    case algorithm::hybrid_single_pass: {
      // Hybrid is single-channel-only. The selector is responsible for not
      // routing multi-channel cells here; if it does, fall through to the
      // GMEM-priv path below.
      if constexpr (NUM_ACTIVE_CHANNELS == 1)
      {
        const auto status =
          dispatch_hybrid_single_pass_staging_smem<NUM_CHANNELS,
                                                   NUM_ACTIVE_CHANNELS,
                                                   hybrid_smem_split_bin_single_channel,
                                                   chunked_smem_bins_max_single_channel>(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_output_histograms,
            output_decode_op,
            privatized_decode_op,
            max_num_output_bins,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            stream,
            policy_selector,
            kernel_source,
            launcher_factory);
        if (status == cudaSuccess || status != cudaErrorNotSupported)
        {
          return status;
        }
        // hybrid setup failed; fall through to the GMEM-priv path.
      }
      [[fallthrough]];
    }
    case algorithm::gmem_priv_cuckoo:
    case algorithm::gmem_priv_sweep:
    case algorithm::gmem_priv_single_probe: {
      // All three go through PRIVATIZED_SMEM_BINS=0 in the deeper `dispatch<>`,
      // which chooses between the direct-atomic-to-output kernel and the
      // SweepPersistent gather-merge kernel based on `disable_direct_atomic`,
      // and (for the direct-atomic path) between the cuckoo and single-probe
      // caches based on `direct_atomic_cache_mode`. We pass both from the
      // selector's pick:
      //   gmem_priv_sweep        -> disable_direct_atomic=true  (sweep)
      //   gmem_priv_cuckoo       -> direct-atomic, cache_mode=0 (cuckoo)
      //   gmem_priv_single_probe -> direct-atomic, cache_mode=1 (single-probe)
      constexpr int PRIVATIZED_SMEM_BINS  = 0;
      const bool disable_direct_atomic_io = (algo == algorithm::gmem_priv_sweep);
      const int direct_atomic_cache_mode  = (algo == algorithm::gmem_priv_single_probe) ? 1 : 0;
      return dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
        d_temp_storage,
        temp_storage_bytes,
        d_samples,
        d_output_histograms,
        num_privatized_levels,
        num_output_levels,
        output_decode_op,
        privatized_decode_op,
        max_num_output_bins,
        num_row_pixels,
        num_rows,
        row_stride_samples,
        stream,
        policy_selector,
        kernel_source,
        launcher_factory,
        disable_direct_atomic_io,
        direct_atomic_cache_mode);
    }
  }
  return cudaErrorInvalidValue; // unreachable
}

template <
  int NUM_CHANNELS,
  int NUM_ACTIVE_CHANNELS,
  typename SampleIteratorT,
  typename CounterT,
  typename LevelT,
  typename OffsetT,
  bool IsByteSample,
  typename PolicySelector,
  typename SampleT = it_value_t<SampleIteratorT>, /// The sample value type of the input iterator
  typename KernelSource =
    DeviceHistogramKernelSource<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, LevelT, OffsetT, SampleT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION static cudaError_t dispatch_range(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
  ::cuda::std::array<const LevelT*, NUM_ACTIVE_CHANNELS> d_levels,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_samples,
  cudaStream_t stream,
  ::cuda::std::bool_constant<IsByteSample>,
  PolicySelector policy_selector,
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  if constexpr (IsByteSample)
  {
    using TransformsT = Transforms<LevelT, OffsetT, SampleT>;

    // Use the pass-thru transform op for converting samples to privatized bins
    using PrivatizedDecodeOpT = typename TransformsT::PassThruTransform;

    // Use the search transform op for converting privatized bins to output bins
    using OutputDecodeOpT = typename TransformsT::template SearchTransform<const LevelT*>;

    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_levels;
    ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op{};
    ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op{};
    int max_levels = num_output_levels[0];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
      num_privatized_levels[channel] = 257;
      output_decode_op[channel].Init(d_levels[channel], num_output_levels[channel]);

      if (num_output_levels[channel] > max_levels)
      {
        max_levels = num_output_levels[channel];
      }
    }
    int max_num_output_bins = max_levels - 1;

    constexpr int PRIVATIZED_SMEM_BINS = 256;

    if (const auto error = CubDebug(
          (detail::histogram::dispatch<NUM_CHANNELS,
                                       NUM_ACTIVE_CHANNELS,
                                       PRIVATIZED_SMEM_BINS,
                                       /* IsDeviceInit = */ false,
                                       /* IsEven = (unused for host-init) */ false,
                                       /* IsByteSample = (unused for host-init) */ false>(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_output_histograms,
            num_privatized_levels,
            num_output_levels,
            output_decode_op,
            privatized_decode_op,
            max_num_output_bins,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            stream,
            policy_selector,
            kernel_source,
            launcher_factory))))
    {
      return error;
    }
  }
  else
  {
    using TransformsT = Transforms<LevelT, OffsetT, SampleT>;

    // Use the search transform op for converting samples to privatized bins
    using PrivatizedDecodeOpT = typename TransformsT::template SearchTransform<const LevelT*>;

    // Use the pass-thru transform op for converting privatized bins to output bins
    using OutputDecodeOpT = typename TransformsT::PassThruTransform;

    ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op{};
    ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op{};
    int max_levels = num_output_levels[0];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
      privatized_decode_op[channel].Init(d_levels[channel], num_output_levels[channel]);
      if (num_output_levels[channel] > max_levels)
      {
        max_levels = num_output_levels[channel];
      }
    }
    const int max_num_output_bins = max_levels - 1;

    selector_features features{};
    features.num_active_channels = NUM_ACTIVE_CHANNELS;
    features.sample_bytes        = sizeof(SampleT);
    features.is_byte_sample      = (sizeof(SampleT) == 1);
    features.is_even             = false;
    features.num_bins            = max_num_output_bins;
    features.num_pixels =
      static_cast<long long>(num_row_pixels) * static_cast<long long>(num_rows);
    const algorithm algo = select_algorithm<false>(features);

    return CubDebug((dispatch_by_algorithm<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      algo,
      d_temp_storage,
      temp_storage_bytes,
      d_samples,
      d_output_histograms,
      num_output_levels,
      num_output_levels,
      output_decode_op,
      privatized_decode_op,
      max_num_output_bins,
      num_row_pixels,
      num_rows,
      row_stride_samples,
      stream,
      policy_selector,
      kernel_source,
      launcher_factory)));
  }

  return cudaSuccess;
}

template <
  int NUM_CHANNELS,
  int NUM_ACTIVE_CHANNELS,
  typename SampleIteratorT,
  typename CounterT,
  typename LevelT,
  typename OffsetT,
  bool IsByteSample,
  typename PolicySelector,
  typename SampleT = it_value_t<SampleIteratorT>, /// The sample value type of the input iterator
  typename KernelSource =
    DeviceHistogramKernelSource<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, LevelT, OffsetT, SampleT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch_even(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
  ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> lower_level,
  ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> upper_level,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_samples,
  cudaStream_t stream,
  ::cuda::std::bool_constant<IsByteSample>,
  PolicySelector policy_selector,
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  if constexpr (IsByteSample)
  {
    using TransformsT = Transforms<LevelT, OffsetT, SampleT>;

    // Use the pass-thru transform op for converting samples to privatized bins
    using PrivatizedDecodeOpT = typename TransformsT::PassThruTransform;

    // Use the scale transform op for converting privatized bins to output bins
    using OutputDecodeOpT = typename TransformsT::ScaleTransform;

    using CommonT = typename TransformsT::ScaleTransform::CommonT;

    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_levels;
    ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op{};
    ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op{};
    int max_levels = num_output_levels[0];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
      num_privatized_levels[channel] = 257;

      int num_levels = num_output_levels[channel];
      if (kernel_source.MayOverflow(num_levels - 1, upper_level, lower_level, channel))
      {
        if (!d_temp_storage)
        {
          temp_storage_bytes = 1U;
        }
        return cudaErrorInvalidValue;
      }

      output_decode_op[channel].Init(num_levels, upper_level[channel], lower_level[channel]);

      if (num_levels > max_levels)
      {
        max_levels = num_levels;
      }
    }
    int max_num_output_bins = max_levels - 1;

    constexpr int PRIVATIZED_SMEM_BINS = 256;

    if (const auto error = CubDebug(
          (detail::histogram::dispatch<NUM_CHANNELS,
                                       NUM_ACTIVE_CHANNELS,
                                       PRIVATIZED_SMEM_BINS,
                                       /* IsDeviceInit = */ false,
                                       /* IsEven = */ false,
                                       /* IsByteSample = */ false>(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_output_histograms,
            num_privatized_levels,
            num_output_levels,
            output_decode_op,
            privatized_decode_op,
            max_num_output_bins,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            stream,
            policy_selector,
            kernel_source,
            launcher_factory))))
    {
      return error;
    }
  }
  else
  {
    using TransformsT = Transforms<LevelT, OffsetT, SampleT>;

    // Use the scale transform op for converting samples to privatized bins
    using PrivatizedDecodeOpT = typename TransformsT::ScaleTransform;

    // Use the pass-thru transform op for converting privatized bins to output bins
    using OutputDecodeOpT = typename TransformsT::PassThruTransform;

    ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op{};
    ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op{};
    int max_levels = num_output_levels[0];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
      int num_levels = num_output_levels[channel];
      if (kernel_source.MayOverflow(num_levels - 1, upper_level, lower_level, channel))
      {
        if (!d_temp_storage)
        {
          temp_storage_bytes = 1U;
        }
        return cudaErrorInvalidValue;
      }

      privatized_decode_op[channel].Init(num_levels, upper_level[channel], lower_level[channel]);

      if (num_levels > max_levels)
      {
        max_levels = num_levels;
      }
    }
    const int max_num_output_bins = max_levels - 1;

    selector_features features{};
    features.num_active_channels = NUM_ACTIVE_CHANNELS;
    features.sample_bytes        = sizeof(SampleT);
    features.is_byte_sample      = (sizeof(SampleT) == 1);
    features.is_even             = true;
    features.num_bins            = max_num_output_bins;
    features.num_pixels =
      static_cast<long long>(num_row_pixels) * static_cast<long long>(num_rows);
    const algorithm algo = select_algorithm<false>(features);

    return CubDebug((dispatch_by_algorithm<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      algo,
      d_temp_storage,
      temp_storage_bytes,
      d_samples,
      d_output_histograms,
      num_output_levels,
      num_output_levels,
      output_decode_op,
      privatized_decode_op,
      max_num_output_bins,
      num_row_pixels,
      num_rows,
      row_stride_samples,
      stream,
      policy_selector,
      kernel_source,
      launcher_factory)));
  }

  return cudaSuccess;
}
} // namespace detail::histogram

/******************************************************************************
 * Dispatch
 ******************************************************************************/

// TODO(bgruber): remove in CCCL 4.0
/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceHistogram
 *
 * Deprecated [Since 3.5]
 *
 * @tparam NUM_CHANNELS
 *   Number of channels interleaved in the input data (may be greater than the number of channels
 *   being actively histogrammed)
 *
 * @tparam NUM_ACTIVE_CHANNELS
 *   Number of channels actively being histogrammed
 *
 * @tparam SampleIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam CounterT
 *   Integer type for counting sample occurrences per histogram bin
 *
 * @tparam LevelT
 *   Type for specifying bin level boundaries
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam PolicyHub
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <
  int NUM_CHANNELS,
  int NUM_ACTIVE_CHANNELS,
  typename SampleIteratorT,
  typename CounterT,
  typename LevelT,
  typename OffsetT,
  typename PolicyHub    = void, // if user passes a custom Policy this should not be void
  typename SampleT      = cub::detail::it_value_t<SampleIteratorT>, /// The sample value type of the input iterator
  typename KernelSource = detail::histogram::
    DeviceHistogramKernelSource<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, LevelT, OffsetT, SampleT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct CCCL_DEPRECATED_BECAUSE("Use the tuning API for DeviceHistogram") DispatchHistogram
{
  static_assert(NUM_CHANNELS <= 4, "Histograms only support up to 4 channels");
  static_assert(NUM_ACTIVE_CHANNELS <= NUM_CHANNELS,
                "Active channels must be at most the number of total channels of the input samples");

  //---------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // Default (host-init) dispatch entrypoints
  // These methods initialize decode operators on the host before kernel launch.
  //---------------------------------------------------------------------

  /**
   * Dispatch routine for HistogramRange with host-side decode operator initialization.
   * This variant initializes the decode operators on the host before kernel launch.
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to `temp_storage_bytes` and
   *   no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_samples
   *   The pointer to the multi-channel input sequence of data samples.
   *   The samples from different channels are assumed to be interleaved
   *   (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
   *
   * @param d_output_histograms
   *   The pointers to the histogram counter output arrays, one for each active channel.
   *   For channel<sub><em>i</em></sub>, the allocation length of `d_histograms[i]` should be
   *   `num_output_levels[i] - 1`.
   *
   * @param num_output_levels
   *   The number of boundaries (levels) for delineating histogram samples in each active channel.
   *   Implies that the number of bins for channel<sub><em>i</em></sub> is
   *   `num_output_levels[i] - 1`.
   *
   * @param d_levels
   *   The pointers to the arrays of boundaries (levels), one for each active channel.
   *   Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are
   *   inclusive and upper sample value boundaries are exclusive.
   *
   * @param num_row_pixels
   *   The number of multi-channel pixels per row in the region of interest
   *
   * @param num_rows
   *   The number of rows in the region of interest
   *
   * @param row_stride_samples
   *   The number of samples between starts of consecutive rows in the region of interest
   *
   * @param stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  template <typename MaxPolicyT = typename ::cuda::std::_If<
              ::cuda::std::is_void_v<PolicyHub>,
              /* fallback_policy_hub */
              detail::histogram::policy_hub<SampleT, CounterT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, /* isEven */ false>,
              PolicyHub>::MaxPolicy,
            bool IsByteSample>
  CUB_RUNTIME_FUNCTION static cudaError_t DispatchRange(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    SampleIteratorT d_samples,
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
    ::cuda::std::array<const LevelT*, NUM_ACTIVE_CHANNELS> d_levels,
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    cudaStream_t stream,
    ::cuda::std::bool_constant<IsByteSample> is_byte_sample,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    [[maybe_unused]] MaxPolicyT max_policy = {})
  {
    return detail::histogram::dispatch_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      d_temp_storage,
      temp_storage_bytes,
      d_samples,
      d_output_histograms,
      num_output_levels,
      d_levels,
      num_row_pixels,
      num_rows,
      row_stride_samples,
      stream,
      is_byte_sample,
      detail::histogram::policy_selector_from_max_policy<MaxPolicyT>{},
      kernel_source,
      launcher_factory);
  }

  /**
   * Dispatch routine for HistogramEven with host-side decode operator initialization.
   * This variant initializes the decode operators on the host before kernel launch.
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to
   *   `temp_storage_bytes` and no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_samples
   *   The pointer to the input sequence of sample items.
   *   The samples from different channels are assumed to be interleaved
   *   (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
   *
   * @param d_output_histograms
   *   The pointers to the histogram counter output arrays, one for each active channel.
   *   For channel<sub><em>i</em></sub>, the allocation length of `d_histograms[i]` should be
   *   `num_output_levels[i] - 1`.
   *
   * @param num_output_levels
   *   The number of bin level boundaries for delineating histogram samples in each active channel.
   *   Implies that the number of bins for channel<sub><em>i</em></sub> is
   *   `num_output_levels[i] - 1`.
   *
   * @param lower_level
   *   The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
   *
   * @param upper_level
   *   The upper sample value bound (exclusive) for the highest histogram bin in each active
   * channel.
   *
   * @param num_row_pixels
   *   The number of multi-channel pixels per row in the region of interest
   *
   * @param num_rows
   *   The number of rows in the region of interest
   *
   * @param row_stride_samples
   *   The number of samples between starts of consecutive rows in the region of interest
   *
   * @param stream
   *   CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
   *
   */
  template <typename MaxPolicyT = typename ::cuda::std::_If<
              ::cuda::std::is_void_v<PolicyHub>,
              /* fallback_policy_hub */
              detail::histogram::policy_hub<SampleT, CounterT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, /* isEven */ true>,
              PolicyHub>::MaxPolicy,
            bool IsByteSample>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t DispatchEven(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    SampleIteratorT d_samples,
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
    ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> lower_level,
    ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> upper_level,
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    cudaStream_t stream,
    ::cuda::std::bool_constant<IsByteSample> is_byte_sample,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    [[maybe_unused]] MaxPolicyT max_policy = {})
  {
    return detail::histogram::dispatch_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      d_temp_storage,
      temp_storage_bytes,
      d_samples,
      d_output_histograms,
      num_output_levels,
      lower_level,
      upper_level,
      num_row_pixels,
      num_rows,
      row_stride_samples,
      stream,
      is_byte_sample,
      detail::histogram::policy_selector_from_max_policy<MaxPolicyT>{},
      kernel_source,
      launcher_factory);
  }
};

CUB_NAMESPACE_END
