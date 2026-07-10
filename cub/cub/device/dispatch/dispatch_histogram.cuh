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
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/array>
#include <cuda/std/limits>
#include <cuda/std/tuple>

#include <nv/target>

#include <cstdio>
#include <cstdlib>
#include <cstring>

CUB_NAMESPACE_BEGIN

namespace detail::histogram
{
template <typename PolicySelector, typename OutputCounterT, typename = void>
struct local_counter
{
  using type = OutputCounterT;
};

template <typename PolicySelector, typename OutputCounterT>
struct local_counter<PolicySelector, OutputCounterT, ::cuda::std::void_t<typename PolicySelector::local_counter_type>>
{
  using type = typename PolicySelector::local_counter_type;
};

//! Counter type used by per-block SMEM/cache/private storage. Tuning selectors
//! may opt into a narrower local counter without changing DeviceHistogram's
//! public output type; legacy selectors retain the output counter type.
template <typename PolicySelector, typename OutputCounterT>
using local_counter_t = typename local_counter<PolicySelector, OutputCounterT>::type;

// Maximum number of bins per channel for which we will use a privatized smem strategy
static constexpr int max_privatized_smem_bins = 256;

// Above the static 256-bin tier, larger privatized histograms are kept on chip in a
// single dynamic-SMEM kernel whose per-block histogram lives in extern __shared__
// sized AT LAUNCH from the runtime bin count. This keeps the histogram in fast SMEM
// (avoiding the GMEM atomicAdd_block of the GMEM-privatized path) without a ladder of
// compile-time-sized kernels. The per-block footprint exceeds the ptxas 48 KB static
// cap (e.g. 16384 bins x 4 B = 64 KB), so the dynamic kernel raises its
// cudaFuncAttributeMaxDynamicSharedMemorySize.
//
// `kDynamicSmemKernelTagBins` is ONLY a compile-time instantiation tag: it is the
// nonzero `PRIVATIZED_SMEM_BINS` template value that selects the dynamic-SMEM kernel
// (vs the static 256-bin kernel at `max_privatized_smem_bins`, vs the GMEM-privatized
// path at 0). It does NOT size any storage on the dynamic path -- ZeroBinCounters /
// StoreOutput / the accumulate loop all bound on the RUNTIME `num_privatized_bins[ch]`
// (see agent_histogram.cuh), and the launch SMEM is `Σ_ch bins * CounterSize()`. So
// the actual on-chip CAPACITY is a runtime byte budget, not this constant.
//
// The runtime routing cap -- the largest bin count the selector keeps on chip -- is
// `detail::histogram::max_dynamic_smem_bins(counter_bytes, channels, device_optin)`
// (tuning_histogram.cuh), derived from `cache_tuning::max_dynamic_smem_bytes`. It was
// previously a single frozen `16384`, which conflated a per-arch HARDWARE byte budget
// with a bin count (silently assuming a 4-byte counter and one channel) and with this
// compile-time tag. Those three roles are now separated.
static constexpr int kDynamicSmemKernelTagBins = 16384;

// Multi-channel privatized-SMEM eligibility is more restrictive than
// single-channel: the per-block histogram footprint and the classify cost both
// scale with the active channel count, so beyond these bin counts multi-channel
// cells route to the high-bin (direct-atomic) path instead. RANGE (expensive
// SearchTransform classify) stays privatized only up to the smaller bound; EVEN
// (cheap ScaleTransform classify) up to the larger.
static constexpr int multi_channel_smem_bins_range = 2048;
static constexpr int multi_channel_smem_bins_even  = 8192;

// Upper bin bound of the single-channel hybrid SMEM+GMEM single-pass kernel: at
// or below this the histogram is small enough that keeping its primary range on
// chip pays off (above it, the high-bin direct-atomic caches take over). Also
// the cap below which single-channel high-bin cells consider the hybrid path.
static constexpr int hybrid_smem_bins_max_single_channel = 65536;

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
// (`dispatch<>`, `dispatch_gmem_privatized_hybrid`) stay; they are
// launchers, not pickers. To add or retire a kernel, add an enumerator here,
// teach `select_algorithm` when to pick it, and add a `case` to
// `dispatch_by_algorithm`.
//
// Centralising the choice keeps the dispatch trade-off surface (single vs
// multi channel, EVEN vs RANGE classify cost, bin count, element count,
// sample width) in one place rather than scattered across cascades in two
// dispatch entry points.
// Histogram algorithm taxonomy, named by (on-chip structure) × (where increments
// land), per cached_privatized_spill_design.md. Three kernel families:
//   * smem_privatized   — whole histogram on chip, atomic-merge to output (low bins).
//   * direct_*          — CacheSpillKernel<Combiner>: device-scope atomics to the SHARED
//                         output (the honest meaning of the old "direct atomic"),
//                         fronted by a Combiner ∈ {NoCache, Cuckoo, SingleProbe}.
//   * gmem_privatized_* — GmemPrivatizedKernel: per-block GMEM-privatized histogram +
//                         atomic-free gather. Combiner is the SMEM front-end:
//                         NoCache (static smem_split tier — the merged hybrid / pure
//                         gather), or Cuckoo / SingleProbe (the design proposal).
enum class algorithm : unsigned char
{
  // --- Privatized-SMEM histograms (low/mid bin counts) ---

  // Whole-histogram-on-chip privatized SMEM. Covers BOTH the old smem_priv_256
  // (static, compile-time-sized __shared__, bins <= 256 and all byte samples) and
  // the old smem_priv_dynamic (extern __shared__ sized at launch, 256 < bins <=
  // max_dynamic_smem_bins); dispatch_by_algorithm recovers the static-vs-dynamic
  // tier from the bin count. Merged to one enumerator per the design doc.
  smem_privatized,

  // --- High-bin algorithms (bins > max_dynamic_smem_bins) ---
  // CacheSpillKernel<Combiner>: combiner-fronted device-scope atomics to the shared output.

  // No on-chip cache: warp-coalesce then device-scope atomicAdd straight to the
  // output. The honest "pure direct atomic". (Not auto-selected; reachable via
  // dispatch + the CUB_HISTO_FORCE_ALGO hook. Isolates the combiner's value.)
  direct_nocache,

  // 2-hash (cuckoo) SMEM cache front-end; absorbs cross-block contention for
  // skewed hot-bin inputs. (Was direct_atomic_cuckoo.)
  direct_cuckoo,

  // Single-probe direct-mapped SMEM cache front-end; the shorter, less-divergent
  // probe wins at very high bin counts where bins vastly outnumber cache slots.
  // (Was direct_atomic_single_probe.)
  direct_single_probe,

  // --- GMEM-privatized algorithms: per-block private histogram + atomic-free gather ---
  // GmemPrivatizedKernel<Combiner, smem_split>.

  // NoCache combiner. Covers BOTH the old gmem_priv_gather (smem_split == 0, whole
  // histogram in per-block GMEM) and the old hybrid_single_pass (smem_split > 0,
  // primary bin range promoted to SMEM, tail in GMEM); the smem_split value chosen
  // by dispatch selects the kernel's HybridSplit instantiation. Merged to one
  // enumerator per the design doc. Single-channel for the hybrid (smem_split>0) sub-case.
  // Reachable via dispatch_by_algorithm + the CUB_HISTO_FORCE_ALGO hook; no longer
  // auto-selected (the high-bin region routes to direct_single_probe).
  gmem_privatized_nocache,

  // (The proposed gmem_privatized_{cuckoo,single_probe} cache+private-spill members were
  // removed: a full-matrix sweep measured them never best nor within 2% of best on any
  // cell, so they were pure dispatch surface. The gather (gmem_privatized_nocache) and the
  // direct-atomic caches cover everything they could.)
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
  // Largest bin count that still fits the whole histogram on chip in the
  // dynamic-SMEM privatized kernel, for THIS counter width and channel count.
  // Derived host-side from the per-arch byte budget + device opt-in SMEM via
  // detail::histogram::max_dynamic_smem_bins(); the caller fills it in (it cannot
  // be computed here -- select_algorithm is _CCCL_HOST_DEVICE and the device query
  // is host-only). Replaces the old frozen 16384 routing threshold.
  int on_chip_bin_cap;
  // Max PRIMARY bins the single-channel hybrid can stage in dyn-SMEM for THIS counter
  // width, from detail::histogram::hybrid_smem_split_bins(). Byte-derived (shrinks for
  // wide counters) so the hybrid is selected only where its on-chip primary is large
  // enough to be worthwhile and, critically, actually fits the per-CTA SMEM cap.
  // Replaces the old frozen 49152 split that assumed a 4-byte counter.
  int hybrid_split_bins;
};

// Pick a single algorithm for one cell. Rules are first-match. The selector
// cannot observe input entropy, so each rule is the geomean winner over the
// input-distribution mix for its (channels, sample width, bin tier, pixels)
// regime, derived by sweeping every algorithm across the benchmark matrix.
//
//   1. Privatized-SMEM region (num_bins <= max_dynamic_smem_bins): all of these
//      return `smem_privatized` (whole histogram on chip); dispatch_by_algorithm
//      recovers the static <=256-bin tier vs the dynamic-SMEM tier from the bin
//      count. Multi-channel eligibility is more restrictive (per-block footprint +
//      classify cost scale with the active channel count), so multi-channel RANGE /
//      >3-channel cells past the relevant cap fall through to the high-bin region.
//
//   2. High-bin region: the single-channel SMEM+GMEM `gmem_privatized_nocache`
//      (smem_split>0, the merged hybrid member) and the two CacheSpillKernel caches
//      (`direct_single_probe` / `direct_cuckoo`). Single-channel uses the on-chip
//      hybrid where the histogram fits AND the input amortizes its setup (the 65536
//      and 131072 tiers above their per-transform pixel floors); above 131072 (the
//      262144 and 1048576 tiers) the histogram exceeds the hybrid's on-chip working
//      set, so direct atomics win. Multi-channel uses the direct caches. The cuckoo
//      and single-probe caches measure within noise, so single-probe (the leaner
//      probe) is the default and cuckoo serves the larger multi bin tiers. (The
//      proposed `gmem_privatized_{cuckoo,single_probe}` are never returned here —
//      they are reachable-but-unselected; see the design doc's Decision.)
template <bool IsByteSample>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE algorithm select_algorithm(selector_features const& f)
{
  // Byte samples: 256-entry pass-thru privatized histograms, then a final
  // scale-transform combine. Bin counts above that are not legal for byte
  // samples (LevelT == SampleT and num_bins fits in [0, 255]).
  if constexpr (IsByteSample)
  {
    return algorithm::smem_privatized;
  }

  // -----------------------------------------------------------------------
  // Privatized-SMEM region. <=256 bins (and all byte samples) use the static
  // fixed-size kernel; above that, a single dynamic-SMEM kernel covers the
  // whole range up to the cap. Multi-channel eligibility is more restrictive
  // (footprint + classify cost scale with channel count); past it, cells fall
  // through to the high-bin region below.
  // -----------------------------------------------------------------------
  if (f.num_bins <= max_privatized_smem_bins)
  {
    return algorithm::smem_privatized;
  }
  if (f.num_active_channels == 1)
  {
    if (f.num_bins <= f.on_chip_bin_cap)
    {
      return algorithm::smem_privatized;
    }
  }
  else
  {
    // RANGE (or any transform) up to the smaller cap; EVEN extends further
    // (cheap classify), with the top of the range gated to <=3 active channels.
    // The multi_channel_smem_bins_* caps are the MEASURED perf crossovers for a 4-byte
    // counter; they must additionally be clamped by `on_chip_bin_cap` (the byte budget
    // / (counter_bytes * channels)) so a wide counter cannot select the dynamic-SMEM
    // privatized kernel at a bin count whose per-CTA footprint exceeds the opt-in cap.
    // For a 4-byte counter on_chip_bin_cap >> these caps, so 4-byte routing is unchanged;
    // for an 8-byte counter (e.g. 4-channel EVEN at 8192 bins = 256 KiB > cap) the clamp
    // routes the cell to the high-bin direct-atomic path instead of crashing the launch.
    const int multi_range_cap =
      (multi_channel_smem_bins_range < f.on_chip_bin_cap) ? multi_channel_smem_bins_range : f.on_chip_bin_cap;
    const int multi_even_cap =
      (multi_channel_smem_bins_even < f.on_chip_bin_cap) ? multi_channel_smem_bins_even : f.on_chip_bin_cap;
    if (f.num_bins <= multi_range_cap)
    {
      return algorithm::smem_privatized;
    }
    if (f.is_even && f.num_bins <= multi_even_cap)
    {
      return algorithm::smem_privatized;
    }
    if (f.is_even && f.num_active_channels <= 3 && f.num_bins <= f.on_chip_bin_cap)
    {
      return algorithm::smem_privatized;
    }
  }

  // -----------------------------------------------------------------------
  // High-bin region (num_bins above the on-chip privatized cap: 65536 / 131072 /
  // 262144 / 1048576).
  //
  // The choice is among: the single-channel SMEM+GMEM hybrid (the smem_split>0
  // member of `gmem_privatized_nocache`); the two CacheSpillKernel caches
  // (`direct_cuckoo`, `direct_single_probe`); and the cooperative pure-gather (the
  // smem_split==0 member of `gmem_privatized_nocache`). The selector cannot observe input entropy (a runtime
  // data property), so each rule picks the algorithm with the best GiB/s GEOMEAN
  // over the input-distribution mix (uniform / skewed / single-hot-bin) for that
  // (channels, sample-width, bin-tier, pixel-count) regime, measured by sweeping
  // every algorithm across the full benchmark matrix on the target architecture.
  //
  // Two facts shape the rules. (1) The cuckoo and single-probe caches perform
  // within noise of each other across the whole single-channel region (both are
  // really direct GMEM atomics once bins far exceed the few-thousand cache
  // slots), so single-probe -- the leaner inner loop -- is the single
  // direct-atomic default. (2) The on-chip hybrid kernel wins the 65536 and 131072
  // tiers at large input, where keeping the whole modest histogram in SMEM beats
  // atomics into a large GMEM output; it loses at small input (its per-block setup
  // is not amortized), and from the 262144 tier up (the histogram exceeds its
  // on-chip working set, so the direct-atomic cache wins at every input size --
  // measured hybrid/direct 0.68..1.00 across the 262144 grid).
  // -----------------------------------------------------------------------

  // Above the on-chip privatized cap, one algorithm wins across the whole high-bin
  // region for every (channels, sample-width, bin-tier, pixel-count) regime: the
  // direct-atomic single-probe cache, run with warp-coalescing DISABLED. Measured on
  // B200 (sm_100) over the full benchmark matrix (run_2026-06-15_coalesce, I32+F64,
  // 14 shapes), it is the per-cell best or within ~2% at every off-chip tier and beats
  // the previous routing (hybrid / gather / cuckoo) by geomean 1.3-1.7x. The hybrid
  // (smem_split>0 gmem_privatized member) and the cuckoo cache were each retired from
  // the selector: single-probe matched or beat them everywhere once coalescing was
  // turned off on the cache kernels (the coalesce penalty -- a __match_any_sync whose
  // dependent atomic stalls when a warp's bins are distinct -- was what made the
  // direct-atomic caches look weak in the older sweeps). single-probe is also the
  // leaner probe, so this is simpler AND faster. Both transforms, both sample widths,
  // both counter widths, single- and multi-channel collapse to this one rule.
  return algorithm::direct_single_probe;
}

// The device's per-CTA opt-in dynamic-SMEM limit (cudaDevAttrMaxSharedMemoryPerBlockOptin),
// or 0 if unavailable -- on the device side (the query is host-only) or on query failure,
// in which case the byte-budget derivations below fall back to their tuned budgets. Mirrors
// the cache path's opt-in query (it already calls cudaDeviceGetAttribute unguarded in the
// same dispatch chain). Queried once per dispatch, before kernel launch; zero device SASS.
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE int query_device_optin_smem_bytes()
{
  int device_optin_smem_bytes = 0;
#if _CCCL_HOSTED()
  NV_IF_TARGET(NV_IS_HOST, ({
                 int dev = 0;
                 if (cudaGetDevice(&dev) != cudaSuccess)
                 {
                   (void) cudaGetLastError();
                   dev = 0;
                 }
                 if (cudaDeviceGetAttribute(&device_optin_smem_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev)
                     != cudaSuccess)
                 {
                   (void) cudaGetLastError();
                   device_optin_smem_bytes = 0; // -> derivations fall back to the tuned budget
                 }
               }));
#endif // _CCCL_HOSTED()
  return device_optin_smem_bytes;
}

// Runtime on-chip bin cap for `selector_features::on_chip_bin_cap`: the largest bin count
// the dynamic-SMEM privatized kernel can hold for this counter width + channel count on
// the current device (byte budget / per-bin footprint).
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE int resolve_on_chip_bin_cap(int counter_bytes, int num_active_channels)
{
  return max_dynamic_smem_bins(counter_bytes, num_active_channels, query_device_optin_smem_bytes());
}

// Runtime hybrid primary split for `selector_features::hybrid_split_bins`: the largest
// PRIMARY (on-chip) bin count the single-channel hybrid can stage for this counter width,
// byte-derived so it shrinks for wide counters rather than overflowing the SMEM cap.
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE int resolve_hybrid_split_bins(int counter_bytes, int num_active_channels)
{
  return hybrid_smem_split_bins(counter_bytes, num_active_channels, query_device_optin_smem_bytes());
}

template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          typename SampleIteratorT,
          typename CounterT,
          typename LevelT,
          typename OffsetT,
          typename SampleT,
          typename OutputCounterT = CounterT>
struct DeviceHistogramKernelSource
{
  static_assert(sizeof(CounterT) <= sizeof(OutputCounterT),
                "The output histogram counter must be at least as wide as the local counter");

  using TransformsT = detail::histogram::Transforms<LevelT, OffsetT, SampleT>;

  template <typename PolicyT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramInitKernel()
  {
    return &DeviceHistogramInitKernel<PolicyT, NUM_ACTIVE_CHANNELS, OutputCounterT, OffsetT>;
  }

  /// Returns the default histogram sweep kernel that receives pre-initialized decode operators from the host.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSmemPrivatizedKernel()
  {
    return &DeviceHistogramSmemPrivatizedKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT,
      OutputCounterT>;
  }

  /// Host-init dynamic-SMEM, NON-staging variant: merges each block's dyn-SMEM
  /// privatized histogram directly into the global output via atomicAdd
  /// (no staging slabs, no combine kernel). Host must launch the init kernel first.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSmemPrivatizedDynamicKernel()
  {
    return &DeviceHistogramSmemPrivatizedDynamicKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT,
      OutputCounterT>;
  }

  /// Host-init FUSED HYBRID single-pass dynamic-SMEM staging+combine sweep kernel. Eliminates
  /// the 2x sample re-read of the dual-chunk kernel by handling both bin "chunks" in a single
  /// sweep: bins in the primary range live in dyn-SMEM, bins in the secondary range live in
  /// per-block GMEM staging slabs. The decode op is the un-chunked privatized op, classifying
  /// each sample once into the full bin space.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramGmemPrivatizedHybridKernel()
  {
    // Hybrid instantiation of the unified GmemPrivatized kernel (HybridSplit=true,
    // smem_split>0): SMEM primary range + GMEM secondary tail + fused reduce.
    return &DeviceHistogramGmemPrivatizedKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT,
      /*HybridSplit=*/true,
      OutputCounterT>;
  }

  /// Pure-gather member of the unified GmemPrivatized kernel (HybridSplit=false,
  /// smem_split=0): per-block GMEM-privatized histogram + grid-sync + atomic-free
  /// gather. Launched cooperatively. C-parallel overrides this to return its JIT
  /// CUkernel.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramGmemPrivatizedKernel()
  {
    return &DeviceHistogramGmemPrivatizedKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT,
      /*HybridSplit=*/false,
      OutputCounterT>;
  }

  /// The cooperative CacheSpillKernel<Combiner, SpillOp> family (cuckoo / single-probe /
  /// no-cache cache front-end x output / private spill). One accessor parameterized
  /// by the probe + spill ops covers all six combinations the dispatch can launch;
  /// C-parallel overrides it to return the matching JIT CUkernel.
  template <typename PolicyT,
            int PRIVATIZED_SMEM_BINS,
            typename PrivatizedDecodeOpT,
            typename OutputDecodeOpT,
            typename ProbeOp,
            typename SpillOp>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramCacheSpillKernel()
  {
    return &DeviceHistogramCacheSpillKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT,
      ProbeOp,
      SpillOp,
      OutputCounterT>;
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
      const ULevelT diff =
        static_cast<ULevelT>(static_cast<ULevelT>(upper_level[channel]) - static_cast<ULevelT>(lower_level[channel]));
      const IntArithmeticT range = static_cast<IntArithmeticT>(diff);
      return range > (::cuda::std::numeric_limits<IntArithmeticT>::max() / static_cast<IntArithmeticT>(num_bins));
    }
    else
    {
      return false;
    }
  }
};

// Occupancy-preserving cache-slot search: the SINGLE definition of the direct-atomic
// SMEM cache slot-count growth loop, shared by the live `dispatch<>` (below) and the
// host-callable `query_direct_atomic_cache_slots` probe so the two can never drift.
// Pick the largest power-of-two slot count in [cache_slots_floor, max_slots_by_smem]
// whose per-SM occupancy for `kernel_ptr` at `direct_atomic_threads_per_block` is no
// lower than the occupancy at the floor size (i.e. spend only FREE SMEM). `kernel_ptr`
// is the direct-atomic kernel that will actually run: its register footprint sets the
// occupancy, so the returned slot count is path-specific (e.g. RANGE's SearchTransform
// classify caps lower than EVEN's ScaleTransform). Callers pass the byte/floor/cap
// values they already derived from {counter width, channels, device}.
template <typename KernelPtr, typename KernelLauncherFactory>
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE int size_direct_atomic_cache_slots(
  KernelPtr kernel_ptr,
  KernelLauncherFactory launcher_factory,
  int num_active_channels,
  int cache_bytes_per_slot,
  int cache_slots_floor,
  int max_slots_by_smem,
  int direct_atomic_threads_per_block)
{
  auto cache_occupancy_for = [&](int slots) -> int {
    const int bytes = num_active_channels * slots * cache_bytes_per_slot;
    if (launcher_factory.set_max_dynamic_smem_size_for(kernel_ptr, bytes) != cudaSuccess)
    {
      (void) cudaGetLastError();
      return 0;
    }
    int occ = 0;
    // Query at the direct-atomic block size against the kernel that will run.
    if (launcher_factory.MaxSmOccupancy(occ, kernel_ptr, direct_atomic_threads_per_block, bytes) != cudaSuccess)
    {
      (void) cudaGetLastError();
      return 0;
    }
    return occ;
  };

  const int floor_occ = cache_occupancy_for(cache_slots_floor);
  int slots           = cache_slots_floor;
  // Grow while occupancy stays at the floor occupancy (free SMEM).
  for (int cand = cache_slots_floor << 1; cand <= max_slots_by_smem; cand <<= 1)
  {
    const int occ = cache_occupancy_for(cand);
    if (floor_occ > 0 && occ >= floor_occ)
    {
      slots = cand;
    }
    else
    {
      break; // growth would cost occupancy; stop.
    }
  }
  return slots;
}

// Host-callable probe returning the per-channel direct-atomic SMEM cache slot count
// the dispatch would choose for {SampleT, CounterT, LevelT, OffsetT, channels, EVEN/
// RANGE} on the current device. It rebuilds the SAME direct-atomic kernel pointer the
// live `dispatch<>` builds (same policy via `policy_selector_from_types` -- the
// selector `DeviceHistogram`'s benchmark/env path uses -- and the same PRIVATIZED_SMEM_BINS=0
// decode ops: ScaleTransform for EVEN, SearchTransform for RANGE) and runs the SAME
// `size_direct_atomic_cache_slots` loop, so the returned value equals what
// CUB_HISTO_DEBUG_SLOTS prints for a real launch. Used by the histogram benchmark's
// stale_resident input generator to size its working set to the ACTUAL cache. Returns
// the slot floor if the device queries fail.
template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          bool IsEven,
          typename SampleT,
          typename CounterT,
          typename LevelT,
          typename OffsetT,
          typename OutputCounterT  = CounterT,
          typename SampleIteratorT = SampleT*,
          typename KernelSource    = DeviceHistogramKernelSource<
               NUM_CHANNELS,
               NUM_ACTIVE_CHANNELS,
               SampleIteratorT,
               CounterT,
               LevelT,
               OffsetT,
               SampleT,
               OutputCounterT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN int
query_direct_atomic_cache_slots(KernelSource kernel_source = {}, KernelLauncherFactory launcher_factory = {})
{
  const int slots_floor = cache_tuning::slots_floor(NUM_ACTIVE_CHANNELS);

  ::cuda::compute_capability cc{};
  if (launcher_factory.PtxComputeCap(cc) != cudaSuccess)
  {
    (void) cudaGetLastError();
    return slots_floor;
  }
  using policy_selector_t = policy_selector_from_types<SampleT, CounterT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, IsEven>;
  const histogram_policy active_policy      = policy_selector_t{}(cc);
  const int direct_atomic_threads_per_block = active_policy.direct_atomic_threads();

  // Decode-op types for the direct-atomic (PRIVATIZED_SMEM_BINS==0, non-byte) path,
  // exactly as dispatch_even / dispatch_range set them.
  using TransformsT     = Transforms<LevelT, OffsetT, SampleT>;
  using OutputDecodeOpT = typename TransformsT::PassThruTransform;
  using PrivatizedDecodeOpT =
    ::cuda::std::conditional_t<IsEven,
                               typename TransformsT::ScaleTransform,
                               typename TransformsT::template SearchTransform<const LevelT*>>;
  constexpr int PRIVATIZED_SMEM_BINS = 0;
  // Representative direct-atomic kernel: the SINGLE-PROBE cache + output spill. This is
  // the kernel the selector actually runs on the high-bin adversarial cells the
  // stale_resident shape targets (direct_single_probe), and it shares the dynamic-SMEM
  // layout with the cuckoo variants. It is NOT interchangeable with the PLAIN 2-probe
  // cuckoo kernel for SIZING: the 2-probe cuckoo carries more registers, so on the
  // tightest occupancy tiers (e.g. multi-channel F64) it sizes the cache one power of two
  // LOWER than single-probe (1024 vs 2048), which would mis-size the working set. The
  // high-bin cuckoo path the dispatch runs is the SECOND-PROBE-GATED variant (register
  // footprint like single-probe), so single-probe is the faithful representative for both.
  auto kernel_ptr = kernel_source.template HistogramCacheSpillKernel<
    policy_selector_t,
    PRIVATIZED_SMEM_BINS,
    PrivatizedDecodeOpT,
    OutputDecodeOpT,
    single_probe_cache,
    output_atomic_spill>();

  int device_ordinal = 0;
  if (cudaGetDevice(&device_ordinal) != cudaSuccess)
  {
    (void) cudaGetLastError();
    device_ordinal = 0;
  }
  const int kCountReplicas       = cache_tuning::replicas(NUM_ACTIVE_CHANNELS);
  const int cache_bytes_per_slot = static_cast<int>(sizeof(int)) + kCountReplicas * static_cast<int>(sizeof(CounterT));
  int max_optin_smem             = 0;
  if (cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_ordinal) != cudaSuccess
      || max_optin_smem <= 0)
  {
    (void) cudaGetLastError();
    max_optin_smem = cache_tuning::smem_fallback_bytes;
  }
  const int cache_smem_budget =
    (max_optin_smem > cache_tuning::smem_reserve_bytes)
      ? (max_optin_smem - cache_tuning::smem_reserve_bytes)
      : max_optin_smem;
  const int max_slots_by_smem = cache_smem_budget / (NUM_ACTIVE_CHANNELS * cache_bytes_per_slot);

  const int queried = size_direct_atomic_cache_slots(
    kernel_ptr,
    launcher_factory,
    NUM_ACTIVE_CHANNELS,
    cache_bytes_per_slot,
    slots_floor,
    max_slots_by_smem,
    direct_atomic_threads_per_block);
#if _CCCL_HOSTED()
  // CUB_HISTO_DEBUG_QUERY_SLOTS: print the probe's chosen slot count and inputs (mirrors
  // CUB_HISTO_DEBUG_SLOTS for the live dispatch), so the benchmark's queried S can be
  // cross-checked against what a real launch sizes. Host-only; zero device SASS.
  NV_IF_TARGET(NV_IS_HOST, ({
                 if (::std::getenv("CUB_HISTO_DEBUG_QUERY_SLOTS"))
                 {
                   ::std::fprintf(
                     stderr,
                     "[CUB_HISTO_DEBUG_QUERY_SLOTS] IsEven=%d active_ch=%d offset_bytes=%d threads=%d "
                     "floor=%d max_by_smem=%d queried=%d\n",
                     static_cast<int>(IsEven),
                     NUM_ACTIVE_CHANNELS,
                     static_cast<int>(sizeof(OffsetT)),
                     direct_atomic_threads_per_block,
                     slots_floor,
                     max_slots_by_smem,
                     queried);
                 }
               }));
#endif // _CCCL_HOSTED()
  return queried;
}

// Extent-aware wrapper. The dispatch DOWN-CONVERTS its OffsetT to `int` when the
// input's byte extent (num_rows * row_stride_bytes) fits in `int` (see
// DeviceHistogram::MultiHistogram*), and the direct-atomic kernel's register footprint
// -- hence its occupancy and the sized slot count -- can differ between the int and
// wide-OffsetT instantiations (e.g. single-channel RANGE sizes to 4096 with `int` but
// 8192 with `int64_t`). So the benchmark MUST query the slot count for the SAME OffsetT
// the dispatch will actually launch at this input size. Pass the byte extent
// (sizeof(SampleT) * total_channels * elements) and the compile-time OffsetT; this
// selects the matching instantiation. Mirrors the facade's `< INT_MAX` rule exactly.
template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          bool IsEven,
          typename SampleT,
          typename CounterT,
          typename LevelT,
          typename OffsetT,
          typename OutputCounterT = CounterT>
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN int
query_direct_atomic_cache_slots_for_extent(unsigned long long byte_extent)
{
  if (sizeof(OffsetT) > sizeof(int) && byte_extent < static_cast<unsigned long long>(INT_MAX))
  {
    return query_direct_atomic_cache_slots<
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      IsEven,
      SampleT,
      CounterT,
      LevelT,
      int,
      OutputCounterT>();
  }
  return query_direct_atomic_cache_slots<
    NUM_CHANNELS,
    NUM_ACTIVE_CHANNELS,
    IsEven,
    SampleT,
    CounterT,
    LevelT,
    OffsetT,
    OutputCounterT>();
}

// This `dispatch<>` is the HOST-INIT histogram sweep dispatcher: it receives
// pre-built decode operators (constructed host-side by `dispatch_even` /
// `dispatch_range`, or by the C Parallel Library's tag-dispatch) and launches the
// init + privatized-SMEM-sweep (or, for PRIVATIZED_SMEM_BINS==0, the high-bin
// direct-atomic / GMEM-privatized kernels). The former device-init variant (which
// built decode ops inside the kernel from raw level arrays) has been removed, so
// there is no longer an `IsDeviceInit` / `IsEven` / `IsByteSample` switch here --
// those only fed the deleted device-init branch.
template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          int PRIVATIZED_SMEM_BINS,
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
  // How dispatch decides between the direct-atomic-to-output path and the
  // cooperative gather-merge / Init+Sweep path:
  //   0 (kHeuristic)   -- no explicit algorithm choice was made; apply the
  //                       `direct_atomic_bin_threshold` bin-count heuristic. Used by
  //                       the C-parallel host-init entry, which does not run
  //                       select_algorithm.
  //   1 (kForceDirect) -- the caller (select_algorithm / dispatch_by_algorithm /
  //                       the force hook) explicitly chose a direct-atomic algorithm;
  //                       run it UNCONDITIONALLY (no bin-count veto). This is what
  //                       makes a forced/selected direct_cuckoo actually run cuckoo
  //                       even below the heuristic threshold.
  //   2 (kForceGather) -- the caller explicitly chose the gather/hybrid path; never
  //                       take direct-atomic.
  // (Replaces the old ambiguous `disable_direct_atomic` bool, where cache_mode==0
  // could not distinguish "selector chose direct_cuckoo" from "no choice -> apply
  // heuristic", so an explicitly-chosen direct_cuckoo was silently vetoed below the
  // threshold and ran gather instead.)
  int direct_atomic_choice = 0,
  // Cache policy for the direct-atomic-to-output kernel (only consulted when the
  // direct-atomic path is taken). Both select the same DeviceHistogramCacheSpillKernel
  // with a different probe op:
  //   0 -> 2-hash cuckoo cache (cuckoo_cache_probe)
  //   1 -> single-probe direct-mapped cache (single_probe_cache)
  // The two share the same dynamic-SMEM cache layout and the dispatch-chosen
  // `cache_slots_per_channel`; only the probe op differs.
  int direct_atomic_cache_mode = 0)
{
  using LocalCounterT = local_counter_t<PolicySelector, CounterT>;

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

  const auto init_kernel = kernel_source.template HistogramInitKernel<PolicySelector>();

  // The privatized-SMEM histogram for the "256 < bins <= on-chip cap" tier lives
  // in extern __shared__ sized at launch, so its per-CTA SMEM footprint can
  // exceed the ptxas 48 KB static cap (16384 bins x 4 B = 64 KB). The dispatch
  // raises the kernel's cudaFuncAttributeMaxDynamicSharedMemorySize and passes
  // the byte budget as the third launch parameter. The 256-bin / byte tier and
  // the GMEM-privatized path (PRIVATIZED_SMEM_BINS == 0) use static SMEM.
  // `kDynamicSmemKernelTagBins` is the compile-time tag marking this dynamic tier
  // (the on-chip CAPACITY is a runtime byte budget, not this value).
  static constexpr bool kUseDynamicSmem = (PRIVATIZED_SMEM_BINS == kDynamicSmemKernelTagBins);

  // The dynamic-SMEM kernel merges each block's privatized histogram directly
  // into the global output via per-block atomicAdd (StoreOutput): at these bin
  // counts the cross-block contention is spread over enough distinct bins that a
  // direct merge beats a GMEM staging round-trip + cross-block gather.
  auto sweep_kernel = [&] {
    if constexpr (kUseDynamicSmem)
    {
      using output_decode_op_t     = typename FirstLevelArrayT::value_type;
      using privatized_decode_op_t = typename SecondLevelArrayT::value_type;
      return kernel_source.template HistogramSmemPrivatizedDynamicKernel<
        PolicySelector,
        PRIVATIZED_SMEM_BINS,
        privatized_decode_op_t,
        output_decode_op_t>();
    }
    else
    {
      // Host-init: decode ops are pre-built (by the host CUB dispatch_even/range,
      // or by the C Parallel Library's host-side tag dispatch) and passed by value.
      using output_decode_op_t     = typename FirstLevelArrayT::value_type;
      using privatized_decode_op_t = typename SecondLevelArrayT::value_type;
      return kernel_source.template HistogramSmemPrivatizedKernel<PolicySelector,
                                                                  PRIVATIZED_SMEM_BINS,
                                                                  privatized_decode_op_t,
                                                                  output_decode_op_t>();
    }
  }();

  // SMEM-priv sweep block size. The STATIC <=256 tier (this is the non-dynamic,
  // PRIVATIZED_SMEM_BINS>0 instantiation -- the dynamic tier is kUseDynamicSmem and the
  // GMEM fallback is PRIVATIZED_SMEM_BINS==0) may narrow the block via the policy's
  // static_smem_threads_per_block override. This MUST match the static kernel's
  // compile-time kSweepThreads / __launch_bounds__ (kernel_histogram.cuh): the SMEM-priv
  // sweep is tile-based (BlockLoad partitions a BLOCK_THREADS*ITEMS_PER_THREAD tile), so
  // the launch block dim has to equal the kernel's compile-time BLOCK_THREADS or the load
  // is wrong. The dynamic and GMEM paths keep the full threads_per_block.
  static constexpr bool kIsStaticSmemTier = (!kUseDynamicSmem && PRIVATIZED_SMEM_BINS > 0);
  const int threads_per_block =
    kIsStaticSmemTier ? active_policy.static_smem_threads() : active_policy.threads_per_block;
  // Match the kernel's compile-time kSweepItems for the static tier so pixels_per_tile
  // (grid sizing) agrees with the kernel's tile. Dynamic/GMEM keep pixels_per_thread.
  const int pixels_per_thread = kIsStaticSmemTier ? active_policy.static_smem_items() : active_policy.pixels_per_thread;
  // Block size for the high-bin direct-atomic (cuckoo / single-probe) kernels.
  // These atomic straight to the output via a pure grid-stride loop, so any
  // block size is correct; a policy may decouple it from the SMEM-priv sweep's
  // `threads_per_block` (0 => inherit). Used below ONLY for the direct-atomic
  // launch's block dims and occupancy/cache-sizing queries; the SMEM-priv sweep
  // grid sizing keeps using `threads_per_block`.
  const int direct_atomic_threads_per_block = active_policy.direct_atomic_threads();

  // Get SM count
  int sm_count;
  if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
  {
    return error;
  }

  // Dynamic-SMEM byte budget for the dynamic-SMEM kernel: the per-block
  // histogram lives in extern __shared__, sized as sum_ch num_privatized_bins[ch]
  // counters (per-channel contiguous). The launch reserves this many bytes and
  // the kernel's cudaFuncAttributeMaxDynamicSharedMemorySize is raised to match.
  int dyn_smem_bytes = 0;
  if constexpr (kUseDynamicSmem)
  {
    int total_bins = 0;
    for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ++ch)
    {
      total_bins += (num_privatized_levels[ch] - 1);
    }
    dyn_smem_bytes = total_bins * static_cast<int>(kernel_source.CounterSize());
  }

  // Get SM occupancy for sweep_kernel. For the dynamic-SMEM path, occupancy must be queried
  // with the dynamic-SMEM byte budget set so the driver accounts for the per-CTA SMEM footprint.
  int histogram_sweep_sm_occupancy;
  if constexpr (kUseDynamicSmem)
  {
    // Raise the kernel's max-dynamic-SMEM cap so the occupancy query accounts for the dyn-SMEM
    // CTA footprint. (The cap also has to be raised before the actual launch below.)
    if (const auto error = CubDebug(launcher_factory.set_max_dynamic_smem_size_for(sweep_kernel, dyn_smem_bytes)))
    {
      return error;
    }
    if (const auto error = CubDebug(launcher_factory.MaxSmOccupancy(
          histogram_sweep_sm_occupancy, sweep_kernel, threads_per_block, dyn_smem_bytes)))
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
  // The direct-atomic-vs-gather decision. When the caller made an EXPLICIT choice
  // (direct_atomic_choice != kHeuristic), honour it unconditionally -- this is the
  // whole point of selecting/forcing an algorithm. Only when no explicit choice was
  // made (kHeuristic, i.e. the C-parallel host-init entry that does not run
  // select_algorithm) do we fall back to the bin-count heuristic.
  //
  // Bin-count heuristic (kHeuristic only). Single-channel threshold is 1<<16
  // (65536): the cuckoo cache lives in dynamic SMEM and grows to use all free shared
  // memory, so the direct-atomic + per-block SMEM cache path is competitive with the
  // gather-merge persistent kernel down to 65536 bins and avoids the gather-merge's
  // O(num_blocks * num_bins) cross-block reduction.
  enum : int
  {
    kHeuristic   = 0,
    kForceDirect = 1,
    kForceGather = 2
  };
  constexpr int direct_atomic_bin_threshold_single = 1 << 16;
  constexpr int direct_atomic_bin_threshold_multi  = 16384;
  const int direct_atomic_bin_threshold =
    (NUM_ACTIVE_CHANNELS > 1) ? direct_atomic_bin_threshold_multi : direct_atomic_bin_threshold_single;
  const bool use_direct_atomic_to_output =
#if _CCCL_HOSTED()
    (PRIVATIZED_SMEM_BINS == 0
     && (direct_atomic_choice == kForceDirect
         || (direct_atomic_choice == kHeuristic && max_num_output_bins >= direct_atomic_bin_threshold)));
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
  ::cuda::std::array<LocalCounterT*, NUM_ACTIVE_CHANNELS> d_privatized_histograms_wrapper;
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_bins_wrapper;
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_bins_wrapper;

  auto* typed_allocations = reinterpret_cast<LocalCounterT**>(allocations);
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
  if constexpr (PRIVATIZED_SMEM_BINS == 0)
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
      // GmemPrivatized kernel, pure-gather member (smem_split=0): the NoCache /
      // split=0 member of the GmemPrivatized family (was GmemPrivGatherKernel).
      auto persistent_kernel_ptr = kernel_source.template HistogramGmemPrivatizedKernel<
        PolicySelector,
        PRIVATIZED_SMEM_BINS,
        privatized_decode_op_t,
        output_decode_op_t>();

      // Force device-side instantiation of the kernel template by referencing
      // it via a dead `<<<>>>` syntax. nvcc emits device code for kernels
      // whose templates are referenced by a chevron call, regardless of
      // whether the call is reachable at runtime. Without this, just taking
      // `&kernel` produces only the host shadow function, and the runtime's
      // kernel-registration table has no device-side entry to match it,
      // causing `cudaLaunchCooperativeKernel` to fail with
      // `cudaErrorInvalidResourceHandle`.
      //
      // Only the HOST-pointer (runtime) launcher needs this: a JIT/driver launcher
      // (C Parallel Library) supplies an already-emitted device entry through its
      // `CUkernel` handle and instantiates this dispatch with type-erased
      // `indirect_arg_t` decode ops that have no device-side decode methods, so the
      // dead reference must be compiled out there. `force_device_kernel_emission`
      // is the per-launcher switch (true for the runtime launcher, false for the
      // driver launcher). The `if constexpr` discards the body for the driver
      // launcher (it can't emit it); the inner `if (false)` keeps the reference
      // compiled-but-never-executed for the runtime launcher (executing it would
      // launch the kernel with dummy args).
      if constexpr (KernelLauncherFactory::force_device_kernel_emission)
      {
        if (false)
        {
          DeviceHistogramGmemPrivatizedKernel<
            PolicySelector,
            PRIVATIZED_SMEM_BINS,
            NUM_CHANNELS,
            NUM_ACTIVE_CHANNELS,
            SampleIteratorT,
            LocalCounterT,
            privatized_decode_op_t,
            output_decode_op_t,
            OffsetT,
            /*HybridSplit=*/false,
            CounterT><<<persistent_grid_dims, dim3{static_cast<unsigned int>(threads_per_block)}, 0, stream>>>(
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
      }

      // The direct-atomic kernel skips per-block privatization entirely
      // and writes atomically to the output histograms. Used only when
      // `use_direct_atomic_to_output` is true (see threshold above).
      auto direct_atomic_kernel_ptr = kernel_source.template HistogramCacheSpillKernel<
        PolicySelector,
        PRIVATIZED_SMEM_BINS,
        privatized_decode_op_t,
        output_decode_op_t,
        cuckoo_cache_probe<>,
        output_atomic_spill>();
      const void* direct_atomic_kernel_ptr_void = reinterpret_cast<const void*>(direct_atomic_kernel_ptr);

      // High-bin variant of the cuckoo kernel with the SECOND PROBE
      // compile-time-gated OFF (`DisableSecondProbe=true`). On the high-bin tier
      // (bins >> cache slots) the cache hit rate is near zero, so the cuckoo's
      // secondary slot rarely holds a useful key: it just reads a second SMEM key
      // -- the largest waste on this SMEM-bound kernel -- before spilling to GMEM
      // anyway, DOUBLING the SMEM key transactions per miss. This variant spills
      // on the first primary collision (one key-read + spill, like the
      // single-probe kernel) WHILE retaining the cuckoo kernel's count-replica
      // de-serialization (routing the cells to the single-probe kernel would lose
      // it). The 2-probe kernel above is kept for the moderate-bin tier where the
      // secondary slot raises the hit rate.
      auto direct_atomic_noprobe2_kernel_ptr = kernel_source.template HistogramCacheSpillKernel<
        PolicySelector,
        PRIVATIZED_SMEM_BINS,
        privatized_decode_op_t,
        output_decode_op_t,
        cuckoo_cache_probe</*DisableSecondProbe=*/true>,
        output_atomic_spill>();
      const void* direct_atomic_noprobe2_kernel_ptr_void =
        reinterpret_cast<const void*>(direct_atomic_noprobe2_kernel_ptr);

      // The single-probe direct-mapped variant of the direct-atomic kernel.
      // It shares the cuckoo kernel's signature (including the runtime
      // `cache_slots_per_channel` dynamic-SMEM cache), so the occupancy-
      // preserving cache sizing, the dynamic-SMEM cap, and the cooperative
      // launch args below are all common to both; only the leader's probe
      // policy differs. Selected when the caller requests
      // `direct_atomic_cache_mode == 1` (the huge-N single-channel high-bin
      // route picked by the unified selector as direct_atomic_single_probe).
      auto direct_atomic_single_probe_kernel_ptr = kernel_source.template HistogramCacheSpillKernel<
        PolicySelector,
        PRIVATIZED_SMEM_BINS,
        privatized_decode_op_t,
        output_decode_op_t,
        single_probe_cache,
        output_atomic_spill>();
      const void* direct_atomic_single_probe_kernel_ptr_void =
        reinterpret_cast<const void*>(direct_atomic_single_probe_kernel_ptr);

      // PRIVATIZED-SPILL variants (design proposal `gmem_privatized_{cuckoo,
      // single_probe}`): same cache front-end, but a cache MISS spills block-scope
      // (`private_block_spill`) into THIS block's private GMEM slab instead of
      // device-scope into the shared output; a Phase-4 atomic-free gather then
      // merges the slabs. The slab reuses the `d_privatized_histograms` temp
      // allocation already sized at `num_thread_blocks * num_bins`. Reached via
      // `direct_atomic_cache_mode` 3 (cuckoo) / 4 (single-probe), which
      // dispatch_by_algorithm sets for the first-class algorithm enumerators
      // `gmem_privatized_{cuckoo,single_probe}` (and the CUB_HISTO_FORCE_ALGO hook).
      // `select_algorithm` never returns them (measured to lose outside one
      // multi_even/powerlaw cell, unexploitable by a shape-blind selector), so they
      // are selectable-but-unselected -- a normal dispatch is byte-identical to
      // before. See cached_privatized_spill_design.md.
      auto direct_atomic_priv_cuckoo_kernel_ptr = kernel_source.template HistogramCacheSpillKernel<
        PolicySelector,
        PRIVATIZED_SMEM_BINS,
        privatized_decode_op_t,
        output_decode_op_t,
        cuckoo_cache_probe<>,
        private_block_spill>();
      auto direct_atomic_priv_single_probe_kernel_ptr = kernel_source.template HistogramCacheSpillKernel<
        PolicySelector,
        PRIVATIZED_SMEM_BINS,
        privatized_decode_op_t,
        output_decode_op_t,
        single_probe_cache,
        private_block_spill>();
      // No-cache combiner, output spill (algorithm::direct_nocache, cache_mode 2):
      // warp-coalesce then device-scope atomicAdd straight to the output, no SMEM
      // cache. Isolates the combiner's contribution.
      auto direct_atomic_no_cache_kernel_ptr = kernel_source.template HistogramCacheSpillKernel<
        PolicySelector,
        PRIVATIZED_SMEM_BINS,
        privatized_decode_op_t,
        output_decode_op_t,
        no_cache_probe,
        output_atomic_spill>();
      const void* direct_atomic_no_cache_kernel_ptr_void =
        reinterpret_cast<const void*>(direct_atomic_no_cache_kernel_ptr);
      const void* direct_atomic_priv_cuckoo_kernel_ptr_void =
        reinterpret_cast<const void*>(direct_atomic_priv_cuckoo_kernel_ptr);
      const void* direct_atomic_priv_single_probe_kernel_ptr_void =
        reinterpret_cast<const void*>(direct_atomic_priv_single_probe_kernel_ptr);

      // `direct_atomic_cache_mode` (set by dispatch_by_algorithm from the algorithm
      // enum) encodes BOTH the combiner and the spill policy:
      //   0 -> cuckoo,       output spill   (direct_cuckoo)
      //   1 -> single-probe, output spill   (direct_single_probe)
      //   2 -> no-cache,     output spill   (direct_nocache)
      //   3 -> cuckoo,       private spill  (gmem_privatized_cuckoo)
      //   4 -> single-probe, private spill  (gmem_privatized_single_probe)
      const bool use_private_spill      = (direct_atomic_cache_mode >= 3);
      const bool use_single_probe_cache = (direct_atomic_cache_mode == 1 || direct_atomic_cache_mode == 4);
      const bool use_no_cache           = (direct_atomic_cache_mode == 2);
      // When the CUCKOO kernel is selected with OUTPUT spill, drop its second probe
      // on the high-bin tier (bins >> any achievable cache slot count, where the
      // secondary slot can't raise the hit rate -- it just doubles the SMEM key
      // transactions per miss). The cache floor is 1024 (multi) / 4096 (single)
      // slots and only grows from there, so a bin count at/above
      // `kSecondProbeBinThreshold` is already >> the floor; the gate is decided up
      // front (no dependence on the final auto-sized slot count) and is a pure
      // pointer swap. Only applies to the output-spill cuckoo selection (mode 0).
      constexpr int kSecondProbeBinThreshold = 262144;
      const bool use_gated_cuckoo =
        (direct_atomic_cache_mode == 0) && (max_num_output_bins >= kSecondProbeBinThreshold);
      const void* active_direct_atomic_kernel_ptr_void =
        (direct_atomic_cache_mode == 4)   ? direct_atomic_priv_single_probe_kernel_ptr_void
        : (direct_atomic_cache_mode == 3) ? direct_atomic_priv_cuckoo_kernel_ptr_void
        : use_no_cache                    ? direct_atomic_no_cache_kernel_ptr_void
        : use_single_probe_cache          ? direct_atomic_single_probe_kernel_ptr_void
        : use_gated_cuckoo
          ? direct_atomic_noprobe2_kernel_ptr_void
          : direct_atomic_kernel_ptr_void;

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
      // GMEM-atomic spills on the high-bin path (the bottleneck there). The
      // floor is the legacy static size (4096 single-channel / 1024
      // multi-channel) so we never regress below the previous behaviour.
      //
      // The cache COUNT array is split into `kCountReplicas` warp-strided replicas
      // to de-serialize the cross-warp atomicAdd_block on hot slots (see the
      // kernel-side rationale). The per-slot footprint is therefore one shared int
      // key plus `kCountReplicas * CounterSize()` count bytes. The replica factor
      // and the slot floor come from `cache_tuning` so this host-side sizing and
      // the kernels' compile-time `kCountReplicas` are guaranteed to agree on one
      // definition; R stays compile-time (a runtime R regressed the
      // register-pinned cuckoo kernel).
      const int kCountReplicas = cache_tuning::replicas(NUM_ACTIVE_CHANNELS);
      const int cache_bytes_per_slot =
        static_cast<int>(sizeof(int)) + kCountReplicas * static_cast<int>(kernel_source.CounterSize());
      const int cache_slots_floor = cache_tuning::slots_floor(NUM_ACTIVE_CHANNELS);
      // Cap the per-CTA dynamic SMEM for the cache: query the device opt-in max
      // (B200/SM100 ~228 KiB) and stay under it, falling back if the query fails.
      int max_optin_smem = 0;
      if (cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_ordinal)
            != cudaSuccess
          || max_optin_smem <= 0)
      {
        (void) cudaGetLastError();
        max_optin_smem = cache_tuning::smem_fallback_bytes;
      }
      // Reserve some SMEM for static/driver shared use; the rest is for the cache.
      const int cache_smem_budget =
        (max_optin_smem > cache_tuning::smem_reserve_bytes)
          ? (max_optin_smem - cache_tuning::smem_reserve_bytes)
          : max_optin_smem;
      const int max_slots_by_smem = cache_smem_budget / (NUM_ACTIVE_CHANNELS * cache_bytes_per_slot);

      // Occupancy-preserving sizing. Simply "fitting the cooperative grid" lets
      // the cache grow until occupancy collapses to 1 block/SM, which slows the
      // latency-bound multi-channel paths. So we only spend SMEM that is FREE:
      // pick the largest power-of-two slot count whose per-SM occupancy is no
      // lower than the occupancy at the floor size. This grows the cache on
      // single-channel high-bin paths (where the extra slots are free) without
      // trading away occupancy on the multi-channel paths.
      //
      // The query / attribute target is the ACTIVE direct-atomic kernel
      // (cuckoo or single-probe per `direct_atomic_cache_mode`): they have the
      // same dynamic-SMEM layout but can differ in register usage, so the
      // free-SMEM occupancy budget must be measured against the kernel that
      // will actually run. `size_direct_atomic_cache_slots` (above) is the single
      // definition of the occupancy-growth loop, shared with the host-callable
      // `query_direct_atomic_cache_slots` probe so the two can never drift.
      auto size_cache_for = [&](auto kernel_ptr) -> int {
        return size_direct_atomic_cache_slots(
          kernel_ptr,
          launcher_factory,
          NUM_ACTIVE_CHANNELS,
          cache_bytes_per_slot,
          cache_slots_floor,
          max_slots_by_smem,
          direct_atomic_threads_per_block);
      };

      int cache_slots_per_channel =
        (direct_atomic_cache_mode == 4)   ? size_cache_for(direct_atomic_priv_single_probe_kernel_ptr)
        : (direct_atomic_cache_mode == 3) ? size_cache_for(direct_atomic_priv_cuckoo_kernel_ptr)
        : use_no_cache                    ? size_cache_for(direct_atomic_no_cache_kernel_ptr)
        : use_single_probe_cache          ? size_cache_for(direct_atomic_single_probe_kernel_ptr)
        : use_gated_cuckoo
          ? size_cache_for(direct_atomic_noprobe2_kernel_ptr)
          : size_cache_for(direct_atomic_kernel_ptr);
#  if _CCCL_HOSTED()
      // Tuning/debug env hooks (host only): (1) CUB_HISTO_FORCE_SLOTS overrides
      // the occupancy-sizer's per-channel slot count with a fixed power-of-two
      // value, clamped to [cache_slots_floor, max_slots_by_smem] so every forced
      // value remains a legal dynamic-SMEM reservation that still admits >=1
      // block/SM; (2) CUB_HISTO_DEBUG_SLOTS prints the chosen (auto or forced)
      // slot count and the sizing inputs to stderr. These let the cache slot
      // count be swept against a single build without recompiling.
      NV_IF_TARGET(NV_IS_HOST, ({
                     if (const char* env = ::std::getenv("CUB_HISTO_FORCE_SLOTS"))
                     {
                       const int forced = ::std::atoi(env);
                       // power-of-two check + clamp into the legal window.
                       if (forced > 0 && (forced & (forced - 1)) == 0)
                       {
                         int clamped = forced;
                         if (clamped < cache_slots_floor)
                         {
                           clamped = cache_slots_floor;
                         }
                         if (clamped > max_slots_by_smem)
                         {
                           // round down to the largest power-of-two <= max_slots_by_smem
                           int hi = cache_slots_floor;
                           while ((hi << 1) <= max_slots_by_smem)
                           {
                             hi <<= 1;
                           }
                           clamped = hi;
                         }
                         cache_slots_per_channel = clamped;
                       }
                     }
                     if (::std::getenv("CUB_HISTO_DEBUG_SLOTS"))
                     {
                       ::std::fprintf(
                         stderr,
                         "[CUB_HISTO_DEBUG_SLOTS] active_ch=%d single_probe=%d R=%d "
                         "auto_or_forced_slots=%d floor=%d max_by_smem=%d bytes/slot=%d\n",
                         NUM_ACTIVE_CHANNELS,
                         static_cast<int>(use_single_probe_cache),
                         kCountReplicas,
                         cache_slots_per_channel,
                         cache_slots_floor,
                         max_slots_by_smem,
                         cache_bytes_per_slot);
                     }
                   }));
#  endif // _CCCL_HOSTED()
      const int cuckoo_cache_smem_bytes = NUM_ACTIVE_CHANNELS * cache_slots_per_channel * cache_bytes_per_slot;
      // Make sure the ACTIVE kernel's dynamic-SMEM attribute matches the final
      // chosen size (the sizing loop above may have left a larger size set on a
      // rejected candidate). Set it on whichever kernel `direct_atomic_cache_mode`
      // selected (0-4). Use the typed kernel pointer (the launcher's
      // set_max_dynamic_smem_size_for takes the kernel handle type, which differs
      // between the CUDA-runtime and CUDA-driver launchers).
      auto set_active_smem = [&](auto kernel_ptr) {
        if (launcher_factory.set_max_dynamic_smem_size_for(kernel_ptr, cuckoo_cache_smem_bytes) != cudaSuccess)
        {
          (void) cudaGetLastError();
        }
      };
      switch (direct_atomic_cache_mode)
      {
        case 4:
          set_active_smem(direct_atomic_priv_single_probe_kernel_ptr);
          break;
        case 3:
          set_active_smem(direct_atomic_priv_cuckoo_kernel_ptr);
          break;
        case 2:
          set_active_smem(direct_atomic_no_cache_kernel_ptr);
          break;
        case 1:
          set_active_smem(direct_atomic_single_probe_kernel_ptr);
          break;
        default:
          if (use_gated_cuckoo)
          {
            set_active_smem(direct_atomic_noprobe2_kernel_ptr);
          }
          else
          {
            set_active_smem(direct_atomic_kernel_ptr);
          }
          break;
      }

      // See the gather-kernel emission note above: only the host-pointer (runtime)
      // launcher needs the dead `<<<>>>` reference to force a matching device
      // entry; the JIT/driver launcher compiles it out (its type-erased
      // `indirect_arg_t` decode ops have no device decode methods). The inner
      // `if (false)` keeps these compiled-but-never-executed for the runtime
      // launcher (executing them would launch with dummy args).
      if constexpr (KernelLauncherFactory::force_device_kernel_emission)
      {
        if (false)
        {
          DeviceHistogramCacheSpillKernel<
            PolicySelector,
            PRIVATIZED_SMEM_BINS,
            NUM_CHANNELS,
            NUM_ACTIVE_CHANNELS,
            SampleIteratorT,
            LocalCounterT,
            privatized_decode_op_t,
            output_decode_op_t,
            OffsetT,
            cuckoo_cache_probe<>,
            output_atomic_spill,
            CounterT>
            <<<persistent_grid_dims, dim3{static_cast<unsigned int>(threads_per_block)}, cuckoo_cache_smem_bytes, stream>>>(
              d_samples,
              num_output_bins_wrapper,
              d_output_histograms,
              second_level_array,
              num_row_pixels,
              num_rows,
              row_stride_samples,
              cache_slots_per_channel);
          DeviceHistogramCacheSpillKernel<
            PolicySelector,
            PRIVATIZED_SMEM_BINS,
            NUM_CHANNELS,
            NUM_ACTIVE_CHANNELS,
            SampleIteratorT,
            LocalCounterT,
            privatized_decode_op_t,
            output_decode_op_t,
            OffsetT,
            single_probe_cache,
            output_atomic_spill,
            CounterT>
            <<<persistent_grid_dims, dim3{static_cast<unsigned int>(threads_per_block)}, cuckoo_cache_smem_bytes, stream>>>(
              d_samples,
              num_output_bins_wrapper,
              d_output_histograms,
              second_level_array,
              num_row_pixels,
              num_rows,
              row_stride_samples,
              cache_slots_per_channel);
          // Force device-side emission of the second-probe-gated cuckoo variant so
          // `cudaLaunchCooperativeKernel` can resolve its device entry.
          DeviceHistogramCacheSpillKernel<
            PolicySelector,
            PRIVATIZED_SMEM_BINS,
            NUM_CHANNELS,
            NUM_ACTIVE_CHANNELS,
            SampleIteratorT,
            LocalCounterT,
            privatized_decode_op_t,
            output_decode_op_t,
            OffsetT,
            cuckoo_cache_probe</*DisableSecondProbe=*/true>,
            output_atomic_spill,
            CounterT>
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
      }

      int cooperative_supported = 0;
      const bool coop_query_ok =
        (cudaDeviceGetAttribute(&cooperative_supported, cudaDevAttrCooperativeLaunch, device_ordinal) == cudaSuccess
         && cooperative_supported != 0);
      // The persistent / direct-atomic kernels may have lower per-SM occupancy
      // than the sweep kernel that was used to size `num_thread_blocks`,
      // so we must verify the chosen kernel's occupancy fits the requested
      // cooperative grid; otherwise `cudaLaunchCooperativeKernel` will fail
      // with cudaErrorCooperativeLaunchTooLarge and we fall back to the legacy
      // two-kernel path (which is much slower for high-bin GMEM-priv configs).
      int persistent_sm_occupancy    = 0;
      int direct_atomic_sm_occupancy = 0;
      const auto persist_occ_err =
        launcher_factory.MaxSmOccupancy(persistent_sm_occupancy, persistent_kernel_ptr, threads_per_block);
      if (persist_occ_err != cudaSuccess)
      {
        (void) cudaGetLastError();
        persistent_sm_occupancy = 0;
      }
      // Query occupancy of the ACTUAL kernel that will run (mode 0-4), so the
      // free-SMEM grid sizing is measured against it.
      const auto direct_occ_err =
        (direct_atomic_cache_mode == 4)
          ? launcher_factory.MaxSmOccupancy(
              direct_atomic_sm_occupancy,
              direct_atomic_priv_single_probe_kernel_ptr,
              direct_atomic_threads_per_block,
              cuckoo_cache_smem_bytes)
        : (direct_atomic_cache_mode == 3)
          ? launcher_factory.MaxSmOccupancy(
              direct_atomic_sm_occupancy,
              direct_atomic_priv_cuckoo_kernel_ptr,
              direct_atomic_threads_per_block,
              cuckoo_cache_smem_bytes)
        : use_no_cache
          ? launcher_factory.MaxSmOccupancy(
              direct_atomic_sm_occupancy,
              direct_atomic_no_cache_kernel_ptr,
              direct_atomic_threads_per_block,
              cuckoo_cache_smem_bytes)
        : use_single_probe_cache
          ? launcher_factory.MaxSmOccupancy(
              direct_atomic_sm_occupancy,
              direct_atomic_single_probe_kernel_ptr,
              direct_atomic_threads_per_block,
              cuckoo_cache_smem_bytes)
        : use_gated_cuckoo
          ? launcher_factory.MaxSmOccupancy(
              direct_atomic_sm_occupancy,
              direct_atomic_noprobe2_kernel_ptr,
              direct_atomic_threads_per_block,
              cuckoo_cache_smem_bytes)
          : launcher_factory.MaxSmOccupancy(
              direct_atomic_sm_occupancy,
              direct_atomic_kernel_ptr,
              direct_atomic_threads_per_block,
              cuckoo_cache_smem_bytes);
      if (direct_occ_err != cudaSuccess)
      {
        (void) cudaGetLastError();
        direct_atomic_sm_occupancy = 0;
      }
      const int persistent_capacity    = persistent_sm_occupancy * sm_count;
      const int direct_atomic_capacity = direct_atomic_sm_occupancy * sm_count;
      const bool persistent_fits       = (persistent_sm_occupancy > 0) && (num_thread_blocks <= persistent_capacity);
      // The OUTPUT-spill direct-atomic kernel distributes work via a pure grid-stride
      // loop, so ANY co-resident block count is correct -- it does NOT need the grid to
      // reach `num_thread_blocks`; it only needs a positive co-resident capacity (the
      // grid is clamped to `direct_atomic_capacity` below). This matters for wide
      // counters: an 8-byte counter inflates the cache's per-CTA SMEM, lowering
      // occupancy so `direct_atomic_capacity < num_thread_blocks` -- which previously
      // failed `selected_fits` and fell through to the (now error-returning) high-bin
      // guard, so multi-channel 64-bit high-bin histograms could not launch at all. The
      // PRIVATE-spill and gather paths still need the strict fit: their per-block slabs
      // are sized for exactly `num_thread_blocks` and the kernel indexes them by
      // block_id, so the co-resident grid must actually be that large.
      const bool direct_atomic_needs_full_grid = use_private_spill;
      const bool direct_atomic_fits            = (direct_atomic_sm_occupancy > 0)
                                   && (!direct_atomic_needs_full_grid || num_thread_blocks <= direct_atomic_capacity);
      const bool selected_fits = use_direct_atomic_to_output ? direct_atomic_fits : persistent_fits;

      // Grid size for the direct-atomic kernels. Unlike the gather-merge
      // kernel, the direct-atomic cuckoo / single-probe kernels distribute work
      // via a pure grid-stride loop over `total_pixels` and use neither
      // `tile_queue` nor `tiles_per_row`, so ANY block count produces correct
      // counts -- more blocks simply means more resident warps. The shared
      // `num_thread_blocks` above is sized off the gather-merge kernel's per-SM
      // occupancy, which is lower (the gather-merge kernel is register-heavy).
      // The direct-atomic kernel admits more blocks/SM and is
      // bound by SMEM-atomic scoreboard latency, so grow the grid to the
      // direct-atomic kernel's OWN co-resident capacity so the extra warps hide
      // that latency, capped by the available work (no point launching blocks
      // that would process zero pixels) and never shrinking below the sweep
      // grid.
      dim3 direct_atomic_grid_dims = persistent_grid_dims;
      // Private-spill must NOT resize the grid: its per-block slab is sized for exactly
      // `num_thread_blocks` (= persistent_grid_dims), and the kernel indexes/gathers
      // slabs by `block_id < blocks_per_grid`, so a different grid would read/write past
      // the slab. The OUTPUT-spill variant has no such constraint (grid-stride loop,
      // any block count correct), so we set its grid to the direct-atomic kernel's OWN
      // co-resident capacity, clamped to the available work and floored at 1. This both
      // GROWS the grid past the (lower) gather-sized num_thread_blocks to hide
      // SMEM-atomic latency, and SHRINKS it to fit when a wide counter lowers the
      // cache kernel's occupancy below num_thread_blocks (the multi-channel 64-bit case)
      // -- a cooperative launch requires the grid to be co-resident, so it must not
      // exceed `direct_atomic_capacity`.
      if (use_direct_atomic_to_output && !use_private_spill && direct_atomic_capacity > 0)
      {
        // Upper bound on useful blocks for the grid-stride direct-atomic kernel:
        // one thread per pixel needs ceil(total_pixels / block_size) blocks;
        // beyond that, additional blocks would have no pixels to process. We use
        // the DIRECT-ATOMIC block size here (not the sweep's), so a smaller
        // direct-atomic block is not artificially capped below the sweep's
        // tile-granularity work bound.
        const long long total_pixels_ll = static_cast<long long>(num_row_pixels) * static_cast<long long>(num_rows);
        const long long work_tiles =
          (total_pixels_ll + direct_atomic_threads_per_block - 1) / direct_atomic_threads_per_block;
        long long target = static_cast<long long>(direct_atomic_capacity);
        if (target > work_tiles)
        {
          target = work_tiles; // no point launching blocks with no pixels
        }
        if (target < 1)
        {
          target = 1;
        }
        direct_atomic_grid_dims = dim3{static_cast<unsigned int>(target), 1u, 1u};
      }

#  if _CCCL_HOSTED()
      if (::std::getenv("CUB_HISTO_DEBUG_GRID"))
      {
        ::std::fprintf(
          stderr,
          "[CUB_HISTO_DEBUG_GRID] blocks=%u threads=%d total_threads=%lld sm_count=%d "
          "direct_occ=%d slots=%d\n",
          direct_atomic_grid_dims.x,
          direct_atomic_threads_per_block,
          static_cast<long long>(direct_atomic_grid_dims.x) * direct_atomic_threads_per_block,
          sm_count,
          direct_atomic_sm_occupancy,
          cache_slots_per_channel);
      }
#  endif

      if (coop_query_ok && selected_fits)
      {
        cudaError_t coop_status = cudaSuccess;
        if (use_direct_atomic_to_output)
        {
#  if CUB_HISTO_TRACK_HITRATE
          // Zero the grid-wide cache hit/miss accumulators before this (single)
          // cooperative launch so the read-back below reflects only this launch.
          // Instrumented build only (CUB_HISTO_TRACK_HITRATE); see kernel_histogram.cuh.
          {
            const unsigned long long zero = 0;
            (void) cudaMemcpyToSymbol(detail::histogram::g_cub_histo_cache_hits, &zero, sizeof(zero));
            (void) cudaMemcpyToSymbol(detail::histogram::g_cub_histo_cache_misses, &zero, sizeof(zero));
          }
#  endif
          // For the very-high-bin GMEM-privatized path, dispatch the direct-atomic
          // kernel instead of the gather-merge persistent kernel. It needs the
          // output histograms, the privatized decode op (in second_level_array),
          // input geometry, and cache slot count. The private-spill variant takes
          // one EXTRA trailing arg: the per-block private slab base
          // (`d_privatized_histograms_wrapper`); the output-spill variants ignore
          // it, but a cooperative launch marshals args positionally, so we pass the
          // extra arg only for the private-spill variant.
          // Route the cooperative launch through the launcher abstraction so the C
          // Parallel Library (CUDA-driver launcher + JIT CUkernels) can run these
          // kernels too -- regular CUB's runtime launcher reinterprets the host
          // function pointer; the driver launcher resolves the CUkernel handle.
          // `DeviceHistogramCacheSpillKernel` has 9 parameters; the 9th
          // (`d_private_histograms_wrapper`) is a C++ default arg used only by the
          // private-spill variant. cudaLaunchCooperativeKernel / cuLaunchCooperative
          // marshal args POSITIONALLY by the kernel's true parameter count and
          // ignore C++ defaults (same rule as the 16-param gather kernel below), so
          // we MUST physically pass all 9 for every variant. The output-spill
          // variants compile out the 9th (`kPrivateSpill == false`), so handing them
          // the valid privatized-slab base is harmless; passing only 8 reads an
          // out-of-bounds arg slot (the prior segfault on the high-bin output-spill
          // path -- it happened to "work" only when that stack slot held a readable
          // pointer).
          coop_status = launcher_factory.doit_cooperative(
            direct_atomic_grid_dims,
            dim3{static_cast<unsigned int>(direct_atomic_threads_per_block)},
            static_cast<unsigned int>(cuckoo_cache_smem_bytes),
            stream,
            active_direct_atomic_kernel_ptr_void,
            d_samples,
            num_output_bins_wrapper,
            d_output_histograms,
            second_level_array,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            cache_slots_per_channel,
            d_privatized_histograms_wrapper);
        }
        else
        {
          // Pure-gather instantiation of the unified GmemPrivatized kernel
          // (HybridSplit=false, smem_split=0). The kernel has 16 params; the 3
          // hybrid-only trailing ones (d_secondary, smem_split, secondary_size) are
          // unused here but MUST be physically present — cudaLaunchCooperativeKernel
          // marshals positionally and ignores C++ defaults.
          ::cuda::std::array<LocalCounterT*, NUM_ACTIVE_CHANNELS> gather_no_secondary{};
          int gather_zero_split     = 0;
          int gather_zero_secondary = 0;
          coop_status               = launcher_factory.doit_cooperative(
            persistent_grid_dims,
            dim3{static_cast<unsigned int>(threads_per_block)},
            /*sharedMem=*/0u,
            stream,
            reinterpret_cast<const void*>(persistent_kernel_ptr),
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
            max_num_output_bins,
            gather_no_secondary,
            gather_zero_split,
            gather_zero_secondary);
        }
        if (coop_status == cudaSuccess)
        {
          launched_persistent = true;
#  if _CCCL_HOSTED()
          // Sweep-only: emit the canonical name of the algorithm that ACTUALLY ran,
          // so the force sweep can drop cells where the requested algo was not the
          // one launched (e.g. a forced direct_cuckoo that fell back to gather below
          // the direct-atomic bin threshold). Host-only, env-gated -> zero device SASS
          // impact. This cooperative block ran either the direct-atomic kernel (by
          // cache_mode) or the pure-gather kernel.
          NV_IF_TARGET(NV_IS_HOST, ({
                         if (::std::getenv("CUB_HISTO_LOG_LAUNCH"))
                         {
                           const char* ran = "gmem_privatized_nocache"; // the gather branch
                           if (use_direct_atomic_to_output)
                           {
                             ran = (direct_atomic_cache_mode == 1) ? "direct_single_probe"
                                 : (direct_atomic_cache_mode == 2) ? "direct_nocache"
                                 : (direct_atomic_cache_mode == 3) ? "gmem_privatized_cuckoo"
                                 : (direct_atomic_cache_mode == 4)
                                   ? "gmem_privatized_single_probe"
                                   : "direct_cuckoo"; // mode 0
                           }
                           ::std::fprintf(
                             stderr, "[launch] bins=%d ch=%d ran=%s\n", max_num_output_bins, NUM_ACTIVE_CHANNELS, ran);
                         }
                       }));
#  endif
#  if CUB_HISTO_TRACK_HITRATE
          // Read back the grid-wide cache hit/miss totals for this launch (cached
          // kernels only; gather / no_cache report 0 hits). Sync first so the kernel
          // has finished accumulating. Emitted per launch under CUB_HISTO_LOG_HITRATE
          // so the hit-rate sweep can parse a rate per (bins, elements).
          if (use_direct_atomic_to_output && !use_no_cache && ::std::getenv("CUB_HISTO_LOG_HITRATE"))
          {
            (void) cudaStreamSynchronize(stream);
            unsigned long long h = 0, m = 0;
            (void) cudaMemcpyFromSymbol(&h, detail::histogram::g_cub_histo_cache_hits, sizeof(h));
            (void) cudaMemcpyFromSymbol(&m, detail::histogram::g_cub_histo_cache_misses, sizeof(m));
            const double rate      = (h + m) ? (static_cast<double>(h) / static_cast<double>(h + m)) : 0.0;
            const long long pixels = static_cast<long long>(num_row_pixels) * static_cast<long long>(num_rows);
            ::std::fprintf(
              stderr,
              "[hitrate] bins=%d ch=%d pixels=%lld hits=%llu misses=%llu rate=%.6f\n",
              max_num_output_bins,
              NUM_ACTIVE_CHANNELS,
              pixels,
              h,
              m,
              rate);
          }
#  endif
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

  // For the high-bin path (PRIVATIZED_SMEM_BINS == 0: GMEM-privatized gather and the
  // direct-atomic caches) the ONLY correct kernel is the cooperative one launched
  // above -- it relies on a grid-wide `grid.sync()` to order init -> sweep -> gather.
  // The non-cooperative fallback below launches `sweep_kernel`, which for
  // PRIVATIZED_SMEM_BINS == 0 is the STATIC <=256-bin privatized kernel -- running it
  // on a high-bin problem would neither gather the per-block slabs nor size the
  // histogram correctly. So if the cooperative launch did not happen (occupancy query
  // failed, or the grid did not fit the co-resident capacity -- `coop_query_ok` /
  // `selected_fits` false above), the requested high-bin algorithm genuinely cannot
  // run here: return cudaErrorNotSupported rather than silently launching the wrong
  // (smem-privatized) kernel. Previously this fell through and the launch tag
  // mislabeled the run as `ran=smem_privatized` (the sweep-only DROP(ran=smem_privatized)
  // anomaly), and for a FORCED algorithm it silently substituted a different one --
  // a forced request must error, never substitute. The smem-privatized tiers
  // (PRIVATIZED_SMEM_BINS > 0) legitimately use the non-cooperative path below.
  if constexpr (PRIVATIZED_SMEM_BINS == 0)
  {
    if (!launched_persistent)
    {
      return cudaErrorNotSupported;
    }
  }

  // Non-cooperative path: a standalone init kernel followed by the sweep kernel.
  // Taken when the cooperative GMEM-privatized gather-merge above did not launch
  // (every privatized-SMEM tier; the high-bin path errored out just above).
  if (!launched_persistent)
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

    // The dynamic-SMEM kernel's per-block histogram lives in extern __shared__,
    // so the launch passes the dyn-SMEM byte budget (its cap was already raised
    // in the occupancy-query branch above); the static-SMEM tiers pass 0. The
    // kernel merges each block's histogram into the output via per-block
    // atomicAdd, so no follow-on combine launch is needed.
    const int sweep_smem_bytes = kUseDynamicSmem ? dyn_smem_bytes : 0;
    if (const auto error = CubDebug(
          launcher_factory(sweep_grid_dims,
                           threads_per_block,
                           sweep_smem_bytes,
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
#if _CCCL_HOSTED()
    // Sweep-only launch tag (see the cooperative block above): the non-cooperative
    // path ran the privatized-SMEM sweep kernel. The `:static` / `:dynamic` suffix
    // records WHICH smem-privatized kernel instantiation ran (compile-time
    // kUseDynamicSmem), so the static-vs-dynamic comparison sweep can tell them apart
    // at the same bin count. Host-only, env-gated.
    NV_IF_TARGET(NV_IS_HOST, ({
                   if (::std::getenv("CUB_HISTO_LOG_LAUNCH"))
                   {
                     ::std::fprintf(stderr,
                                    "[launch] bins=%d ch=%d ran=smem_privatized:%s\n",
                                    max_num_output_bins,
                                    NUM_ACTIVE_CHANNELS,
                                    kUseDynamicSmem ? "dynamic" : "static");
                   }
                 }));
#endif
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
 * HOST-INIT HistogramEven dispatch for the C Parallel Library (no device-init).
 *
 * Takes PRE-BUILT decode operators (constructed host-side by the caller --
 * C-parallel builds the real `Transforms<LevelT,OffsetT,SampleT>::ScaleTransform`
 * via a runtime type-tag dispatch and hands their bytes through
 * `OutputDecodeOpArrayT` / `PrivatizedDecodeOpArrayT` holders). It then routes
 * through the host-init `dispatch<>` / `DeviceHistogramSmemPrivatizedKernel`, which
 * is why C-parallel needs no device-side decode-op initialization kernel.
 *
 * The decode-op holders only need a working `operator&` (yielding the bytes the
 * kernel reads as its grid-constant decode-op argument); their static `value_type`
 * is unused on the host (C-parallel names the JIT kernel + decode-op types itself).
 */
template <
  int NUM_CHANNELS,
  int NUM_ACTIVE_CHANNELS,
  typename SampleIteratorT,
  typename CounterT,
  typename LevelT,
  typename OffsetT,
  typename PolicySelector,
  typename OutputDecodeOpArrayT,
  typename PrivatizedDecodeOpArrayT,
  typename SampleT = it_value_t<SampleIteratorT>,
  typename KernelSource =
    DeviceHistogramKernelSource<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, LevelT, OffsetT, SampleT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t __dispatch_even_host_init(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
  // Pre-built decode-op holders (host-constructed). For non-byte EVEN:
  // output = PassThruTransform array, privatized = ScaleTransform array.
  OutputDecodeOpArrayT output_decode_op,
  PrivatizedDecodeOpArrayT privatized_decode_op,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_samples,
  cudaStream_t stream,
  bool is_byte_sample                    = false,
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  int max_levels = num_output_levels[0];
  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    if (num_output_levels[channel] > max_levels)
    {
      max_levels = num_output_levels[channel];
    }
  }
  const int max_num_output_bins = max_levels - 1;

  // num_privatized_levels: for BYTE samples the privatized op is a 256-entry
  // pass-thru staging histogram (257 levels), reduced to the output bins by the
  // ScaleTransform OUTPUT op. For non-byte EVEN the privatized op IS the
  // ScaleTransform (privatized bins == output bins), so privatized == output.
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_levels = num_output_levels;
  if (is_byte_sample)
  {
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
      num_privatized_levels[channel] = 257;
    }
  }

  const auto run = [&](auto privatized_smem_bins) {
    constexpr int PRIVATIZED_SMEM_BINS = decltype(privatized_smem_bins)::value;
    return CubDebug((detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS>(
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
      launcher_factory)));
  };

  if (max_num_output_bins > detail::histogram::max_privatized_smem_bins)
  {
    return run(::cuda::std::integral_constant<int, 0>{});
  }
  return run(::cuda::std::integral_constant<int, detail::histogram::max_privatized_smem_bins>{});
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
// path covers the same bin range and is the known-correct fallback).
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
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_gmem_privatized_hybrid(
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
  constexpr int kPrivatizedSmemBins = kDynamicSmemKernelTagBins;
  using LocalCounterT               = local_counter_t<PolicySelector, CounterT>;

  // The hybrid kernel stages a PRIMARY range of low bins in dyn-SMEM and the secondary
  // tail in per-block GMEM. The primary size (`hybrid_split_bin`) is byte-derived for
  // the actual counter width, NOT the frozen template `kSplitBin` (which assumed a
  // 4-byte counter -- 49152 bins -> 192 KiB -- and overflowed the per-CTA SMEM cap at
  // an 8-byte counter, crashing the launch). `kSplitBin` is now only an UPPER BOUND on
  // the primary; the launch SMEM is `hybrid_split_bin * sizeof(LocalCounterT) * channels`,
  // which fits the opt-in cap by construction. Clamp also to max_num_output_bins - 1 so
  // the GMEM secondary tail is non-empty (the hybrid requires both regions).
  int hybrid_split_bin =
    hybrid_smem_split_bins(int(sizeof(LocalCounterT)), NUM_ACTIVE_CHANNELS, query_device_optin_smem_bytes());
  if (hybrid_split_bin > kSplitBin)
  {
    hybrid_split_bin = kSplitBin; // template upper bound (the tuned/measured primary size)
  }
  if (hybrid_split_bin > max_num_output_bins - 1)
  {
    hybrid_split_bin = max_num_output_bins - 1; // leave a non-empty secondary tail
  }
  if (hybrid_split_bin <= 0 || max_num_output_bins <= hybrid_split_bin)
  {
    // No usable split (the whole histogram would fit the primary, or the budget is too
    // small for even one bin). The caller's selector gates hybrid on a worthwhile split,
    // so this is the defensive fallback: decline so the caller uses the direct path.
    return cudaErrorNotSupported;
  }

  const int hybrid_secondary_size = max_num_output_bins - hybrid_split_bin;

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

  // dyn-SMEM bytes per block: per-channel hybrid_split_bin counters (byte-derived).
  const int dyn_smem_bytes_for_staging = int(sizeof(LocalCounterT)) * hybrid_split_bin * NUM_ACTIVE_CHANNELS;

  // Calculate occupancy and grid size.
  int fused_sm_occupancy       = 0;
  auto fused_hybrid_kernel_ptr = kernel_source.template HistogramGmemPrivatizedHybridKernel<
    PolicySelector,
    kPrivatizedSmemBins,
    InnerPrivatizedDecodeOpT,
    OutputDecodeOpT>();

  // Force device-side instantiation of the hybrid kernel template via a dead `<<<>>>` call.
  // Without this, just taking `&kernel` produces only the host shadow function and the runtime
  // kernel-registration table has no device-side entry to match it (cooperative launch then
  // fails with cudaErrorInvalidResourceHandle).
  if (false)
  {
    // Dead instantiation of the unified GmemPrivatized kernel, HybridSplit=true.
    // Unified arg order: (samples, num_output_bins[=split-sized], num_privatized_bins,
    // d_output, d_privatized[=primary], output_decode, priv_decode, geometry...,
    // tiles_per_row, tile_queue, max_num_output_bins, d_secondary, smem_split,
    // secondary_size).
    DeviceHistogramGmemPrivatizedKernel<
      PolicySelector,
      kPrivatizedSmemBins,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      LocalCounterT,
      InnerPrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT,
      /*HybridSplit=*/true,
      CounterT><<<1, 1, 0, stream>>>(
      d_samples,
      ::cuda::std::array<int, NUM_ACTIVE_CHANNELS>{},
      ::cuda::std::array<int, NUM_ACTIVE_CHANNELS>{},
      ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS>{},
      ::cuda::std::array<LocalCounterT*, NUM_ACTIVE_CHANNELS>{},
      ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS>{},
      ::cuda::std::array<InnerPrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS>{},
      num_row_pixels,
      num_rows,
      row_stride_samples,
      int{},
      GridQueue<int>{nullptr},
      int{},
      ::cuda::std::array<LocalCounterT*, NUM_ACTIVE_CHANNELS>{},
      int{},
      int{});
  }

  if (const auto error =
        launcher_factory.set_max_dynamic_smem_size_for(fused_hybrid_kernel_ptr, dyn_smem_bytes_for_staging))
  {
    (void) error;
    return cudaErrorNotSupported;
  }

  if (const auto error = CubDebug(launcher_factory.MaxSmOccupancy(
        fused_sm_occupancy, fused_hybrid_kernel_ptr, threads_per_block, dyn_smem_bytes_for_staging)))
  {
    (void) error;
    return cudaErrorNotSupported;
  }

  if (fused_sm_occupancy <= 0)
  {
    return cudaErrorNotSupported;
  }

  // Calculate launch geometry: pixels per block and tile counts. total_pixels and the
  // tile counts MUST be 64-bit: num_row_pixels is an OffsetT (which is 64-bit for large
  // inputs), and at >= 2^31 elements a 32-bit `total_pixels` overflows to a negative
  // value, making tiles_per_row negative and num_thread_blocks <= 0 -> a malformed
  // cooperative launch ("operation not supported"). num_thread_blocks itself is bounded
  // (by occupancy and an explicit INT_MAX/2 clamp), so it stays int, but it is derived
  // from 64-bit intermediates.
  const int pixels_per_block    = threads_per_block * pixels_per_thread;
  const long long total_pixels  = static_cast<long long>(num_row_pixels);
  const long long tiles_per_row = (total_pixels + pixels_per_block - 1) / pixels_per_block;

  // Number of blocks: max grid that fits both occupancy and tiles_per_row * num_rows
  // (matches the persistent-grid sizing in the existing fused kernel).
  const int max_blocks_per_grid_by_occupancy = sm_count * fused_sm_occupancy;
  const long long total_tiles                = static_cast<long long>(num_rows) * tiles_per_row;
  const int max_blocks_for_work =
    static_cast<int>(::cuda::std::min<long long>(total_tiles, ::cuda::std::numeric_limits<int>::max() / 2));
  const int num_thread_blocks = ::cuda::std::min(max_blocks_per_grid_by_occupancy, max_blocks_for_work);

  if (num_thread_blocks <= 0)
  {
    return cudaErrorNotSupported;
  }

  // Allocate per-block staging slabs:
  //   primary slab:   num_thread_blocks * NUM_ACTIVE_CHANNELS * hybrid_split_bin * sizeof(LocalCounterT)
  //   secondary slab: num_thread_blocks * NUM_ACTIVE_CHANNELS * hybrid_secondary_size * sizeof(LocalCounterT)
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

  ::cuda::std::array<LocalCounterT*, NUM_ACTIVE_CHANNELS> d_primary_staging_array{};
  ::cuda::std::array<LocalCounterT*, NUM_ACTIVE_CHANNELS> d_secondary_staging_array{};
  for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ++ch)
  {
    d_primary_staging_array[ch]   = static_cast<LocalCounterT*>(allocations[ch]);
    d_secondary_staging_array[ch] = static_cast<LocalCounterT*>(allocations[ch + NUM_ACTIVE_CHANNELS]);
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

  // Args for the unified GmemPrivatized kernel, HybridSplit=true. Unified arg
  // order (16 args; cudaLaunchCooperativeKernel marshals positionally and ignores
  // C++ defaults, so all must be present):
  //   samples, num_output_bins[=split-sized num_smem_bins], num_privatized_bins
  //   [unused by hybrid; pass num_smem too], d_output, d_privatized[=PRIMARY
  //   staging], output_decode, priv_decode, num_row_pixels, num_rows,
  //   row_stride_samples, tiles_per_row, tile_queue, max_num_output_bins[unused],
  //   d_secondary[=SECONDARY staging], smem_split, secondary_size.
  const int hybrid_max_num_output_bins = hybrid_split_bin + hybrid_secondary_size;
  cudaError_t launch_error             = launcher_factory.doit_cooperative(
    grid_dims,
    block_dims,
    static_cast<unsigned int>(dyn_smem_bytes_for_staging),
    stream,
    reinterpret_cast<const void*>(fused_hybrid_kernel_ptr),
    d_samples,
    num_smem_bins_wrapper,
    num_gmem_bins_wrapper,
    d_output_histograms,
    d_primary_staging_array,
    output_decode_op,
    inner_privatized_decode_op,
    num_row_pixels,
    num_rows,
    row_stride_samples,
    static_cast<int>(tiles_per_row), // kernel param is int; value is bounded (<< INT_MAX)
    tile_queue,
    hybrid_max_num_output_bins,
    d_secondary_staging_array,
    hybrid_split_bin,
    hybrid_secondary_size);

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

#  if _CCCL_HOSTED()
  // Sweep-only launch tag: the hybrid (smem_split>0) member of the GmemPrivatized
  // kernel ran. It serves the `gmem_privatized_nocache` enum, so the sweep validator
  // (which forces by enum name) accepts `gmem_privatized_nocache`; the `:hybrid`
  // suffix records the member for diagnostics. Host-only, env-gated.
  NV_IF_TARGET(NV_IS_HOST, ({
                 if (::std::getenv("CUB_HISTO_LOG_LAUNCH"))
                 {
                   ::std::fprintf(stderr,
                                  "[launch] bins=%d ch=%d ran=gmem_privatized_nocache:hybrid\n",
                                  max_num_output_bins,
                                  NUM_ACTIVE_CHANNELS);
                 }
               }));
#  endif
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
// `dispatch_gmem_privatized_hybrid`), this helper falls through to
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
  using LocalCounterT = local_counter_t<PolicySelector, CounterT>;

  // Experimental sweep hook: CUB_HISTO_FORCE_ALGO overrides `select_algorithm`'s
  // pick so EVERY algorithm can be measured at EVERY cell (apples-to-apples per
  // cell), not just the one the selector happens to choose. Accepted values:
  //   hybrid            -> hybrid_single_pass (single-channel only)
  //   direct_cuckoo     -> direct-atomic cuckoo cache, output spill
  //   direct_single_probe -> direct-atomic single-probe cache, output spill
  //   gmem_priv_gather  -> per-block GMEM-privatized + gather (no cache)
  //   priv_cuckoo       -> cuckoo cache + private_block_spill (PROPOSAL)
  //   priv_single_probe -> single-probe cache + private_block_spill (PROPOSAL)
  // The priv_* and direct_* values route to the PRIVATIZED_SMEM_BINS==0 case; the
  // deeper `dispatch<>` hook reads the same env var to pick the exact kernel
  // (cache variant + spill policy). Off by default -> normal dispatch unchanged.
  // Some forced (algo, cell) pairs may be illegal (e.g. hybrid needs the bin
  // count to fit its SMEM split, direct-atomic needs a cooperative launch that
  // fits); those fall through to their built-in fallback, so a forced run that
  // silently lands on a different kernel should be cross-checked with
  // CUB_HISTO_DEBUG_SLOTS (nonzero => a direct-atomic kernel actually ran).
  // See cached_privatized_spill_design.md.
#if _CCCL_HOSTED()
  NV_IF_TARGET(NV_IS_HOST, ({
                 // Honor a forced algorithm UNCONDITIONALLY, at any bin count. If a
                 // request is forced it must run -- a force that is silently ignored is
                 // a bug (the caller measures a different algorithm than it asked for).
                 //
                 // This previously gated forcing to bins above the on-chip cap, on the
                 // theory that forcing a high-bin (cooperative direct-atomic /
                 // gmem-privatized) kernel at a tiny bin count drove a "degenerate
                 // cooperative launch that can crash". That is not true on this dispatch:
                 // every high-bin algo (gmem_privatized_nocache, direct_{nocache,cuckoo,
                 // single_probe}, gmem_privatized_{cuckoo,single_probe}) forced at 256 /
                 // 1024 / 16384 bins runs the requested kernel, returns success, and
                 // passes the benchmark's built-in correctness check. The cooperative
                 // launch already has a safe non-cooperative fallback if it cannot be set
                 // up, so there is nothing to guard against. (A genuinely unsupported
                 // (algo, cell) pair -- e.g. multi-channel hybrid -- returns
                 // cudaErrorNotSupported from its dispatch helper so the cell is dropped,
                 // which is the correct way to decline a force: error, never substitute.)
                 if (const char* env = ::std::getenv("CUB_HISTO_FORCE_ALGO"))
                 {
                   // Force any algorithm at any cell (apples-to-apples sweeps). Names
                   // match the algorithm enum; the legacy aliases (hybrid,
                   // gmem_priv_gather, priv_*) are kept so older sweep scripts work.
                   if (::std::strcmp(env, "hybrid") == 0 || ::std::strcmp(env, "gmem_priv_gather") == 0
                       || ::std::strcmp(env, "gmem_privatized_nocache") == 0)
                   {
                     algo = algorithm::gmem_privatized_nocache;
                   }
                   else if (::std::strcmp(env, "direct_nocache") == 0)
                   {
                     algo = algorithm::direct_nocache;
                   }
                   else if (::std::strcmp(env, "direct_cuckoo") == 0)
                   {
                     algo = algorithm::direct_cuckoo;
                   }
                   else if (::std::strcmp(env, "direct_single_probe") == 0)
                   {
                     algo = algorithm::direct_single_probe;
                   }
                 }
               }));
#endif // _CCCL_HOSTED()

  // Sweep-only: the gmem_privatized_nocache enum covers BOTH the hybrid (smem_split>0)
  // and pure-gather (smem_split==0) members; the force hook lets a sweep pin one
  // member so they can be measured as distinct algorithms. `hybrid` -> hybrid member
  // only (no gather fallback, so a failed setup is dropped, not silently mislabeled);
  // `gmem_privatized_nocache` / `gmem_priv_gather` -> pure-gather member only. With no
  // force (normal dispatch) both stay false -> the default try-hybrid-then-gather.
  bool force_hybrid_member = false;
  bool force_gather_member = false;
  // Sweep-only: the smem_privatized enum runs the STATIC (<=256-bin, compile-time
  // sized) kernel at <=256 bins and the DYNAMIC (extern __shared__, launch-sized)
  // kernel above. CUB_HISTO_FORCE_SMEM={static,dynamic} pins which kernel runs so a
  // sweep can measure static vs dynamic at the SAME (low) bin count -- used to decide
  // whether the dynamic kernel can replace the static one. 0 = no override (bin-count
  // rule stands). Host-only; zero device SASS.
  int force_smem_kind = 0; // 0 = auto, 1 = force static, 2 = force dynamic
#if _CCCL_HOSTED()
  NV_IF_TARGET(
    NV_IS_HOST, ({
      if (const char* env = ::std::getenv("CUB_HISTO_FORCE_ALGO"))
      {
        force_hybrid_member = (::std::strcmp(env, "hybrid") == 0);
        force_gather_member =
          (::std::strcmp(env, "gmem_privatized_nocache") == 0 || ::std::strcmp(env, "gmem_priv_gather") == 0);
      }
      if (const char* env = ::std::getenv("CUB_HISTO_FORCE_SMEM"))
      {
        force_smem_kind = (::std::strcmp(env, "static") == 0) ? 1 : (::std::strcmp(env, "dynamic") == 0) ? 2 : 0;
      }
    }));
#endif // _CCCL_HOSTED()

  switch (algo)
  {
    case algorithm::smem_privatized: {
      // Whole-histogram-on-chip privatized SMEM. Recover the static (compile-time
      // sized, bins <= 256) vs dynamic (extern __shared__, sized at launch) tier from
      // the bin count — the two were separate enumerators before the merge.
      // PRIVATIZED_SMEM_BINS is the compile-time marker selecting the path in dispatch<>;
      // the dynamic path's actual bin count comes from the runtime level arrays.
      // CUB_HISTO_FORCE_SMEM (force_smem_kind) overrides the rule below for the
      // static-vs-dynamic comparison sweep: force_static (1) keeps <=256 on the static
      // kernel; force_dynamic (2) routes EVEN <=256-bin cells through the dynamic kernel
      // (which sizes its extern __shared__ from the runtime bin count, so a small bin
      // count just allocates a small dyn-SMEM region). Default (0) uses the rule.
      //
      // The default rule is counter-width-aware. The static <=256-bin kernel carries a
      // compile-time `CounterT[NUM_ACTIVE_CHANNELS][256+1]` __shared__ array sized for the
      // FULL 256 bins regardless of the actual bin count. At a 4-byte counter that is the
      // faster path (measured ~6% over dynamic on the low-bin tier: no launch-time
      // dynamic-SMEM setup, compile-time bin-loop addressing). At a wider counter it is
      // pathological: the fixed 8-byte array measures 2-8x SLOWER than the dynamic kernel
      // across the whole <=256 tier (run_2026-06-13_u64: dyn/static geomean 7.6x even,
      // 5.4x multi_even, 2.3x range, 1.7x multi_range; the dynamic kernel sizes its extern
      // __shared__ to the runtime bin count instead). So only take the static kernel when
      // the counter is 4 bytes; wider counters route <=256 through the dynamic kernel,
      // which is both correct and the measured-faster path there.
      const bool counter_prefers_static = (sizeof(LocalCounterT) <= 4);
      const bool use_static_smem =
        (force_smem_kind == 2) ? false
        : (force_smem_kind == 1)
          ? true
          : (counter_prefers_static && max_num_output_bins <= max_privatized_smem_bins);
      if (use_static_smem)
      {
        constexpr int PRIVATIZED_SMEM_BINS = max_privatized_smem_bins;
        return dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS>(
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
      constexpr int PRIVATIZED_SMEM_BINS = kDynamicSmemKernelTagBins;
      return dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS>(
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
    case algorithm::gmem_privatized_nocache: {
      // GmemPrivatized<NoCache>. smem_split>0 (hybrid) is the single-channel
      // SMEM-primary + GMEM-tail staging path; smem_split==0 (pure gather) is the
      // whole-histogram-in-GMEM fallback. Default (no force): try the hybrid member
      // for single-channel, fall through to pure-gather on setup failure.
      // Force `gmem_privatized_nocache`/`gmem_priv_gather` -> skip hybrid (pure gather
      // only). Force `hybrid` -> hybrid only, NO gather fallback (so a failed setup is
      // reported and the sweep drops the cell rather than recording gather under the
      // hybrid label).
      if constexpr (NUM_ACTIVE_CHANNELS == 1)
      {
        if (!force_gather_member)
        {
          const auto status = dispatch_gmem_privatized_hybrid<NUM_CHANNELS,
                                                              NUM_ACTIVE_CHANNELS,
                                                              hybrid_smem_split_bin_single_channel,
                                                              hybrid_smem_bins_max_single_channel>(
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
          if (status == cudaSuccess || status != cudaErrorNotSupported || force_hybrid_member)
          {
            return status; // success, a hard error, or hybrid was force-pinned (no fallback)
          }
          // hybrid setup unsupported here; fall through to the pure-gather member.
        }
      }
      else if (force_hybrid_member)
      {
        // Hybrid is single-channel-only; a multi-channel force-hybrid request cannot
        // run -> report unsupported so the sweep drops it (never silently gather).
        return cudaErrorNotSupported;
      }
      // Pure-gather member: kForceGather routes dispatch<> to the GmemPrivatized
      // gather kernel (HybridSplit=false), never direct-atomic.
      constexpr int PRIVATIZED_SMEM_BINS = 0;
      return dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS>(
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
        /*direct_atomic_choice=*/2 /*kForceGather*/,
        /*direct_atomic_cache_mode=*/0);
    }
    case algorithm::direct_nocache:
    case algorithm::direct_cuckoo:
    case algorithm::direct_single_probe: {
      // CacheSpillKernel<Combiner> family, PRIVATIZED_SMEM_BINS=0. The deeper dispatch<>
      // picks the kernel from `direct_atomic_cache_mode`, which encodes the combiner
      // (cuckoo / single-probe / no-cache); the spill is always device-scope to the
      // shared output (the private-spill modes 3/4 of the removed gmem_privatized_*
      // members are gone):
      //   0 -> cuckoo,       output spill   (direct_cuckoo)
      //   1 -> single-probe, output spill   (direct_single_probe)
      //   2 -> no-cache,     output spill   (direct_nocache)
      // kForceDirect: these are explicit direct-atomic choices, so dispatch<> runs
      // the direct-atomic kernel UNCONDITIONALLY (no bin-count veto) -- a forced or
      // selected direct_cuckoo therefore actually runs cuckoo at any high-bin count.
      constexpr int PRIVATIZED_SMEM_BINS = 0;
      const int direct_atomic_cache_mode =
        (algo == algorithm::direct_cuckoo) ? 0
        : (algo == algorithm::direct_single_probe)
          ? 1
          : /* algorithm::direct_nocache */ 2;
      return dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS>(
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
        /*direct_atomic_choice=*/1 /*kForceDirect*/,
        direct_atomic_cache_mode);
    }
  }
  return cudaErrorInvalidValue; // unreachable
}

template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          typename SampleIteratorT,
          typename CounterT,
          typename LevelT,
          typename OffsetT,
          bool IsByteSample,
          typename PolicySelector,
          typename SampleT      = it_value_t<SampleIteratorT>, /// The sample value type of the input iterator
          typename KernelSource = DeviceHistogramKernelSource<
            NUM_CHANNELS,
            NUM_ACTIVE_CHANNELS,
            SampleIteratorT,
            local_counter_t<PolicySelector, CounterT>,
            LevelT,
            OffsetT,
            SampleT,
            CounterT>,
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
  using LocalCounterT = local_counter_t<PolicySelector, CounterT>;

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

    if (const auto error =
          CubDebug((detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS>(
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
    features.num_pixels          = static_cast<long long>(num_row_pixels) * static_cast<long long>(num_rows);
    features.on_chip_bin_cap     = resolve_on_chip_bin_cap(int{sizeof(LocalCounterT)}, NUM_ACTIVE_CHANNELS);
    features.hybrid_split_bins   = resolve_hybrid_split_bins(int{sizeof(LocalCounterT)}, NUM_ACTIVE_CHANNELS);
    const algorithm algo         = select_algorithm<false>(features);

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

template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          typename SampleIteratorT,
          typename CounterT,
          typename LevelT,
          typename OffsetT,
          bool IsByteSample,
          typename PolicySelector,
          typename SampleT      = it_value_t<SampleIteratorT>, /// The sample value type of the input iterator
          typename KernelSource = DeviceHistogramKernelSource<
            NUM_CHANNELS,
            NUM_ACTIVE_CHANNELS,
            SampleIteratorT,
            local_counter_t<PolicySelector, CounterT>,
            LevelT,
            OffsetT,
            SampleT,
            CounterT>,
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
  using LocalCounterT = local_counter_t<PolicySelector, CounterT>;

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

    if (const auto error =
          CubDebug((detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS>(
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
    features.num_pixels          = static_cast<long long>(num_row_pixels) * static_cast<long long>(num_rows);
    features.on_chip_bin_cap     = resolve_on_chip_bin_cap(int{sizeof(LocalCounterT)}, NUM_ACTIVE_CHANNELS);
    features.hybrid_split_bins   = resolve_hybrid_split_bins(int{sizeof(LocalCounterT)}, NUM_ACTIVE_CHANNELS);
    const algorithm algo         = select_algorithm<false>(features);

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
