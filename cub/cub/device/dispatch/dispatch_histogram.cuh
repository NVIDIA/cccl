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
#include <cstdlib>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::histogram
{
// Maximum number of bins per channel for which we will use a privatized smem strategy
static constexpr int max_privatized_smem_bins = 256;

// Above the static 256-bin tier, larger privatized histograms (up to this cap)
// are kept on chip in a single dynamic-SMEM kernel whose per-block histogram
// lives in extern __shared__ sized at launch. This keeps the histogram in fast
// SMEM (avoiding the GMEM atomicAdd_block of the GMEM-privatized path) without a
// ladder of compile-time-sized kernels. 16384 bins x 4 bytes = 64 KB/block
// exceeds the ptxas 48 KB static cap, so the dynamic kernel raises its
// cudaFuncAttributeMaxDynamicSharedMemorySize; the per-CTA SMEM budget on
// SM90/SM100 is large enough for this to launch with reasonable occupancy.
static constexpr int max_dynamic_smem_bins = 16384;

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
  // --- Privatized-SMEM histograms (low/mid bin counts) ---

  // Static fixed-size privatized-SMEM histogram: bins <= 256, and all byte
  // samples. The per-block histogram is a compile-time-sized __shared__ array.
  smem_priv_256,

  // Dynamic privatized-SMEM histogram for 256 < bins <= max_dynamic_smem_bins.
  // The per-block histogram lives in extern __shared__ sized at launch (so one
  // kernel covers the whole range instead of a ladder of compile-time tiers)
  // and is merged into the output with a per-block atomicAdd.
  smem_priv_dynamic,

  // Single-channel SMEM+GMEM single-pass kernel: a primary range of bins stays
  // on chip while the tail spills to a per-block GMEM region, merged in one
  // cooperative launch. Bridges the privatized-SMEM and high-bin regimes.
  hybrid_single_pass,

  // --- High-bin algorithms (bins > max_dynamic_smem_bins) ---

  // Direct atomics to the output, fronted by a per-block 2-hash (cuckoo) SMEM
  // cache that absorbs cross-block contention for skewed hot-bin inputs.
  direct_atomic_cuckoo,

  // Direct atomics to the output, fronted by a per-block single-probe
  // direct-mapped SMEM cache. The shorter, less-divergent probe wins over the
  // cuckoo chain at very high bin counts, where bins vastly outnumber cache
  // slots so the cache holds almost nothing and the extra cuckoo probe is wasted.
  direct_atomic_single_probe,

  // Per-block privatized histograms in GMEM, combined by an atomic-free
  // gather-merge after a grid sync (one cooperative launch). A distinct
  // algorithm that the current tuning does not select (the direct-atomic caches
  // win or tie it across the measured matrix); kept selectable through
  // dispatch_by_algorithm for workloads or architectures where it pays off.
  gmem_priv_gather,
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

// Pick a single algorithm for one cell. Rules are first-match. The selector
// cannot observe input entropy, so each rule is the geomean winner over the
// input-distribution mix for its (channels, sample width, bin tier, pixels)
// regime, derived by sweeping every algorithm across the benchmark matrix.
//
//   1. Privatized-SMEM region (num_bins <= max_dynamic_smem_bins): <=256 bins
//      (and all byte samples) use the static smem_priv_256 kernel; above that, a
//      single dynamic-SMEM kernel (smem_priv_dynamic). Multi-channel eligibility
//      is more restrictive (per-block footprint + classify cost scale with the
//      active channel count), so multi-channel RANGE / >3-channel cells past the
//      relevant cap fall through to the high-bin region.
//
//   2. High-bin region: the single-channel SMEM+GMEM hybrid_single_pass and the
//      two direct-atomic caches (single_probe / cuckoo). Single-channel uses the
//      on-chip hybrid where the histogram fits and the input amortizes its setup
//      (the 65536 cap tier at N>4M, the 262144 mid tier once amortized), and
//      direct atomics elsewhere. Multi-channel uses the direct-atomic caches.
//      The cuckoo and single-probe caches measure within noise, so single-probe
//      (the leaner probe) is the default and cuckoo serves the larger multi
//      bin tiers.
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
  // Privatized-SMEM region. <=256 bins (and all byte samples) use the static
  // fixed-size kernel; above that, a single dynamic-SMEM kernel covers the
  // whole range up to the cap. Multi-channel eligibility is more restrictive
  // (footprint + classify cost scale with channel count); past it, cells fall
  // through to the high-bin region below.
  // -----------------------------------------------------------------------
  if (f.num_bins <= max_privatized_smem_bins)
  {
    return algorithm::smem_priv_256;
  }
  if (f.num_active_channels == 1)
  {
    if (f.num_bins <= max_dynamic_smem_bins)
    {
      return algorithm::smem_priv_dynamic;
    }
  }
  else
  {
    // RANGE (or any transform) up to the smaller cap; EVEN extends further
    // (cheap classify), with the top of the range gated to <=3 active channels.
    if (f.num_bins <= multi_channel_smem_bins_range)
    {
      return algorithm::smem_priv_dynamic;
    }
    if (f.is_even && f.num_bins <= multi_channel_smem_bins_even)
    {
      return algorithm::smem_priv_dynamic;
    }
    if (f.is_even && f.num_active_channels <= 3 && f.num_bins <= max_dynamic_smem_bins)
    {
      return algorithm::smem_priv_dynamic;
    }
  }

  // -----------------------------------------------------------------------
  // High-bin region (num_bins > max_dynamic_smem_bins, i.e. one of 65536 /
  // 262144 / 1048576).
  //
  // The choice is among four algorithms: the single-channel SMEM+GMEM
  // `hybrid_single_pass`; the two direct-atomic-to-output kernels with a
  // per-block cuckoo or single-probe SMEM cache (`direct_atomic_cuckoo`,
  // `direct_atomic_single_probe`); and the cooperative privatized gather-merge
  // `gmem_priv_gather`. The selector cannot observe input entropy (a runtime
  // data property), so each rule picks the algorithm with the best GiB/s GEOMEAN
  // over the input-distribution mix (uniform / skewed / single-hot-bin) for that
  // (channels, sample-width, bin-tier, pixel-count) regime, measured by sweeping
  // every algorithm across the full benchmark matrix on the target architecture.
  //
  // Two facts shape the rules. (1) The cuckoo and single-probe caches perform
  // within noise of each other across the whole single-channel region (both are
  // really direct GMEM atomics once bins far exceed the few-thousand cache
  // slots), so single-probe -- the leaner inner loop -- is the single
  // direct-atomic default. (2) The on-chip hybrid kernel dominates the 65536 and
  // 262144 tiers at large input, where keeping the whole modest histogram in
  // SMEM beats atomics into a large GMEM output; it loses only at small input
  // (its per-block setup is not amortized) and at the 1M-bin tier (too large to
  // stay on chip).
  // -----------------------------------------------------------------------

  // Boundary thresholds. ~256M / ~64M / ~16M / ~4M pixels.
  constexpr long long kSweepPixelThreshold         = 1LL << 28; // 256M
  constexpr long long kRangeF64SweepPixelThreshold = 1LL << 26; // 64M: EVEN mid-tier flips to hybrid
  constexpr long long kHybridMidTierPixels         = 1LL << 24; // 16M: cap-tier flips to hybrid
  constexpr long long kSmallNHighBinPixels         = 1LL << 22; // 4M: below this, setup cost dominates
  constexpr int kMidHighBinTier                    = 262144;    // the middle high-bin tier

  if (f.num_active_channels == 1)
  {
    // Small input: the privatized/hybrid setup (zeroing + merging a wide
    // histogram) is not amortized, so atomic straight to the output wins.
    if (f.num_pixels <= kSmallNHighBinPixels)
    {
      return algorithm::direct_atomic_single_probe;
    }

    // Cap tier (<=65536): the whole histogram fits the on-chip hybrid kernel,
    // which dominates here for both transforms once N > 4M.
    if (f.num_bins <= hybrid_smem_bins_max_single_channel)
    {
      return algorithm::hybrid_single_pass;
    }

    // Mid tier (<=262144): hybrid wins once its setup is amortized over enough
    // pixels. EVEN's cheap ScaleTransform classify reaches that point at >=64M;
    // RANGE's costlier SearchTransform needs >=256M. Below that, direct atomics.
    if (f.num_bins <= kMidHighBinTier)
    {
      const long long hybrid_pixels = f.is_even ? kRangeF64SweepPixelThreshold : kSweepPixelThreshold;
      return (f.num_pixels >= hybrid_pixels) ? algorithm::hybrid_single_pass
                                             : algorithm::direct_atomic_single_probe;
    }

    // Top tier (1048576): bins far exceed the SMEM cache, so it is effectively a
    // pure direct GMEM atomic.
    return algorithm::direct_atomic_single_probe;
  }

  // ---- Multi-channel (hybrid is single-channel-only) ----
  // The per-block privatized intermediate scales with the active channel count,
  // so the high-bin region is served by the direct-atomic caches. (A cooperative
  // gather-merge rule was evaluated for multi-EVEN at the largest inputs but
  // dropped: it wins a narrow uniform/skew geomean yet collapses on the
  // adversarial cache-stress distributions -- e.g. ~4x slower at the cache
  // capacity cliff -- which the direct-atomic caches absorb.)
  //
  // EVEN I32 favours the single-probe cache across all N (cheap classify, leaner
  // probe). EVEN F64 at the cap tier is the one place cuckoo's count replicas pay
  // off, so it is excluded here and falls through to the cuckoo default.
  if (f.is_even && f.sample_bytes < 8)
  {
    return algorithm::direct_atomic_single_probe;
  }
  // Cap tier (<=65536) at moderate/large input: single-probe, except the EVEN F64
  // case handled above. Covers RANGE (both widths) and is a measured tie for the
  // cells it overlaps, so it never regresses.
  if (f.num_bins <= hybrid_smem_bins_max_single_channel && f.num_pixels >= kHybridMidTierPixels
      && !(f.is_even && f.sample_bytes >= 8))
  {
    return algorithm::direct_atomic_single_probe;
  }
  // Default: the cuckoo cache (its 2-hash probe + count replicas win the larger
  // bin tiers and the smallest inputs).
  return algorithm::direct_atomic_cuckoo;
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
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSmemPrivKernel()
  {
    return &DeviceHistogramSmemPrivKernel<
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
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSmemPrivDeviceInitKernel()
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

    return &DeviceHistogramSmemPrivDeviceInitKernel<
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

  /// Host-init dynamic-SMEM, NON-staging variant: merges each block's dyn-SMEM
  /// privatized histogram directly into the global output via atomicAdd
  /// (no staging slabs, no combine kernel). Host must launch the init kernel first.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSmemPrivDynamicKernel()
  {
    return &DeviceHistogramSmemPrivDynamicKernel<
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
  HistogramHybridSinglePassKernel()
  {
    return &DeviceHistogramHybridSinglePassKernel<
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
  // (cuckoo) path and the cooperative gather-merge path via the unified
  // algorithm selector, this overrides the legacy `direct_atomic_bin_threshold`
  // heuristic. Set true to force the gather-merge / Init+Sweep path regardless
  // of bin count.
  bool disable_direct_atomic = false,
  // Cache policy for the direct-atomic-to-output kernel (only consulted when
  // the direct-atomic path is taken, i.e. !disable_direct_atomic):
  //   0 -> 2-hash cuckoo cache (DeviceHistogramDirectAtomicCuckooKernel)
  //   1 -> single-probe direct-mapped cache
  //        (DeviceHistogramDirectAtomicSingleProbeKernel)
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

  const auto init_kernel = kernel_source.template HistogramInitKernel<PolicySelector>();

  // The privatized-SMEM histogram for 256 < bins <= max_dynamic_smem_bins lives
  // in extern __shared__ sized at launch, so its per-CTA SMEM footprint can
  // exceed the ptxas 48 KB static cap (16384 bins x 4 B = 64 KB). The dispatch
  // raises the kernel's cudaFuncAttributeMaxDynamicSharedMemorySize and passes
  // the byte budget as the third launch parameter. The 256-bin / byte tier and
  // the GMEM-privatized path (PRIVATIZED_SMEM_BINS == 0) use static SMEM.
  static constexpr bool kUseDynamicSmem = (PRIVATIZED_SMEM_BINS == max_dynamic_smem_bins);

  // The dynamic-SMEM kernel merges each block's privatized histogram directly
  // into the global output via per-block atomicAdd (StoreOutput): at these bin
  // counts the cross-block contention is spread over enough distinct bins that a
  // direct merge beats a GMEM staging round-trip + cross-block gather.
  auto sweep_kernel = [&] {
    if constexpr (kUseDynamicSmem)
    {
      using output_decode_op_t     = typename FirstLevelArrayT::value_type;
      using privatized_decode_op_t = typename SecondLevelArrayT::value_type;
      return kernel_source.template HistogramSmemPrivDynamicKernel<
             PolicySelector,
             PRIVATIZED_SMEM_BINS,
             privatized_decode_op_t,
             output_decode_op_t>();
    }
    else if constexpr (IsDeviceInit)
    {
      return kernel_source.template HistogramSmemPrivDeviceInitKernel<
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
        .template HistogramSmemPrivKernel<PolicySelector, PRIVATIZED_SMEM_BINS, privatized_decode_op_t, output_decode_op_t>();
    }
  }();

  const int threads_per_block = active_policy.threads_per_block;
  const int pixels_per_thread = active_policy.pixels_per_thread;
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
    if (const auto error =
          CubDebug(launcher_factory.set_max_dynamic_smem_size_for(sweep_kernel, dyn_smem_bytes)))
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
  // Caller may have already picked sweep over direct-atomic via the unified
  // algorithm selector; honour that. Otherwise fall back to the bin-count-
  // based heuristic for backwards compatibility.
  //
  // Single-channel threshold is 1<<16 (65536): because the cuckoo cache lives in
  // dynamic SMEM and grows to use all free shared memory, the direct-atomic +
  // per-block SMEM cache path is competitive with the gather-merge persistent
  // kernel down to 65536 bins, and it avoids the gather-merge's
  // O(num_blocks * num_bins) cross-block reduction. This routes the 262144-bin
  // single-channel cells (the weakest high-bin cells, gather-merge-bound on
  // uniform input) through the larger cache.
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
      auto persistent_kernel_ptr = &DeviceHistogramGmemPrivGatherKernel<PolicySelector,
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
        DeviceHistogramGmemPrivGatherKernel<PolicySelector,
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
        &DeviceHistogramDirectAtomicCuckooKernel<PolicySelector,
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
      auto direct_atomic_noprobe2_kernel_ptr =
        &DeviceHistogramDirectAtomicCuckooKernel<PolicySelector,
                                                          PRIVATIZED_SMEM_BINS,
                                                          NUM_CHANNELS,
                                                          NUM_ACTIVE_CHANNELS,
                                                          SampleIteratorT,
                                                          CounterT,
                                                          privatized_decode_op_t,
                                                          output_decode_op_t,
                                                          OffsetT,
                                                          /*DisableSecondProbe=*/true>;
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
      auto direct_atomic_single_probe_kernel_ptr =
        &DeviceHistogramDirectAtomicSingleProbeKernel<PolicySelector,
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

      // Pick the active direct-atomic kernel per the cache mode. The three
      // variants (single-probe, 2-probe cuckoo, 2-probe-gated cuckoo) are all
      // PRIVATIZED_SMEM_BINS==0 host-init cooperative kernels with the same
      // dynamic-SMEM cache layout; the rest of the launch path treats them
      // uniformly through `active_direct_atomic_kernel_ptr(_void)`.
      const bool use_single_probe_cache = (direct_atomic_cache_mode == 1);
      // When the CUCKOO kernel is selected, drop its second probe on the high-bin
      // tier (bins >> any achievable cache slot count, where the secondary slot
      // can't raise the hit rate -- it just doubles the SMEM key transactions per
      // miss). The cache floor is 1024 (multi) / 4096 (single)
      // slots and only grows from there, so a bin count at/above
      // `kSecondProbeBinThreshold` is already >> the floor; the gate is decided
      // up front (no dependence on the final auto-sized slot count) and is a
      // pure pointer swap. The single-probe kernel already does one probe, so
      // the gate only applies to the cuckoo selection.
      constexpr int kSecondProbeBinThreshold = 262144;
      const bool use_gated_cuckoo =
        (!use_single_probe_cache) && (max_num_output_bins >= kSecondProbeBinThreshold);
      const void* active_direct_atomic_kernel_ptr_void =
        use_single_probe_cache ? direct_atomic_single_probe_kernel_ptr_void
        : use_gated_cuckoo     ? direct_atomic_noprobe2_kernel_ptr_void
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
      if (cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_ordinal) != cudaSuccess
          || max_optin_smem <= 0)
      {
        (void) cudaGetLastError();
        max_optin_smem = cache_tuning::smem_fallback_bytes;
      }
      // Reserve some SMEM for static/driver shared use; the rest is for the cache.
      const int cache_smem_budget =
        (max_optin_smem > cache_tuning::smem_reserve_bytes) ? (max_optin_smem - cache_tuning::smem_reserve_bytes)
                                                            : max_optin_smem;
      const int max_slots_by_smem =
        cache_smem_budget / (NUM_ACTIVE_CHANNELS * cache_bytes_per_slot);

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
      // will actually run.
      auto cache_occupancy_for = [&](auto kernel_ptr, int slots) -> int {
        const int bytes = NUM_ACTIVE_CHANNELS * slots * cache_bytes_per_slot;
        if (launcher_factory.set_max_dynamic_smem_size_for(kernel_ptr, bytes) != cudaSuccess)
        {
          (void) cudaGetLastError();
          return 0;
        }
        int occ = 0;
        // Query at the direct-atomic block size: these lambdas only size the
        // cuckoo / single-probe kernels' dynamic-SMEM cache against their own
        // occupancy.
        if (launcher_factory.MaxSmOccupancy(occ, kernel_ptr, direct_atomic_threads_per_block, bytes) != cudaSuccess)
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

      int cache_slots_per_channel =
        use_single_probe_cache ? size_cache_for(direct_atomic_single_probe_kernel_ptr)
        : use_gated_cuckoo     ? size_cache_for(direct_atomic_noprobe2_kernel_ptr)
                               : size_cache_for(direct_atomic_kernel_ptr);
#if _CCCL_HOSTED()
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
                       ::std::fprintf(stderr,
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
#endif // _CCCL_HOSTED()
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
      else if (use_gated_cuckoo)
      {
        if (launcher_factory.set_max_dynamic_smem_size_for(direct_atomic_noprobe2_kernel_ptr, cuckoo_cache_smem_bytes)
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
        DeviceHistogramDirectAtomicCuckooKernel<PolicySelector,
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
        DeviceHistogramDirectAtomicSingleProbeKernel<PolicySelector,
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
        // Force device-side emission of the second-probe-gated cuckoo variant so
        // `cudaLaunchCooperativeKernel` can resolve its device entry.
        DeviceHistogramDirectAtomicCuckooKernel<PolicySelector,
                                                         PRIVATIZED_SMEM_BINS,
                                                         NUM_CHANNELS,
                                                         NUM_ACTIVE_CHANNELS,
                                                         SampleIteratorT,
                                                         CounterT,
                                                         privatized_decode_op_t,
                                                         output_decode_op_t,
                                                         OffsetT,
                                                         /*DisableSecondProbe=*/true>
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
      // than the sweep kernel that was used to size `num_thread_blocks`,
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
          ? launcher_factory.MaxSmOccupancy(direct_atomic_sm_occupancy,
                                            direct_atomic_single_probe_kernel_ptr,
                                            direct_atomic_threads_per_block,
                                            cuckoo_cache_smem_bytes)
          : use_gated_cuckoo
          ? launcher_factory.MaxSmOccupancy(direct_atomic_sm_occupancy,
                                            direct_atomic_noprobe2_kernel_ptr,
                                            direct_atomic_threads_per_block,
                                            cuckoo_cache_smem_bytes)
          : launcher_factory.MaxSmOccupancy(direct_atomic_sm_occupancy,
                                            direct_atomic_kernel_ptr,
                                            direct_atomic_threads_per_block,
                                            cuckoo_cache_smem_bytes);
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
      if (use_direct_atomic_to_output && direct_atomic_capacity > num_thread_blocks)
      {
        // Upper bound on useful blocks for the grid-stride direct-atomic kernel:
        // one thread per pixel needs ceil(total_pixels / block_size) blocks;
        // beyond that, additional blocks would have no pixels to process. We use
        // the DIRECT-ATOMIC block size here (not the sweep's), so a smaller
        // direct-atomic block is not artificially capped below the sweep's
        // tile-granularity work bound.
        const long long total_pixels_ll =
          static_cast<long long>(num_row_pixels) * static_cast<long long>(num_rows);
        const long long work_tiles =
          (total_pixels_ll + direct_atomic_threads_per_block - 1) / direct_atomic_threads_per_block;
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
            dim3{static_cast<unsigned int>(direct_atomic_threads_per_block)},
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

  // Non-cooperative path: a standalone init kernel followed by the sweep kernel.
  // Taken when the cooperative GMEM-privatized gather-merge above did not launch
  // (every privatized-SMEM tier, plus the direct-atomic and legacy fallbacks).
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
  constexpr int kPrivatizedSmemBins = max_dynamic_smem_bins;

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
    kernel_source.template HistogramHybridSinglePassKernel<
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
    DeviceHistogramHybridSinglePassKernel<PolicySelector,
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
    case algorithm::smem_priv_dynamic: {
      // One dynamic-SMEM kernel for the whole 256 < bins <= max_dynamic_smem_bins
      // range: the per-block histogram lives in extern __shared__ sized at launch
      // and merges into the output with a per-block atomicAdd. PRIVATIZED_SMEM_BINS
      // here is only the "use SMEM privatization, dynamically sized" marker (its
      // value selects the dynamic path in dispatch<>; the actual bin count comes
      // from the runtime level arrays).
      constexpr int PRIVATIZED_SMEM_BINS = max_dynamic_smem_bins;
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
        if (status == cudaSuccess || status != cudaErrorNotSupported)
        {
          return status;
        }
        // hybrid setup failed; fall through to the GMEM-priv path.
      }
      [[fallthrough]];
    }
    case algorithm::direct_atomic_cuckoo:
    case algorithm::gmem_priv_gather:
    case algorithm::direct_atomic_single_probe: {
      // All three go through PRIVATIZED_SMEM_BINS=0 in the deeper `dispatch<>`,
      // which chooses between the direct-atomic-to-output kernel and the
      // gather-merge kernel based on `disable_direct_atomic`, and (for the
      // direct-atomic path) between the cuckoo and single-probe caches based on
      // `direct_atomic_cache_mode`. We pass both from the selector's pick:
      //   gmem_priv_gather        -> disable_direct_atomic=true  (sweep)
      //   direct_atomic_cuckoo       -> direct-atomic, cache_mode=0 (cuckoo)
      //   direct_atomic_single_probe -> direct-atomic, cache_mode=1 (single-probe)
      constexpr int PRIVATIZED_SMEM_BINS = 0;
      const bool disable_direct_atomic   = (algo == algorithm::gmem_priv_gather);
      const int direct_atomic_cache_mode = (algo == algorithm::direct_atomic_single_probe) ? 1 : 0;
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
        disable_direct_atomic,
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
