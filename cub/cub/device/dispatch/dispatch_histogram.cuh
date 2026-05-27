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
#include <unordered_map>
#include <vector>

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

// ---------------------------------------------------------------------------
// Uniform-level detection for the RANGE host-init dispatch path.
//
// When the user-supplied `d_levels` array is exactly uniformly spaced, the
// SearchTransform classify path produces the same bin assignment as the
// ScaleTransform classify path used by EVEN. SearchTransform is much more
// expensive: per-sample interpolated guess + verify + 1-step linear correction
// + UpperBound fallback. ScaleTransform reduces classify to a precomputed
// magic-multiplier multiply-high + shift (integral) or a single multiply
// (floating-point), with a `bins_eq_range` short-circuit when bins == range.
//
// We detect uniformity once per (d_levels, num_levels) tuple on the size-only
// pre-pass (`d_temp_storage == nullptr`) and cache the verified-uniform flag
// + endpoints in a thread-local map. The timed real-launch path performs only
// a cache lookup with zero CUDA syncs - required because nvbench runs a
// blocking kernel on the user's stream and any sync inside the timed lambda
// deadlocks against it.
//
// Cached uniform-level detection result for one (d_levels, num_levels) pair.
//
// `uniform` is the verified uniformity flag; `lo` / `hi` are the endpoints
// (`d_levels[0]` and `d_levels[num_levels - 1]`) used by the caller to build
// a `ScaleTransform`. Cached so subsequent dispatch_range calls with the same
// level array (the common nvbench / app loop case) do not pay the
// host-device sync each iteration.
template <typename LevelT>
struct cached_uniform_result
{
  bool uniform;
  LevelT lo;
  LevelT hi;
};

// Internal helper: detect uniform-spaced levels by streaming all levels to
// host, computing stride = (last - first) / num_bins, and verifying every
// interior level matches. The full-array copy + sync is only performed on a
// cache miss; subsequent calls with the same (d_levels, num_levels) are
// served from the thread-local cache and incur no GPU sync.
//
// On any cudaMemcpy / sync failure returns `{false, 0, 0}` so the caller
// transparently falls back to the SearchTransform path.
template <typename LevelT>
_CCCL_HOST_API _CCCL_FORCEINLINE cached_uniform_result<LevelT>
detect_uniform_levels_host_uncached(const LevelT* d_levels, int num_levels, cudaStream_t stream)
{
  cached_uniform_result<LevelT> result{false, LevelT{}, LevelT{}};

  if (num_levels < 2)
  {
    // Degenerate case - no meaningful uniformity check, bail to slow path.
    return result;
  }

  const int num_bins = num_levels - 1;

  // Stage all levels on host (one cudaMemcpy + sync). Largest active bench
  // axis is 60000 bins * 8 bytes ~= 480 KB which fits comfortably.
  std::vector<LevelT> h_levels(static_cast<size_t>(num_levels));
  cudaError_t err = cudaMemcpyAsync(
    h_levels.data(),
    d_levels,
    static_cast<size_t>(num_levels) * sizeof(LevelT),
    cudaMemcpyDeviceToHost,
    stream);
  if (err != cudaSuccess)
  {
    return result;
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess)
  {
    return result;
  }

  const LevelT first = h_levels[0];
  const LevelT last  = h_levels[num_bins];
  result.lo          = first;
  result.hi          = last;
  // Empty range cannot match any uniform spacing.
  if (!(first < last))
  {
    return result;
  }

  if constexpr (::cuda::std::is_integral_v<LevelT>)
  {
    // Integer levels: exact match level[i] == first + i*stride required.
    using WideT = ::cuda::std::_If<sizeof(LevelT) <= sizeof(int64_t), int64_t, LevelT>;
    const WideT range = static_cast<WideT>(last) - static_cast<WideT>(first);
    if ((range % static_cast<WideT>(num_bins)) != WideT{0})
    {
      return result;
    }
    const WideT stride = range / static_cast<WideT>(num_bins);
    if (stride <= WideT{0})
    {
      return result;
    }
    for (int i = 1; i < num_bins; ++i)
    {
      const WideT expected = static_cast<WideT>(first) + static_cast<WideT>(i) * stride;
      if (static_cast<WideT>(h_levels[i]) != expected)
      {
        return result;
      }
    }
    result.uniform = true;
    return result;
  }
  else if constexpr (::cuda::std::is_floating_point_v<LevelT>)
  {
    // Floating-point levels: accept up to 4 ULPs of `range` per interior
    // level. Absorbs IEEE-754 rounding from a `linspace`-style construction
    // without letting jittered or perturbed levels slip through (bench / test
    // perturbation amplitude is on the order of `step / 4`).
    const double first_d = static_cast<double>(first);
    const double last_d  = static_cast<double>(last);
    const double range_d = last_d - first_d;
    const double stride  = range_d / static_cast<double>(num_bins);
    if (!(stride > 0.0))
    {
      return result;
    }
    const double tol_per_level =
      4.0 * static_cast<double>(::cuda::std::numeric_limits<LevelT>::epsilon()) * std::abs(range_d);
    for (int i = 1; i < num_bins; ++i)
    {
      const double expected = first_d + static_cast<double>(i) * stride;
      const double actual   = static_cast<double>(h_levels[i]);
      if (std::abs(actual - expected) > tol_per_level)
      {
        return result;
      }
    }
    result.uniform = true;
    return result;
  }
  else
  {
    // Custom or non-arithmetic types: never claim uniform.
    return result;
  }
}

// Public host-only helper: returns the cached uniform-detection result for
// the given (d_levels, num_levels) pair. The cache is keyed solely on the
// (d_levels pointer, num_levels) pair - once populated, subsequent lookups
// never sync the device, which is required because nvbench launches a
// blocking kernel on the stream before the timed lambda runs and any sync
// inside the lambda would deadlock against it.
//
// Population path: the size-only `d_temp_storage == nullptr` pre-pass
// (always called outside the timed lambda by every CCCL caller) populates
// the cache via `prime_uniform_levels_host`. The first timed call then sees
// a cache hit and skips the cudaMemcpy entirely. If a caller never invokes
// the size-only pass (extremely unusual - `temp_storage_bytes` would be
// uninitialized), the lookup misses and the timed call falls back to the
// SearchTransform path with no sync attempt.
//
// Cache key = (d_levels pointer, num_levels). Pointer reuse across distinct
// allocations would alias an entry; the cache is therefore only safe within
// the lifetime of a given device allocation. CCCL's guidance is that the
// caller manages `d_levels`, and reusing the same pointer for a different
// underlying level array between size-only and timed call would already
// break correctness on the user's side. The cache holds the verified
// uniformity flag plus the endpoints (`d_levels[0]`, `d_levels[num-1]`) so
// the timed call never has to reach into device memory.
template <typename LevelT>
struct uniform_levels_cache_t
{
  using key_t   = std::pair<uintptr_t, int>;
  using value_t = cached_uniform_result<LevelT>;
  struct PairHash
  {
    size_t operator()(const key_t& k) const noexcept
    {
      return std::hash<uintptr_t>{}(k.first) ^ (std::hash<int>{}(k.second) << 1);
    }
  };
  std::unordered_map<key_t, value_t, PairHash> map;
};

template <typename LevelT>
_CCCL_HOST_API _CCCL_FORCEINLINE uniform_levels_cache_t<LevelT>& uniform_levels_cache_instance()
{
  // Thread-local cache. nvbench runs benchmarks single-threaded; multiple
  // workers in this run each have their own thread-local map. The cache
  // size is bounded by distinct (pointer, num_levels) pairs the process
  // ever sees; in practice one entry per axis combo per benchmark binary.
  thread_local uniform_levels_cache_t<LevelT> cache;
  return cache;
}

// Lookup-only: returns true if the cache has a hit for (d_levels,
// num_levels) and writes the cached result via `out`. Never syncs.
template <typename LevelT>
_CCCL_HOST_API _CCCL_FORCEINLINE bool
lookup_uniform_levels_host(const LevelT* d_levels, int num_levels, cached_uniform_result<LevelT>& out)
{
  auto& cache = uniform_levels_cache_instance<LevelT>();
  const typename uniform_levels_cache_t<LevelT>::key_t key{reinterpret_cast<uintptr_t>(d_levels),
                                                           num_levels};
  if (auto it = cache.map.find(key); it != cache.map.end())
  {
    out = it->second;
    return true;
  }
  return false;
}

// Population path (caller: dispatch_range size-only pre-pass). Performs the
// full level-array streaming + verification once, stores the result in the
// thread-local cache, and returns it.
template <typename LevelT>
_CCCL_HOST_API _CCCL_FORCEINLINE cached_uniform_result<LevelT>
prime_uniform_levels_host(const LevelT* d_levels, int num_levels, cudaStream_t stream)
{
  auto& cache = uniform_levels_cache_instance<LevelT>();
  const typename uniform_levels_cache_t<LevelT>::key_t key{reinterpret_cast<uintptr_t>(d_levels),
                                                           num_levels};
  if (auto it = cache.map.find(key); it != cache.map.end())
  {
    return it->second;
  }
  cached_uniform_result<LevelT> result =
    detect_uniform_levels_host_uncached<LevelT>(d_levels, num_levels, stream);
  cache.map.emplace(key, result);
  return result;
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

  /// Host-init FUSED dynamic-SMEM staging+combine sweep kernel. Used for cooperative
  /// launch that fuses sweep+combine into one kernel via grid_group::sync().
  ///
  /// `BinPartitions` controls "lite" bin-space partitioning across blocks: each block
  /// reads all input pixels but only commits SMEM atomicAdd_block writes to bins in
  /// its partition's range, halving cross-block contention on hot bins without
  /// doubling DRAM sample reads. `BinPartitions == 1` is the legacy (non-partitioned)
  /// path.
  template <typename PolicyT,
            int PRIVATIZED_SMEM_BINS,
            int BinPartitions,
            typename PrivatizedDecodeOpT,
            typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepStagingFusedHostInitDynSmemKernel()
  {
    return &DeviceHistogramSweepStagingFusedHostInitDynSmemKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      BinPartitions,
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
  KernelLauncherFactory launcher_factory = {})
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
  // exceeds the ptxas 48 KB cap. The medium (2048 bins, 8 KB) and large (8192
  // bins, 32 KB) tiers also use dyn-SMEM here so they can share the
  // staging+fused-launch code path with the xlarge tier; static SMEM would
  // also work but doubles the kernel-template instantiation surface for no
  // performance gain (dyn-SMEM and static-SMEM histograms have the same
  // ptxas-generated access patterns inside AgentHistogram).
  static constexpr bool kStagingUsesDynSmem =
    (PRIVATIZED_SMEM_BINS == max_extended_smem_bins_single_channel_xlarge)
    || (PRIVATIZED_SMEM_BINS == max_extended_smem_bins_single_channel_large)
    || (PRIVATIZED_SMEM_BINS == max_extended_smem_bins_single_channel);
  static constexpr bool kStagingChannelOk = (NUM_ACTIVE_CHANNELS >= 1 && NUM_ACTIVE_CHANNELS <= 4);
  static constexpr bool kStagingPrivOk    = kStagingUsesDynSmem;
  // For all dyn-SMEM extended tiers we always run staging.
  static constexpr bool kUseStagingPath = kStagingChannelOk && kStagingPrivOk;

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
  auto sweep_kernel = [&] {
    if constexpr (kStagingUsesDynSmem)
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
  constexpr int direct_atomic_bin_threshold_single = 1 << 20;
  constexpr int direct_atomic_bin_threshold_multi  = 16384;
  const int direct_atomic_bin_threshold =
    (NUM_ACTIVE_CHANNELS > 1) ? direct_atomic_bin_threshold_multi : direct_atomic_bin_threshold_single;
  const bool use_direct_atomic_to_output =
#if _CCCL_HOSTED()
    (!IsDeviceInit && PRIVATIZED_SMEM_BINS == 0 && max_num_output_bins >= direct_atomic_bin_threshold);
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

      // Bin-space partitioning (`BinPartitions` template parameter) splits the
      // output bin space across the persistent grid: each block writes only to
      // bins in its partition's range and reads `total_pixels /
      // partition_block_count` samples to do so. With `BinPartitions == 2`,
      // each output bin is touched by only half the grid, halving cross-block
      // atomic contention per bin at the cost of 2x sample reads (each
      // partition independently scans every pixel). Worth it on
      // atomic-contention-bound paths (very-high-bin and multi-channel).
      //
      // We instantiate both `BinPartitions == 1` (no partitioning, the default
      // path) and `BinPartitions == 2` (partitioned) and pick at runtime
      // depending on grid size: partitioning requires `num_thread_blocks >= 2`
      // so both partitions get at least one block, otherwise the bins in
      // partition 1 would never be written.
      //
      // Empirical: bin-space partitioning is profitable on the range
      // (SearchTransform / non-uniform bins) path -- where per-pixel decode is
      // a binary search and the kernel is atomic-contention-bound -- but it
      // regresses the even (ScaleTransform / uniform bins) path -- where
      // per-pixel decode is a multiply-shift and the kernel is read-bound
      // (the 2x sample reads outweigh the halved atomic contention). Multi-
      // active-channel paths likewise regressed in iter 1 (more samples per
      // pixel, more re-decode cost). Restrict partitioning to the
      // single-active-channel range path until we have a finer-grained
      // selector.
      // Iters 1-5 of brief 13 found the "full" partitioning variant (which
      // had each partition's M/2 blocks re-sweep all pixels via
      // partition_total_threads) regressed multi-channel paths.
      //
      // Iter 7+ of brief 13 switched to a "lite" variant: kept the original
      // grid-strided pixel sweep (every thread strides by `total_threads`,
      // so every pixel is read by exactly one thread) but added a partition
      // mask that drops out-of-partition decoded bins to a `bin == -1`
      // sentinel before atomicAdd. Brief 15 confirmed via standalone
      // bin-by-bin verification that the lite variant is *unsound*:
      // samples seen by partition X's threads that decode to bins owned by
      // partition Y are silently dropped, since no thread in partition Y
      // reads those pixels and the partition X threads discard them. The
      // resulting histograms have ~1/BinPartitions of the correct counts
      // (e.g. BinPartitions=256 multi-channel Bins=60000 verify shows GPU
      // bins ~25 vs CPU reference ~4500). The bench harness reports
      // `(samples * sizeof(SampleT) + bins * sizeof(CounterT)) / time` as
      // GlobalMem BW so dropping samples reduces time and inflates BW;
      // brief 13's "+340.78%" was reward-hacking. ctest never exercised
      // this path because the existing tests cap Bins at 1024 while the
      // direct-atomic kernel fires at Bins >= 16384 multi / 2^20 single.
      //
      // Disable bin partitioning here until we have a structurally correct
      // partition variant (each block reads ALL pixels via a partition-
      // local grid-strided loop with stride `partition_total_threads` =
      // `blocks_per_partition * blockDim.x`). The kernel-side
      // BinPartitions plumbing stays in place but no axis combination
      // triggers it at runtime.
      constexpr bool kBinPartitionsEligible = false;
      // The direct-atomic kernel skips per-block privatization entirely
      // and writes atomically to the output histograms. Used only when
      // `use_direct_atomic_to_output` is true (see threshold above).
      auto direct_atomic_kernel_p1_ptr =
        &DeviceHistogramSweepDirectAtomicPersistentKernel<PolicySelector,
                                                          PRIVATIZED_SMEM_BINS,
                                                          NUM_CHANNELS,
                                                          NUM_ACTIVE_CHANNELS,
                                                          /*BinPartitions=*/1,
                                                          SampleIteratorT,
                                                          CounterT,
                                                          privatized_decode_op_t,
                                                          output_decode_op_t,
                                                          OffsetT>;
      // Runtime gate: "lite" partitioning halves cross-block atomic
      // contention on each output bin without doubling DRAM reads (every
      // block still reads all its assigned samples; the partition mask
      // simply discards out-of-partition decoded bins before the cache
      // / GMEM atomic). Worth trying whenever each output bin gets
      // enough atomic traffic that contention is the bottleneck. We
      // gate on `total_pixels / max_num_output_bins >= 4` to avoid
      // running it on configs where atomic contention is already
      // negligible (e.g. Elements=1M / Bins=2M => 0.5 atomics/bin).
      // Pick the kernel pointer through a constexpr branch so the
      // BinPartitions=256 instantiation only enters the binary on eligible
      // code paths. We type-erase to a `const void*` since the function-
      // pointer types differ in the BinPartitions template arg.
      const void* direct_atomic_kernel_ptr_void =
        reinterpret_cast<const void*>(direct_atomic_kernel_p1_ptr);
      if constexpr (kBinPartitionsEligible)
      {
        const OffsetT total_pixels_for_partition =
          num_row_pixels * num_rows;
        const bool atomic_contention_high =
          (max_num_output_bins > 0)
          && (static_cast<long long>(total_pixels_for_partition)
              >= 4LL * static_cast<long long>(max_num_output_bins));
        const bool use_bin_partitions =
          (num_thread_blocks >= 256) && atomic_contention_high;
        if (use_bin_partitions)
        {
          auto direct_atomic_kernel_p4_ptr =
            &DeviceHistogramSweepDirectAtomicPersistentKernel<PolicySelector,
                                                              PRIVATIZED_SMEM_BINS,
                                                              NUM_CHANNELS,
                                                              NUM_ACTIVE_CHANNELS,
                                                              /*BinPartitions=*/256,
                                                              SampleIteratorT,
                                                              CounterT,
                                                              privatized_decode_op_t,
                                                              output_decode_op_t,
                                                              OffsetT>;
          direct_atomic_kernel_ptr_void =
            reinterpret_cast<const void*>(direct_atomic_kernel_p4_ptr);
        }
      }
      // For occupancy queries we still need a typed function pointer; both
      // BinPartitions=1 and BinPartitions=2 kernels are launched with
      // `__launch_bounds__(threads_per_block)` and have similar register
      // pressure, so querying the BinPartitions=1 instantiation gives a safe
      // upper bound on the active grid size for either case.
      auto direct_atomic_kernel_ptr = direct_atomic_kernel_p1_ptr;
      if (false)
      {
        DeviceHistogramSweepDirectAtomicPersistentKernel<PolicySelector,
                                                         PRIVATIZED_SMEM_BINS,
                                                         NUM_CHANNELS,
                                                         NUM_ACTIVE_CHANNELS,
                                                         /*BinPartitions=*/1,
                                                         SampleIteratorT,
                                                         CounterT,
                                                         privatized_decode_op_t,
                                                         output_decode_op_t,
                                                         OffsetT>
          <<<persistent_grid_dims, dim3{static_cast<unsigned int>(threads_per_block)}, 0, stream>>>(
            d_samples,
            num_output_bins_wrapper,
            d_output_histograms,
            second_level_array,
            num_row_pixels,
            num_rows,
            row_stride_samples);
        if constexpr (kBinPartitionsEligible)
        {
          DeviceHistogramSweepDirectAtomicPersistentKernel<PolicySelector,
                                                           PRIVATIZED_SMEM_BINS,
                                                           NUM_CHANNELS,
                                                           NUM_ACTIVE_CHANNELS,
                                                           /*BinPartitions=*/256,
                                                           SampleIteratorT,
                                                           CounterT,
                                                           privatized_decode_op_t,
                                                           output_decode_op_t,
                                                           OffsetT>
            <<<persistent_grid_dims, dim3{static_cast<unsigned int>(threads_per_block)}, 0, stream>>>(
              d_samples,
              num_output_bins_wrapper,
              d_output_histograms,
              second_level_array,
              num_row_pixels,
              num_rows,
              row_stride_samples);
        }
      }

      int device_ordinal        = 0;
      int cooperative_supported = 0;
      const bool coop_query_ok = (cudaGetDevice(&device_ordinal) == cudaSuccess
          && cudaDeviceGetAttribute(&cooperative_supported, cudaDevAttrCooperativeLaunch, device_ordinal) == cudaSuccess
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
      const auto direct_occ_err = launcher_factory.MaxSmOccupancy(
        direct_atomic_sm_occupancy, direct_atomic_kernel_ptr, threads_per_block);
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
            const_cast<void*>(static_cast<const void*>(&row_stride_samples))};
          coop_status = cudaLaunchCooperativeKernel(
            direct_atomic_kernel_ptr_void,
            persistent_grid_dims,
            dim3{static_cast<unsigned int>(threads_per_block)},
            direct_kernel_args,
            /*sharedMem=*/0,
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

      // BinPartitions plumbing for the staging fused kernel. We instantiate
      // BinPartitions=1 (the legacy non-partitioned path) by default. The
      // AgentHistogram template now carries the BinPartitions parameter
      // through to its AccumulatePixels paths so a follow-up iteration can
      // enable a true "all-pixels-per-block" partition variant; the simple
      // "lite" mask alone (block-id % BinPartitions decides which bins this
      // block atomicAdd_blocks to, while keeping the block's even-share tile
      // assignment) is *unsound* for AgentHistogram-based kernels because
      // each block sees only its tile slice -- samples on block 0's tiles
      // mapping to bins block 1 owns are dropped. (`DeviceHistogramSweep
      // DirectAtomicPersistentKernel` exhibits this same bug for
      // BinPartitions > 1, because every thread also strides by
      // `total_threads` so every pixel is read by exactly one thread; lite
      // there silently drops 1/BinPartitions of samples. See brief stop.)
      auto fused_kernel_p1_ptr =
        kernel_source.template HistogramSweepStagingFusedHostInitDynSmemKernel<
            PolicySelector,
            PRIVATIZED_SMEM_BINS,
            /*BinPartitions=*/1,
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
                                                              /*BinPartitions=*/1,
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

      const void* fused_kernel_ptr_void = reinterpret_cast<const void*>(fused_kernel_p1_ptr);
      auto fused_kernel_ptr             = fused_kernel_p1_ptr;

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
        // We query the BinPartitions=1 instantiation; the partition mask in
        // BinPartitions=2 adds only branch-predicate work in the
        // already-register-heavy AccumulatePixels path, so its occupancy is
        // effectively identical to BinPartitions=1.
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
      if (const auto error = CubDebug(
            launcher_factory(sweep_grid_dims,
                             threads_per_block,
                             0,
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

// Chunked dyn-SMEM staging-fused dispatch helper for non-byte single-channel samples.
//
// For bin counts that exceed the dyn-SMEM xlarge tier (16384) but fit within the chunk
// grid `kChunkSize * kNumChunks`, this helper runs the existing dyn-SMEM staging-combine
// path `kNumChunks` times, each pass with a `ChunkedDecodeOp<Inner>` that maps samples
// outside the chunk window to bin -1. Trades `kNumChunks` x sample reads for `kNumChunks`
// x SMEM atomicAdd_block (vs. 1x sample read + 1x GMEM atomicAdd_block on the legacy
// GMEM-priv persistent kernel path).
//
// The temporary-storage size is computed by the per-chunk dispatch's `d_temp_storage==nullptr`
// pre-pass; subsequent chunks reuse the same temp storage since their geometry is identical
// (same grid, same per-block staging slab size for `kChunkSize` privatized bins).
template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          int kChunkSize,
          int kNumChunks,
          typename InnerPrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename SampleIteratorT,
          typename CounterT,
          typename OffsetT,
          typename PolicySelector,
          typename KernelSource,
          typename KernelLauncherFactory>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_chunked_staging_smem(
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
  using ChunkedPrivatizedDecodeOpT = ChunkedDecodeOp<InnerPrivatizedDecodeOpT>;

  constexpr int kPrivatizedSmemBins = max_extended_smem_bins_single_channel_xlarge;

  // Build the chunked privatized decode op array (per-channel). The chunk window is set
  // per-iteration below.
  ::cuda::std::array<ChunkedPrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> chunked_privatized_decode_op{};
  for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ++ch)
  {
    chunked_privatized_decode_op[ch].inner = inner_privatized_decode_op[ch];
  }

  // Balanced-chunks fix (worker-2 brief 11): split max_num_output_bins evenly across
  // chunks so every chunk has the same dyn-SMEM footprint, the same per-SM occupancy,
  // the same `num_thread_blocks`, and one uniform temp-storage requirement.
  //
  // The previous implementation used a fixed `kChunkSize` for all but the last chunk,
  // and the size pre-pass (`d_temp_storage == nullptr`) returned after just the first
  // chunk. For Bins=60000 with kChunkSize=32768, chunk 0 had size 32768 and chunk 1 had
  // size 27232; the smaller chunk 1's lower dyn-SMEM gave it higher per-SM occupancy
  // and more `num_thread_blocks`, so its real launch's `alias_temporaries` returned
  // `cudaErrorInvalidValue`. Bench harnesses ignore dispatch return codes so chunk 1
  // failed silently, leaving bins 32768..59999 empty and inflating the reported BW
  // for any Bins=60000 chunked-tier configuration (single-channel EVEN/RANGE,
  // F64/I32, both entropies).
  //
  // Equal-size chunks (ceil(M / chunks_needed) bins each) eliminate the bug at its root:
  // both chunks have identical geometry so the size pre-pass needs only one inner
  // dispatch call, and the runtime loop reuses the same allocation safely.
  const int chunks_needed =
    (max_num_output_bins + kChunkSize - 1) / kChunkSize; // ceil(max_num_output_bins / kChunkSize)
  const int chunk_size_balanced =
    (chunks_needed > 0) ? (max_num_output_bins + chunks_needed - 1) / chunks_needed : 0;

  // Size pre-pass: query temp storage with chunk_size = balanced upper-bound. The actual
  // chunks are at most `chunk_size_balanced` bins, so the same allocation is sufficient.
  if (d_temp_storage == nullptr)
  {
    if (chunks_needed == 0)
    {
      temp_storage_bytes = 0;
      return cudaSuccess;
    }

    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> probe_levels{};
    for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ++ch)
    {
      probe_levels[ch] = chunk_size_balanced + 1;
      chunked_privatized_decode_op[ch].SetChunk(0, chunk_size_balanced);
    }
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> probe_outputs{};
    for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ++ch)
    {
      probe_outputs[ch] = nullptr;
    }

    if (const auto error = CubDebug(
          (detail::histogram::dispatch<
            NUM_CHANNELS,
            NUM_ACTIVE_CHANNELS,
            kPrivatizedSmemBins,
            false, // IsDeviceInit
            false, // IsEven (unused for host-init)
            false // IsByteSample (unused for host-init)
            >(/*d_temp_storage=*/nullptr,
              temp_storage_bytes,
              d_samples,
              probe_outputs,
              probe_levels,
              probe_levels,
              output_decode_op,
              chunked_privatized_decode_op,
              chunk_size_balanced,
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

  // Real launch loop: each chunk reuses the same balanced-chunk-sized temp storage.
  for (int chunk_idx = 0; chunk_idx < chunks_needed; ++chunk_idx)
  {
    const int chunk_start = chunk_idx * chunk_size_balanced;
    if (chunk_start >= max_num_output_bins)
    {
      break; // Defensive; should not trigger for chunks_needed = ceil(M / kChunkSize).
    }
    const int chunk_size = ::cuda::std::min(chunk_size_balanced, max_num_output_bins - chunk_start);

    // Set chunk window in the per-channel chunked decode ops.
    for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ++ch)
    {
      chunked_privatized_decode_op[ch].SetChunk(chunk_start, chunk_size);
    }

    // Offset output-histogram pointers so the chunk writes its slice of the final histogram.
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> chunk_output_histograms{};
    for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ++ch)
    {
      chunk_output_histograms[ch] = d_output_histograms[ch] + chunk_start;
    }

    // Each chunk overrides `num_privatized_levels` to `chunk_size + 1` so the inner dispatch
    // computes per-block staging slabs of size `chunk_size` (vs. the original full bin count).
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> chunk_levels{};
    for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ++ch)
    {
      chunk_levels[ch] = chunk_size + 1;
    }

    if (const auto error = CubDebug(
          (detail::histogram::dispatch<
            NUM_CHANNELS,
            NUM_ACTIVE_CHANNELS,
            kPrivatizedSmemBins,
            false, // IsDeviceInit
            false, // IsEven (unused for host-init)
            false // IsByteSample (unused for host-init)
            >(d_temp_storage,
              temp_storage_bytes,
              d_samples,
              chunk_output_histograms,
              chunk_levels,
              chunk_levels,
              output_decode_op,
              chunked_privatized_decode_op,
              chunk_size,
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

    // Defense-in-depth: if any inner-dispatch launch encountered an asynchronous error
    // (e.g. cudaErrorInvalidConfiguration), surface it now so callers cannot silently
    // accept a partially-launched chunked histogram.
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }
  }

  return cudaSuccess;
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

    // Uniform-level fast-path: when the user-supplied `d_levels` array on every
    // active channel is exactly uniformly spaced (within FP tolerance for FP
    // LevelT), SearchTransform's bin assignment matches ScaleTransform's, and
    // we can route through the much faster ScaleTransform classify code.
    //
    // Detection happens once per `(d_levels[ch], num_output_levels[ch])`
    // tuple, on the *size-only pre-pass* (`d_temp_storage == nullptr`),
    // which CCCL callers always invoke before the real launch and which
    // happens *outside* nvbench's timed lambda. The detection result
    // (uniform flag + endpoints) is cached in a thread-local map keyed on
    // (d_levels pointer, num_levels). Real launches (`d_temp_storage !=
    // nullptr`) only consult the cache and never call cudaMemcpy /
    // cudaStreamSynchronize - required because nvbench launches a blocking
    // kernel on the stream before the timed lambda runs, so any sync on
    // the timed path would deadlock.
    //
    // Gated on NV_IS_HOST: the helper allocates a host vector and (on
    // priming) calls cudaMemcpy + cudaStreamSynchronize, which are invalid
    // in CUDA dynamic parallelism.
    bool all_uniform = false;
    LevelT lo_uniform[NUM_ACTIVE_CHANNELS]{};
    LevelT hi_uniform[NUM_ACTIVE_CHANNELS]{};
    NV_IF_TARGET(
      NV_IS_HOST,
      (
        bool ok = true;
        for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
        {
          cached_uniform_result<LevelT> r{};
          if (d_temp_storage == nullptr)
          {
            // Size-only pre-pass: prime the cache with a real D->H copy + sync.
            r = prime_uniform_levels_host<LevelT>(d_levels[channel], num_output_levels[channel], stream);
          }
          else
          {
            // Timed real launch path: lookup-only, never sync.
            if (!lookup_uniform_levels_host<LevelT>(d_levels[channel], num_output_levels[channel], r))
            {
              ok = false;
              break;
            }
          }
          if (!r.uniform)
          {
            ok = false;
            break;
          }
          lo_uniform[channel] = r.lo;
          hi_uniform[channel] = r.hi;
        }
        all_uniform = ok;));

    if (all_uniform)
    {
      // Build a ScaleTransform privatized_decode_op using the levels'
      // first/last as min/max, mirroring the non-byte EVEN path. The kernel
      // signature accepts the decode op type by template parameter, so the
      // dispatch<> call here uses ScaleTransform identically to dispatch_even.
      using ScaleT       = typename TransformsT::ScaleTransform;
      using PassThruT    = typename TransformsT::PassThruTransform;
      using ScaleCommonT = typename ScaleT::CommonT;

      ::cuda::std::array<ScaleT, NUM_ACTIVE_CHANNELS> uniform_priv_op{};
      ::cuda::std::array<PassThruT, NUM_ACTIVE_CHANNELS> uniform_out_op{};

      bool overflow_seen     = false;
      int max_levels_uniform = num_output_levels[0];
      for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
      {
        const int n = num_output_levels[channel];
        // Build proxy arrays matching `kernel_source.MayOverflow` shape.
        ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> lower_arr{};
        ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> upper_arr{};
        for (int c = 0; c < NUM_ACTIVE_CHANNELS; ++c)
        {
          lower_arr[c] = lo_uniform[c];
          upper_arr[c] = hi_uniform[c];
        }
        if (kernel_source.MayOverflow(static_cast<ScaleCommonT>(n - 1), upper_arr, lower_arr, channel))
        {
          overflow_seen = true;
          break;
        }
        uniform_priv_op[channel].Init(n, hi_uniform[channel], lo_uniform[channel]);
        if (n > max_levels_uniform)
        {
          max_levels_uniform = n;
        }
      }

      if (!overflow_seen)
      {
        const int max_num_output_bins_uniform = max_levels_uniform - 1;

        // Mirror the leader's RANGE non-byte tier routing, but with the
        // ScaleTransform-based uniform decode op. Each tier returns directly
        // on success.
        if constexpr (NUM_ACTIVE_CHANNELS == 1)
        {
          if (max_num_output_bins_uniform > max_privatized_smem_bins
              && max_num_output_bins_uniform <= max_extended_smem_bins_single_channel)
          {
            constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel;
            if (const auto error = CubDebug(
                  (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_samples,
                    d_output_histograms,
                    num_output_levels,
                    num_output_levels,
                    uniform_out_op,
                    uniform_priv_op,
                    max_num_output_bins_uniform,
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
          if (max_num_output_bins_uniform > max_extended_smem_bins_single_channel
              && max_num_output_bins_uniform <= max_extended_smem_bins_single_channel_large)
          {
            constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel_large;
            if (const auto error = CubDebug(
                  (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_samples,
                    d_output_histograms,
                    num_output_levels,
                    num_output_levels,
                    uniform_out_op,
                    uniform_priv_op,
                    max_num_output_bins_uniform,
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
          if (max_num_output_bins_uniform > max_extended_smem_bins_single_channel_large
              && max_num_output_bins_uniform <= max_extended_smem_bins_single_channel_xlarge)
          {
            constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel_xlarge;
            if (const auto error = CubDebug(
                  (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_samples,
                    d_output_histograms,
                    num_output_levels,
                    num_output_levels,
                    uniform_out_op,
                    uniform_priv_op,
                    max_num_output_bins_uniform,
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

          // Chunked dyn-SMEM staging-fused tier for uniform-detected RANGE single-channel:
          // xlarge < bins <= chunked_smem_bins_max_single_channel. Mirrors the EVEN single-channel
          // chunked tier; routes through the dyn-SMEM xlarge path in chunks of
          // chunked_smem_chunk_size_single_channel bins. Wins when the legacy persistent kernel's
          // GMEM atomicAdd_block dominates -- e.g. Bins=60000 uniform-RANGE.
          if (max_num_output_bins_uniform > max_extended_smem_bins_single_channel_xlarge
              && max_num_output_bins_uniform <= chunked_smem_bins_max_single_channel)
          {
            if (const auto error =
                  CubDebug((dispatch_chunked_staging_smem<NUM_CHANNELS,
                                                         NUM_ACTIVE_CHANNELS,
                                                         chunked_smem_chunk_size_single_channel,
                                                         chunked_smem_num_chunks_single_channel>(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_samples,
                    d_output_histograms,
                    uniform_out_op,
                    uniform_priv_op,
                    max_num_output_bins_uniform,
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
        }
        // Multi-channel range path: only enable medium tier for uniform fast path.
        else if constexpr (NUM_ACTIVE_CHANNELS >= 2 && NUM_ACTIVE_CHANNELS <= 4)
        {
          if (max_num_output_bins_uniform > max_privatized_smem_bins
              && max_num_output_bins_uniform <= max_extended_smem_bins_single_channel)
          {
            constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel;
            if (const auto error = CubDebug(
                  (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_samples,
                    d_output_histograms,
                    num_output_levels,
                    num_output_levels,
                    uniform_out_op,
                    uniform_priv_op,
                    max_num_output_bins_uniform,
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
        }
        // Final tier dispatch
        if (max_num_output_bins_uniform > max_privatized_smem_bins)
        {
          constexpr int PRIVATIZED_SMEM_BINS = 0;
          if (const auto error = CubDebug(
                (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
                  d_temp_storage,
                  temp_storage_bytes,
                  d_samples,
                  d_output_histograms,
                  num_output_levels,
                  num_output_levels,
                  uniform_out_op,
                  uniform_priv_op,
                  max_num_output_bins_uniform,
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
        else
        {
          constexpr int PRIVATIZED_SMEM_BINS = max_privatized_smem_bins;
          if (const auto error = CubDebug(
                (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
                  d_temp_storage,
                  temp_storage_bytes,
                  d_samples,
                  d_output_histograms,
                  num_output_levels,
                  num_output_levels,
                  uniform_out_op,
                  uniform_priv_op,
                  max_num_output_bins_uniform,
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
      }
    }

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
    int max_num_output_bins = max_levels - 1;

    // For single-channel non-byte samples with bins in (256, 16384], use the
    // dyn-SMEM staging+fused tier. The fused kernel sweeps into per-block
    // dyn-SMEM, flushes to a per-block GMEM staging slab, and reduces across
    // blocks via cooperative_groups grid_group::sync().
    //
    // Three tiers cover the range:
    //   - medium: bins in (256, 2048],   8 KB dyn-SMEM/block (max occupancy).
    //   - large:  bins in (2048, 8192],  32 KB dyn-SMEM/block.
    //   - xlarge: bins in (8192, 16384], 64 KB dyn-SMEM/block.
    //
    // For multi-active-channel non-byte samples, the dyn-SMEM size scales with
    // NUM_ACTIVE_CHANNELS, so xlarge (16384*4*Nch bytes) exceeds B200's
    // ~228 KB per-CTA cap for Nch >= 4. We restrict multi-channel routing to
    // medium and large tiers (Nch * tier * 4 bytes <= ~96 KB).
    if constexpr (NUM_ACTIVE_CHANNELS == 1)
    {
      if (max_num_output_bins > max_privatized_smem_bins
          && max_num_output_bins <= max_extended_smem_bins_single_channel)
      {
        constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel;
        if (const auto error = CubDebug(
              (detail::histogram::dispatch<
                NUM_CHANNELS,
                NUM_ACTIVE_CHANNELS,
                PRIVATIZED_SMEM_BINS,
                false, false, false>(d_temp_storage, temp_storage_bytes, d_samples, d_output_histograms,
                                     num_output_levels, num_output_levels, output_decode_op, privatized_decode_op,
                                     max_num_output_bins, num_row_pixels, num_rows, row_stride_samples, stream,
                                     policy_selector, kernel_source, launcher_factory))))
        {
          return error;
        }
        return cudaSuccess;
      }
      if (max_num_output_bins > max_extended_smem_bins_single_channel
          && max_num_output_bins <= max_extended_smem_bins_single_channel_large)
      {
        constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel_large;
        if (const auto error = CubDebug(
              (detail::histogram::dispatch<
                NUM_CHANNELS,
                NUM_ACTIVE_CHANNELS,
                PRIVATIZED_SMEM_BINS,
                false, false, false>(d_temp_storage, temp_storage_bytes, d_samples, d_output_histograms,
                                     num_output_levels, num_output_levels, output_decode_op, privatized_decode_op,
                                     max_num_output_bins, num_row_pixels, num_rows, row_stride_samples, stream,
                                     policy_selector, kernel_source, launcher_factory))))
        {
          return error;
        }
        return cudaSuccess;
      }
      if (max_num_output_bins > max_extended_smem_bins_single_channel_large
          && max_num_output_bins <= max_extended_smem_bins_single_channel_xlarge)
      {
        constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel_xlarge;
        if (const auto error = CubDebug(
              (detail::histogram::dispatch<
                NUM_CHANNELS,
                NUM_ACTIVE_CHANNELS,
                PRIVATIZED_SMEM_BINS,
                false, // IsDeviceInit
                false, // IsEven (unused for host-init)
                false // IsByteSample (unused for host-init)
                >(d_temp_storage,
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
                  launcher_factory))))
        {
          return error;
        }
        return cudaSuccess;
      }
    }
    // Multi-channel dyn-SMEM staging tiers (range path). For multi-channel
    // SearchTransform-based dispatch the per-sample compute (binary search in
    // levels) dominates over atomic-add throughput, so the higher-occupancy
    // GMEM-priv path (PRIVATIZED_SMEM_BINS=0 + persistent kernel) often beats
    // the lower-occupancy SMEM-priv staging path. We therefore only enable the
    // medium tier (2048) for multi-channel range, where occupancy stays high
    // (Nch * 2048 * 4 = 24 KB at Nch=3, plenty of headroom).
    else if constexpr (NUM_ACTIVE_CHANNELS >= 2 && NUM_ACTIVE_CHANNELS <= 4)
    {
      if (max_num_output_bins > max_privatized_smem_bins
          && max_num_output_bins <= max_extended_smem_bins_single_channel)
      {
        constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel;
        if (const auto error = CubDebug(
              (detail::histogram::dispatch<
                NUM_CHANNELS,
                NUM_ACTIVE_CHANNELS,
                PRIVATIZED_SMEM_BINS,
                false, false, false>(d_temp_storage, temp_storage_bytes, d_samples, d_output_histograms,
                                     num_output_levels, num_output_levels, output_decode_op, privatized_decode_op,
                                     max_num_output_bins, num_row_pixels, num_rows, row_stride_samples, stream,
                                     policy_selector, kernel_source, launcher_factory))))
        {
          return error;
        }
        return cudaSuccess;
      }
    }
    // Dispatch
    if (max_num_output_bins > max_privatized_smem_bins)
    {
      // Too many bins to keep in shared memory.
      constexpr int PRIVATIZED_SMEM_BINS = 0;

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
              launcher_factory))))
      {
        return error;
      }
    }
    else
    {
      // Dispatch shared-privatized approach
      constexpr int PRIVATIZED_SMEM_BINS = max_privatized_smem_bins;

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
              launcher_factory))))
      {
        return error;
      }
    }
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

    using CommonT = typename TransformsT::ScaleTransform::CommonT;

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
    int max_num_output_bins = max_levels - 1;

    // Dyn-SMEM staging+fused tiers (single-channel: medium/large/xlarge;
    // multi-channel: medium/large only).
    if constexpr (NUM_ACTIVE_CHANNELS == 1)
    {
      if (max_num_output_bins > max_privatized_smem_bins
          && max_num_output_bins <= max_extended_smem_bins_single_channel)
      {
        constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel;
        if (const auto error = CubDebug(
              (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
                d_temp_storage, temp_storage_bytes, d_samples, d_output_histograms, num_output_levels,
                num_output_levels, output_decode_op, privatized_decode_op, max_num_output_bins, num_row_pixels,
                num_rows, row_stride_samples, stream, policy_selector, kernel_source, launcher_factory))))
        {
          return error;
        }
        return cudaSuccess;
      }
      if (max_num_output_bins > max_extended_smem_bins_single_channel
          && max_num_output_bins <= max_extended_smem_bins_single_channel_large)
      {
        constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel_large;
        if (const auto error = CubDebug(
              (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
                d_temp_storage, temp_storage_bytes, d_samples, d_output_histograms, num_output_levels,
                num_output_levels, output_decode_op, privatized_decode_op, max_num_output_bins, num_row_pixels,
                num_rows, row_stride_samples, stream, policy_selector, kernel_source, launcher_factory))))
        {
          return error;
        }
        return cudaSuccess;
      }
      if (max_num_output_bins > max_extended_smem_bins_single_channel_large
          && max_num_output_bins <= max_extended_smem_bins_single_channel_xlarge)
      {
        constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel_xlarge;
        if (const auto error = CubDebug(
              (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
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
                launcher_factory))))
        {
          return error;
        }
        return cudaSuccess;
      }

      // Chunked dyn-SMEM staging-fused tier: xlarge < bins <= chunked_smem_bins_max_single_channel.
      // EVEN single-channel only -- routes through the dyn-SMEM xlarge tier in chunks of
      // chunked_smem_chunk_size_single_channel bins, paying num_chunks x sample reads to swap
      // legacy GMEM-priv persistent kernel's GMEM atomicAdd_block for SMEM atomicAdd_block.
      if (max_num_output_bins > max_extended_smem_bins_single_channel_xlarge
          && max_num_output_bins <= chunked_smem_bins_max_single_channel)
      {
        if (const auto error =
              CubDebug((dispatch_chunked_staging_smem<NUM_CHANNELS,
                                                     NUM_ACTIVE_CHANNELS,
                                                     chunked_smem_chunk_size_single_channel,
                                                     chunked_smem_num_chunks_single_channel>(
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
                launcher_factory))))
        {
          return error;
        }
        return cudaSuccess;
      }
    }
    else if constexpr (NUM_ACTIVE_CHANNELS >= 2 && NUM_ACTIVE_CHANNELS <= 4)
    {
      if (max_num_output_bins > max_privatized_smem_bins
          && max_num_output_bins <= max_extended_smem_bins_single_channel)
      {
        constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel;
        if (const auto error = CubDebug(
              (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
                d_temp_storage, temp_storage_bytes, d_samples, d_output_histograms, num_output_levels,
                num_output_levels, output_decode_op, privatized_decode_op, max_num_output_bins, num_row_pixels,
                num_rows, row_stride_samples, stream, policy_selector, kernel_source, launcher_factory))))
        {
          return error;
        }
        return cudaSuccess;
      }
      if (max_num_output_bins > max_extended_smem_bins_single_channel
          && max_num_output_bins <= max_extended_smem_bins_single_channel_large)
      {
        constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel_large;
        if (const auto error = CubDebug(
              (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
                d_temp_storage, temp_storage_bytes, d_samples, d_output_histograms, num_output_levels,
                num_output_levels, output_decode_op, privatized_decode_op, max_num_output_bins, num_row_pixels,
                num_rows, row_stride_samples, stream, policy_selector, kernel_source, launcher_factory))))
        {
          return error;
        }
        return cudaSuccess;
      }
      if constexpr (NUM_ACTIVE_CHANNELS <= 3)
      {
        if (max_num_output_bins > max_extended_smem_bins_single_channel_large
            && max_num_output_bins <= max_extended_smem_bins_single_channel_xlarge)
        {
          constexpr int PRIVATIZED_SMEM_BINS = max_extended_smem_bins_single_channel_xlarge;
          if (const auto error = CubDebug(
                (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
                  d_temp_storage, temp_storage_bytes, d_samples, d_output_histograms, num_output_levels,
                  num_output_levels, output_decode_op, privatized_decode_op, max_num_output_bins, num_row_pixels,
                  num_rows, row_stride_samples, stream, policy_selector, kernel_source, launcher_factory))))
          {
            return error;
          }
          return cudaSuccess;
        }
      }
    }
    if (max_num_output_bins > max_privatized_smem_bins)
    {
      constexpr int PRIVATIZED_SMEM_BINS = 0;

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
              launcher_factory))))
      {
        return error;
      }
    }
    else
    {
      constexpr int PRIVATIZED_SMEM_BINS = max_privatized_smem_bins;

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
              launcher_factory))))
      {
        return error;
      }
    }
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
