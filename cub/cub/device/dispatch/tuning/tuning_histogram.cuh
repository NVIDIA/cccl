// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_histogram.cuh>
#include <cub/block/block_load.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__host_stdlib/ostream>

CUB_NAMESPACE_BEGIN

//! The tuning policy for all algorithms in @ref DeviceHistogram.
struct HistogramPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int pixels_per_thread; //!< Number of pixels processed per thread
  int vec_size; //!< Vectorization size for loading samples
  BlockLoadAlgorithm load_algorithm; //!< The @ref BlockLoadAlgorithm used for loading samples from global memory
  CacheLoadModifier load_modifier; //!< The @ref CacheLoadModifier used for loading samples from global memory
  bool rle_compress; //!< Whether to perform localized RLE to compress samples before histogramming
  BlockHistogramMemoryPreference mem_preference; //!< Whether to prefer privatized shared-memory or global-memory bins,
                                                 //!< or a mix of both
  bool use_work_stealing; //!< Whether to dequeue tiles from a global work queue
  int init_kernel_pdl_trigger_max_bins; //!< Maximum number of bins for the init kernel to trigger the histogram kernel
                                        //!< early using PDL
  int direct_atomic_threads_per_block = 0; //!< Thread count for direct-atomic kernels; 0 inherits threads_per_block

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int direct_atomic_threads() const
  {
    return direct_atomic_threads_per_block != 0 ? direct_atomic_threads_per_block : threads_per_block;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const HistogramPolicy& lhs, const HistogramPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.pixels_per_thread == rhs.pixels_per_thread
        && lhs.vec_size == rhs.vec_size && lhs.load_algorithm == rhs.load_algorithm
        && lhs.load_modifier == rhs.load_modifier && lhs.rle_compress == rhs.rle_compress
        && lhs.mem_preference == rhs.mem_preference && lhs.use_work_stealing == rhs.use_work_stealing
        && lhs.init_kernel_pdl_trigger_max_bins == rhs.init_kernel_pdl_trigger_max_bins
        && lhs.direct_atomic_threads_per_block == rhs.direct_atomic_threads_per_block;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const HistogramPolicy& lhs, const HistogramPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const HistogramPolicy& p)
  {
    return os
        << "HistogramPolicy { .threads_per_block = " << p.threads_per_block << ", .pixels_per_thread = "
        << p.pixels_per_thread << ", .vec_size = " << p.vec_size << ", .load_algorithm = " << p.load_algorithm
        << ", .load_modifier = " << p.load_modifier << ", .rle_compress = " << p.rle_compress
        << ", .mem_preference = " << p.mem_preference << ", .use_work_stealing = " << p.use_work_stealing
        << ", .init_kernel_pdl_trigger_max_bins = " << p.init_kernel_pdl_trigger_max_bins
        << ", .direct_atomic_threads_per_block = " << p.direct_atomic_threads_per_block << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::histogram
{
// ---------------------------------------------------------------------------
// Device-tuning constants for the high-bin direct-atomic cache.
//
// These are the hardware-shaped knobs for the per-block SMEM cache that the
// direct-atomic-to-output kernels use to absorb cross-block contention. They
// live here, in the tuning header, so the dispatch sizing logic and the kernel
// bodies read ONE definition instead of repeating the literals (the count-replica
// factor in particular was previously duplicated across the host sizer and both
// direct-atomic kernels and had to be kept in sync by hand). Defaults are tuned
// for SM90/SM100 (B200) and can be revisited per architecture.
struct cache_tuning
{
  // Count-array replica factor for multi-active-channel direct-atomic caches.
  // The per-slot count array is split into this many warp-strided replicas so
  // cross-warp atomicAdd_block traffic on a hot slot is de-serialised; single
  // channel uses 1 (no replication, byte-identical to an unreplicated cache).
  // Kept compile-time (a runtime replica factor regressed the register-pinned
  // multi-channel kernel), so callers select via `replicas(num_active_channels)`.
  static constexpr int multi_channel_count_replicas = 4;

  _CCCL_HOST_DEVICE static constexpr int replicas(int num_active_channels)
  {
    return num_active_channels > 1 ? multi_channel_count_replicas : 1;
  }

  // Per-channel SMEM cache slot floor (power of two). The dispatch grows the
  // slot count above this only while it stays free of occupancy cost. Single
  // channel affords a larger floor than multi (which pays per active channel).
  static constexpr int slots_floor_single_channel = 4096;
  static constexpr int slots_floor_multi_channel  = 1024;

  _CCCL_HOST_DEVICE static constexpr int slots_floor(int num_active_channels)
  {
    return num_active_channels == 1 ? slots_floor_single_channel : slots_floor_multi_channel;
  }

  // Per-CTA opt-in dynamic-SMEM budget for the cache. The dispatch queries the
  // device's cudaDevAttrMaxSharedMemoryPerBlockOptin (B200/SM100 ~228 KiB) and
  // reserves `smem_reserve_bytes` for static/driver use; if the query fails it
  // falls back to `smem_fallback_bytes`.
  static constexpr int smem_fallback_bytes = 96 * 1024;
  static constexpr int smem_reserve_bytes  = 4096;
};

// TODO(bgruber): drop in CCCL 4.0
enum class primitive_sample
{
  no,
  yes
};

// TODO(bgruber): drop in CCCL 4.0
enum class sample_size
{
  _1,
  _2,
  _4,
  _8,
  unknown
};

// TODO(bgruber): drop in CCCL 4.0
enum class counter_size
{
  _4,
  unknown
};

// TODO(bgruber): drop in CCCL 4.0
template <class T>
_CCCL_HOST_DEVICE_API constexpr primitive_sample is_primitive_sample()
{
  return is_primitive<T>::value ? primitive_sample::yes : primitive_sample::no;
}

// TODO(bgruber): drop in CCCL 4.0
template <class CounterT>
_CCCL_HOST_DEVICE_API constexpr counter_size classify_counter_size()
{
  return sizeof(CounterT) == 4 ? counter_size::_4 : counter_size::unknown;
}

// TODO(bgruber): drop in CCCL 4.0
template <class SampleT>
_CCCL_HOST_DEVICE_API constexpr sample_size classify_sample_size()
{
  return sizeof(SampleT) == 1   ? sample_size::_1
       : sizeof(SampleT) == 2   ? sample_size::_2
       : sizeof(SampleT) == 4   ? sample_size::_4
       : sizeof(SampleT) == 8   ? sample_size::_8
                                : sample_size::unknown;
}

// TODO(bgruber): drop in CCCL 4.0
template <class SampleT,
          int NumChannels,
          int NumActiveChannels,
          counter_size CounterSize,
          primitive_sample PrimitiveSample = is_primitive_sample<SampleT>(),
          sample_size SampleSize           = classify_sample_size<SampleT>()>
struct sm90_tuning;

template <class SampleT>
struct sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  static constexpr int threads = 768;
  static constexpr int items   = 12;

  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool rle_compress      = false;
  static constexpr bool use_work_stealing = false;
};

template <class SampleT>
struct sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_2>
{
  static constexpr int threads = 960;
  static constexpr int items   = 10;

  static constexpr CacheLoadModifier load_modifier               = LOAD_DEFAULT;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool rle_compress      = true;
  static constexpr bool use_work_stealing = false;
};

// TODO(bgruber): drop in CCCL 4.0
template <bool IsEven,
          class SampleT,
          int NumChannels,
          int NumActiveChannels,
          counter_size CounterSize,
          primitive_sample PrimitiveSample = is_primitive_sample<SampleT>(),
          sample_size SampleSize           = classify_sample_size<SampleT>()>
struct sm100_tuning;

// even
template <class SampleT>
struct sm100_tuning<true, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  // ipt_12.tpb_928.rle_0.ws_0.mem_1.ld_2.laid_0.vec_2 1.033332  0.940517  1.031835  1.195876
  static constexpr int items                                     = 12;
  static constexpr int threads                                   = 928;
  static constexpr bool rle_compress                             = false;
  static constexpr bool use_work_stealing                        = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_CA;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_DIRECT;
  static constexpr int vec_size                                  = 1 << 2;
};

// sample_size 2/4/8 showed no benefit over SM90 during verification benchmarks

// range
template <class SampleT>
struct sm100_tuning<false, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  // ipt_12.tpb_448.rle_0.ws_0.mem_1.ld_1.laid_0.vec_2 1.078987  0.985542  1.085118  1.175637
  static constexpr int items                                     = 12;
  static constexpr int threads                                   = 448;
  static constexpr bool rle_compress                             = false;
  static constexpr bool use_work_stealing                        = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_DIRECT;
  static constexpr int vec_size                                  = 1 << 2;
};

// SM100 sample_size 4 (I32) single-channel non-byte tuning. The default Policy500 fallback
// {384 threads, t_scale(16)=16 ipt} is suboptimal for the dyn-SMEM 16384-bin tier where
// SMEM atomicAdd_block contention dominates. Use 768 threads / 12 ipt (matching SM90 sample_size=1
// shape) to spread atomic contention across more concurrent issues per CTA.
//
// Verified empirically vs 512/12 (-1.27%), 928/12 (-0.79%), 768/16 (-1.43%); 768/12 wins.
// LOAD_LDG was verified vs LOAD_CA (-0.50%); LOAD_LDG wins. BLOCK_LOAD_VECTORIZE slightly beats
// BLOCK_LOAD_DIRECT (+0.08%, within noise).
template <bool IsEven, class SampleT>
struct sm100_tuning<IsEven, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_4>
{
  static constexpr int items                                     = 12;
  static constexpr int threads                                   = 768;
  static constexpr bool rle_compress                             = true;
  static constexpr bool work_stealing                            = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_VECTORIZE;
  static constexpr int vec_size                                  = 1 << 2;
};

// SM100 sample_size 8 (F64) single-channel non-byte tuning. F64 has half the throughput per byte
// and the dyn-SMEM 16384 tier already saturates at ~5 TB/s for F64 entropy=1.0; aim for a balanced
// {threads, ipt} that doesn't regress lower-bin tiers either.
//
// Verified empirically vs 768/8 (-0.37% overall, but -3% on range): 512/8 wins because it
// trades a small even-path regression for a larger range-path improvement.
template <bool IsEven, class SampleT>
struct sm100_tuning<IsEven, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_8>
{
  static constexpr int items                                     = 8;
  static constexpr int threads                                   = 512;
  static constexpr bool rle_compress                             = true;
  static constexpr bool work_stealing                            = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_VECTORIZE;
  static constexpr int vec_size                                  = 1 << 2;
};

// multi.even and multi.range: none of the found tunings surpassed the SM90 tuning during verification benchmarks

// TODO(bgruber): drop in CCCL 4.0
template <class SampleT, class CounterT, int NumChannels, int NumActiveChannels, bool IsEven>
struct policy_hub
{
  // TODO(bgruber): move inside t_scale in C++14
  static constexpr int v_scale = (sizeof(SampleT) + sizeof(int) - 1) / sizeof(int);

  _CCCL_HOST_DEVICE_API static constexpr int t_scale(int nominalItemsPerThread)
  {
    return (::cuda::std::max) (nominalItemsPerThread / NumActiveChannels / v_scale, 1);
  }

  // SM50
  struct Policy500 : detail::chained_policy<500, Policy500, Policy500>
  {
    // TODO This might be worth it to separate usual histogram and the multi one
    using AgentHistogramPolicyT =
      agent_histogram_policy<384, t_scale(16), BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false>;
  };

  // SM90
  struct Policy900 : detail::chained_policy<900, Policy900, Policy500>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy500
    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(int)
      -> agent_histogram_policy<Tuning::threads,
                                Tuning::items,
                                Tuning::load_algorithm,
                                Tuning::load_modifier,
                                Tuning::rle_compress,
                                Tuning::mem_preference,
                                Tuning::use_work_stealing>;

    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(long) -> typename Policy500::AgentHistogramPolicyT;

    using AgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm90_tuning<SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(0));

    static constexpr int init_kernel_pdl_trigger_max_bins = 2048;
  };

  struct Policy1000 : detail::chained_policy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(int) -> agent_histogram_policy<
      Tuning::threads,
      Tuning::items,
      Tuning::load_algorithm,
      Tuning::load_modifier,
      Tuning::rle_compress,
      Tuning::mem_preference,
      Tuning::use_work_stealing,
      Tuning::vec_size>;

    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(long) -> typename Policy900::AgentHistogramPolicyT;

    using AgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm100_tuning<IsEven, SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(
        0));

    static constexpr int init_kernel_pdl_trigger_max_bins = 2048;
  };

  using MaxPolicy = Policy1000;
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept histogram_policy_selector = policy_selector<T, HistogramPolicy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  bool sample_is_primitive;
  int sample_size;
  int counter_size;
  int sample_size_bytes;
  int num_channels;
  int num_active_channels;
  bool is_even;

private:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int t_scale(int nominal_items_per_thread) const
  {
    const int sample_scale = (sample_size_bytes + int{sizeof(int)} - 1) / int{sizeof(int)};
    return (::cuda::std::max) (nominal_items_per_thread / num_active_channels / sample_scale, 1);
  }

public:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> HistogramPolicy
  {
    if (cc >= ::cuda::compute_capability{10, 0})
    {
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive && sample_size == 1)
      {
        if (is_even)
        {
          // ipt_12.tpb_928.rle_0.ws_0.mem_1.ld_2.laid_0.vec_2 1.033332  0.940517  1.031835  1.195876
          return HistogramPolicy{928, 12, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_CA, false, SMEM, false, 2048};
        }
        else
        {
          // ipt_12.tpb_448.rle_0.ws_0.mem_1.ld_1.laid_0.vec_2 1.078987  0.985542  1.085118  1.175637
          return HistogramPolicy{448, 12, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_LDG, false, SMEM, false, 2048};
        }
      }

      // SM100 single-channel NON-byte tuning (I32 sample_size==4, F64 sample_size==8).
      //
      // Before this arm, single-channel I32/F64 fell all the way through to the SM50
      // Policy500 fallback {384 threads, t_scale(16) ipt, rle=true, LDG} -- a launch
      // shape never tuned for SM100. (The byte-sample arm above only matches
      // sample_size==1, and the SM90 arm below only matches sample_size 1/2; the
      // sm100_tuning sample_size _4/_8 structs are read ONLY by
      // policy_selector_from_max_policy, which DeviceHistogram's env API does not use --
      // the benchmark drives policy_selector_from_types, i.e. THIS struct.)
      //
      // This non-byte single-channel policy drives the SMEM-priv sweep AND the high-bin
      // direct-atomic (cuckoo / single-probe) kernels. A wider launch than 384 threads
      // gives those kernels more resident warps to hide SMEM-atomic and SearchTransform
      // latency, mirroring the multi-channel SM100 arm below. EVEN and RANGE are
      // decoupled per transform (RANGE's per-sample binary search is latency-heavier).
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive
          && (sample_size == 4 || sample_size == 8))
      {
        if (is_even)
        {
          // EVEN: 768 threads. I32 -> 12 ipt, F64 -> 6 ipt (t_scale(12)).
          return histogram_policy{768, t_scale(12), BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 1 << 2, 2048};
        }
        else
        {
          // RANGE: 768 threads for the SMEM-priv sweep tiers (bins 64/2000/16384).
          //
          // direct_atomic_threads_per_block=512: the high-bin direct-atomic
          // (cuckoo / single-probe) single-channel RANGE cells run a SEPARATE
          // kernel from the SMEM-priv sweep -- the cuckoo kernel for bins>65536
          // (and all F64 high-bin) at <256M pixels, and the single-probe kernel
          // for 1M bins at >=256M pixels. These atomic straight to the output via
          // a pure grid-stride loop, so their block size is decoupled from the
          // sweep without affecting correctness.
          //
          // A structured thread sweep over the 57 direct-atomic cells (I32+F64,
          // bins 65536/262144/1M, all entropy/input-size; per-cell GiB/s geomean,
          // measured against an identical build differing only in this field so
          // the ~140 unaffected cells stay fixed at ratio 1.000) is UNIMODAL with
          // a clear peak at 512: 256=0.814, 384=0.971, 512=1.024, 768=1.000
          // (relative to the 768-thread inherit). 512 beats the inherited 768 by
          // +2.4% and 384 by +5.4%; both the cuckoo and single-probe sub-groups
          // agree on 512.
          //
          // ncu on the 1M-bin/256M/uniform single-probe cell explains it: at 768
          // threads (30 regs) the kernel is pinned to 2 blocks/SM (Block Limit
          // Registers/SMEM/Warps all == 2) -> 75% achieved occupancy, 17.9%
          // issue-slot utilisation, 79.9% of warp cycles stalled on the
          // long-scoreboard SMEM-cache atomic dependency (44.8 GB/s, latency- not
          // bandwidth-bound). At 512 threads (28 regs) all three limiters jump to
          // 4 blocks/SM -> 100% achieved occupancy (63.9 warps), 47.1 GB/s, 23.0
          // vs 24.2 ms: the extra co-resident blocks (each its own dynamic-SMEM
          // cache partition + a shorter per-block CAS/atomicAdd_block dependency
          // chain) hide the scoreboard-stall-dominated latency. 384/256 starve the
          // issue pipeline (too few warps/block); 768 caps occupancy at 75%.
          //
          // Note this differs from the multi-channel RANGE single-probe path
          // (worker-3 brief-4), which peaked at 384: there the 3-active-channel
          // SearchTransform makes it latency-bound and the 1024-slot/channel cache
          // is small, so the SM wanted even more, smaller blocks. Single-channel
          // RANGE has one SearchTransform and a 4096-slot cache, so it is more
          // throughput-bound and 512 (more warps/block) wins -- the per-transform
          // decouple is itself per-channel-count.
          //
          // The sweep tiers keep the 768-thread shape (SMEM-priv occupancy-bound,
          // not direct-atomic latency-bound). EVEN inherits its sweep thread count
          // (override stays 0): its cheap ScaleTransform makes the high-bin
          // direct-atomic cells throughput-bound, where the wider block is fine.
          return histogram_policy{768, t_scale(12), BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 1 << 2, 2048, 512};
        }
      }

      // sample_size 2 showed no benefit over SM90 during verification benchmarks

      // SM100 multi-channel (num_channels >= 2) tuning, decoupled per transform.
      // Previously every multi-channel configuration fell through to the SM50
      // Policy500 fallback {384 threads, t_scale(16) ipt, rle=true} -- a launch
      // shape never tuned for SM90/SM100. That shape is genuinely strong for the
      // EVEN path (cheap ScaleTransform classify => the SMEM-priv tiers are
      // contention-bound on shared-memory atomics, where intra-thread RLE
      // compression of same-bin runs plus a modest 384-thread launch minimise
      // atomic pressure), so EVEN keeps the fallback shape verbatim. The RANGE
      // path is different: its per-sample SearchTransform binary search over the
      // level boundaries is latency-heavy and paid per active channel, so the
      // sweep is classify-bound rather than atomic-bound and a wider launch hides
      // that latency. We therefore give RANGE a wider 768-thread shape while
      // keeping rle=true (free when runs are absent, e.g. uniform entropy).
      if (num_channels >= 2 && counter_size == 4 && sample_is_primitive)
      {
        if (!is_even)
        {
          // RANGE: 1024 threads. worker-3 found 768==1024 within noise on the
          // SMEM-priv mid-bin sweep cells (where 768 gets better occupancy), but
          // this policy ALSO drives the high-bin direct-atomic (cuckoo/single-probe)
          // kernels, which atomic directly to the output and are GMEM-atomic/
          // classify-latency bound rather than SMEM-priv occupancy bound. A wider
          // 1024-thread launch gives those kernels more resident warps to hide that
          // latency (matching EVEN's 1024 pick). rle=true is free. Keep LOAD_LDG:
          // an LOAD_CA ablation regressed multi_range -26.2% (cache-all thrashes
          // L1/L2 caching the SearchTransform level-array + sample loads across the
          // 3 active channels on the SMEM-priv mid-bin sweep cells; LDG's streaming
          // loads avoid the eviction churn).
          //
          // direct_atomic_threads_per_block=384 (COMBINE / brief-8): the
          // multi-channel direct-atomic RANGE decouple from worker-3 brief-4
          // (parent C 614970dd). The high-bin direct-atomic cuckoo/single-probe
          // kernels share this policy but run a SEPARATE grid-stride kernel from
          // the SMEM-priv sweep. ncu on the 1M-bin/256M/uniform single-probe
          // cell at 1024 threads showed ~100% achieved occupancy (2 blocks/SM,
          // 64 warps) but 89.6%-of-cycles long-scoreboard SMEM-cache atomic
          // stalls at 15% issue util: 64 warps hammer ONE block's single-probe
          // cache. A 384-thread block lets the SM hold more, smaller blocks,
          // each with its own dynamic-SMEM cache partition and a shorter atomic
          // dependency chain. The SMEM-priv mid-bin sweep tiers keep the
          // 1024-thread shape (threads_per_block stays 1024); only the
          // direct-atomic kernels see 384. iter4: C's 384 REVERTED to 0 (inherit
          // 1024) -- decompose found C's 384 and cuckoo R=2 are antagonistic on
          // multi_range (the larger R=2 per-slot count footprint + a 384-thread
          // launch together starve the cuckoo cache). Keep cuckoo R=2 at 1024 to
          // capture B's multi_range gain without the antagonism.
          return histogram_policy{1024, t_scale(16), BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 4, 0, 0};
        }
        else
        {
          // EVEN: 1024 threads. The SMEM-priv even sweep was pinned to 1 block/SM
          // (32 warps, 50% occ): at t_scale(16) the per-thread accumulate holds
          // samples[pixels_per_thread][NumChannels] + bins[pixels_per_thread]
          // (for I32, 3 active channels: 5*4 + 5 = 25 live ints) which compiles to
          // ~58 registers => 58*2048 > 65536 regs/SM, so only one CTA is resident
          // and there is no second block to hide the shared-memory atomicAdd
          // latency this contention-bound sweep is dominated by.
          //
          // Halve the nominal items (t_scale(8): 2 pixels/thread for I32, 1 for
          // F64). That shrinks the live samples/bins arrays enough for ptxas to
          // hold the kernel in <= 32 registers WITHOUT spilling (forcing 32 regs
          // at t_scale(16) instead spills ~88 B and nets a regression -- measured),
          // which together with the DeviceHistogramSweepKernel __launch_bounds__
          // min-blocks=2 hint admits a 2nd resident CTA (64 warps, 100% occ) on
          // the register-limited low-bin even tiers (256/2048 bins, 4 KB/25 KB
          // SMEM -- registers, not SMEM, gate occupancy here). rle=true is
          // preserved (load-bearing: dropping the same-bin RLE coalescing
          // collapses multi_even). LOAD_CA matches the single-channel SM100 even
          // tuning.
          return histogram_policy{1024, t_scale(8), BLOCK_LOAD_DIRECT, LOAD_CA, true, SMEM, false, 4, 0};
        }
      }
    }

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive)
      {
        if (sample_size == 1)
        {
          return HistogramPolicy{768, 12, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_LDG, false, SMEM, false, 2048};
        }
        else if (sample_size == 2)
        {
          return HistogramPolicy{960, 10, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, true, SMEM, false, 2048};
        }
      }
    }

    // fallback from SM50
    return HistogramPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 0};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(histogram_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <class SampleT, class CounterT, int NumChannels, int NumActiveChannels, bool IsEven>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> HistogramPolicy
  {
    constexpr auto policies = policy_selector{
      is_primitive_v<SampleT>,
      int{sizeof(SampleT)},
      int{sizeof(CounterT)},
      int{sizeof(SampleT)},
      NumChannels,
      NumActiveChannels,
      IsEven};
    return policies(cc);
  }
};
} // namespace detail::histogram

CUB_NAMESPACE_END
