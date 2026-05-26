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
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
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
          <<<persistent_grid_dims, dim3{static_cast<unsigned int>(threads_per_block)}, 0, stream>>>(
            d_samples,
            num_output_bins_wrapper,
            d_output_histograms,
            second_level_array,
            num_row_pixels,
            num_rows,
            row_stride_samples);
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
            reinterpret_cast<const void*>(direct_atomic_kernel_ptr),
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

      // Raise the dyn-SMEM cap on the fused kernel before launch (the previous
      // `set_max_dynamic_smem_size_for(sweep_kernel, ...)` call set it on the
      // staging-only kernel, but that's a different function pointer).
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
            reinterpret_cast<const void*>(fused_kernel_ptr),
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
