// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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
#include <cub/device/dispatch/tuning/tuning_histogram.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_type.cuh>

#include <cuda/__numeric/sub_overflow.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__numeric/reduce.h>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cooperative_groups.h>

CUB_NAMESPACE_BEGIN
namespace detail::histogram
{
template <typename LevelT, typename OffsetT, typename SampleT>
struct Transforms
{
  //---------------------------------------------------------------------
  // Transform functors for converting samples to bin-ids
  //---------------------------------------------------------------------

  // Searches for bin given a list of bin-boundary levels
  template <typename LevelIteratorT>
  struct SearchTransform
  {
    LevelIteratorT d_levels; // Pointer to levels array
    int num_output_levels; // Number of levels in array

    //! @brief Initializer
    //!
    //! @param d_levels_ Pointer to levels array
    //! @param num_output_levels_ Number of levels in array
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(LevelIteratorT d_levels_, int num_output_levels_)
    {
      this->d_levels          = d_levels_;
      this->num_output_levels = num_output_levels_;
    }

    // Method for converting samples to bin-ids
    template <CacheLoadModifier LoadModifier, typename Sample>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(Sample sample, int& bin, bool valid) const
    {
      /// Level iterator wrapper type
      // Wrap the native input pointer with CacheModifiedInputIterator
      // or Directly use the supplied input iterator type
      using WrappedLevelIteratorT =
        ::cuda::std::_If<::cuda::std::is_pointer_v<LevelIteratorT>,
                         CacheModifiedInputIterator<LoadModifier, LevelT, OffsetT>,
                         LevelIteratorT>;

      const WrappedLevelIteratorT wrapped_levels(d_levels);

      const int num_bins = num_output_levels - 1;
      if (valid)
      {
        bin = UpperBound(wrapped_levels, num_output_levels, static_cast<LevelT>(sample)) - 1;
        if (bin >= num_bins)
        {
          bin = -1;
        }
      }
    }
  };

  // Scales samples to evenly-spaced bins
  struct ScaleTransform
  {
    using CommonT = ::cuda::std::common_type_t<LevelT, SampleT>;
    static_assert(::cuda::std::is_convertible_v<CommonT, int>,
                  "The common type of `LevelT` and `SampleT` must be "
                  "convertible to `int`.");
    static_assert(::cuda::is_trivially_copyable_v<CommonT>,
                  "The common type of `LevelT` and `SampleT` must be "
                  "trivially copyable.");

    // An arithmetic type that's used for bin computation of integral types, guaranteed to not
    // overflow for (max_level - min_level) * scale.fraction.bins. Since we drop invalid samples
    // of less than min_level, (sample - min_level) is guaranteed to be non-negative. We use the
    // rule: 2^l * 2^r = 2^(l + r) to determine a sufficiently large type to hold the
    // multiplication result.
    // If CommonT used to be a 128-bit wide integral type already, we use CommonT's arithmetic
    using IntArithmeticT = ::cuda::std::_If< //
      sizeof(SampleT) + sizeof(CommonT) <= sizeof(uint32_t), //
      uint32_t, //
#if _CCCL_HAS_INT128()
      ::cuda::std::_If< //
        (::cuda::std::is_same_v<CommonT, __int128_t> || //
         ::cuda::std::is_same_v<CommonT, __uint128_t>), //
        CommonT, //
        uint64_t> //
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
      uint64_t
#endif // !_CCCL_HAS_INT128()
      >;

  private:
    // Alias template that excludes __[u]int128 from the integral types
    template <typename T>
    using is_integral_excl_int128 =
#if _CCCL_HAS_INT128()
      ::cuda::std::_If<::cuda::std::is_same_v<T, __int128_t> || ::cuda::std::is_same_v<T, __uint128_t>,
                       ::cuda::std::false_type,
                       ::cuda::std::is_integral<T>>;
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
      ::cuda::std::is_integral<T>;
#endif // !_CCCL_HAS_INT128()

    // prefer uint32 for performance reasons when possible
    // bool, 8-bit, 16-bit, 32-bit integers -> uint32_t
    // 64-bit integers                      -> uint64_t
    // Other types                          -> IntArithmeticT
    [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr auto FractionStorageType()
    {
      if constexpr (is_integral_excl_int128<CommonT>::value)
      {
        if constexpr (sizeof(CommonT) < sizeof(uint32_t))
        {
          return uint32_t{};
        }
        else
        {
          return ::cuda::std::make_unsigned_t<CommonT>{};
        }
      }
      else
      {
        return IntArithmeticT{};
      }
    }

    using FractionStorageT = decltype(FractionStorageType());

    template <typename T>
    [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static auto subtract_as_unsigned(T lhs, T rhs) noexcept
    {
      if constexpr (::cuda::std::is_same_v<T, bool>)
      {
        return ::cuda::__sub_as_unsigned<uint8_t>(lhs, rhs);
      }
      else
      {
        return ::cuda::__sub_as_unsigned<::cuda::std::make_unsigned_t<T>>(lhs, rhs);
      }
    }

    union ScaleT
    {
      // Used when CommonT is not floating-point to avoid intermediate
      // rounding errors (see NVIDIA/cub#489).
      struct FractionT
      {
        FractionStorageT bins;
        FractionStorageT range;
      } fraction;

      // Used when CommonT is floating-point as an optimization.
      CommonT reciprocal;
    };

    CommonT m_max; // Max sample level (exclusive)
    CommonT m_min; // Min sample level (inclusive)
    ScaleT m_scale; // Bin scaling

    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScaleT
    ComputeScale(int num_levels, T max_level, T min_level, ::cuda::std::true_type /* is_fp */)
    {
      // The active scale representation is assigned below.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      ScaleT result;
      result.reciprocal = static_cast<T>(static_cast<T>(num_levels - 1) / static_cast<T>(max_level - min_level));
      return result;
    }

    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScaleT
    ComputeScale(int num_levels, T max_level, T min_level, ::cuda::std::false_type /* is_fp */)
    {
      // The active scale representation is assigned below.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      ScaleT result;
      result.fraction.bins = static_cast<FractionStorageT>(num_levels - 1);
      if constexpr (is_integral_excl_int128<T>::value)
      {
        result.fraction.range = FractionStorageT{subtract_as_unsigned(max_level, min_level)};
      }
      else
      {
        result.fraction.range = static_cast<FractionStorageT>(max_level) - static_cast<FractionStorageT>(min_level);
      }
      return result;
    }

    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScaleT ComputeScale(int num_levels, T max_level, T min_level)
    {
      return this->ComputeScale(num_levels, max_level, min_level, ::cuda::std::is_floating_point<T>{});
    }

#if _CCCL_HAS_NVFP16()
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScaleT ComputeScale(int num_levels, __half max_level, __half min_level)
    {
      // The active scale representation is assigned below.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      ScaleT result;
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
                        (result.reciprocal = __hdiv(__float2half(num_levels - 1), __hsub(max_level, min_level));),
                        (result.reciprocal = __float2half(
                           static_cast<float>(num_levels - 1) / (__half2float(max_level) - __half2float(min_level)));))
      return result;
    }
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
    _CCCL_HOST_DEVICE
    _CCCL_FORCEINLINE ScaleT ComputeScale(int num_levels, __nv_bfloat16 max_level, __nv_bfloat16 min_level)
    {
      // The active scale representation is assigned below.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      ScaleT result;
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_80,
        (result.reciprocal = __hdiv(__float2bfloat16(num_levels - 1), __hsub(max_level, min_level));),
        (result.reciprocal = __float2bfloat16(
           static_cast<float>(num_levels - 1) / (__bfloat162float(max_level) - __bfloat162float(min_level)));))
      return result;
    }
#endif // _CCCL_HAS_NVBF16()

    template <typename T>
    [[nodiscard]] _CCCL_HOST_DEVICE_API
      _CCCL_FORCEINLINE int SampleIsValid(T sample, T max_level, T min_level) const noexcept
    {
#if _CCCL_HAS_NVFP16()
      if constexpr (::cuda::std::is_same_v<T, ::__half>)
      { // NOLINT(bugprone-branch-clone)
        NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
                          (return ::__hge(sample, min_level) && ::__hlt(sample, max_level);),
                          (return ::__half2float(sample) >= ::__half2float(min_level)
                                 && ::__half2float(sample) < ::__half2float(max_level);));
      }
      else
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
        if constexpr (::cuda::std::is_same_v<T, ::__nv_bfloat16>)
      {
        NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
                          (return ::__hge(sample, min_level) && ::__hlt(sample, max_level);),
                          (return ::__bfloat162float(sample) >= ::__bfloat162float(min_level)
                                 && ::__bfloat162float(sample) < ::__bfloat162float(max_level);));
      }
      else
#endif // _CCCL_HAS_NVBF16()
      {
        return sample >= min_level && sample < max_level;
      }
    }

    template <typename T>
    [[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int
    ComputeBin(T sample, T min_level, ScaleT scale) const noexcept
    {
      if constexpr (is_integral_excl_int128<T>::value)
      {
        const auto offset = subtract_as_unsigned(sample, min_level);
        return static_cast<int>(
          (IntArithmeticT{offset} * IntArithmeticT{scale.fraction.bins}) / IntArithmeticT{scale.fraction.range});
      }
#if _CCCL_HAS_NVFP16()
      else if constexpr (::cuda::std::is_same_v<T, ::__half>)
      {
        NV_IF_ELSE_TARGET(
          NV_PROVIDES_SM_53,
          (return static_cast<int>(::__hmul(::__hsub(sample, min_level), scale.reciprocal));),
          (return static_cast<int>(
                    (::__half2float(sample) - ::__half2float(min_level)) * ::__half2float(scale.reciprocal));));
      }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
      else if constexpr (::cuda::std::is_same_v<T, ::__nv_bfloat16>)
      {
        // Compute in float on all architectures: bfloat16 cannot represent all bin indices beyond 256.
        return static_cast<int>(
          (::__bfloat162float(sample) - ::__bfloat162float(min_level)) * ::__bfloat162float(scale.reciprocal));
      }
#endif // _CCCL_HAS_NVBF16()
      else if constexpr (::cuda::std::is_floating_point_v<T>)
      {
        return static_cast<int>((sample - min_level) * scale.reciprocal);
      }
      else
      {
        // Custom types and __[u]int128
        return static_cast<int>(((sample - min_level) * static_cast<CommonT>(scale.fraction.bins))
                                / static_cast<CommonT>(scale.fraction.range));
      }
    }

  public:
    //! @brief Initializes the ScaleTransform for the given parameters
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(int num_levels, LevelT max_level, LevelT min_level)
    {
      m_max = static_cast<CommonT>(max_level);
      m_min = static_cast<CommonT>(min_level);

      m_scale = this->ComputeScale(num_levels, m_max, m_min);
    }

    // Method for converting samples to bin-ids. The sample type is a template parameter because the
    // agent also feeds privatized bin indices through this op, which must not round-trip through SampleT.
    template <CacheLoadModifier LoadModifier, typename Sample>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(Sample sample, int& bin, bool valid) const
    {
      const CommonT common_sample = static_cast<CommonT>(sample);

      if (valid && this->SampleIsValid(common_sample, m_max, m_min))
      {
        bin = this->ComputeBin(common_sample, m_min, m_scale);
      }
    }
  };

  // Pass-through bin transform operator
  struct PassThruTransform
  {
// GCC 14 rightfully warns that when a value-initialized array of this struct is copied using memcpy, uninitialized
// bytes may be accessed. To avoid this, we add a dummy member, so value initialization actually initializes the memory.
#if _CCCL_COMPILER(GCC, >=, 13)
    char dummy;
#endif

    // No-op Init for uniformity with ScaleTransform
    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(int, T, T)
    {}

    // No-op Init for uniformity with SearchTransform
    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(T, int)
    {}

    // Method for converting samples to bin-ids
    template <CacheLoadModifier LoadModifier, typename Sample>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(Sample sample, int& bin, bool valid) const
    {
      if (valid)
      {
        bin = static_cast<int>(sample);
      }
    }
  };
};

/******************************************************************************
 * Histogram kernel entry points
 *****************************************************************************/

//! Histogram initialization kernel entry point
//!
//! @tparam PolicySelector
//!   Selects the tuning policy
//!
//! @tparam NumActiveChannels
//!   Number of channels actively being histogrammed
//!
//! @tparam CounterT
//!   Integer type for counting sample occurrences per histogram bin
//!
//! @tparam OffsetT
//!   Signed integer type for global offsets
//!
//! @param num_output_bins_wrapper
//!   Number of output histogram bins per channel
//!
//! @param d_output_histograms_wrapper
//!   Histogram counter data having logical dimensions `CounterT[NUM_ACTIVE_CHANNELS][num_bins.array[CHANNEL]]`
//!
//! @param tile_queue
//!   Drain queue descriptor for dynamically mapping tile data onto thread blocks
template <typename PolicySelector, int NumActiveChannels, typename CounterT, typename OffsetT>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
_CCCL_KERNEL_ATTRIBUTES void DeviceHistogramInitKernel(
  ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
  ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
  GridQueue<int> tile_queue)
{
  [[maybe_unused]] static constexpr HistogramPolicy policy = current_policy<PolicySelector>();
  _CCCL_PDL_GRID_DEPENDENCY_SYNC(); // TODO(bgruber): if we had the guarantee that there would be no pending
                                    // writes/reads to the temp storage, we could omit the sync here

  // we trigger the sweep kernel only if we have a small number of remaining writes in this kernel
  NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                 if (::cuda::std::reduce(num_output_bins_wrapper.begin(), num_output_bins_wrapper.end())
                     <= policy.init_kernel_pdl_trigger_max_bins)
                 {
                   _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
                 }
               }));

  if ((threadIdx.x == 0) && (blockIdx.x == 0))
  {
    tile_queue.ResetDrain();
  }

  const int output_bin = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    if (output_bin < num_output_bins_wrapper[ch])
    {
      d_output_histograms_wrapper[ch][output_bin] = 0;
    }
  }
}

//! Histogram privatized sweep kernel entry point (multi-block).
//! Computes privatized histograms, one per thread block.
//! This kernel receives pre-initialized decode operators from the host.
//!
//! @tparam PolicySelector
//!   Selects the tuning policy
//!
//! @tparam PrivatizedSmemBins
//!   Maximum number of histogram bins per channel (e.g., up to 256)
//!
//! @tparam NumChannels
//!   Number of channels interleaved in the input data (may be greater than the number of channels
//!   being actively histogrammed)
//!
//! @tparam NumActiveChannels
//!   Number of channels actively being histogrammed
//!
//! @tparam SampleIteratorT
//!   The input iterator type. @iterator.
//!
//! @tparam CounterT
//!   Integer type for counting sample occurrences per histogram bin
//!
//! @tparam PrivatizedDecodeOpT
//!   The transform operator type for determining privatized counter indices from samples,
//!   one for each channel
//!
//! @tparam OutputDecodeOpT
//!   The transform operator type for determining output bin-ids from privatized counter indices,
//!   one for each channel
//!
//! @tparam OffsetT
//!   Integer type for global offsets
//!
//! @param d_samples
//!   Input data to reduce
//!
//! @param num_output_bins_wrapper
//!   The number of bins per final output histogram
//!
//! @param num_privatized_bins_wrapper
//!   The number of bins per privatized histogram
//!
//! @param d_output_histograms_wrapper
//!   Reference to final output histograms
//!
//! @param d_privatized_histograms_wrapper
//!   Reference to privatized histograms
//!
//! @param output_decode_op_wrapper
//!   The transform operator for determining output bin-ids from privatized counter indices,
//!   one for each channel (pre-initialized on host)
//!
//! @param privatized_decode_op_wrapper
//!   The transform operator for determining privatized counter indices from samples,
//!   one for each channel (pre-initialized on host)
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
//!   Drain queue descriptor for dynamically mapping tile data onto thread blocks
template <typename PolicySelector,
          int PrivatizedSmemBins,
          int NumChannels,
          int NumActiveChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepKernel(
    const SampleIteratorT d_samples,
    const ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    const ::cuda::std::array<int, NumActiveChannels> num_privatized_bins_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
    const ::cuda::std::array<OutputDecodeOpT, NumActiveChannels> output_decode_op_wrapper,
    const ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op_wrapper,
    const OffsetT num_row_pixels,
    const OffsetT num_rows,
    const OffsetT row_stride_samples,
    const int tiles_per_row,
    GridQueue<int> tile_queue)
{
  static constexpr HistogramPolicy hp = current_policy<PolicySelector>();

  // Thread block type for compositing input tiles
  using AgentHistogramPolicyT = agent_histogram_policy<
    hp.threads_per_block,
    hp.pixels_per_thread,
    hp.load_algorithm,
    hp.load_modifier,
    hp.rle_compress,
    hp.mem_preference,
    hp.use_work_stealing,
    hp.vec_size>;
  using AgentHistogramT =
    AgentHistogram<AgentHistogramPolicyT,
                   PrivatizedSmemBins,
                   NumChannels,
                   NumActiveChannels,
                   SampleIteratorT,
                   CounterT,
                   PrivatizedDecodeOpT,
                   OutputDecodeOpT,
                   OffsetT>;

  // Shared memory for AgentHistogram
  __shared__ typename AgentHistogramT::TempStorage temp_storage;

  AgentHistogramT agent(
    temp_storage,
    d_samples,
    num_output_bins_wrapper.data(),
    num_privatized_bins_wrapper.data(),
    d_output_histograms_wrapper.data(),
    d_privatized_histograms_wrapper.data(),
    output_decode_op_wrapper.data(),
    privatized_decode_op_wrapper.data());

  // Initialize counters
  agent.InitBinCounters();

  // Consume input tiles
  agent.ConsumeTiles(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);

  // Store output to global (if necessary)
  agent.StoreOutput();
}

//! Histogram privatized sweep kernel entry point (multi-block) with device-side initialization.
//! Computes privatized histograms, one per thread block.
//! This kernel initializes decode operators from level arrays inside the kernel.
//!
//! @tparam PolicySelector
//!   Selects the tuning policy
//!
//! @tparam PrivatizedSmemBins
//!   Maximum number of histogram bins per channel (e.g., up to 256)
//!
//! @tparam NumChannels
//!   Number of channels interleaved in the input data (may be greater than the number of channels
//!   being actively histogrammed)
//!
//! @tparam NumActiveChannels
//!   Number of channels actively being histogrammed
//!
//! @tparam SampleIteratorT
//!   The input iterator type. @iterator.
//!
//! @tparam CounterT
//!   Integer type for counting sample occurrences per histogram bin
//!
//! @tparam FirstLevelArrayT
//!   For DispatchEven: array of upper level bounds per channel.
//!   For DispatchRange: array of number of output levels per channel.
//!
//! @tparam SecondLevelArrayT
//!   For DispatchEven: array of lower level bounds per channel.
//!   For DispatchRange: array of level pointers per channel.
//!
//! @tparam PrivatizedDecodeOpT
//!   The transform operator type for determining privatized counter indices from samples,
//!   one for each channel
//!
//! @tparam OutputDecodeOpT
//!   The transform operator type for determining output bin-ids from privatized counter indices,
//!   one for each channel
//!
//! @tparam OffsetT
//!   Integer type for global offsets
//!
//! @tparam IsEven
//!   Whether this is a HistogramEven dispatch (true) or HistogramRange dispatch (false).
//!   Affects how decode operators are initialized from the level arrays.
//!
//! @param d_samples
//!   Input data to reduce
//!
//! @param num_output_bins_wrapper
//!   The number of bins per final output histogram
//!
//! @param num_privatized_bins_wrapper
//!   The number of bins per privatized histogram
//!
//! @param d_output_histograms_wrapper
//!   Reference to final output histograms
//!
//! @param d_privatized_histograms_wrapper
//!   Reference to privatized histograms
//!
//! @param first_level_array
//!   For DispatchEven: upper level bounds per channel.
//!   For DispatchRange: number of output levels per channel.
//!
//! @param second_level_array
//!   For DispatchEven: lower level bounds per channel.
//!   For DispatchRange: level pointers per channel.
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
//!   Drain queue descriptor for dynamically mapping tile data onto thread blocks
template <typename PolicySelector,
          int PrivatizedSmemBins,
          int NumChannels,
          int NumActiveChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename FirstLevelArrayT, // Upper level array for DispatchEven; Number of output levels array for
                                     // DispatchRange
          typename SecondLevelArrayT, // Lower level array for DispatchEven; Levels array for DispatchRange
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT,
          bool IsEven>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepDeviceInitKernel(
    const SampleIteratorT d_samples,
    ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    ::cuda::std::array<int, NumActiveChannels> num_privatized_bins_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
    const FirstLevelArrayT first_level_array,
    const SecondLevelArrayT second_level_array,
    const OffsetT num_row_pixels,
    const OffsetT num_rows,
    const OffsetT row_stride_samples,
    const int tiles_per_row,
    const GridQueue<int> tile_queue)
{
  static constexpr HistogramPolicy hp = current_policy<PolicySelector>();

  OutputDecodeOpT output_decode_op[NumActiveChannels];
  PrivatizedDecodeOpT privatized_decode_op[NumActiveChannels];
  if constexpr (IsEven)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int channel = 0; channel < NumActiveChannels; ++channel)
    {
      const int num_levels   = num_output_bins_wrapper[channel] + 1;
      const auto upper_level = first_level_array[channel];
      const auto lower_level = second_level_array[channel];
      privatized_decode_op[channel].Init(num_levels, upper_level, lower_level);
      output_decode_op[channel].Init(num_levels, upper_level, lower_level);
    }
  }
  else
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int channel = 0; channel < NumActiveChannels; ++channel)
    {
      const auto num_output_levels = first_level_array[channel];
      const auto levels            = second_level_array[channel];
      privatized_decode_op[channel].Init(levels, num_output_levels);
      output_decode_op[channel].Init(levels, num_output_levels);
    }
  }

  // Thread block type for compositing input tiles
  using AgentHistogramPolicyT = agent_histogram_policy<
    hp.threads_per_block,
    hp.pixels_per_thread,
    hp.load_algorithm,
    hp.load_modifier,
    hp.rle_compress,
    hp.mem_preference,
    hp.use_work_stealing,
    hp.vec_size>;
  using AgentHistogramT =
    AgentHistogram<AgentHistogramPolicyT,
                   PrivatizedSmemBins,
                   NumChannels,
                   NumActiveChannels,
                   SampleIteratorT,
                   CounterT,
                   PrivatizedDecodeOpT,
                   OutputDecodeOpT,
                   OffsetT>;

  // Shared memory for AgentHistogram
  __shared__ typename AgentHistogramT::TempStorage temp_storage;

  AgentHistogramT agent(
    temp_storage,
    d_samples,
    num_output_bins_wrapper.data(),
    num_privatized_bins_wrapper.data(),
    d_output_histograms_wrapper.data(),
    d_privatized_histograms_wrapper.data(),
    output_decode_op,
    privatized_decode_op);

  // Initialize counters
  agent.InitBinCounters();

  // Consume input tiles
  agent.ConsumeTiles(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);

  // Store output to global (if necessary)
  agent.StoreOutput();
}

//! Spills cache misses directly into the final output histogram with device-scope atomics.
//!
//! A spill operation owns the representation and initialization of its destination. Per-sample misses enter through
//! `consume`, cache flushes bypass probing and per-sample aggregation through `spill`, `finish` flushes per-thread
//! aggregation, and `finalize` performs any cooperative reduction required by the destination.
template <typename CounterT, typename OutputCounterT>
struct global_output_spill
{
  using counter_type = CounterT;
  using target_type  = OutputCounterT;

  target_type* target{};
  int num_bins{};

  template <typename GridGroup>
  _CCCL_DEVICE _CCCL_FORCEINLINE void initialize(
    OutputCounterT* output_histogram,
    CounterT*,
    int channel_num_bins,
    GridGroup grid,
    unsigned int global_thread,
    unsigned int total_threads,
    int,
    int)
  {
    target   = output_histogram;
    num_bins = channel_num_bins;
    for (unsigned int bin = global_thread; bin < static_cast<unsigned int>(num_bins); bin += total_threads)
    {
      target[bin] = OutputCounterT{0};
    }
    grid.sync();
  }

  template <typename ContributionT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void spill(int bin, ContributionT contribution)
  {
    if constexpr (::cuda::std::is_integral_v<OutputCounterT> && sizeof(OutputCounterT) == sizeof(::cuda::std::uint64_t))
    {
      // CUDA provides exact int/unsigned-int overloads for 32-bit counters. Its 64-bit integer overload is spelled in
      // terms of unsigned long long, while uint64_t is unsigned long on LP64 targets, so only this case needs casts.
      atomicAdd(reinterpret_cast<unsigned long long*>(&target[bin]), static_cast<unsigned long long>(contribution));
    }
    else
    {
      atomicAdd(&target[bin], static_cast<OutputCounterT>(contribution));
    }
  }

  template <typename ProbeOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume(ProbeOp& probe_op, int bin, CounterT contribution)
  {
    if (bin >= 0 && probe_op.try_cache(bin, contribution))
    {
      spill(bin, contribution);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void finish() {}

  template <typename GridGroup>
  _CCCL_DEVICE _CCCL_FORCEINLINE void finalize(GridGroup, unsigned int, unsigned int)
  {}
};

//! Spills cache misses into a block-private global-memory histogram with block-scope atomics.
template <typename CounterT, typename OutputCounterT>
struct block_private_spill
{
  using counter_type = CounterT;
  using target_type  = CounterT;

  target_type* target{};
  CounterT* private_histograms{};
  OutputCounterT* output_histogram{};
  int num_bins{};

  template <typename GridGroup>
  _CCCL_DEVICE _CCCL_FORCEINLINE void initialize(
    OutputCounterT* channel_output_histogram,
    CounterT* channel_private_histograms,
    int channel_num_bins,
    GridGroup,
    unsigned int,
    unsigned int,
    int thread_index,
    int block_threads)
  {
    private_histograms = channel_private_histograms;
    output_histogram   = channel_output_histogram;
    num_bins           = channel_num_bins;
    target             = private_histograms + static_cast<size_t>(blockIdx.x) * num_bins;
    for (int bin = thread_index; bin < num_bins; bin += block_threads)
    {
      target[bin] = CounterT{0};
    }
    __syncthreads();
  }

  template <typename ContributionT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void spill(int bin, ContributionT contribution)
  {
    ::cuda::atomic_ref<CounterT, ::cuda::thread_scope_block>{target[bin]}.fetch_add(
      static_cast<CounterT>(contribution), ::cuda::memory_order_relaxed);
  }

  template <typename ProbeOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume(ProbeOp& probe_op, int bin, CounterT contribution)
  {
    if (bin >= 0 && probe_op.try_cache(bin, contribution))
    {
      spill(bin, contribution);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void finish() {}

  template <typename GridGroup>
  _CCCL_DEVICE _CCCL_FORCEINLINE void finalize(GridGroup grid, unsigned int global_thread, unsigned int total_threads)
  {
    grid.sync();
    const unsigned int channel_bins = static_cast<unsigned int>(num_bins);
    for (unsigned int bin = global_thread; bin < channel_bins; bin += total_threads)
    {
      OutputCounterT total = OutputCounterT{0};
      for (unsigned int block = 0; block < gridDim.x; ++block)
      {
        total += static_cast<OutputCounterT>(private_histograms[static_cast<size_t>(block) * channel_bins + bin]);
      }
      output_histogram[bin] = total;
    }
  }
};

//! Coalesces equal bins within a warp before forwarding one combined contribution to `UnderlyingSpillOp`.
template <typename UnderlyingSpillOp>
struct warp_coalesced_spill
{
  using counter_type = typename UnderlyingSpillOp::counter_type;
  using target_type  = typename UnderlyingSpillOp::target_type;

  UnderlyingSpillOp underlying{};

  template <typename OutputCounterT, typename CounterT, typename GridGroup>
  _CCCL_DEVICE _CCCL_FORCEINLINE void initialize(
    OutputCounterT* output_histogram,
    CounterT* private_histograms,
    int num_bins,
    GridGroup grid,
    unsigned int global_thread,
    unsigned int total_threads,
    int thread_index,
    int block_threads)
  {
    underlying.initialize(
      output_histogram, private_histograms, num_bins, grid, global_thread, total_threads, thread_index, block_threads);
  }

  template <typename ContributionT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void spill(int bin, ContributionT contribution)
  {
    underlying.spill(bin, contribution);
  }

  template <typename ProbeOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume(ProbeOp& probe_op, int bin, counter_type contribution)
  {
    if constexpr (sizeof(counter_type) > sizeof(::cuda::std::uint32_t))
    {
      const unsigned int lane_id = threadIdx.x & 0x1f;
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_70,
        (const unsigned int peers = __match_any_sync(__activemask(), static_cast<unsigned int>(bin));
         const int leader         = __ffs(static_cast<int>(peers)) - 1;
         if (bin >= 0 && static_cast<int>(lane_id) == leader) {
           const counter_type coalesced_count = static_cast<counter_type>(__popc(peers));
           if (probe_op.try_cache(bin, coalesced_count))
           {
             spill(bin, coalesced_count);
           }
         }),
        (if (bin >= 0 && probe_op.try_cache(bin, contribution)) { spill(bin, contribution); }));
    }
    else
    {
      const int spill_bin = bin >= 0 && probe_op.try_cache(bin, contribution) ? bin : -1;
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_70,
        (const unsigned int active = __activemask();
         const unsigned int peers  = __match_any_sync(active, static_cast<unsigned int>(spill_bin));
         const int leader          = __ffs(static_cast<int>(peers)) - 1;
         const int lane_id         = static_cast<int>(threadIdx.x & 0x1f);
         if (spill_bin >= 0 && lane_id == leader) {
           spill(spill_bin, static_cast<counter_type>(contribution * static_cast<counter_type>(__popc(peers))));
         }),
        (if (spill_bin >= 0) { spill(spill_bin, contribution); }));
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void finish()
  {
    underlying.finish();
  }

  template <typename GridGroup>
  _CCCL_DEVICE _CCCL_FORCEINLINE void finalize(GridGroup grid, unsigned int global_thread, unsigned int total_threads)
  {
    underlying.finalize(grid, global_thread, total_threads);
  }
};

//! Run-length encodes consecutive misses before forwarding them to `UnderlyingSpillOp`.
template <typename UnderlyingSpillOp>
struct rle_spill
{
  using counter_type = typename UnderlyingSpillOp::counter_type;
  using target_type  = typename UnderlyingSpillOp::target_type;

  UnderlyingSpillOp underlying{};
  int pending_bin            = -1;
  counter_type pending_count = counter_type{0};

  template <typename OutputCounterT, typename CounterT, typename GridGroup>
  _CCCL_DEVICE _CCCL_FORCEINLINE void initialize(
    OutputCounterT* output_histogram,
    CounterT* private_histograms,
    int num_bins,
    GridGroup grid,
    unsigned int global_thread,
    unsigned int total_threads,
    int thread_index,
    int block_threads)
  {
    underlying.initialize(
      output_histogram, private_histograms, num_bins, grid, global_thread, total_threads, thread_index, block_threads);
  }

  template <typename ContributionT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void spill(int bin, ContributionT contribution)
  {
    underlying.spill(bin, contribution);
  }

  template <typename ProbeOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume(ProbeOp& probe_op, int bin, counter_type contribution)
  {
    if (bin < 0 || !probe_op.try_cache(bin, contribution))
    {
      return;
    }
    if (pending_bin == bin)
    {
      pending_count += contribution;
      return;
    }
    flush_pending();
    pending_bin   = bin;
    pending_count = contribution;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void finish()
  {
    flush_pending();
    underlying.finish();
  }

  template <typename GridGroup>
  _CCCL_DEVICE _CCCL_FORCEINLINE void finalize(GridGroup grid, unsigned int global_thread, unsigned int total_threads)
  {
    underlying.finalize(grid, global_thread, total_threads);
  }

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE void flush_pending()
  {
    if (pending_bin >= 0)
    {
      underlying.spill(pending_bin, pending_count);
      pending_bin   = -1;
      pending_count = counter_type{0};
    }
  }
};

//! Shared-memory cache probe used by the cooperative histogram agent.
//!
//! Probe operations own cache initialization, per-thread channel state, insertion/update attempts, and the final cache
//! flush. A cache miss is forwarded through the selected spill operation after any required warp reconvergence.
//! `UseSecondProbe == false` is the single-probe direct-mapped cache; `true` adds the cuckoo fallback probe.
template <bool UseSecondProbe, typename CounterT>
struct cuckoo_cache_probe
{
  ::cuda::std::uint32_t* keys{};
  CounterT* counts_base{};
  CounterT* thread_counts{};
  int slots_per_channel{};
  int count_replicas{};
  int mask{};
  int log2_slots{};

  _CCCL_DEVICE _CCCL_FORCEINLINE void initialize(
    ::cuda::std::uint32_t* cache_keys,
    CounterT* cache_counts,
    int channel,
    int channel_slots,
    int channel_count_replicas,
    int thread_index,
    int block_threads)
  {
    const int replica = static_cast<int>((threadIdx.x >> 5) % channel_count_replicas);
    keys              = cache_keys + static_cast<size_t>(channel) * channel_slots;
    counts_base       = cache_counts + static_cast<size_t>(channel) * channel_count_replicas * channel_slots;
    thread_counts     = counts_base + static_cast<size_t>(replica) * channel_slots;
    slots_per_channel = channel_slots;
    count_replicas    = channel_count_replicas;
    mask              = channel_slots - 1;
    log2_slots        = 31 - ::cuda::std::countl_zero(static_cast<::cuda::std::uint32_t>(channel_slots));
    for (int slot = thread_index; slot < slots_per_channel; slot += block_threads)
    {
      keys[slot] = UINT32_MAX;
    }
    for (int count = thread_index; count < count_replicas * slots_per_channel; count += block_threads)
    {
      counts_base[count] = CounterT{0};
    }
    __syncthreads();
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE bool try_cache(int bin, CounterT contribution)
  {
    constexpr ::cuda::std::uint32_t empty_key = UINT32_MAX;
    const auto bin_key                        = static_cast<::cuda::std::uint32_t>(bin);
    const auto try_slot                       = [&](int slot) {
      ::cuda::std::uint32_t key = keys[slot];
      if (key == bin_key)
      {
        ::cuda::atomic_ref<CounterT, ::cuda::thread_scope_block>{thread_counts[slot]}.fetch_add(
          contribution, ::cuda::memory_order_relaxed);
        return true;
      }
      if (key == empty_key)
      {
        key = atomicCAS_block(&keys[slot], empty_key, bin_key);
        if (key == empty_key || key == bin_key)
        {
          ::cuda::atomic_ref<CounterT, ::cuda::thread_scope_block>{thread_counts[slot]}.fetch_add(
            contribution, ::cuda::memory_order_relaxed);
          return true;
        }
      }
      return false;
    };

    const unsigned int hash = static_cast<unsigned int>(bin) * 2654435761u;
    const int primary       = static_cast<int>((hash >> (32 - log2_slots)) & static_cast<unsigned int>(mask));
    if (try_slot(primary))
    {
      return false;
    }

    if constexpr (UseSecondProbe)
    {
      const unsigned int hash2 = (static_cast<unsigned int>(bin) ^ 0x9e3779b9u) * 2246822519u;
      const int secondary      = static_cast<int>((hash2 >> (32 - log2_slots)) & static_cast<unsigned int>(mask));
      if (try_slot(secondary))
      {
        return false;
      }
    }
    return true;
  }

  template <typename SpillOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void flush(SpillOp& spill_op, int thread_index, int block_threads)
  {
    __syncthreads();
    for (int slot = thread_index; slot < slots_per_channel; slot += block_threads)
    {
      const auto key = keys[slot];
      if (key != UINT32_MAX)
      {
        CounterT count = CounterT{0};
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int replica = 0; replica < count_replicas; ++replica)
        {
          count += counts_base[static_cast<size_t>(replica) * slots_per_channel + slot];
        }
        if (count > CounterT{0})
        {
          spill_op.spill(static_cast<int>(key), count);
        }
      }
    }
  }
};

template <typename CounterT>
using single_probe_cache = cuckoo_cache_probe<false, CounterT>;

template <typename CounterT>
using double_probe_cache = cuckoo_cache_probe<true, CounterT>;

//! Probe operation that bypasses shared-memory caching and forwards every contribution to the spill operation.
//!
//! It implements the same lifecycle as `cuckoo_cache_probe`; initialization and flushing are intentionally no-ops.
template <typename CounterT>
struct no_cache_probe
{
  _CCCL_DEVICE _CCCL_FORCEINLINE void initialize(::cuda::std::uint32_t*, CounterT*, int, int, int, int, int) {}

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE bool try_cache(int, CounterT)
  {
    return true;
  }

  template <typename SpillOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void flush(SpillOp&, int, int)
  {}
};

template <typename PolicySelector,
          int NumChannels,
          int NumActiveChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename OutputCounterT,
          typename PrivatizedDecodeOpT,
          typename OffsetT>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
struct AgentHistogramCooperative
{
  _CCCL_DEVICE _CCCL_FORCEINLINE static void Consume(
    const SampleIteratorT d_samples,
    const ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    ::cuda::std::array<OutputCounterT*, NumActiveChannels> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
    const ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> decode_op_wrapper,
    const OffsetT num_row_pixels,
    const OffsetT num_rows,
    const OffsetT row_stride_samples,
    const int cache_slots_per_channel)
  {
    static constexpr HistogramPolicy policy = current_policy<PolicySelector>();
    static constexpr int count_replicas     = policy.high_bin_cache_count_replicas;
    using ProbeOp                           = ::cuda::std::conditional_t<
      policy.high_bin_cache == HistogramCacheAlgorithm::none,
      no_cache_probe<CounterT>,
      ::cuda::std::conditional_t<policy.high_bin_cache == HistogramCacheAlgorithm::single_probe,
                                 single_probe_cache<CounterT>,
                                 double_probe_cache<CounterT>>>;
    using AtomicSpillOp =
      ::cuda::std::conditional_t<policy.high_bin_spill == HistogramSpillAlgorithm::global_memory_privatized,
                                 block_private_spill<CounterT, OutputCounterT>,
                                 global_output_spill<CounterT, OutputCounterT>>;
    using SpillOp = ::cuda::std::conditional_t<
      policy.high_bin_aggregation == HistogramAggregationAlgorithm::warp_coalesced,
      warp_coalesced_spill<AtomicSpillOp>,
      ::cuda::std::conditional_t<policy.high_bin_aggregation == HistogramAggregationAlgorithm::rle,
                                 rle_spill<AtomicSpillOp>,
                                 AtomicSpillOp>>;
    static_assert(policy.high_bin_pixels_per_thread > 0, "Histogram cooperative pixels_per_thread must be positive");
    static_assert(policy.high_bin_blocks_per_sm >= 0, "Histogram cooperative blocks per SM must not be negative");
    namespace cg        = ::cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const unsigned int tid_global    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_threads = gridDim.x * blockDim.x;

    // Count replicas give different warps independent counters for the same cache slot, reducing shared-memory atomic
    // contention. They are summed when the cache is flushed.
    static_assert(count_replicas > 0, "Histogram cache replication must be positive");

    extern __shared__ unsigned char dynamic_smem[];
    // Bin indices are non-negative `int` values, so uint32_t stores every possible key while preserving UINT32_MAX as
    // an empty-slot sentinel. A wider key would only reduce cache capacity without extending the supported bin range.
    auto* cache_keys = reinterpret_cast<::cuda::std::uint32_t*>(dynamic_smem);
    CounterT* cache_counts =
      reinterpret_cast<CounterT*>(cache_keys + static_cast<size_t>(NumActiveChannels) * cache_slots_per_channel);
    const int thread_idx    = threadIdx.x;
    const int block_threads = blockDim.x;

    using PixelOffsetT =
      ::cuda::std::conditional_t<(sizeof(OffsetT) < sizeof(::cuda::std::int64_t)), ::cuda::std::int64_t, OffsetT>;
    constexpr int pixels_per_thread = policy.high_bin_pixels_per_thread;
    const PixelOffsetT total_pixels = static_cast<PixelOffsetT>(num_rows) * num_row_pixels;
    const PixelOffsetT step         = static_cast<PixelOffsetT>(total_threads);
    const PixelOffsetT chunk        = static_cast<PixelOffsetT>(pixels_per_thread) * step;
    const PixelOffsetT chunk_count  = ::cuda::ceil_div(total_pixels, chunk);
    const bool contiguous_input     = num_rows == 1;

    PrivatizedDecodeOpT decode_op[NumActiveChannels];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      decode_op[ch] = decode_op_wrapper[ch];
    }

    ProbeOp probe_ops[NumActiveChannels];
    SpillOp spill_ops[NumActiveChannels];

    // Phase 1: initialize each channel's probe and spill operations. The operations own their storage and the
    // synchronization required to make that storage ready for use.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      probe_ops[ch].initialize(
        cache_keys, cache_counts, ch, cache_slots_per_channel, count_replicas, thread_idx, block_threads);
      spill_ops[ch].initialize(
        d_output_histograms_wrapper[ch],
        d_privatized_histograms_wrapper[ch],
        num_output_bins_wrapper[ch],
        grid,
        tid_global,
        total_threads,
        thread_idx,
        block_threads);
    }

    // Phase 2: decode input pixels and pass each channel's bin to the selected spill operation. The spill operation
    // owns cache probing and any per-thread or warp-level aggregation; a negative bin denotes an out-of-range sample.
    using SampleValueT = it_value_t<SampleIteratorT>;
    if (contiguous_input)
    {
      if constexpr (NumActiveChannels == 1)
      {
        for (PixelOffsetT chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx)
        {
          const PixelOffsetT first_pixel = static_cast<PixelOffsetT>(tid_global) + chunk_idx * chunk;
          SampleValueT staged_samples[pixels_per_thread];
          bool valid_samples[pixels_per_thread];
          int bins[pixels_per_thread];

          _CCCL_PRAGMA_UNROLL_FULL()
          for (int pixel_index = 0; pixel_index < pixels_per_thread; ++pixel_index)
          {
            const PixelOffsetT pixel      = first_pixel + static_cast<PixelOffsetT>(pixel_index) * step;
            valid_samples[pixel_index]    = pixel < total_pixels;
            const PixelOffsetT safe_pixel = valid_samples[pixel_index] ? pixel : PixelOffsetT{0};
            staged_samples[pixel_index]   = d_samples[safe_pixel * NumChannels];
          }

          _CCCL_PRAGMA_UNROLL_FULL()
          for (int pixel_index = 0; pixel_index < pixels_per_thread; ++pixel_index)
          {
            int bin = -1;
            if (valid_samples[pixel_index])
            {
              decode_op[0].template BinSelect<LOAD_DEFAULT>(staged_samples[pixel_index], bin, true);
              if (bin >= num_output_bins_wrapper[0])
              {
                bin = -1;
              }
            }
            bins[pixel_index] = bin;
          }

          _CCCL_PRAGMA_UNROLL_FULL()
          for (int pixel_index = 0; pixel_index < pixels_per_thread; ++pixel_index)
          {
            spill_ops[0].consume(probe_ops[0], bins[pixel_index], CounterT{1});
          }
        }
      }
      else
      {
        if constexpr ((NumChannels == 2 || NumChannels == 4) && ::cuda::std::is_trivially_copyable_v<SampleValueT>)
        {
          using PixelT = typename CubVector<SampleValueT, NumChannels>::Type;
          const SampleValueT* native_base;
          if constexpr (::cuda::std::is_pointer_v<SampleIteratorT>)
          {
            native_base = d_samples;
          }
          else
          {
            native_base = NativePointer(d_samples);
          }
          const bool vectorizable =
            native_base != nullptr && (reinterpret_cast<size_t>(native_base) & (alignof(PixelT) - 1)) == 0;
          if (vectorizable)
          {
            const PixelT* const pixels = reinterpret_cast<const PixelT*>(native_base);
            for (PixelOffsetT chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx)
            {
              const PixelOffsetT first_pixel = static_cast<PixelOffsetT>(tid_global) + chunk_idx * chunk;
              _CCCL_PRAGMA_UNROLL_FULL()
              for (int pixel_index = 0; pixel_index < pixels_per_thread; ++pixel_index)
              {
                const PixelOffsetT pixel  = first_pixel + static_cast<PixelOffsetT>(pixel_index) * step;
                const bool valid          = pixel < total_pixels;
                const PixelT packed       = pixels[valid ? pixel : PixelOffsetT{0}];
                const SampleValueT* lanes = reinterpret_cast<const SampleValueT*>(&packed);
                int bins[NumActiveChannels];

                _CCCL_PRAGMA_UNROLL_FULL()
                for (int ch = 0; ch < NumActiveChannels; ++ch)
                {
                  int bin = -1;
                  if (valid)
                  {
                    decode_op[ch].template BinSelect<LOAD_DEFAULT>(lanes[ch], bin, true);
                    if (bin >= num_output_bins_wrapper[ch])
                    {
                      bin = -1;
                    }
                  }
                  bins[ch] = bin;
                }
                _CCCL_PRAGMA_UNROLL_FULL()
                for (int ch = 0; ch < NumActiveChannels; ++ch)
                {
                  spill_ops[ch].consume(probe_ops[ch], bins[ch], CounterT{1});
                }
              }
            }
          }
          else
          {
            for (PixelOffsetT chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx)
            {
              const PixelOffsetT first_pixel = static_cast<PixelOffsetT>(tid_global) + chunk_idx * chunk;
              _CCCL_PRAGMA_UNROLL_FULL()
              for (int pixel_index = 0; pixel_index < pixels_per_thread; ++pixel_index)
              {
                const PixelOffsetT pixel      = first_pixel + static_cast<PixelOffsetT>(pixel_index) * step;
                const bool valid              = pixel < total_pixels;
                const PixelOffsetT safe_pixel = valid ? pixel : PixelOffsetT{0};
                int bins[NumActiveChannels];

                _CCCL_PRAGMA_UNROLL_FULL()
                for (int ch = 0; ch < NumActiveChannels; ++ch)
                {
                  const SampleValueT sample = d_samples[safe_pixel * NumChannels + ch];
                  int bin                   = -1;
                  if (valid)
                  {
                    decode_op[ch].template BinSelect<LOAD_DEFAULT>(sample, bin, true);
                    if (bin >= num_output_bins_wrapper[ch])
                    {
                      bin = -1;
                    }
                  }
                  bins[ch] = bin;
                }
                _CCCL_PRAGMA_UNROLL_FULL()
                for (int ch = 0; ch < NumActiveChannels; ++ch)
                {
                  spill_ops[ch].consume(probe_ops[ch], bins[ch], CounterT{1});
                }
              }
            }
          }
        }
        else
        {
          for (PixelOffsetT chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx)
          {
            const PixelOffsetT first_pixel = static_cast<PixelOffsetT>(tid_global) + chunk_idx * chunk;
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int pixel_index = 0; pixel_index < pixels_per_thread; ++pixel_index)
            {
              const PixelOffsetT pixel      = first_pixel + static_cast<PixelOffsetT>(pixel_index) * step;
              const bool valid              = pixel < total_pixels;
              const PixelOffsetT safe_pixel = valid ? pixel : PixelOffsetT{0};
              int bins[NumActiveChannels];

              _CCCL_PRAGMA_UNROLL_FULL()
              for (int ch = 0; ch < NumActiveChannels; ++ch)
              {
                const SampleValueT sample = d_samples[safe_pixel * NumChannels + ch];
                int bin                   = -1;
                if (valid)
                {
                  decode_op[ch].template BinSelect<LOAD_DEFAULT>(sample, bin, true);
                  if (bin >= num_output_bins_wrapper[ch])
                  {
                    bin = -1;
                  }
                }
                bins[ch] = bin;
              }
              _CCCL_PRAGMA_UNROLL_FULL()
              for (int ch = 0; ch < NumActiveChannels; ++ch)
              {
                spill_ops[ch].consume(probe_ops[ch], bins[ch], CounterT{1});
              }
            }
          }
        }
      }
    }
    else
    {
      for (PixelOffsetT pixel = static_cast<PixelOffsetT>(tid_global); pixel < total_pixels; pixel += step)
      {
        const PixelOffsetT row = pixel / num_row_pixels;
        const PixelOffsetT pixel_offset =
          row * static_cast<PixelOffsetT>(row_stride_samples) + (pixel - row * num_row_pixels) * NumChannels;

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int ch = 0; ch < NumActiveChannels; ++ch)
        {
          int bin = -1;
          decode_op[ch].template BinSelect<LOAD_DEFAULT>(d_samples[pixel_offset + ch], bin, true);
          if (bin >= 0 && bin < num_output_bins_wrapper[ch])
          {
            spill_ops[ch].consume(probe_ops[ch], bin, CounterT{1});
          }
        }
      }
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      spill_ops[ch].finish();
    }

    // Phase 3: flush cached counts through the selected spill operation.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      probe_ops[ch].flush(spill_ops[ch], thread_idx, block_threads);
    }

    // Phase 4: finalize the spill destination. Block-private storage performs its cooperative gather here; direct
    // output spilling has no final work.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      spill_ops[ch].finalize(grid, tid_global, total_threads);
    }
  }
};

//! Policy-configurable cooperative high-bin histogram kernel.
template <typename PolicySelector,
          int NumChannels,
          int NumActiveChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename OutputCounterT,
          typename PrivatizedDecodeOpT,
          typename OffsetT>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
_CCCL_KERNEL_ATTRIBUTES void DeviceHistogramCooperativeKernel(
  _CCCL_GRID_CONSTANT const SampleIteratorT d_samples,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
  ::cuda::std::array<OutputCounterT*, NumActiveChannels> d_output_histograms_wrapper,
  ::cuda::std::array<CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> decode_op_wrapper,
  _CCCL_GRID_CONSTANT const OffsetT num_row_pixels,
  _CCCL_GRID_CONSTANT const OffsetT num_rows,
  _CCCL_GRID_CONSTANT const OffsetT row_stride_samples,
  _CCCL_GRID_CONSTANT const int cache_slots_per_channel)
{
  AgentHistogramCooperative<
    PolicySelector,
    NumChannels,
    NumActiveChannels,
    SampleIteratorT,
    CounterT,
    OutputCounterT,
    PrivatizedDecodeOpT,
    OffsetT>::Consume(d_samples,
                      num_output_bins_wrapper,
                      d_output_histograms_wrapper,
                      d_privatized_histograms_wrapper,
                      decode_op_wrapper,
                      num_row_pixels,
                      num_rows,
                      row_stride_samples,
                      cache_slots_per_channel);
}
} // namespace detail::histogram
CUB_NAMESPACE_END
