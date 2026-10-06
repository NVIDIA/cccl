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

#include <cuda/__numeric/sub_overflow.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/cmath>
#include <cuda/std/__numeric/reduce.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/make_unsigned.h>

CUB_NAMESPACE_BEGIN
namespace detail::histogram
{
template <typename LevelT, typename OffsetT, typename InputSampleT>
struct Transforms
{
  //---------------------------------------------------------------------
  // Transform functors for converting samples to bin-ids
  //---------------------------------------------------------------------

  //! @brief Finds a RANGE bin with binary search.
  //!
  //! Uses `UpperBound` without interpolation or per-thread state.
  template <typename LevelIteratorT>
  struct SearchTransform
  {
    LevelIteratorT d_levels; // Pointer to levels array
    int num_output_levels; // Number of levels in array

    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(LevelIteratorT d_levels_, int num_output_levels_)
    {
      d_levels          = d_levels_;
      num_output_levels = num_output_levels_;
    }

    template <CacheLoadModifier LoadModifier, typename SampleT>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(SampleT sample, int& bin, bool valid) const
    {
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
    using CommonT = ::cuda::std::common_type_t<LevelT, InputSampleT>;
    static_assert(::cuda::std::is_convertible_v<CommonT, int>,
                  "The common type of `LevelT` and `InputSampleT` must be "
                  "convertible to `int`.");
    static_assert(::cuda::is_trivially_copyable_v<CommonT>,
                  "The common type of `LevelT` and `InputSampleT` must be "
                  "trivially copyable.");

    // An arithmetic type that's used for bin computation of integral types, guaranteed to not
    // overflow for (max_level - min_level) * scale.fraction.bins. Since we drop invalid samples
    // of less than min_level, (sample - min_level) is guaranteed to be non-negative. We use the
    // rule: 2^l * 2^r = 2^(l + r) to determine a sufficiently large type to hold the
    // multiplication result.
    // If CommonT used to be a 128-bit wide integral type already, we use CommonT's arithmetic
    using IntArithmeticT = ::cuda::std::_If< //
      sizeof(InputSampleT) + sizeof(CommonT) <= sizeof(uint32_t), //
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

  protected:
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

    // Method for converting samples to bin-ids
    template <CacheLoadModifier LoadModifier>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(InputSampleT sample, int& bin, bool valid) const
    {
      const CommonT common_sample = static_cast<CommonT>(sample);

      if (valid && this->SampleIsValid(common_sample, m_max, m_min))
      {
        bin = this->ComputeBin(common_sample, m_min, m_scale);
      }
    }
  };

  //! @brief Scales integral samples to evenly-spaced bins using a precomputed divisor.
  //!
  //! The divisor is initialized on the host and replaces the runtime integer division in the
  //! privatized histogram kernels with the multiply-high sequence provided by `cuda::fast_mod_div`.
  //! Floating-point, extended-integer, and custom types retain `ScaleTransform`'s implementation.
  struct FastScaleTransform : ScaleTransform
  {
    using BaseT = ScaleTransform;
    using FastDivisorValueT =
      ::cuda::std::_If<BaseT::template is_integral_excl_int128<typename BaseT::CommonT>::value,
                       typename BaseT::IntArithmeticT,
                       uint32_t>;
    using FastDivisorT = ::cuda::fast_mod_div<FastDivisorValueT>;

    FastDivisorT m_range_divisor{FastDivisorValueT{1}};

    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(int num_levels, LevelT max_level, LevelT min_level)
    {
      BaseT::Init(num_levels, max_level, min_level);

      if constexpr (BaseT::template is_integral_excl_int128<typename BaseT::CommonT>::value)
      {
        m_range_divisor = FastDivisorT{static_cast<FastDivisorValueT>(BaseT::m_scale.fraction.range)};
      }
    }

    template <CacheLoadModifier LoadModifier>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(InputSampleT sample, int& bin, bool valid) const
    {
      using CommonT = typename BaseT::CommonT;

      const CommonT common_sample = static_cast<CommonT>(sample);
      if (valid && BaseT::SampleIsValid(common_sample, BaseT::m_max, BaseT::m_min))
      {
        if constexpr (BaseT::template is_integral_excl_int128<CommonT>::value)
        {
          const auto offset = BaseT::subtract_as_unsigned(common_sample, BaseT::m_min);
          const typename BaseT::IntArithmeticT numerator =
            typename BaseT::IntArithmeticT{offset} * typename BaseT::IntArithmeticT{BaseT::m_scale.fraction.bins};
          bin = static_cast<int>(numerator / m_range_divisor);
        }
        else
        {
          bin = BaseT::ComputeBin(common_sample, BaseT::m_min, BaseT::m_scale);
        }
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
    template <CacheLoadModifier LoadModifier, typename SampleT>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(SampleT sample, int& bin, bool valid) const
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

  // Trigger the sweep only when the remaining output writes occupy at most the tuned byte threshold.
  NV_IF_TARGET(
    NV_PROVIDES_SM_90, ({
      const auto output_histogram_bytes =
        ::cuda::std::reduce(num_output_bins_wrapper.begin(), num_output_bins_wrapper.end(), ::cuda::std::uint64_t{0})
        * sizeof(CounterT);
      if (output_histogram_bytes
          <= static_cast<::cuda::std::uint64_t>(policy.max_output_histogram_bytes_for_init_kernel_pdl))
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

template <typename PolicySelector, typename PrivatizationMode>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto histogram_privatization_policy() -> HistogramPrivatizationPolicy
{
  if constexpr (is_privatized_static_smem_v<PrivatizationMode>)
  {
    return current_policy<PolicySelector>().static_smem;
  }
  else if constexpr (is_privatized_dynamic_smem_v<PrivatizationMode>)
  {
    return current_policy<PolicySelector>().dynamic_smem;
  }
  else
  {
    return current_policy<PolicySelector>().gmem;
  }
}

template <typename PolicySelector, typename PrivatizationMode>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int histogram_min_blocks_per_sm()
{
  if constexpr (is_privatized_static_smem_v<PrivatizationMode>)
  {
    return current_policy<PolicySelector>().static_smem_min_blocks_per_sm;
  }
  else
  {
    return 0;
  }
}

//! Histogram privatized sweep kernel entry point (multi-block).
//! Computes privatized histograms, one per thread block.
//! This kernel receives pre-initialized decode operators from the host.
//!
//! @tparam PolicySelector
//!   Selects the tuning policy
//!
//! @tparam PrivatizationMode
//!   Storage mode for the privatized histogram
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
          typename PrivatizationMode,
          int NumChannels,
          int NumActiveChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT,
          typename OutputCounterT = CounterT>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(histogram_privatization_policy<PolicySelector, PrivatizationMode>().threads_per_block),
                  int(histogram_min_blocks_per_sm<PolicySelector, PrivatizationMode>()))
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepKernel(
    const SampleIteratorT d_samples,
    const ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    const ::cuda::std::array<int, NumActiveChannels> num_privatized_bins_wrapper,
    ::cuda::std::array<OutputCounterT*, NumActiveChannels> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
    ::cuda::std::array<OutputDecodeOpT, NumActiveChannels> output_decode_op_wrapper,
    ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op_wrapper,
    const OffsetT num_row_pixels,
    const OffsetT num_rows,
    const OffsetT row_stride_samples,
    const int tiles_per_row,
    GridQueue<int> tile_queue)
{
  using AgentHistogramT =
    AgentHistogram<PolicySelector,
                   PrivatizationMode,
                   NumChannels,
                   NumActiveChannels,
                   SampleIteratorT,
                   CounterT,
                   PrivatizedDecodeOpT,
                   OutputDecodeOpT,
                   OffsetT,
                   OutputCounterT>;

  __shared__ typename AgentHistogramT::TempStorage static_smem;
  extern __shared__ __align__(16) unsigned char dynamic_smem[];

  CounterT* dynamic_smem_privatized_histograms = nullptr;
  if constexpr (is_privatized_dynamic_smem_v<PrivatizationMode>)
  {
    dynamic_smem_privatized_histograms = reinterpret_cast<CounterT*>(dynamic_smem);
  }

  AgentHistogramT agent(
    static_smem,
    d_samples,
    num_output_bins_wrapper.data(),
    num_privatized_bins_wrapper.data(),
    d_output_histograms_wrapper.data(),
    d_privatized_histograms_wrapper.data(),
    output_decode_op_wrapper.data(),
    privatized_decode_op_wrapper.data(),
    dynamic_smem_privatized_histograms);

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
//! @tparam PrivatizationMode
//!   Storage mode for the privatized histogram
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
          typename PrivatizationMode,
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
          bool IsEven,
          typename OutputCounterT = CounterT>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(histogram_privatization_policy<PolicySelector, PrivatizationMode>().threads_per_block),
                  int(histogram_min_blocks_per_sm<PolicySelector, PrivatizationMode>()))
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepDeviceInitKernel(
    const SampleIteratorT d_samples,
    ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    ::cuda::std::array<int, NumActiveChannels> num_privatized_bins_wrapper,
    ::cuda::std::array<OutputCounterT*, NumActiveChannels> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
    const FirstLevelArrayT first_level_array,
    const SecondLevelArrayT second_level_array,
    const OffsetT num_row_pixels,
    const OffsetT num_rows,
    const OffsetT row_stride_samples,
    const int tiles_per_row,
    const GridQueue<int> tile_queue)
{
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

  using AgentHistogramT =
    AgentHistogram<PolicySelector,
                   PrivatizationMode,
                   NumChannels,
                   NumActiveChannels,
                   SampleIteratorT,
                   CounterT,
                   PrivatizedDecodeOpT,
                   OutputDecodeOpT,
                   OffsetT,
                   OutputCounterT>;

  __shared__ typename AgentHistogramT::TempStorage static_smem;
  extern __shared__ __align__(16) unsigned char dynamic_smem[];

  CounterT* dynamic_smem_privatized_histograms = nullptr;
  if constexpr (is_privatized_dynamic_smem_v<PrivatizationMode>)
  {
    dynamic_smem_privatized_histograms = reinterpret_cast<CounterT*>(dynamic_smem);
  }

  AgentHistogramT agent(
    static_smem,
    d_samples,
    num_output_bins_wrapper.data(),
    num_privatized_bins_wrapper.data(),
    d_output_histograms_wrapper.data(),
    d_privatized_histograms_wrapper.data(),
    output_decode_op,
    privatized_decode_op,
    dynamic_smem_privatized_histograms);

  // Initialize counters
  agent.InitBinCounters();

  // Consume input tiles
  agent.ConsumeTiles(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);

  // Store output to global (if necessary)
  agent.StoreOutput();
}
} // namespace detail::histogram
CUB_NAMESPACE_END
