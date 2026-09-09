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

#include <cuda/__type_traits/is_trivially_copyable.h>
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
    static constexpr bool is_range_transform = false;
    struct BracketCacheT
    {};

    LevelIteratorT d_levels; // Pointer to levels array
    int num_output_levels; // Number of levels in array

    _CCCL_DEVICE _CCCL_FORCEINLINE void PrecomputeOnDevice(int) {}

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
    template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(_SampleT sample, int& bin, bool valid) const
    {
      /// Level iterator wrapper type
      // Wrap the native input pointer with CacheModifiedInputIterator
      // or Directly use the supplied input iterator type
      using WrappedLevelIteratorT =
        ::cuda::std::_If<::cuda::std::is_pointer_v<LevelIteratorT>,
                         CacheModifiedInputIterator<LOAD_MODIFIER, LevelT, OffsetT>,
                         LevelIteratorT>;

      WrappedLevelIteratorT wrapped_levels(d_levels);

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

  //! @brief Finds a RANGE bin with piecewise-linear interpolation and a per-thread bracket cache.
  template <typename LevelIteratorT>
  struct CachedSearchTransform
  {
    static constexpr bool is_range_transform = true;

    template <typename T>
    [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static auto interpolation_difference(T lhs, T rhs)
    {
      if constexpr (::cuda::std::is_integral_v<T>)
      {
        using UnsignedT = ::cuda::std::make_unsigned_t<T>;
        return static_cast<UnsignedT>(lhs) - static_cast<UnsignedT>(rhs);
      }
      else
      {
        return lhs - rhs;
      }
    }

    struct BracketCacheT
    {
      LevelT lo{};
      LevelT hi{};
      int bin = -1;
    };

    LevelIteratorT d_levels;
    int num_output_levels;
    LevelT first{};
    LevelT middle{};
    LevelT last{};
    float inverse_scale{};
    float inverse_scale_low{};
    float inverse_scale_high{};
    int middle_bin{};
    bool has_precompute{};

    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(LevelIteratorT d_levels_, int num_output_levels_)
    {
      d_levels          = d_levels_;
      num_output_levels = num_output_levels_;
      has_precompute    = false;
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE void PrecomputeOnDevice(int interpolation_min_level_bytes)
    {
      const int num_bins = num_output_levels - 1;
      if (static_cast<size_t>(num_output_levels) * sizeof(LevelT) < static_cast<size_t>(interpolation_min_level_bytes))
      {
        return;
      }

      using WrappedLevelIteratorT =
        ::cuda::std::_If<::cuda::std::is_pointer_v<LevelIteratorT>,
                         CacheModifiedInputIterator<LOAD_LDG, LevelT, OffsetT>,
                         LevelIteratorT>;
      WrappedLevelIteratorT wrapped_levels(d_levels);
      const LevelT first_level = wrapped_levels[0];
      const LevelT last_level  = wrapped_levels[num_bins];
      if (!(first_level < last_level))
      {
        return;
      }

      first              = first_level;
      last               = last_level;
      inverse_scale      = static_cast<float>(num_bins) / static_cast<float>(interpolation_difference(last, first));
      middle_bin         = 0;
      inverse_scale_low  = inverse_scale;
      inverse_scale_high = inverse_scale;
      middle             = first;

      const int split = num_bins >> 1;
      if (split > 0 && split < num_bins)
      {
        const LevelT split_level = wrapped_levels[split];
        if (first < split_level && split_level < last)
        {
          middle     = split_level;
          middle_bin = split;
          inverse_scale_low =
            static_cast<float>(split) / static_cast<float>(interpolation_difference(split_level, first));
          inverse_scale_high =
            static_cast<float>(num_bins - split) / static_cast<float>(interpolation_difference(last, split_level));
        }
      }
      has_precompute = true;
    }

    template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void BinSelect(_SampleT sample, int& bin, bool valid, BracketCacheT& bracket) const
    {
      if (!valid)
      {
        return;
      }

      using WrappedLevelIteratorT =
        ::cuda::std::_If<::cuda::std::is_pointer_v<LevelIteratorT>,
                         CacheModifiedInputIterator<LOAD_MODIFIER, LevelT, OffsetT>,
                         LevelIteratorT>;
      WrappedLevelIteratorT wrapped_levels(d_levels);
      const int num_bins = num_output_levels - 1;
      const LevelT value = static_cast<LevelT>(sample);

      if (bracket.bin >= 0 && !(value < bracket.lo) && value < bracket.hi)
      {
        bin = bracket.bin;
        return;
      }

      if (!has_precompute)
      {
        bin = UpperBound(wrapped_levels, num_output_levels, value) - 1;
        if (bin >= num_bins)
        {
          bin = -1;
        }
      }
      else if (value < first || !(value < last))
      {
        bin = -1;
      }
      else
      {
        int guess =
          value < middle || middle_bin == 0
            ? static_cast<int>(static_cast<float>(interpolation_difference(value, first)) * inverse_scale_low)
            : middle_bin
                + static_cast<int>(static_cast<float>(interpolation_difference(value, middle)) * inverse_scale_high);
        guess           = guess < 0 ? 0 : (guess < num_bins ? guess : num_bins - 1);
        const LevelT lo = wrapped_levels[guess];
        const LevelT hi = wrapped_levels[guess + 1];

        if (!(value < lo) && value < hi)
        {
          bin     = guess;
          bracket = BracketCacheT{lo, hi, guess};
          return;
        }

        if (value < lo && guess > 0)
        {
          const LevelT adjacent_lo = wrapped_levels[guess - 1];
          if (!(value < adjacent_lo))
          {
            bin     = guess - 1;
            bracket = BracketCacheT{adjacent_lo, lo, bin};
            return;
          }
        }
        else if (!(value < hi) && guess + 1 < num_bins)
        {
          const LevelT adjacent_hi = wrapped_levels[guess + 2];
          if (value < adjacent_hi)
          {
            bin     = guess + 1;
            bracket = BracketCacheT{hi, adjacent_hi, bin};
            return;
          }
        }

        bin = UpperBound(wrapped_levels, num_output_levels, value) - 1;
        if (bin >= num_bins)
        {
          bin = -1;
        }
      }

      if (bin >= 0)
      {
        bracket = BracketCacheT{wrapped_levels[bin], wrapped_levels[bin + 1], bin};
      }
    }

    template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void BinSelect(_SampleT sample, int& bin, bool valid) const
    {
      if (!valid)
      {
        return;
      }

      using WrappedLevelIteratorT =
        ::cuda::std::_If<::cuda::std::is_pointer_v<LevelIteratorT>,
                         CacheModifiedInputIterator<LOAD_MODIFIER, LevelT, OffsetT>,
                         LevelIteratorT>;
      WrappedLevelIteratorT wrapped_levels(d_levels);
      const int num_bins = num_output_levels - 1;
      const LevelT value = static_cast<LevelT>(sample);

      if (!has_precompute)
      {
        bin = UpperBound(wrapped_levels, num_output_levels, value) - 1;
        if (bin >= num_bins)
        {
          bin = -1;
        }
        return;
      }
      if (value < first || !(value < last))
      {
        bin = -1;
        return;
      }

      int guess =
        value < middle || middle_bin == 0
          ? static_cast<int>(static_cast<float>(interpolation_difference(value, first)) * inverse_scale_low)
          : middle_bin
              + static_cast<int>(static_cast<float>(interpolation_difference(value, middle)) * inverse_scale_high);
      guess           = guess < 0 ? 0 : (guess < num_bins ? guess : num_bins - 1);
      const LevelT lo = wrapped_levels[guess];
      const LevelT hi = wrapped_levels[guess + 1];

      if (!(value < lo) && value < hi)
      {
        bin = guess;
        return;
      }
      if (value < lo && guess > 0)
      {
        const LevelT adjacent_lo = wrapped_levels[guess - 1];
        if (!(value < adjacent_lo))
        {
          bin = guess - 1;
          return;
        }
      }
      else if (!(value < hi) && guess + 1 < num_bins)
      {
        const LevelT adjacent_hi = wrapped_levels[guess + 2];
        if (value < adjacent_hi)
        {
          bin = guess + 1;
          return;
        }
      }

      bin = UpperBound(wrapped_levels, num_output_levels, value) - 1;
      if (bin >= num_bins)
      {
        bin = -1;
      }
    }
  };

  // Scales samples to evenly-spaced bins.
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

    using FractionStorageT =
      typename ::cuda::std::_If<is_integral_excl_int128<CommonT>::value,
                                ::cuda::std::make_unsigned<CommonT>,
                                ::cuda::std::type_identity<CommonT>>::type;

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
      ScaleT result;
      result.reciprocal = static_cast<T>(static_cast<T>(num_levels - 1) / static_cast<T>(max_level - min_level));
      return result;
    }

    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScaleT
    ComputeScale(int num_levels, T max_level, T min_level, ::cuda::std::false_type /* is_fp */)
    {
      ScaleT result;
      result.fraction.bins = static_cast<FractionStorageT>(num_levels - 1);
      if constexpr (::cuda::std::is_integral_v<T>)
      {
        using UnsignedT = ::cuda::std::make_unsigned_t<T>;
        const UnsignedT distance =
          static_cast<UnsignedT>(static_cast<UnsignedT>(max_level) - static_cast<UnsignedT>(min_level));
        result.fraction.range = static_cast<FractionStorageT>(distance);
      }
      else
      {
        result.fraction.range = static_cast<FractionStorageT>(max_level - min_level);
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
      ScaleT result;
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_80,
        (result.reciprocal = __hdiv(__float2bfloat16(num_levels - 1), __hsub(max_level, min_level));),
        (result.reciprocal = __float2bfloat16(
           static_cast<float>(num_levels - 1) / (__bfloat162float(max_level) - __bfloat162float(min_level)));))
      return result;
    }
#endif // _CCCL_HAS_NVBF16()

    // All types but __half:
    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int SampleIsValid(T sample, T max_level, T min_level) const
    {
      return sample >= min_level && sample < max_level;
    }

#if _CCCL_HAS_NVFP16()
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int SampleIsValid(__half sample, __half max_level, __half min_level) const
    {
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_53,
        (return __hge(sample, min_level) && __hlt(sample, max_level);),
        (return __half2float(sample) >= __half2float(min_level) && __half2float(sample) < __half2float(max_level);));
    }
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
    _CCCL_HOST_DEVICE
    _CCCL_FORCEINLINE int SampleIsValid(__nv_bfloat16 sample, __nv_bfloat16 max_level, __nv_bfloat16 min_level)
    {
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
                        (return __hge(sample, min_level) && __hlt(sample, max_level);),
                        (return __bfloat162float(sample) >= __bfloat162float(min_level)
                               && __bfloat162float(sample) < __bfloat162float(max_level);));
    }
#endif // _CCCL_HAS_NVBF16()

    //! @brief Bin computation for floating point (and extended floating point) types
    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int
    ComputeBin(T sample, T min_level, ScaleT scale, ::cuda::std::true_type /* is_fp */) const
    {
      return static_cast<int>((sample - min_level) * scale.reciprocal);
    }

    //! @brief Bin computation for custom types and __[u]int128
    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int
    ComputeBin(T sample, T min_level, ScaleT scale, ::cuda::std::false_type /* is_fp */) const
    {
      return static_cast<int>(((sample - min_level) * scale.fraction.bins) / scale.fraction.range);
    }

    //! @brief Bin computation for integral types of up to 64-bit types
    template <typename T, ::cuda::std::enable_if_t<is_integral_excl_int128<T>::value, int> = 0>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int ComputeBin(T sample, T min_level, ScaleT scale) const
    {
      using UnsignedT               = ::cuda::std::make_unsigned_t<T>;
      const IntArithmeticT distance = static_cast<IntArithmeticT>(
        static_cast<UnsignedT>(static_cast<UnsignedT>(sample) - static_cast<UnsignedT>(min_level)));
      return static_cast<int>((distance * static_cast<IntArithmeticT>(scale.fraction.bins))
                              / static_cast<IntArithmeticT>(scale.fraction.range));
    }

    template <typename T, ::cuda::std::enable_if_t<!is_integral_excl_int128<T>::value, int> = 0>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int ComputeBin(T sample, T min_level, ScaleT scale) const
    {
      return this->ComputeBin(sample, min_level, scale, ::cuda::std::is_floating_point<T>{});
    }

#if _CCCL_HAS_NVFP16()
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int ComputeBin(__half sample, __half min_level, ScaleT scale) const
    {
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_53,
        (return static_cast<int>(__hmul(__hsub(sample, min_level), scale.reciprocal));),
        (return static_cast<int>((__half2float(sample) - __half2float(min_level)) * __half2float(scale.reciprocal));));
    }
#endif // _CCCL_HAS_NVFP16()

  public:
    static constexpr bool is_range_transform = false;
    struct BracketCacheT
    {};

    _CCCL_DEVICE _CCCL_FORCEINLINE void PrecomputeOnDevice(int) {}

    //! @brief Initializes the ScaleTransform for the given parameters
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(int num_levels, LevelT max_level, LevelT min_level)
    {
      m_max = static_cast<CommonT>(max_level);
      m_min = static_cast<CommonT>(min_level);

      m_scale = this->ComputeScale(num_levels, m_max, m_min);
    }

    // Method for converting samples to bin-ids
    template <CacheLoadModifier LOAD_MODIFIER>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(SampleT sample, int& bin, bool valid) const
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
    static constexpr bool is_range_transform = false;
    struct BracketCacheT
    {};

    _CCCL_DEVICE _CCCL_FORCEINLINE void PrecomputeOnDevice(int) {}
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
    template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(_SampleT sample, int& bin, bool valid) const
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

struct output_atomic_spill
{
  static constexpr bool is_private              = false;
  static constexpr bool defer_until_reconverged = false;
  static constexpr bool coalesce_before_probe   = false;

  template <typename CounterT, typename OutputCounterT>
  using target_type = OutputCounterT;

  template <typename CounterT>
  struct state
  {};

  template <typename OutputCounterT, typename ContributionT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void direct_spill(OutputCounterT* target, int bin, ContributionT contribution)
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

  template <typename CounterT, typename OutputCounterT, typename ContributionT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  spill(state<CounterT>&, OutputCounterT* target, int bin, ContributionT contribution)
  {
    if (bin >= 0)
    {
      direct_spill(target, bin, contribution);
    }
  }

  template <typename CounterT, typename OutputCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void finish(state<CounterT>&, OutputCounterT*)
  {}
};

struct private_block_spill
{
  static constexpr bool is_private              = true;
  static constexpr bool defer_until_reconverged = false;
  static constexpr bool coalesce_before_probe   = false;

  template <typename CounterT, typename OutputCounterT>
  using target_type = CounterT;

  template <typename CounterT>
  struct state
  {};

  template <typename CounterT, typename ContributionT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void direct_spill(CounterT* target, int bin, ContributionT contribution)
  {
    atomicAdd_block(&target[bin], static_cast<CounterT>(contribution));
  }

  template <typename CounterT, typename ContributionT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  spill(state<CounterT>&, CounterT* target, int bin, ContributionT contribution)
  {
    if (bin >= 0)
    {
      direct_spill(target, bin, contribution);
    }
  }

  template <typename CounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void finish(state<CounterT>&, CounterT*)
  {}
};

template <typename AtomicSpillOp>
struct warp_coalesced_spill
{
  static constexpr bool is_private              = AtomicSpillOp::is_private;
  static constexpr bool defer_until_reconverged = true;
  static constexpr bool coalesce_before_probe   = true;

  template <typename CounterT, typename OutputCounterT>
  using target_type = typename AtomicSpillOp::template target_type<CounterT, OutputCounterT>;

  template <typename CounterT>
  struct state
  {};

  template <typename CounterT, typename SpillCounterT, typename ContributionT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  spill(state<CounterT>&, SpillCounterT* target, int bin, ContributionT contribution)
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_70,
      (const unsigned int active = __activemask();
       const unsigned int peers  = __match_any_sync(active, static_cast<unsigned int>(bin));
       const int leader          = __ffs(static_cast<int>(peers)) - 1;
       const int lane_id         = static_cast<int>(threadIdx.x & 0x1f);
       if (bin >= 0 && lane_id == leader) {
         AtomicSpillOp::direct_spill(
           target, bin, static_cast<ContributionT>(contribution * static_cast<ContributionT>(__popc(peers))));
       }),
      (if (bin >= 0) { AtomicSpillOp::direct_spill(target, bin, contribution); }));
  }

  template <typename CounterT, typename SpillCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void finish(state<CounterT>&, SpillCounterT*)
  {}

  template <typename SpillCounterT, typename ContributionT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void direct_spill(SpillCounterT* target, int bin, ContributionT contribution)
  {
    AtomicSpillOp::direct_spill(target, bin, contribution);
  }
};

template <typename AtomicSpillOp>
struct rle_spill
{
  static constexpr bool is_private              = AtomicSpillOp::is_private;
  static constexpr bool defer_until_reconverged = false;
  static constexpr bool coalesce_before_probe   = false;

  template <typename CounterT, typename OutputCounterT>
  using target_type = typename AtomicSpillOp::template target_type<CounterT, OutputCounterT>;

  template <typename CounterT>
  struct state
  {
    int pending_bin        = -1;
    CounterT pending_count = CounterT{0};
  };

  template <typename CounterT, typename SpillCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void finish(state<CounterT>& pending, SpillCounterT* target)
  {
    if (pending.pending_bin >= 0)
    {
      AtomicSpillOp::direct_spill(target, pending.pending_bin, pending.pending_count);
      pending.pending_bin   = -1;
      pending.pending_count = CounterT{0};
    }
  }

  template <typename CounterT, typename SpillCounterT, typename ContributionT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  spill(state<CounterT>& pending, SpillCounterT* target, int bin, ContributionT contribution)
  {
    if (bin < 0)
    {
      return;
    }
    if (pending.pending_bin == bin)
    {
      pending.pending_count += static_cast<CounterT>(contribution);
      return;
    }
    finish(pending, target);
    pending.pending_bin   = bin;
    pending.pending_count = static_cast<CounterT>(contribution);
  }

  template <typename SpillCounterT, typename ContributionT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void direct_spill(SpillCounterT* target, int bin, ContributionT contribution)
  {
    AtomicSpillOp::direct_spill(target, bin, contribution);
  }
};

template <typename SpillOp, typename CounterT, typename SpillCounterT, typename ContributionT>
_CCCL_DEVICE _CCCL_FORCEINLINE bool probe_miss(
  typename SpillOp::template state<CounterT>& spill_state, SpillCounterT* target, int bin, ContributionT contribution)
{
  if constexpr (!SpillOp::defer_until_reconverged)
  {
    SpillOp::spill(spill_state, target, bin, contribution);
  }
  return true;
}

template <bool DisableSecondProbe = false>
struct cuckoo_cache_probe
{
  template <typename CounterT, typename SpillOp, typename SpillCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static bool apply(
    ::cuda::std::uint32_t* keys,
    CounterT* counts,
    typename SpillOp::template state<CounterT>& spill_state,
    SpillCounterT* spill_target,
    int bin,
    CounterT contribution,
    int cache_mask,
    int cache_log2)
  {
    constexpr ::cuda::std::uint32_t empty_key = UINT32_MAX;
    const auto bin_key                        = static_cast<::cuda::std::uint32_t>(bin);
    const auto try_slot                       = [&](int slot) {
      ::cuda::std::uint32_t key = keys[slot];
      if (key == bin_key)
      {
        atomicAdd_block(&counts[slot], contribution);
        return true;
      }
      if (key == empty_key)
      {
        key = atomicCAS_block(&keys[slot], empty_key, bin_key);
        if (key == empty_key || key == bin_key)
        {
          atomicAdd_block(&counts[slot], contribution);
          return true;
        }
      }
      return false;
    };

    const unsigned int hash = static_cast<unsigned int>(bin) * 2654435761u;
    const int primary       = static_cast<int>((hash >> (32 - cache_log2)) & static_cast<unsigned int>(cache_mask));
    if (try_slot(primary))
    {
      return false;
    }

    if constexpr (!DisableSecondProbe)
    {
      const unsigned int hash2 = (static_cast<unsigned int>(bin) ^ 0x9e3779b9u) * 2246822519u;
      const int secondary      = static_cast<int>((hash2 >> (32 - cache_log2)) & static_cast<unsigned int>(cache_mask));
      if (try_slot(secondary))
      {
        return false;
      }
    }
    return probe_miss<SpillOp>(spill_state, spill_target, bin, contribution);
  }
};

struct single_probe_cache
{
  template <typename CounterT, typename SpillOp, typename SpillCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static bool apply(
    ::cuda::std::uint32_t* keys,
    CounterT* counts,
    typename SpillOp::template state<CounterT>& spill_state,
    SpillCounterT* spill_target,
    int bin,
    CounterT contribution,
    int cache_mask,
    int cache_log2)
  {
    constexpr ::cuda::std::uint32_t empty_key = UINT32_MAX;
    const auto bin_key                        = static_cast<::cuda::std::uint32_t>(bin);
    const unsigned int hash                   = static_cast<unsigned int>(bin) * 2654435761u;
    const int slot            = static_cast<int>((hash >> (32 - cache_log2)) & static_cast<unsigned int>(cache_mask));
    ::cuda::std::uint32_t key = keys[slot];
    if (key == bin_key)
    {
      atomicAdd_block(&counts[slot], contribution);
      return false;
    }
    if (key == empty_key)
    {
      key = atomicCAS_block(&keys[slot], empty_key, bin_key);
      if (key == empty_key || key == bin_key)
      {
        atomicAdd_block(&counts[slot], contribution);
        return false;
      }
    }
    return probe_miss<SpillOp>(spill_state, spill_target, bin, contribution);
  }
};

struct no_cache_probe
{
  template <typename CounterT, typename SpillOp, typename SpillCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static bool apply(
    ::cuda::std::uint32_t*,
    CounterT*,
    typename SpillOp::template state<CounterT>& spill_state,
    SpillCounterT* spill_target,
    int bin,
    CounterT contribution,
    int,
    int)
  {
    return probe_miss<SpillOp>(spill_state, spill_target, bin, contribution);
  }
};

//! Agent for the policy-configurable cooperative high-bin histogram kernel.
template <typename PolicySelector,
          int NumChannels,
          int NumActiveChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename OutputCounterT,
          typename PrivatizedDecodeOpT,
          typename OffsetT,
          typename ProbeOp,
          typename SpillOp>
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
    static constexpr bool is_private_spill  = SpillOp::is_private;
    static constexpr bool is_no_cache       = ::cuda::std::is_same_v<ProbeOp, no_cache_probe>;
    using SpillCounterT                     = typename SpillOp::template target_type<CounterT, OutputCounterT>;
    static_assert(policy.high_bin_pixels_per_thread > 0, "Histogram cooperative unroll must be positive");
    static_assert(policy.high_bin_blocks_per_sm >= 0, "Histogram cooperative blocks per SM must not be negative");
    namespace cg        = ::cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const unsigned int tid_global    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_threads = gridDim.x * blockDim.x;

    if constexpr (!is_private_spill)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        for (unsigned int bin = tid_global; bin < static_cast<unsigned int>(num_output_bins_wrapper[ch]);
             bin += total_threads)
        {
          d_output_histograms_wrapper[ch][bin] = OutputCounterT{0};
        }
      }
      grid.sync();
    }

    static_assert(count_replicas > 0, "Histogram cache replication must be positive");

    extern __shared__ unsigned char dynamic_smem[];
    // Bin indices are non-negative `int` values, so uint32_t stores every possible key while preserving UINT32_MAX as
    // an empty-slot sentinel. A wider key would only reduce cache capacity without extending the supported bin range.
    auto* cache_keys = reinterpret_cast<::cuda::std::uint32_t*>(dynamic_smem);
    CounterT* cache_counts =
      reinterpret_cast<CounterT*>(cache_keys + static_cast<size_t>(NumActiveChannels) * cache_slots_per_channel);
    const int cache_mask = cache_slots_per_channel > 0 ? cache_slots_per_channel - 1 : 0;
    const int cache_log2 =
      cache_slots_per_channel > 0
        ? 31 - ::cuda::std::countl_zero(static_cast<::cuda::std::uint32_t>(cache_slots_per_channel))
        : 0;
    const int thread_idx    = static_cast<int>(threadIdx.x);
    const int block_threads = static_cast<int>(blockDim.x);

    if constexpr (!is_no_cache)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        auto* channel_keys       = cache_keys + static_cast<size_t>(ch) * cache_slots_per_channel;
        CounterT* channel_counts = cache_counts + static_cast<size_t>(ch) * count_replicas * cache_slots_per_channel;
        for (int slot = thread_idx; slot < cache_slots_per_channel; slot += block_threads)
        {
          channel_keys[slot] = UINT32_MAX;
        }
        for (int count = thread_idx; count < count_replicas * cache_slots_per_channel; count += block_threads)
        {
          channel_counts[count] = CounterT{0};
        }
      }
    }
    if constexpr (is_private_spill)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        CounterT* block_histogram =
          d_privatized_histograms_wrapper[ch] + static_cast<size_t>(blockIdx.x) * num_output_bins_wrapper[ch];
        for (int bin = thread_idx; bin < num_output_bins_wrapper[ch]; bin += block_threads)
        {
          block_histogram[bin] = CounterT{0};
        }
      }
    }
    __syncthreads();

    constexpr int unroll        = policy.high_bin_pixels_per_thread;
    const OffsetT total_pixels  = num_rows * num_row_pixels;
    const OffsetT step          = static_cast<OffsetT>(total_threads);
    const OffsetT chunk         = static_cast<OffsetT>(unroll) * step;
    const OffsetT chunk_count   = ::cuda::ceil_div(total_pixels, chunk);
    const bool contiguous_input = num_rows == 1;

    PrivatizedDecodeOpT decode_op[NumActiveChannels];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      decode_op[ch] = decode_op_wrapper[ch];
      decode_op[ch].PrecomputeOnDevice(policy.high_bin_interpolation_min_level_bytes);
    }

    constexpr bool use_mru_cache = NumActiveChannels == 1 && PrivatizedDecodeOpT::is_range_transform;
    [[maybe_unused]] typename PrivatizedDecodeOpT::BracketCacheT bracket_cache[NumActiveChannels];
    ::cuda::std::uint32_t* channel_keys[NumActiveChannels];
    CounterT* thread_counts[NumActiveChannels];
    SpillCounterT* spill_targets[NumActiveChannels];
    typename SpillOp::template state<CounterT> spill_states[NumActiveChannels]{};

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      channel_keys[ch]         = cache_keys + static_cast<size_t>(ch) * cache_slots_per_channel;
      CounterT* channel_counts = cache_counts + static_cast<size_t>(ch) * count_replicas * cache_slots_per_channel;
      const int replica        = static_cast<int>((threadIdx.x >> 5) % count_replicas);
      thread_counts[ch]        = channel_counts + static_cast<size_t>(replica) * cache_slots_per_channel;
      if constexpr (is_private_spill)
      {
        spill_targets[ch] =
          d_privatized_histograms_wrapper[ch] + static_cast<size_t>(blockIdx.x) * num_output_bins_wrapper[ch];
      }
      else
      {
        spill_targets[ch] = d_output_histograms_wrapper[ch];
      }
    }

    const auto update_bin = [&](int ch, int selected_bin, CounterT contribution) {
      if constexpr (SpillOp::defer_until_reconverged)
      {
        const bool should_spill = ProbeOp::template apply<CounterT, SpillOp, SpillCounterT>(
          channel_keys[ch],
          thread_counts[ch],
          spill_states[ch],
          spill_targets[ch],
          selected_bin,
          contribution,
          cache_mask,
          cache_log2);
        SpillOp::spill(spill_states[ch], spill_targets[ch], should_spill ? selected_bin : -1, contribution);
      }
      else
      {
        (void) ProbeOp::template apply<CounterT, SpillOp, SpillCounterT>(
          channel_keys[ch],
          thread_counts[ch],
          spill_states[ch],
          spill_targets[ch],
          selected_bin,
          contribution,
          cache_mask,
          cache_log2);
      }
    };

    const auto consume_bin = [&](int ch, int bin) {
      constexpr bool coalesce_before_probe =
        !is_no_cache && sizeof(CounterT) > sizeof(::cuda::std::uint32_t) && SpillOp::coalesce_before_probe;
      if constexpr (coalesce_before_probe)
      {
        const unsigned int lane_id = threadIdx.x & 0x1f;
        NV_IF_ELSE_TARGET(
          NV_PROVIDES_SM_70,
          (const unsigned int peers = __match_any_sync(0xffffffffu, static_cast<unsigned int>(bin));
           const int leader         = __ffs(static_cast<int>(peers)) - 1;
           if (bin >= 0 && static_cast<int>(lane_id) == leader) {
             update_bin(ch, bin, static_cast<CounterT>(__popc(peers)));
           }),
          (if (bin >= 0) { update_bin(ch, bin, CounterT{1}); }));
      }
      else if (bin >= 0)
      {
        update_bin(ch, bin, CounterT{1});
      }
    };

    using SampleValueT = it_value_t<SampleIteratorT>;
    if (contiguous_input)
    {
      if constexpr (NumActiveChannels == 1)
      {
        for (OffsetT chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx)
        {
          const OffsetT first_pixel = static_cast<OffsetT>(tid_global) + chunk_idx * chunk;
          SampleValueT staged_samples[unroll];
          bool valid_samples[unroll];
          int bins[unroll];

          _CCCL_PRAGMA_UNROLL_FULL()
          for (int item = 0; item < unroll; ++item)
          {
            const OffsetT pixel      = first_pixel + static_cast<OffsetT>(item) * step;
            valid_samples[item]      = pixel < total_pixels;
            const OffsetT safe_pixel = valid_samples[item] ? pixel : OffsetT{0};
            staged_samples[item]     = d_samples[safe_pixel * NumChannels];
          }

          _CCCL_PRAGMA_UNROLL_FULL()
          for (int item = 0; item < unroll; ++item)
          {
            int bin = -1;
            if (valid_samples[item])
            {
              if constexpr (use_mru_cache)
              {
                decode_op[0].template BinSelect<LOAD_DEFAULT>(staged_samples[item], bin, true, bracket_cache[0]);
              }
              else
              {
                decode_op[0].template BinSelect<LOAD_DEFAULT>(staged_samples[item], bin, true);
              }
              if (bin >= num_output_bins_wrapper[0])
              {
                bin = -1;
              }
            }
            bins[item] = bin;
          }

          _CCCL_PRAGMA_UNROLL_FULL()
          for (int item = 0; item < unroll; ++item)
          {
            consume_bin(0, bins[item]);
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
            for (OffsetT chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx)
            {
              const OffsetT first_pixel = static_cast<OffsetT>(tid_global) + chunk_idx * chunk;
              _CCCL_PRAGMA_UNROLL_FULL()
              for (int item = 0; item < unroll; ++item)
              {
                const OffsetT pixel       = first_pixel + static_cast<OffsetT>(item) * step;
                const bool valid          = pixel < total_pixels;
                const PixelT packed       = pixels[valid ? pixel : OffsetT{0}];
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
                  consume_bin(ch, bins[ch]);
                }
              }
            }
          }
          else
          {
            for (OffsetT chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx)
            {
              const OffsetT first_pixel = static_cast<OffsetT>(tid_global) + chunk_idx * chunk;
              _CCCL_PRAGMA_UNROLL_FULL()
              for (int item = 0; item < unroll; ++item)
              {
                const OffsetT pixel      = first_pixel + static_cast<OffsetT>(item) * step;
                const bool valid         = pixel < total_pixels;
                const OffsetT safe_pixel = valid ? pixel : OffsetT{0};
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
                  consume_bin(ch, bins[ch]);
                }
              }
            }
          }
        }
        else
        {
          for (OffsetT chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx)
          {
            const OffsetT first_pixel = static_cast<OffsetT>(tid_global) + chunk_idx * chunk;
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int item = 0; item < unroll; ++item)
            {
              const OffsetT pixel      = first_pixel + static_cast<OffsetT>(item) * step;
              const bool valid         = pixel < total_pixels;
              const OffsetT safe_pixel = valid ? pixel : OffsetT{0};
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
                consume_bin(ch, bins[ch]);
              }
            }
          }
        }
      }
    }
    else
    {
      for (OffsetT pixel = static_cast<OffsetT>(tid_global); pixel < total_pixels; pixel += step)
      {
        const OffsetT row          = pixel / num_row_pixels;
        const OffsetT pixel_offset = row * row_stride_samples + (pixel - row * num_row_pixels) * NumChannels;

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int ch = 0; ch < NumActiveChannels; ++ch)
        {
          int bin = -1;
          if constexpr (use_mru_cache)
          {
            decode_op[ch].template BinSelect<LOAD_DEFAULT>(d_samples[pixel_offset + ch], bin, true, bracket_cache[ch]);
          }
          else
          {
            decode_op[ch].template BinSelect<LOAD_DEFAULT>(d_samples[pixel_offset + ch], bin, true);
          }
          if (bin >= 0 && bin < num_output_bins_wrapper[ch])
          {
            SpillOp::spill(spill_states[ch], spill_targets[ch], bin, CounterT{1});
          }
        }
      }
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      SpillOp::finish(spill_states[ch], spill_targets[ch]);
    }

    if constexpr (!is_no_cache)
    {
      __syncthreads();
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        auto* channel_keys       = cache_keys + static_cast<size_t>(ch) * cache_slots_per_channel;
        CounterT* channel_counts = cache_counts + static_cast<size_t>(ch) * count_replicas * cache_slots_per_channel;
        for (int slot = thread_idx; slot < cache_slots_per_channel; slot += block_threads)
        {
          const auto key = channel_keys[slot];
          if (key != UINT32_MAX)
          {
            CounterT count = CounterT{0};
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int replica = 0; replica < count_replicas; ++replica)
            {
              count += channel_counts[static_cast<size_t>(replica) * cache_slots_per_channel + slot];
            }
            if (count > CounterT{0})
            {
              SpillOp::direct_spill(spill_targets[ch], static_cast<int>(key), count);
            }
          }
        }
      }
    }

    if constexpr (is_private_spill)
    {
      grid.sync();
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        const unsigned int num_bins = static_cast<unsigned int>(num_output_bins_wrapper[ch]);
        for (unsigned int bin = tid_global; bin < num_bins; bin += total_threads)
        {
          OutputCounterT total = OutputCounterT{0};
          for (unsigned int block = 0; block < gridDim.x; ++block)
          {
            total += static_cast<OutputCounterT>(
              d_privatized_histograms_wrapper[ch][static_cast<size_t>(block) * num_bins + bin]);
          }
          d_output_histograms_wrapper[ch][bin] = total;
        }
      }
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
          typename OffsetT,
          typename ProbeOp,
          typename SpillOp>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().high_bin_threads()),
                  int(current_policy<PolicySelector>().high_bin_min_blocks()))
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
    OffsetT,
    ProbeOp,
    SpillOp>::Consume(d_samples,
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
