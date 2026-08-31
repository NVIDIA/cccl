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
#include <cuda/std/__numeric/reduce.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cooperative_groups.h>

CUB_NAMESPACE_BEGIN
namespace detail::histogram
{
template <typename UInt>
struct fast_divide_by_constant
{
  static_assert(::cuda::std::is_unsigned_v<UInt>, "fast_divide_by_constant requires an unsigned integer divisor type");
  static_assert(sizeof(UInt) == 4 || sizeof(UInt) == 8, "fast_divide_by_constant supports 32-bit or 64-bit divisors");

  static constexpr int bits = static_cast<int>(sizeof(UInt) * 8);

  UInt magic;
  unsigned char shift;
  unsigned char mode;

  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static int count_leading_zeros(::cuda::std::uint64_t value)
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return value == 0 ? 64 : __clzll(static_cast<long long>(value));),
                      (return value == 0 ? 64 : __builtin_clzll(value);));
  }

  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static int count_leading_zeros(::cuda::std::uint32_t value)
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return value == 0 ? 32 : __clz(static_cast<int>(value));),
                      (return value == 0 ? 32 : __builtin_clz(value);));
  }

  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static int ceil_log2(UInt divisor)
  {
    if (divisor <= UInt{1})
    {
      return 0;
    }
    if constexpr (sizeof(UInt) == 4)
    {
      return bits - count_leading_zeros(static_cast<::cuda::std::uint32_t>(divisor - UInt{1}));
    }
    else
    {
      return bits - count_leading_zeros(static_cast<::cuda::std::uint64_t>(divisor - UInt{1}));
    }
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(UInt divisor)
  {
    if (divisor <= UInt{1})
    {
      magic = UInt{0};
      shift = 0;
      mode  = 0;
      return;
    }
    if ((divisor & (divisor - UInt{1})) == UInt{0})
    {
      magic = UInt{0};
      shift = static_cast<unsigned char>(ceil_log2(divisor));
      mode  = 1;
      return;
    }

    const int log2_divisor = ceil_log2(divisor);
    if (log2_divisor == bits)
    {
      magic = divisor;
      shift = 0;
      mode  = 3;
      return;
    }
    if constexpr (sizeof(UInt) == 8)
    {
#if _CCCL_HAS_INT128()
      const __uint128_t numerator   = static_cast<__uint128_t>(1) << (bits + log2_divisor);
      const __uint128_t denominator = static_cast<__uint128_t>(divisor);
      magic                         = static_cast<UInt>((numerator + denominator - 1) / denominator);
#else
      UInt quotient  = 0;
      UInt remainder = 0;
      for (int bit = bits + log2_divisor; bit >= 0; --bit)
      {
        UInt next_remainder     = (remainder << 1) | (bit == bits + log2_divisor ? UInt{1} : UInt{0});
        const bool carry        = (remainder >> (bits - 1)) != 0;
        const UInt quotient_bit = (carry || next_remainder >= divisor) ? UInt{1} : UInt{0};
        if (quotient_bit != 0)
        {
          next_remainder -= divisor;
        }
        remainder = next_remainder;
        quotient  = (quotient << 1) | quotient_bit;
      }
      magic = quotient + (remainder != 0 ? UInt{1} : UInt{0});
#endif
    }
    else
    {
      const ::cuda::std::uint64_t numerator   = ::cuda::std::uint64_t{1} << (bits + log2_divisor);
      const ::cuda::std::uint64_t denominator = static_cast<::cuda::std::uint64_t>(divisor);
      magic                                   = static_cast<UInt>((numerator + denominator - 1) / denominator);
    }
    shift = static_cast<unsigned char>(log2_divisor);
    mode  = 2;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UInt Divide(UInt numerator) const
  {
    if (mode == 0)
    {
      return numerator;
    }
    if (mode == 1)
    {
      return numerator >> shift;
    }
    if (mode == 3)
    {
      return numerator / magic;
    }

    UInt high;
    if constexpr (sizeof(UInt) == 8)
    {
      NV_IF_ELSE_TARGET(
        NV_IS_DEVICE,
        (high = static_cast<UInt>(
           __umul64hi(static_cast<unsigned long long>(magic), static_cast<unsigned long long>(numerator)));),
        ({
#if _CCCL_HAS_INT128()
          high = static_cast<UInt>((static_cast<__uint128_t>(magic) * static_cast<__uint128_t>(numerator)) >> bits);
#else
          const ::cuda::std::uint64_t a_low     = static_cast<::cuda::std::uint32_t>(magic);
          const ::cuda::std::uint64_t a_high    = magic >> 32;
          const ::cuda::std::uint64_t b_low     = static_cast<::cuda::std::uint32_t>(numerator);
          const ::cuda::std::uint64_t b_high    = numerator >> 32;
          const ::cuda::std::uint64_t low_low   = a_low * b_low;
          const ::cuda::std::uint64_t low_high  = a_low * b_high;
          const ::cuda::std::uint64_t high_low  = a_high * b_low;
          const ::cuda::std::uint64_t high_high = a_high * b_high;
          const ::cuda::std::uint64_t middle =
            (low_low >> 32) + static_cast<::cuda::std::uint32_t>(low_high)
            + static_cast<::cuda::std::uint32_t>(high_low);
          high = high_high + (low_high >> 32) + (high_low >> 32) + (middle >> 32);
#endif
        }));
    }
    else
    {
      high = static_cast<UInt>(
        (static_cast<::cuda::std::uint64_t>(magic) * static_cast<::cuda::std::uint64_t>(numerator)) >> bits);
    }
    return (((numerator - high) >> 1) + high) >> (shift - 1);
  }
};

template <typename SampleValueT, typename SampleIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE const SampleValueT* sample_native_pointer(SampleIteratorT itr)
{
  if constexpr (::cuda::std::is_pointer_v<SampleIteratorT>)
  {
    return itr;
  }
  else
  {
    return NativePointer(itr);
  }
  _CCCL_UNREACHABLE();
}

template <typename CounterT>
_CCCL_DEVICE _CCCL_FORCEINLINE void histogram_atomic_add(CounterT* address, CounterT value)
{
  if constexpr (::cuda::std::is_integral_v<CounterT> && sizeof(CounterT) == sizeof(::cuda::std::uint64_t))
  {
    // CUDA's 64-bit integer atomic overload is spelled in terms of unsigned long long.
    // Keep the width decision explicit and use that spelling only at the API boundary.
    atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value));
  }
  else
  {
    atomicAdd(address, value);
  }
}

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
    static constexpr bool is_range_transform = true;

    struct BracketCacheT
    {
      LevelT lo{};
      LevelT hi{};
      int bin = -1;
    };

    LevelIteratorT d_levels; // Pointer to levels array
    int num_output_levels; // Number of levels in array
    LevelT first{};
    LevelT middle{};
    LevelT last{};
    float inverse_scale_low  = 0.0f;
    float inverse_scale_high = 0.0f;
    int middle_bin           = 0;
    bool has_precompute      = false;

    //! @brief Initializer
    //!
    //! @param d_levels_ Pointer to levels array
    //! @param num_output_levels_ Number of levels in array
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(LevelIteratorT d_levels_, int num_output_levels_)
    {
      this->d_levels          = d_levels_;
      this->num_output_levels = num_output_levels_;
      this->has_precompute    = false;
    }

    template <typename T>
    [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static float level_distance(T upper, T lower)
    {
      if constexpr (::cuda::std::is_integral_v<T>)
      {
        using UnsignedT = ::cuda::std::make_unsigned_t<T>;
        return static_cast<float>(static_cast<UnsignedT>(upper) - static_cast<UnsignedT>(lower));
      }
      else
      {
        return static_cast<float>(upper - lower);
      }
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE void PrecomputeOnDevice(int interpolation_min_bins)
    {
      const int num_bins = num_output_levels - 1;
      if (num_bins < interpolation_min_bins)
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

      first      = first_level;
      last       = last_level;
      middle_bin = num_bins >> 1;
      middle     = wrapped_levels[middle_bin];
      if (!(first < middle) || !(middle < last))
      {
        middle_bin        = 0;
        inverse_scale_low = static_cast<float>(num_bins) / level_distance(last, first);
      }
      else
      {
        inverse_scale_low  = static_cast<float>(middle_bin) / level_distance(middle, first);
        inverse_scale_high = static_cast<float>(num_bins - middle_bin) / level_distance(last, middle);
      }
      has_precompute = true;
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
      if (!valid)
      {
        return;
      }

      const LevelT value = static_cast<LevelT>(sample);
      if (!has_precompute)
      {
        bin = UpperBound(wrapped_levels, num_output_levels, value) - 1;
        bin = bin < num_bins ? bin : -1;
        return;
      }
      if (value < first || !(value < last))
      {
        bin = -1;
        return;
      }

      int guess       = value < middle || middle_bin == 0
                        ? static_cast<int>(level_distance(value, first) * inverse_scale_low)
                        : middle_bin + static_cast<int>(level_distance(value, middle) * inverse_scale_high);
      guess           = guess < 0 ? 0 : (guess < num_bins ? guess : num_bins - 1);
      const LevelT lo = wrapped_levels[guess];
      const LevelT hi = wrapped_levels[guess + 1];
      if (!(value < lo) && value < hi)
      {
        bin = guess;
        return;
      }
      bin = UpperBound(wrapped_levels, num_output_levels, value) - 1;
    }

    template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void
    BinSelect(_SampleT sample, int& bin, bool valid, BracketCacheT& bracket) const
    {
      const LevelT value = static_cast<LevelT>(sample);
      if (valid && bracket.bin >= 0 && !(value < bracket.lo) && value < bracket.hi)
      {
        bin = bracket.bin;
        return;
      }

      BinSelect<LOAD_MODIFIER>(sample, bin, valid);
      if (bin >= 0)
      {
        using WrappedLevelIteratorT =
          ::cuda::std::_If<::cuda::std::is_pointer_v<LevelIteratorT>,
                           CacheModifiedInputIterator<LOAD_MODIFIER, LevelT, OffsetT>,
                           LevelIteratorT>;
        WrappedLevelIteratorT wrapped_levels(d_levels);
        bracket.lo  = wrapped_levels[bin];
        bracket.hi  = wrapped_levels[bin + 1];
        bracket.bin = bin;
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

    using FractionStorageT = ::cuda::std::_If<is_integral_excl_int128<CommonT>::value, IntArithmeticT, CommonT>;

    union ScaleT
    {
      // Used when CommonT is not floating-point to avoid intermediate
      // rounding errors (see NVIDIA/cub#489).
      struct FractionT
      {
        FractionStorageT bins;
        FractionStorageT range;
        fast_divide_by_constant<IntArithmeticT> range_divider;
        bool bins_equal_range;
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
      result.fraction.bins_equal_range = result.fraction.bins == result.fraction.range;
      result.fraction.range_divider.Init(static_cast<IntArithmeticT>(result.fraction.range));
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
      if (scale.fraction.bins_equal_range)
      {
        return static_cast<int>(distance);
      }
      const IntArithmeticT numerator = distance * static_cast<IntArithmeticT>(scale.fraction.bins);
      return static_cast<int>(scale.fraction.range_divider.Divide(numerator));
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

template <typename CounterT>
_CCCL_DEVICE _CCCL_FORCEINLINE bool histogram_cache_probe(
  ::cuda::std::uint32_t* keys,
  CounterT* counts,
  int count_replicas,
  int bin,
  CounterT contribution,
  int cache_mask,
  int cache_log2,
  HistogramCacheAlgorithm cache_algorithm,
  bool use_second_probe)
{
  if (cache_algorithm == HistogramCacheAlgorithm::none)
  {
    return false;
  }

  constexpr ::cuda::std::uint32_t empty_key = UINT32_MAX;
  const auto bin_key                        = static_cast<::cuda::std::uint32_t>(bin);
  const auto try_slot                       = [&](int slot) {
    ::cuda::std::uint32_t key = keys[slot];
    if (key == empty_key)
    {
      key = atomicCAS_block(&keys[slot], empty_key, bin_key);
    }
    if (key == empty_key || key == bin_key)
    {
      const int replica = static_cast<int>((threadIdx.x >> 5) % count_replicas);
      atomicAdd_block(&counts[replica * (cache_mask + 1) + slot], contribution);
      return true;
    }
    return false;
  };

  const unsigned int hash = static_cast<unsigned int>(bin) * 2654435761u;
  const int primary       = static_cast<int>((hash >> (32 - cache_log2)) & static_cast<unsigned int>(cache_mask));
  if (try_slot(primary))
  {
    return true;
  }

  if (use_second_probe)
  {
    const unsigned int hash2 = (static_cast<unsigned int>(bin) ^ 0x9e3779b9u) * 2246822519u;
    const int secondary      = static_cast<int>((hash2 >> (32 - cache_log2)) & static_cast<unsigned int>(cache_mask));
    return try_slot(secondary);
  }
  return false;
}

//! Agent for the policy-configurable cooperative high-bin histogram kernel.
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
    static_assert(policy.high_bin_pixels_per_thread > 0, "Histogram cooperative unroll must be positive");
    static_assert(
      policy.high_bin_cache == HistogramCacheAlgorithm::none
        || (policy.high_bin_cache_entries_per_channel >= 32
            && (policy.high_bin_cache_entries_per_channel & (policy.high_bin_cache_entries_per_channel - 1)) == 0),
      "Histogram cache entries per channel must be a power of two of at least 32");
    namespace cg        = ::cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const unsigned int tid_global    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_threads = gridDim.x * blockDim.x;

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

    static_assert(count_replicas > 0, "Histogram cache replication must be positive");

    extern __shared__ unsigned char dynamic_smem[];
    auto* cache_keys = reinterpret_cast<::cuda::std::uint32_t*>(dynamic_smem);
    CounterT* cache_counts =
      reinterpret_cast<CounterT*>(cache_keys + static_cast<size_t>(NumActiveChannels) * cache_slots_per_channel);
    const int cache_mask = cache_slots_per_channel > 0 ? cache_slots_per_channel - 1 : 0;
    const int cache_log2 =
      cache_slots_per_channel > 0 ? 31 - __clz(static_cast<unsigned int>(cache_slots_per_channel)) : 0;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      auto* channel_keys       = cache_keys + static_cast<size_t>(ch) * cache_slots_per_channel;
      CounterT* channel_counts = cache_counts + static_cast<size_t>(ch) * count_replicas * cache_slots_per_channel;
      for (int slot = threadIdx.x; slot < cache_slots_per_channel; slot += blockDim.x)
      {
        channel_keys[slot] = ~::cuda::std::uint32_t{0};
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int replica = 0; replica < count_replicas; ++replica)
        {
          channel_counts[static_cast<size_t>(replica) * cache_slots_per_channel + slot] = CounterT{0};
        }
      }

      if constexpr (policy.high_bin_spill == HistogramSpillAlgorithm::global_memory_privatized)
      {
        CounterT* block_histogram =
          d_privatized_histograms_wrapper[ch] + static_cast<size_t>(blockIdx.x) * num_output_bins_wrapper[ch];
        for (int bin = threadIdx.x; bin < num_output_bins_wrapper[ch]; bin += blockDim.x)
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
    const unsigned int lane_id  = threadIdx.x & 0x1f;
    const bool contiguous_input = num_rows == 1;

    PrivatizedDecodeOpT decode_op[NumActiveChannels];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      decode_op[ch] = decode_op_wrapper[ch];
      decode_op[ch].PrecomputeOnDevice(policy.high_bin_interpolation_min_bins);
    }

    constexpr bool use_mru_cache = NumActiveChannels == 1 && PrivatizedDecodeOpT::is_range_transform;
    typename PrivatizedDecodeOpT::BracketCacheT bracket_cache[NumActiveChannels];
    int pending_bin[NumActiveChannels];
    CounterT pending_count[NumActiveChannels];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      pending_bin[ch]   = -1;
      pending_count[ch] = CounterT{0};
    }

    const auto spill_bin = [&](int ch, int selected_bin, CounterT contribution) {
      if constexpr (policy.high_bin_aggregation == HistogramAggregationAlgorithm::rle)
      {
        if (pending_bin[ch] == selected_bin)
        {
          pending_count[ch] += contribution;
          return;
        }
        if (pending_bin[ch] >= 0)
        {
          if constexpr (policy.high_bin_spill == HistogramSpillAlgorithm::global_memory_privatized)
          {
            CounterT* block_histogram =
              d_privatized_histograms_wrapper[ch] + static_cast<size_t>(blockIdx.x) * num_output_bins_wrapper[ch];
            atomicAdd_block(&block_histogram[pending_bin[ch]], pending_count[ch]);
          }
          else
          {
            histogram_atomic_add(&d_output_histograms_wrapper[ch][pending_bin[ch]],
                                 static_cast<OutputCounterT>(pending_count[ch]));
          }
        }
        pending_bin[ch]   = selected_bin;
        pending_count[ch] = contribution;
      }
      else if constexpr (policy.high_bin_spill == HistogramSpillAlgorithm::global_memory_privatized)
      {
        CounterT* block_histogram =
          d_privatized_histograms_wrapper[ch] + static_cast<size_t>(blockIdx.x) * num_output_bins_wrapper[ch];
        atomicAdd_block(&block_histogram[selected_bin], contribution);
      }
      else
      {
        histogram_atomic_add(&d_output_histograms_wrapper[ch][selected_bin], static_cast<OutputCounterT>(contribution));
      }
    };

    const auto update_bin = [&](int ch, int selected_bin, CounterT contribution) {
      auto* channel_keys          = cache_keys + static_cast<size_t>(ch) * cache_slots_per_channel;
      CounterT* channel_counts    = cache_counts + static_cast<size_t>(ch) * count_replicas * cache_slots_per_channel;
      const bool use_second_probe = policy.high_bin_cache == HistogramCacheAlgorithm::cuckoo
                                 && num_output_bins_wrapper[ch] < policy.high_bin_cache_cuckoo_max_bins;
      if (!histogram_cache_probe(
            channel_keys,
            channel_counts,
            count_replicas,
            selected_bin,
            contribution,
            cache_mask,
            cache_log2,
            policy.high_bin_cache,
            use_second_probe))
      {
        spill_bin(ch, selected_bin, contribution);
      }
    };

    const auto consume_bin = [&](int ch, int bin) {
      if constexpr (policy.high_bin_aggregation == HistogramAggregationAlgorithm::warp_coalesced)
      {
        NV_IF_ELSE_TARGET(
          NV_PROVIDES_SM_70,
          (const unsigned int active = __activemask();
           const unsigned int peers  = __match_any_sync(active, static_cast<unsigned int>(bin));
           const int leader          = __ffs(static_cast<int>(peers)) - 1;
           if (bin >= 0 && static_cast<int>(lane_id) == leader) {
             update_bin(ch, bin, static_cast<CounterT>(__popc(peers)));
           }),
          (if (bin >= 0) { update_bin(ch, bin, CounterT{1}); }));
      }
      else if constexpr (policy.high_bin_aggregation == HistogramAggregationAlgorithm::rle)
      {
        if (bin >= 0)
        {
          update_bin(ch, bin, CounterT{1});
        }
      }
      else if (bin >= 0)
      {
        update_bin(ch, bin, CounterT{1});
      }
    };

    using SampleValueT = it_value_t<SampleIteratorT>;
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
          const OffsetT pixel = first_pixel + static_cast<OffsetT>(item) * step;
          valid_samples[item] = pixel < total_pixels;
          OffsetT pixel_offset{};
          if (valid_samples[item])
          {
            if (contiguous_input)
            {
              pixel_offset = pixel * NumChannels;
            }
            else
            {
              const OffsetT row = pixel / num_row_pixels;
              pixel_offset      = row * row_stride_samples + (pixel - row * num_row_pixels) * NumChannels;
            }
            staged_samples[item] = d_samples[pixel_offset];
          }
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
      const auto consume_pixels = [&](auto load_pixel) {
        for (OffsetT chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx)
        {
          const OffsetT first_pixel = static_cast<OffsetT>(tid_global) + chunk_idx * chunk;
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int item = 0; item < unroll; ++item)
          {
            const OffsetT pixel = first_pixel + static_cast<OffsetT>(item) * step;
            const bool valid    = pixel < total_pixels;
            SampleValueT samples[NumActiveChannels];
            int bins[NumActiveChannels];
            load_pixel(pixel, valid, samples);

            _CCCL_PRAGMA_UNROLL_FULL()
            for (int ch = 0; ch < NumActiveChannels; ++ch)
            {
              int bin = -1;
              if (valid)
              {
                decode_op[ch].template BinSelect<LOAD_DEFAULT>(samples[ch], bin, true);
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
      };

      if constexpr ((NumChannels == 2 || NumChannels == 4) && ::cuda::std::is_trivially_copyable_v<SampleValueT>)
      {
        using PixelT            = typename CubVector<SampleValueT, NumChannels>::Type;
        const auto* native_base = sample_native_pointer<SampleValueT>(d_samples);
        const bool vectorizable = contiguous_input && native_base != nullptr
                               && (reinterpret_cast<size_t>(native_base) & (alignof(PixelT) - 1)) == 0;
        if (vectorizable)
        {
          const PixelT* pixels = reinterpret_cast<const PixelT*>(native_base);
          consume_pixels([&](OffsetT pixel, bool valid, SampleValueT(&samples)[NumActiveChannels]) {
            if (valid)
            {
              const PixelT packed_pixel       = pixels[pixel];
              const SampleValueT* pixel_lanes = reinterpret_cast<const SampleValueT*>(&packed_pixel);
              _CCCL_PRAGMA_UNROLL_FULL()
              for (int ch = 0; ch < NumActiveChannels; ++ch)
              {
                samples[ch] = pixel_lanes[ch];
              }
            }
          });
        }
        else
        {
          consume_pixels([&](OffsetT pixel, bool valid, SampleValueT(&samples)[NumActiveChannels]) {
            OffsetT pixel_offset{};
            if (valid)
            {
              const OffsetT row = contiguous_input ? OffsetT{0} : pixel / num_row_pixels;
              const OffsetT col = contiguous_input ? pixel : pixel - row * num_row_pixels;
              pixel_offset      = row * row_stride_samples + col * NumChannels;
            }
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int ch = 0; ch < NumActiveChannels; ++ch)
            {
              if (valid)
              {
                samples[ch] = d_samples[pixel_offset + ch];
              }
            }
          });
        }
      }
      else
      {
        consume_pixels([&](OffsetT pixel, bool valid, SampleValueT(&samples)[NumActiveChannels]) {
          OffsetT pixel_offset{};
          if (valid)
          {
            const OffsetT row = contiguous_input ? OffsetT{0} : pixel / num_row_pixels;
            const OffsetT col = contiguous_input ? pixel : pixel - row * num_row_pixels;
            pixel_offset      = row * row_stride_samples + col * NumChannels;
          }
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int ch = 0; ch < NumActiveChannels; ++ch)
          {
            if (valid)
            {
              samples[ch] = d_samples[pixel_offset + ch];
            }
          }
        });
      }
    }

    if constexpr (policy.high_bin_aggregation == HistogramAggregationAlgorithm::rle)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        if (pending_bin[ch] >= 0)
        {
          if constexpr (policy.high_bin_spill == HistogramSpillAlgorithm::global_memory_privatized)
          {
            CounterT* block_histogram =
              d_privatized_histograms_wrapper[ch] + static_cast<size_t>(blockIdx.x) * num_output_bins_wrapper[ch];
            atomicAdd_block(&block_histogram[pending_bin[ch]], pending_count[ch]);
          }
          else
          {
            histogram_atomic_add(&d_output_histograms_wrapper[ch][pending_bin[ch]],
                                 static_cast<OutputCounterT>(pending_count[ch]));
          }
        }
      }
    }

    __syncthreads();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      auto* channel_keys       = cache_keys + static_cast<size_t>(ch) * cache_slots_per_channel;
      CounterT* channel_counts = cache_counts + static_cast<size_t>(ch) * count_replicas * cache_slots_per_channel;
      for (int slot = threadIdx.x; slot < cache_slots_per_channel; slot += blockDim.x)
      {
        const auto key = channel_keys[slot];
        if (key != ~::cuda::std::uint32_t{0})
        {
          CounterT count = CounterT{0};
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int replica = 0; replica < count_replicas; ++replica)
          {
            count += channel_counts[static_cast<size_t>(replica) * cache_slots_per_channel + slot];
          }
          if (count > CounterT{0})
          {
            if constexpr (policy.high_bin_spill == HistogramSpillAlgorithm::global_memory_privatized)
            {
              CounterT* block_histogram =
                d_privatized_histograms_wrapper[ch] + static_cast<size_t>(blockIdx.x) * num_output_bins_wrapper[ch];
              atomicAdd_block(&block_histogram[key], count);
            }
            else
            {
              histogram_atomic_add(&d_output_histograms_wrapper[ch][key], static_cast<OutputCounterT>(count));
            }
          }
        }
      }
    }

    if constexpr (policy.high_bin_spill == HistogramSpillAlgorithm::global_memory_privatized)
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
          typename OffsetT>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().high_bin_threads()))
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
