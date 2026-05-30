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

#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__numeric/reduce.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#if !_CCCL_COMPILER(NVRTC)
#  include <cooperative_groups.h>
#endif // !_CCCL_COMPILER(NVRTC)

CUB_NAMESPACE_BEGIN
namespace detail::histogram
{

//! @brief Self-contained "round-up" / libdivide-style fast unsigned division
//! by a runtime constant divisor.
//!
//! Replaces a 64-bit integer divide in the hot path of `ScaleTransform::ComputeBin`
//! (the `EVEN`-integer histogram classify) with a multiply-high + shift sequence.
//! Magic-multiplier and shift are precomputed on the host inside
//! `ScaleTransform::Init` and propagated to the device via the per-channel
//! decode-op argument.
//!
//! The implementation follows the classic Granlund-Möller / Hacker's-Delight
//! "round-up" form (libdivide's branchfree variant): for a divisor `d >= 2`,
//! `n / d = ((((n - mulhi(M, n)) >> 1) + mulhi(M, n)) >> (L-1))` where
//! `L = ceil(log2(d))` and `M = ceil(2^(N+L) / d) - 2^N` fits in `N` bits.
//! For `d == 1` we return `n` directly; for `d` a power of two we degenerate
//! to a plain shift.
//!
//! Default-constructible so a zero-initialised instance is well-defined
//! (acts as the identity divider, divisor==1). `Init` overwrites the state
//! before any `Divide` call on the device.
template <typename UInt>
struct fast_divide_by_constant
{
  static_assert(::cuda::std::is_unsigned_v<UInt>, "fast_divide_by_constant requires an unsigned integer divisor type");
  static_assert(sizeof(UInt) == 4 || sizeof(UInt) == 8, "fast_divide_by_constant supports 32-bit or 64-bit divisors");

  static constexpr int kBits = static_cast<int>(sizeof(UInt) * 8);

  UInt magic; // multiplier (low N bits of the round-up multiplier)
  unsigned char shift; // shift amount; for power-of-two divisors this is log2(d)
  unsigned char mode; // 0: identity (d == 1); 1: power-of-two; 2: general round-up

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static int CountLeadingZeros64(::cuda::std::uint64_t x)
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return x == 0 ? 64 : __clzll(static_cast<long long>(x));),
                      (return x == 0 ? 64 : __builtin_clzll(x);));
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static int CountLeadingZeros32(::cuda::std::uint32_t x)
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return x == 0 ? 32 : __clz(static_cast<int>(x));), (return x == 0 ? 32 : __builtin_clz(x);));
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static int CeilLog2(UInt d)
  {
    if (d <= UInt{1})
    {
      return 0;
    }
    if constexpr (sizeof(UInt) == 4)
    {
      return kBits - CountLeadingZeros32(static_cast<::cuda::std::uint32_t>(d - UInt{1}));
    }
    else
    {
      return kBits - CountLeadingZeros64(static_cast<::cuda::std::uint64_t>(d - UInt{1}));
    }
  }

  //! @brief Computes the magic-multiplier and shift for divisor `d`.
  //!
  //! Must be called before any `Divide` call. Computed on host (or device, if
  //! constructible from device code), but only the host call site is exercised
  //! today.
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(UInt d)
  {
    if (d <= UInt{1})
    {
      magic = UInt{0};
      shift = 0;
      mode  = 0; // identity
      return;
    }
    // Power of two?
    if ((d & (d - UInt{1})) == UInt{0})
    {
      magic = UInt{0};
      // shift = log2(d); CeilLog2 gives that for power-of-two.
      shift = static_cast<unsigned char>(CeilLog2(d));
      mode  = 1;
      return;
    }
    // General round-up form. L = ceil(log2(d)); 2^(L-1) < d < 2^L.
    const int L = CeilLog2(d);
    // M_full = ceil(2^(N+L) / d). For d not a power of two, 2^N < M_full < 2^(N+1),
    // so M_low = M_full - 2^N fits in N bits.
    if constexpr (sizeof(UInt) == 8)
    {
      // 128-bit arithmetic. Use compiler __uint128_t when available.
#if _CCCL_HAS_INT128()
      const __uint128_t numer = (static_cast<__uint128_t>(1) << (kBits + L));
      const __uint128_t denom = static_cast<__uint128_t>(d);
      // ceil(numer / denom) == (numer + denom - 1) / denom
      const __uint128_t M_full = (numer + denom - 1) / denom;
      magic                    = static_cast<UInt>(M_full); // truncates the high bit (==1 by construction)
#else
      // Fallback: long division of 2^(N+L) by d via Newton-style iteration.
      // For our histogram divisors (~2^31 max) this branch never runs, but keep
      // it defensive for portability.
      UInt q = 0;
      UInt r = 0;
      for (int b = kBits + L; b >= 0; --b)
      {
        // (r << 1) | bit_of_2^(N+L) at position b
        UInt new_r = (r << 1) | (b == kBits + L ? UInt{1} : UInt{0});
        bool carry = (r >> (kBits - 1)) != 0;
        UInt qbit  = (carry || new_r >= d) ? UInt{1} : UInt{0};
        if (qbit)
        {
          new_r -= d;
        }
        r = new_r;
        q = (q << 1) | qbit;
      }
      // q == floor(2^(N+L) / d); add (remainder != 0 ? 1 : 0) for ceil.
      magic = q + (r != 0 ? UInt{1} : UInt{0});
#endif
    }
    else
    {
      // 32-bit divisor: do the magic in 64-bit.
      const ::cuda::std::uint64_t numer  = (::cuda::std::uint64_t{1} << (kBits + L));
      const ::cuda::std::uint64_t denom  = static_cast<::cuda::std::uint64_t>(d);
      const ::cuda::std::uint64_t M_full = (numer + denom - 1) / denom;
      magic                              = static_cast<UInt>(M_full); // truncates the high bit
    }
    shift = static_cast<unsigned char>(L);
    mode  = 2;
  }

  //! @brief Computes `n / divisor` exactly for any non-negative `n` representable
  //! in `UInt`.
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UInt Divide(UInt n) const
  {
    if (mode == 0)
    {
      return n; // identity (divisor == 1)
    }
    if (mode == 1)
    {
      return n >> shift; // power-of-two divisor
    }
    // General round-up form: n / d = ((((n - hi) >> 1) + hi) >> (L-1)) with hi = mulhi(magic, n).
    UInt hi;
    if constexpr (sizeof(UInt) == 8)
    {
      NV_IF_ELSE_TARGET(
        NV_IS_DEVICE,
        (hi = static_cast<UInt>(__umul64hi(static_cast<unsigned long long>(magic),
                                           static_cast<unsigned long long>(n)));),
        ({
#if _CCCL_HAS_INT128()
          hi = static_cast<UInt>((static_cast<__uint128_t>(magic) * static_cast<__uint128_t>(n)) >> kBits);
#else
          // Manual 64x64->128 high mul, host fallback.
          const ::cuda::std::uint64_t a_lo = static_cast<::cuda::std::uint32_t>(magic);
          const ::cuda::std::uint64_t a_hi = magic >> 32;
          const ::cuda::std::uint64_t b_lo = static_cast<::cuda::std::uint32_t>(n);
          const ::cuda::std::uint64_t b_hi = n >> 32;
          const ::cuda::std::uint64_t ll   = a_lo * b_lo;
          const ::cuda::std::uint64_t lh   = a_lo * b_hi;
          const ::cuda::std::uint64_t hl   = a_hi * b_lo;
          const ::cuda::std::uint64_t hh   = a_hi * b_hi;
          const ::cuda::std::uint64_t mid  = (ll >> 32) + static_cast<::cuda::std::uint32_t>(lh) + static_cast<::cuda::std::uint32_t>(hl);
          hi                               = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
#endif
        }));
    }
    else
    {
      hi = static_cast<UInt>((static_cast<::cuda::std::uint64_t>(magic) * static_cast<::cuda::std::uint64_t>(n)) >> kBits);
    }
    return (((n - hi) >> 1) + hi) >> (shift - 1);
  }
};

// Detect whether a decode op is the pass-through transform (any specialization of
// Transforms<L,O,S>::PassThruTransform). Identifies transforms that map identically
// from input bin to output bin, which is required for the combine staging path.
template <typename T, typename = void>
struct is_pass_thru_transform : ::cuda::std::false_type
{};

template <typename T>
struct is_pass_thru_transform<T, ::cuda::std::void_t<typename T::is_pass_thru_transform>> : ::cuda::std::true_type
{};

template <typename T>
inline constexpr bool is_pass_thru_transform_v = is_pass_thru_transform<T>::value;

template <typename LevelT, typename OffsetT, typename SampleT>
struct Transforms
{
  //---------------------------------------------------------------------
  // Transform functors for converting samples to bin-ids
  //---------------------------------------------------------------------

  // Searches for bin given a list of bin-boundary levels.
  //
  // For roughly uniformly-spaced levels we replace a 22-iteration UpperBound
  // binary search with an interpolated first-guess plus a short linear
  // correction window. If the correction window does not converge within a
  // small fixed number of steps, we fall back to UpperBound so non-uniform
  // level distributions still produce correct results.
  template <typename LevelIteratorT>
  struct SearchTransform
  {
    LevelIteratorT d_levels; // Pointer to levels array
    int num_output_levels; // Number of levels in array

    // Precomputed (loop-invariant) interpolation state, populated by
    // `PrecomputeOnDevice()`. The interpolation slope `num_bins / (last -
    // first)` and the boundary levels are uniform across all samples a thread
    // classifies, but the original `BinSelect` recomputed them per sample
    // (two cache loads for the endpoints plus a `__fdividef` MUFU.RCP on the
    // critical dependency chain). Hoisting them out turns the per-sample
    // first-guess into a single `(float)delta * m_inv_scale` FMA and removes
    // the two endpoint loads, which is the dominant cost on the ALU/XU-bound
    // RANGE classify. `m_have_precompute == false` keeps the original
    // per-sample path so host-only initialization (no device pointer to
    // dereference) and tiny bin counts remain correct.
    float m_inv_scale; // num_bins / (float)(last - first); valid iff m_have_precompute
    LevelT m_first; // cached d_levels[0]
    LevelT m_last; // cached d_levels[num_bins]
    bool m_have_precompute; // whether the fields above are valid

    //! @brief Initializer
    //!
    //! @param d_levels_ Pointer to levels array
    //! @param num_output_levels_ Number of levels in array
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(LevelIteratorT d_levels_, int num_output_levels_)
    {
      this->d_levels          = d_levels_;
      this->num_output_levels = num_output_levels_;
      this->m_have_precompute = false;
      this->m_inv_scale       = 0.0f;
    }

    //! @brief Hoist the loop-invariant interpolation state out of `BinSelect`.
    //!
    //! Must be called on the device (it dereferences the device level array)
    //! once per thread before the sweep loop. Reads the first and last level,
    //! validates strict monotonicity of the endpoints and a usable bin count,
    //! and on success precomputes the float reciprocal slope so the hot path
    //! avoids a per-sample `__fdividef` and two endpoint loads. On any
    //! degenerate input it leaves `m_have_precompute == false`, so `BinSelect`
    //! transparently falls back to the original (fully general) path.
    _CCCL_DEVICE _CCCL_FORCEINLINE void PrecomputeOnDevice()
    {
      const int num_bins = num_output_levels - 1;
      if (num_bins < 4)
      {
        m_have_precompute = false;
        return;
      }

      using WrappedLevelIteratorT =
        ::cuda::std::_If<::cuda::std::is_pointer_v<LevelIteratorT>,
                         CacheModifiedInputIterator<LOAD_LDG, LevelT, OffsetT>,
                         LevelIteratorT>;
      WrappedLevelIteratorT wrapped_levels(d_levels);

      const LevelT first = wrapped_levels[0];
      const LevelT last  = wrapped_levels[num_bins];
      if (!(first < last))
      {
        m_have_precompute = false;
        return;
      }

      m_first     = first;
      m_last      = last;
      m_inv_scale = static_cast<float>(num_bins) / static_cast<float>(last - first);
      m_have_precompute = true;
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

      const LevelT s = static_cast<LevelT>(sample);

      // For very small bin counts, the interpolation overhead is not worth
      // it; fall back to the original binary search.
      if (num_bins < 4)
      {
        bin = UpperBound(wrapped_levels, num_output_levels, s) - 1;
        if (bin >= num_bins)
        {
          bin = -1;
        }
        return;
      }

      // Read first and last levels. When `PrecomputeOnDevice()` has run we use
      // the cached endpoints (and the precomputed reciprocal slope below),
      // removing two per-sample endpoint loads and the per-sample
      // `__fdividef`. Otherwise (host-only init, or a degenerate level array
      // that PrecomputeOnDevice rejected) we read them per sample as before.
      // These are warp/CTA-uniform and land in L1 / texture cache after the
      // first read, so even the fallback amortizes across samples.
      const LevelT first_level = m_have_precompute ? m_first : wrapped_levels[0];
      const LevelT last_level  = m_have_precompute ? m_last : wrapped_levels[num_bins];

      // Defensive: if a user-supplied level array has non-monotonic endpoints
      // (e.g. `last_level <= first_level`), the boundary check below would
      // misclassify all samples as out-of-range. Fall back to UpperBound,
      // which uses ordered comparisons only and produces correct results
      // regardless of endpoint ordering. (PrecomputeOnDevice already enforces
      // `first < last` before setting m_have_precompute, so this only fires on
      // the non-precomputed path.)
      if (!(first_level < last_level))
      {
        bin = UpperBound(wrapped_levels, num_output_levels, s) - 1;
        if (bin >= num_bins)
        {
          bin = -1;
        }
        return;
      }

      // Out-of-range samples map to bin -1.
      if (s < first_level || !(s < last_level))
      {
        bin = -1;
        return;
      }

      // Interpolated first-guess index. We always use a fast 32-bit float
      // divide (MUFU.RCP) for the slope: the divide does not have to be
      // accurate, only close enough that the verify-or-1-step-correct path
      // hits a handful of bins. The full UpperBound fallback catches any
      // remaining mismatch from precision loss or non-uniform spacing.
      // For wide-ranged 64-bit types we still compute (sample - first) in
      // the level type to avoid float overflow on the difference itself.
      //
      // On the precomputed path the slope `num_bins / (last - first)` is a
      // loop-invariant `m_inv_scale`, so the guess collapses to a single
      // `(float)delta * m_inv_scale` FMA (no per-sample MUFU.RCP). The result
      // is bit-identical in intent to `__fdividef(delta*num_bins, range)`:
      // both are approximate first guesses validated by the bracket check
      // below, so any rounding difference is absorbed by the same verify /
      // 1-step / UpperBound correction ladder.
      const auto delta = (s - first_level);
      int guess;
      if (m_have_precompute)
      {
        NV_IF_ELSE_TARGET(
          NV_IS_DEVICE,
          (guess = static_cast<int>(static_cast<float>(delta) * m_inv_scale);),
          (guess = static_cast<int>(static_cast<float>(delta) * m_inv_scale);));
      }
      else
      {
        const auto range = (last_level - first_level);
        NV_IF_ELSE_TARGET(
          NV_IS_DEVICE,
          (guess = static_cast<int>(
             __fdividef(static_cast<float>(delta) * static_cast<float>(num_bins), static_cast<float>(range)));),
          (guess = static_cast<int>(
             (static_cast<float>(delta) * static_cast<float>(num_bins)) / static_cast<float>(range));));
      }
      if (guess < 0)
      {
        guess = 0;
      }
      else if (guess > num_bins - 1)
      {
        guess = num_bins - 1;
      }

      // Verify the guess: d_levels[guess] <= s < d_levels[guess + 1]. We
      // load both bracketing levels in parallel to expose memory-level
      // parallelism and branch on the result. The level array has length
      // num_bins + 1, so wrapped_levels[guess + 1] is always in-bounds for
      // guess <= num_bins - 1.
      const LevelT lvl_lo = wrapped_levels[guess];
      const LevelT lvl_hi = wrapped_levels[guess + 1];

      if (!(s < lvl_lo) && (s < lvl_hi))
      {
        bin = guess;
        return;
      }

      // One-step linear correction: try a single neighbor before falling
      // back to a binary search. If the guess was high, try guess - 1; if
      // low, try guess + 1.
      if (s < lvl_lo)
      {
        // guess too high; check guess - 1.
        const int g2 = guess - 1;
        if (g2 >= 0)
        {
          const LevelT lvl2_lo = wrapped_levels[g2];
          // lvl2_hi is lvl_lo (loaded already).
          if (!(s < lvl2_lo))
          {
            bin = g2;
            return;
          }
        }
      }
      else
      {
        // s >= lvl_hi: guess too low; check guess + 1.
        const int g2 = guess + 1;
        if (g2 <= num_bins - 1)
        {
          // lvl2_lo is lvl_hi (loaded already).
          const LevelT lvl2_hi = wrapped_levels[g2 + 1];
          if (s < lvl2_hi)
          {
            bin = g2;
            return;
          }
        }
      }

      // Fall back to binary search for irregular level distributions.
      bin = UpperBound(wrapped_levels, num_output_levels, s) - 1;
      if (bin >= num_bins)
      {
        bin = -1;
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

    // Storage type for the precomputed `range = max_level - min_level` and
    // `bins = num_levels - 1` used by the integer ComputeBin path. For
    // narrow integer CommonT (e.g. int8_t with full range), `max - min`
    // overflows CommonT and would silently produce wrong bins; widening to
    // `IntArithmeticT` (uint32_t / uint64_t) holds the difference without
    // overflow. For 128-bit and non-integer types, IntArithmeticT == CommonT
    // (or wider), so this is also correct.
    using FractionStorageT =
      ::cuda::std::_If<is_integral_excl_int128<CommonT>::value, IntArithmeticT, CommonT>;

    // The integral path replaces a 64-bit divide-by-runtime-constant in
    // `ComputeBin` with a precomputed multiply-high + shift sequence. The
    // precomputation runs on the host inside `Init` and is propagated to
    // the device via the per-channel decode-op argument.
    using FastDivideT = fast_divide_by_constant<IntArithmeticT>;

    union ScaleT
    {
      // Used when CommonT is not floating-point to avoid intermediate
      // rounding errors (see NVIDIA/cub#489).
      struct FractionT
      {
        FractionStorageT bins;
        FractionStorageT range;
        FastDivideT range_divider;
        // True iff bins == range, the common benchmark case (e.g. uniform
        // even-spaced bins where one bin == one sample value). When set,
        // ComputeBin short-circuits to `sample - min_level` and skips both
        // the multiply by `bins` and the divide-by-range.
        bool bins_eq_range;
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
      // Compute `max - min` without overflowing T. For signed integer T
      // with full range (e.g. int8_t with [-128, 127]), the signed
      // difference `127 - (-128) = 255` overflows int8_t. Cast each
      // operand to its unsigned counterpart of the same width and
      // subtract, then assign back to that unsigned type to truncate via
      // modular wrap-around: e.g. for int8_t with max=127, min=-128, the
      // unsigned reinterpretations are uint8_t(127)=127 and
      // uint8_t(-128)=128 (two's complement bit pattern). C++ integer
      // promotion lifts the subtraction to int (127 - 128 = -1), and
      // truncating that back to uint8_t yields 255 — the correct
      // difference in [0, 2^N - 1]. The intermediate ULevelT is required
      // because casting the int result directly to FractionStorageT (a
      // wider unsigned type) would sign-extend -1 into a giant value.
      if constexpr (::cuda::std::is_integral_v<T>)
      {
        using UT              = ::cuda::std::make_unsigned_t<T>;
        const UT diff         = static_cast<UT>(static_cast<UT>(max_level) - static_cast<UT>(min_level));
        result.fraction.range = static_cast<FractionStorageT>(diff);
      }
      else
      {
        result.fraction.range = static_cast<FractionStorageT>(max_level - min_level);
      }
      // Precompute the magic multiplier + shift for fast (sample - min_level) * bins / range
      // in `ComputeBin`. This is a no-op for non-integral CommonT (e.g. user types),
      // where IntArithmeticT may still be uint64_t but the integral overload is not used.
      result.fraction.range_divider.Init(static_cast<IntArithmeticT>(result.fraction.range));
      result.fraction.bins_eq_range = (result.fraction.bins == result.fraction.range);
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
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScaleT
    ComputeScale(int num_levels, __nv_bfloat16 max_level, __nv_bfloat16 min_level)
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
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int
    SampleIsValid(__nv_bfloat16 sample, __nv_bfloat16 max_level, __nv_bfloat16 min_level)
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

    //! @brief Bin computation for integral types of up to 64-bit types.
    //! Uses a precomputed magic-multiplier + shift to avoid the runtime
    //! 64-bit integer divide that previously dominated the EVEN-path
    //! classify. The host-side `Init` populates `scale.fraction.range_divider`
    //! with a libdivide-style "round-up" multiplier that gives an exact
    //! `floor(numerator / range)` for any non-negative numerator
    //! representable in `IntArithmeticT`.
    //!
    //! Fast path: when `bins == range` (the dispatched-for-uniform-bins case
    //! that dominates our benchmarks), bin equals `sample - min_level` and we
    //! skip both the multiply by `bins` and the divide-by-range entirely.
    //!
    //! Compute `sample - min_level` via the unsigned representation of T,
    //! mirroring `ComputeScale`. For signed integer T with negative
    //! `min_level` (e.g. `min_level = INT_MIN`), the signed difference
    //! `sample - min_level` overflows T and is undefined behaviour; the
    //! resulting numerator on two's complement is a wildly wrong magnitude
    //! and produces an incorrect bin. The unsigned subtraction wraps
    //! modularly and yields the correct non-negative difference exactly
    //! the way `ComputeScale` computes `max_level - min_level`.
    template <typename T, ::cuda::std::enable_if_t<is_integral_excl_int128<T>::value, int> = 0>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int ComputeBin(T sample, T min_level, ScaleT scale) const
    {
      using UT                  = ::cuda::std::make_unsigned_t<T>;
      const IntArithmeticT diff = static_cast<IntArithmeticT>(
        static_cast<UT>(static_cast<UT>(sample) - static_cast<UT>(min_level)));
      if (scale.fraction.bins_eq_range)
      {
        return static_cast<int>(diff);
      }
      const IntArithmeticT numerator = diff * static_cast<IntArithmeticT>(scale.fraction.bins);
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
    //! @brief Initializes the ScaleTransform for the given parameters
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(int num_levels, LevelT max_level, LevelT min_level)
    {
      m_max = static_cast<CommonT>(max_level);
      m_min = static_cast<CommonT>(min_level);

      m_scale = this->ComputeScale(num_levels, m_max, m_min);
    }

    // No-op for uniformity with SearchTransform::PrecomputeOnDevice.
    _CCCL_DEVICE _CCCL_FORCEINLINE void PrecomputeOnDevice() {}

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
    // Tag for detecting the pass-through transform without depending on its template
    // parameters. Used by dispatch to decide whether the combine staging path is safe
    // (the combine kernel assumes output_decode_op is identity).
    using is_pass_thru_transform = ::cuda::std::true_type;

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

    // No-op for uniformity with SearchTransform::PrecomputeOnDevice.
    _CCCL_DEVICE _CCCL_FORCEINLINE void PrecomputeOnDevice() {}

    // Method for converting samples to bin-ids
    template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(_SampleT sample, int& bin, bool valid) const
    {
      if (valid)
      {
        // The byte-sample privatized histogram has 256 bins indexed by the
        // sample's unsigned byte value. For signed integer samples this
        // reinterprets the bit pattern: int8_t(-128..127) -> uint8_t(128..255, 0..127).
        // Without this reinterpretation, negative samples cast directly to
        // `int` produce negative bin indices and are silently dropped.
        if constexpr (::cuda::std::is_integral_v<_SampleT>)
        {
          using UT = ::cuda::std::make_unsigned_t<_SampleT>;
          bin      = static_cast<int>(static_cast<UT>(sample));
        }
        else
        {
          bin = static_cast<int>(sample);
        }
      }
    }
  };
};

//! @brief Chunked privatized-bin decode op wrapper.
//!
//! Wraps an inner decode op (`ScaleTransform` for EVEN paths and the uniform-detected
//! RANGE fast-path; `SearchTransform` for non-uniform RANGE paths) so the per-block
//! dyn-SMEM histogram only counts samples whose full-domain bin falls in
//! `[chunk_start, chunk_start + chunk_size)`. The returned local bin is shifted down by
//! `chunk_start` so the agent indexes a SMEM histogram of size `chunk_size`.
//!
//! Used by the chunked dyn-SMEM staging-fused path: the dispatch loops `num_chunks` times,
//! each pass paying 1x sample read but only 1x SMEM atomicAdd_block (instead of the legacy
//! 1x sample read + 1x GMEM atomicAdd_block on the GMEM-priv persistent kernel). For
//! Bins=60000 single-channel this trades 2x sample reads for 2x SMEM atomic latency, which
//! pays off when the persistent-kernel atomic phase dominates.
template <typename Inner>
struct ChunkedDecodeOp
{
  Inner inner;
  int chunk_start;
  int chunk_size;

  // Forwards to inner ScaleTransform::Init(num_levels, max_level, min_level).
  template <typename L>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(int num_levels, L max_level, L min_level)
  {
    inner.Init(num_levels, max_level, min_level);
  }

  // Forwards to inner SearchTransform::Init(d_levels, num_output_levels).
  template <typename It>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(It d_levels, int num_output_levels)
  {
    inner.Init(d_levels, num_output_levels);
  }

  // Sets the chunk window in privatized-bin space. Called by the chunked-dispatch loop
  // before the kernel launch, so the kernel sees the chunk via the by-value GRID_CONSTANT
  // decode-op argument.
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void SetChunk(int start, int size)
  {
    chunk_start = start;
    chunk_size  = size;
  }

  // Forward to inner so the RANGE SearchTransform's loop-invariant
  // interpolation state is hoisted out of BinSelect on chunked paths too.
  _CCCL_DEVICE _CCCL_FORCEINLINE void PrecomputeOnDevice()
  {
    inner.PrecomputeOnDevice();
  }

  template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(_SampleT sample, int& bin, bool valid) const
  {
    int full_bin = -1;
    inner.template BinSelect<LOAD_MODIFIER>(sample, full_bin, valid);
    // Map to chunk-local bin if in window, else -1 so the agent's accumulate path skips it.
    const int local_bin = full_bin - chunk_start;
    bin                 = (full_bin >= 0 && local_bin >= 0 && local_bin < chunk_size) ? local_bin : -1;
  }
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
// Request a minimum of 2 resident blocks/SM. This binds only for the wide
// launch shapes: the register cap it imposes is 65536/(threads_per_block*2), so
// it is loose (>= 64 regs) for the narrow single-channel policies (384/448
// threads) -- leaving their bare-form register counts untouched -- and tight
// (<= 32 regs) for the wide 1024-thread multi-channel EVEN policy. That EVEN
// sweep is contention-bound on shared-memory atomicAdd and, with a bare launch
// bound, ptxas left it at ~58 registers => only 1 block/SM (58*2048 > 65536),
// with no second resident block to hide the atomic/scoreboard latency. At its
// 256/2048-bin tiers the SMEM footprint is small (4 KB / 25 KB) so registers,
// not SMEM, gate occupancy; the hint forces ptxas to fit 2 CTAs and doubles
// resident warps. NB: a literal `,1` would *relax* the cap (it regressed the
// 448/768-thread RANGE kernels), and `2048/threads_per_block` over-constrains
// the 384-thread single-channel EVEN kernel into register spills -- `,2` is the
// sweet spot that lifts the 1024-thread kernel without touching the narrow ones.
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block), 2)
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

  // This is the smem_priv_256 tier (privatized bins <= 256). The level array a
  // RANGE SearchTransform searches here is tiny, so the classify is not the
  // bottleneck and the per-thread cached-interpolation state would only add
  // register pressure (ncu showed regs 52->76 and occupancy 51->37% when this
  // kernel localized + precomputed the decode ops). Keep the lean path: use
  // the grid-constant decode ops directly (read from constant memory) without
  // the device-side precompute. The other (higher-bin) sweep kernels, where
  // the classify dominates, still precompute.
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

  // No follow-on kernel reads our writes; emit the trigger so any
  // downstream PDL-launched kernel in the stream sees a completion signal.
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
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

  // Hoist the RANGE SearchTransform's loop-invariant interpolation slope and
  // boundary levels out of the per-sample classify. No-op for EVEN
  // (ScaleTransform) and byte-sample (PassThruTransform) decode ops.
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int channel = 0; channel < NumActiveChannels; ++channel)
  {
    privatized_decode_op[channel].PrecomputeOnDevice();
    output_decode_op[channel].PrecomputeOnDevice();
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

  // No follow-on kernel reads our writes; emit the trigger so any
  // downstream PDL-launched kernel in the stream sees a completion signal.
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
}

//! Persistent grid-resident histogram sweep kernel that fuses output-histogram
//! initialization, drain-counter reset, and the sweep+store phase into a single
//! cooperative kernel launch. It uses `cooperative_groups::this_grid()` and
//! `grid.sync()` to synchronize the initialization phase with the sweep phase,
//! eliminating the separate `DeviceHistogramInitKernel` launch and its
//! associated launch overhead.
//!
//! This is the host-init variant: it mirrors `DeviceHistogramSweepKernel`'s
//! interface and accepts pre-initialized decode operators, plus the
//! `max_num_output_bins` argument used to bound the output-histogram
//! initialization stride loop.
//!
//! The kernel must be launched cooperatively via `cudaLaunchCooperativeKernel`
//! so that all blocks are guaranteed to be co-resident on the device, which is
//! a precondition of `grid_group::sync()`. The dispatch layer is responsible
//! for verifying that the requested grid fits on the device before selecting
//! this kernel.
//!
//! Phase 1 (no synchronization): every thread cooperatively zeroes the output
//! histograms across all active channels via a grid-wide stride loop. Thread 0
//! of block 0 also zeroes the work-stealing drain counter inside the
//! `tile_queue` so that the subsequent sweep can use it as a shared
//! work-stealing counter.
//!
//! Phase 2 (`grid.sync()`): all blocks synchronize so that the zeroed output
//! histograms and the reset drain counter are visible to every block before
//! the sweep+store phase begins.
//!
//! Phase 3 (block-local): each block runs the standard `AgentHistogram`
//! pipeline (`InitBinCounters`, `ConsumeTiles`, `StoreOutput`).
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
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepPersistentKernel(
    _CCCL_GRID_CONSTANT const SampleIteratorT d_samples,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_privatized_bins_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<OutputDecodeOpT, NumActiveChannels> output_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const OffsetT num_row_pixels,
    _CCCL_GRID_CONSTANT const OffsetT num_rows,
    _CCCL_GRID_CONSTANT const OffsetT row_stride_samples,
    _CCCL_GRID_CONSTANT const int tiles_per_row,
    GridQueue<int> tile_queue,
    _CCCL_GRID_CONSTANT const int max_num_output_bins)
{
  static constexpr histogram_policy hp = current_policy<PolicySelector>();

  namespace cg = ::cooperative_groups;

  cg::grid_group grid = cg::this_grid();

  // ---------------------------------------------------------------------
  // Phase 1: zero the output histograms via a grid-wide stride loop, and
  // reset the work-stealing drain counter on a single thread.
  // ---------------------------------------------------------------------
  const unsigned int blocks_per_grid = gridDim.x * gridDim.y * gridDim.z;
  const unsigned int block_id        = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const unsigned int tid_global      = block_id * blockDim.x + threadIdx.x;
  const unsigned int total_threads   = blocks_per_grid * blockDim.x;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    const int channel_bins = num_output_bins_wrapper[ch];
    for (unsigned int bin = tid_global; bin < static_cast<unsigned int>(channel_bins); bin += total_threads)
    {
      d_output_histograms_wrapper[ch][bin] = 0;
    }
  }

  if (tid_global == 0)
  {
    // Reset the drain counter so that the sweep phase below can use the
    // queue as a shared work-stealing counter when work-stealing is enabled.
    GridQueue<int> queue = tile_queue;
    queue.ResetDrain();
  }

  // ---------------------------------------------------------------------
  // Phase 2: grid-wide synchronization so that all output-histogram zeros
  // and the drain-counter reset are visible to every block before the
  // sweep+store phase begins.
  // ---------------------------------------------------------------------
  grid.sync();

  // ---------------------------------------------------------------------
  // Phase 3: standard AgentHistogram sweep — InitBinCounters (zero
  // privatized counters) and ConsumeTiles (atomic-add into privatized
  // counters). For the GMEM-privatized path (PrivatizedSmemBins == 0)
  // each block's privatized histogram lives in global memory at
  // `d_privatized_histograms[ch] + block_id * num_privatized_bins[ch]`.
  // ---------------------------------------------------------------------
  using AgentHistogramPolicyT =
    AgentHistogramPolicy<hp.threads_per_block,
                         hp.pixels_per_thread,
                         hp.load_algorithm,
                         hp.load_modifier,
                         hp.rle_compress,
                         hp.mem_preference,
                         hp.work_stealing,
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

  __shared__ typename AgentHistogramT::TempStorage temp_storage;

  // The agent stores the per-channel base pointer of this block's privatized
  // histogram in `d_privatized_histograms`, which it sets in its constructor
  // by adding `block_id * num_privatized_bins[ch]`. We need the per-channel
  // base of the *all-blocks* privatized array later for the gather merge,
  // so save it here before constructing the agent.
  CounterT* d_privatized_base[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    d_privatized_base[ch] = d_privatized_histograms_wrapper[ch];
  }

  // Host-init path: hoist the RANGE SearchTransform's loop-invariant
  // interpolation state out of the per-sample classify. No-op for EVEN /
  // byte-sample decode ops.
  OutputDecodeOpT output_decode_op[NumActiveChannels];
  PrivatizedDecodeOpT privatized_decode_op[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    output_decode_op[ch] = output_decode_op_wrapper[ch];
    privatized_decode_op[ch] = privatized_decode_op_wrapper[ch];
    output_decode_op[ch].PrecomputeOnDevice();
    privatized_decode_op[ch].PrecomputeOnDevice();
  }

  {
    AgentHistogramT agent(
      temp_storage,
      d_samples,
      num_output_bins_wrapper.data(),
      num_privatized_bins_wrapper.data(),
      d_output_histograms_wrapper.data(),
      d_privatized_histograms_wrapper.data(),
      output_decode_op,
      privatized_decode_op);

    // Initialize per-block privatized counters
    agent.InitBinCounters();

    // Consume input tiles
    agent.ConsumeTiles(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);

    if constexpr (PrivatizedSmemBins != 0)
    {
      // Block-private SMEM path: privatized counters are in shared memory
      // and are lost when the block exits, so the merge into the output
      // histograms must happen here using the agent's standard
      // atomic-add `StoreOutput`.
      agent.StoreOutput();
    }
  }

  if constexpr (PrivatizedSmemBins == 0)
  {
    // GMEM-privatized merge: every block's privatized histogram is now in
    // global memory at `d_privatized_base[ch] + block_id * num_privatized_bins[ch]`.
    // We need a grid-wide barrier so every block's `ConsumeTiles` writes are
    // visible to every other block before the gather merge below.
    grid.sync();

    // Phase 4: gather-merge. Each thread takes a slice of OUTPUT bins and
    // sums the corresponding privatized counters across all blocks, writing
    // the total to the output histogram. This converts the original
    // `num_blocks * num_output_bins` atomicAdds into plain reads + writes,
    // eliminating cross-block atomic contention on the output histogram.
    //
    // This optimization assumes that `num_privatized_bins == num_output_bins`
    // and that `output_decode_op` is the identity (PassThruTransform), which
    // is the case for the host-init non-byte-sample dispatch path that
    // selects `PRIVATIZED_SMEM_BINS = 0` for `max_num_output_bins > 256`.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      const int num_bins             = num_privatized_bins_wrapper[ch];
      const CounterT* base           = d_privatized_base[ch];
      CounterT* d_out                = d_output_histograms_wrapper[ch];
      const unsigned int num_bins_u  = static_cast<unsigned int>(num_bins);
      for (unsigned int bin = tid_global; bin < num_bins_u; bin += total_threads)
      {
        CounterT total = 0;
        for (unsigned int b = 0; b < blocks_per_grid; ++b)
        {
          total += base[b * num_bins_u + bin];
        }
        d_out[bin] = total;
      }
    }
  }

  // Emit the trigger so any PDL-launched downstream kernel in the stream
  // sees a completion signal. (Cooperative launches typically do not use
  // PDL, so this is a no-op in the common case.)
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
}

//! Persistent grid-resident histogram sweep kernel that fuses output-histogram
//! initialization with a direct-atomic sweep. For the very-high-bin
//! GMEM-privatized path the per-block privatization storage is so large
//! (`num_blocks * num_bins * sizeof(CounterT)`) that the
//! `InitBinCounters` zero-fill plus the gather-merge dominate runtime. At
//! these bin counts atomic contention on the final output histogram is
//! also low (each output bin only sees a tiny fraction of the input
//! samples), so it is faster to:
//!
//! 1. Cooperatively zero the output histogram once (Phase 1).
//! 2. `grid.sync()` to make the zeros visible (Phase 2).
//! 3. Have every thread atomic-add (device-scope) directly into the
//!    output histogram (Phase 3).
//!
//! This avoids ~`num_blocks * num_bins * sizeof(CounterT)` bytes of
//! temporary GMEM writes (init) and reads + writes (gather merge), and
//! also lets the dispatch layer skip the per-block privatization
//! allocation entirely.
//!
//! This kernel must be launched cooperatively
//! (`cudaLaunchCooperativeKernel`) so that all blocks are co-resident on
//! the device, which is a precondition of `grid_group::sync()`.
//!
//! Unlike `DeviceHistogramSweepPersistentKernel`, this kernel does not
//! use `AgentHistogram`. The agent's `AccumulatePixels` uses
//! `atomicAdd_block` (block-scope), which is undefined for memory shared
//! across blocks. We therefore implement a small stand-alone sweep that
//! reads samples directly from `d_samples` and uses device-scope
//! `atomicAdd` against `d_output_histograms`.
//!
//! The sweep iterates `OffsetT` total samples; the dispatch layer
//! flattens `(num_row_pixels, num_rows, row_stride_samples)` into a
//! single linear input region when possible, but here we always treat
//! the input as a single linear array of `total_pixels = num_rows *
//! num_row_pixels` pixels and skip any padding columns explicitly when
//! `row_stride_samples != num_row_pixels * NumChannels`.
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
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepDirectAtomicPersistentKernel(
    _CCCL_GRID_CONSTANT const SampleIteratorT d_samples,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const OffsetT num_row_pixels,
    _CCCL_GRID_CONSTANT const OffsetT num_rows,
    _CCCL_GRID_CONSTANT const OffsetT row_stride_samples,
    _CCCL_GRID_CONSTANT const int cache_slots_per_channel)
{
  namespace cg = ::cooperative_groups;

  cg::grid_group grid = cg::this_grid();

  // ---------------------------------------------------------------------
  // Phase 1: zero the output histograms via a grid-wide stride loop.
  // ---------------------------------------------------------------------
  const unsigned int blocks_per_grid = gridDim.x * gridDim.y * gridDim.z;
  const unsigned int block_id        = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const unsigned int tid_global      = block_id * blockDim.x + threadIdx.x;
  const unsigned int total_threads   = blocks_per_grid * blockDim.x;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    const int channel_bins = num_output_bins_wrapper[ch];
    for (unsigned int bin = tid_global; bin < static_cast<unsigned int>(channel_bins); bin += total_threads)
    {
      d_output_histograms_wrapper[ch][bin] = 0;
    }
  }

  // ---------------------------------------------------------------------
  // Phase 2: grid-wide synchronization so that all output-histogram
  // zeros are visible to every block before the atomic-sweep phase
  // begins.
  // ---------------------------------------------------------------------
  grid.sync();

  // ---------------------------------------------------------------------
  // Phase 3: direct-atomic sweep. Each thread strides over input pixels
  // and atomic-adds into the output histogram for every active channel.
  // No per-block privatization storage is needed.
  //
  // The dispatch layer flattens `(num_row_pixels, num_rows,
  // row_stride_samples)` into a single linear array of pixels when
  // possible (`num_row_pixels * NumChannels == row_stride_samples`), so
  // the common path has `num_rows == 1` and we can skip the per-pixel
  // (row, col) reconstruction. We expose a fast path for that case so
  // the inner loop has no integer division.
  //
  // We also unroll the sweep so that each thread holds several samples
  // and several `atomicAdd` operations in flight at once. This is the
  // primary mechanism for hiding atomic latency on the very-high-bin
  // path: with one atomic per iteration the kernel was bottlenecked on
  // L1TEX scoreboard dependencies (~94% CPI stall in profiling).
  // ---------------------------------------------------------------------
  constexpr int unroll = 4;
  const OffsetT total_pixels = num_rows * num_row_pixels;

  // Per-warp atomic coalescing: when multiple lanes in a warp produce
  // the same bin id, fold their contributions into one device-scope
  // atomicAdd issued by the lane with the lowest matching lane id.
  // This converts up to 32 contended atomics on a hot bin into 1,
  // dramatically reducing atomic traffic for low-entropy distributions.
  //
  // We always pass the full warp mask `0xffffffffu` to
  // `__match_any_sync` so the coalescer doesn't depend on
  // `__activemask()` in possibly-divergent code. The pixel sweep loop
  // is structured so every lane in a warp executes the same number of
  // iterations: we use a single grid-strided loop with a per-iteration
  // bounds check that issues a sentinel `bin == -1` for past-the-end
  // pixels, keeping all lanes in lockstep through the coalescer.
  //
  // Per-block SMEM cache: after warp-coalescing, the warp leader's
  // atomicAdd to the GLOBAL output histogram still incurs cross-block
  // contention (multiple blocks racing on the same hot bins). To absorb
  // this contention, we maintain a per-block SMEM cache that maps a hash
  // of the bin id to a (bin_key, accumulated_count) slot. Leaders probe
  // their slot: on hit (slot key matches bin), they atomicAdd_block
  // (block-scope, ~10x cheaper than device-scope) into the cache slot's
  // count. On miss (slot key differs or empty), the leader evicts the
  // current slot (atomicAdd to global with the slot's accumulated count)
  // and claims the slot for its own bin. After the sweep ends, all slots
  // are flushed cooperatively to the global histogram.
  //
  // Sizing: the cache lives in DYNAMIC shared memory so the dispatch layer
  // can pick the largest power-of-two slot count per channel that still
  // fits the cooperative grid's required per-SM occupancy (it queries
  // `cudaOccupancyMaxActiveBlocksPerMultiprocessor` with the chosen dynamic
  // SMEM size). Profiling showed the high-bin / high-entropy cells
  // (e.g. Bins=262144, Entropy=1.0) are bound by scattered GMEM-atomic
  // spills (DRAM ~17% of peak, latency-bound), and that neither packing
  // the slot nor changing the probe count moves them -- the spill rate is
  // fixed by how many distinct hot bins fit in cache. Growing the slot
  // count is the one lever that raises the hit rate and pulls spills off
  // the contended global histogram onto cheap block-scope SMEM atomics.
  //
  // `cache_slots_per_channel` is a runtime power of two (mask = slots-1).
  // The extern __shared__ region holds, per channel, a key array (int)
  // followed by a count array (CounterT): keys for all channels first,
  // then counts for all channels. Two multiplicative-hash probes are
  // retained (the single-probe variant was verified to collapse skewed
  // distributions by 25-30%). Keys are write-once / immutable after a CAS
  // claim, so the cache is race-free (no hit-vs-evict window).
  const int cache_mask = cache_slots_per_channel - 1;
  extern __shared__ unsigned char s_cuckoo_raw[];
  int* const s_cache_keys_base       = reinterpret_cast<int*>(s_cuckoo_raw);
  CounterT* const s_cache_counts_base = reinterpret_cast<CounterT*>(
    s_cuckoo_raw + static_cast<size_t>(NumActiveChannels) * cache_slots_per_channel * sizeof(int));
  // Per-channel slot bases.
  int* s_cache_keys[NumActiveChannels];
  CounterT* s_cache_counts[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    s_cache_keys[ch]   = s_cache_keys_base + static_cast<size_t>(ch) * cache_slots_per_channel;
    s_cache_counts[ch] = s_cache_counts_base + static_cast<size_t>(ch) * cache_slots_per_channel;
  }

  // Initialize cache: keys = -1 (empty sentinel), counts = 0.
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    for (int slot = threadIdx.x; slot < cache_slots_per_channel; slot += blockDim.x)
    {
      s_cache_keys[ch][slot]   = -1;
      s_cache_counts[ch][slot] = CounterT{0};
    }
  }
  __syncthreads();

  // Host-init path: hoist the RANGE SearchTransform's loop-invariant
  // interpolation state out of the per-sample classify. No-op for EVEN /
  // byte-sample decode ops. Used by both the num_rows==1 and the general
  // sweep below.
  PrivatizedDecodeOpT decode_op[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    decode_op[ch] = privatized_decode_op_wrapper[ch];
    decode_op[ch].PrecomputeOnDevice();
  }

  if (num_rows == 1)
  {
    // Pixel sweep parameters: every thread strides over the whole pixel
    // space using `total_threads` (the standard grid-strided loop).
    const OffsetT step         = static_cast<OffsetT>(total_threads);
    const OffsetT start        = static_cast<OffsetT>(tid_global);
    const unsigned int lane_id = threadIdx.x & 0x1f;

    // Determine the maximum number of `unroll`-sized chunks any thread
    // in the grid will run, so every thread iterates the same number
    // of times. Past-the-end pixels yield `bin = -1`, which the
    // coalescer treats as a no-op group.
    const OffsetT chunk           = static_cast<OffsetT>(unroll) * step;
    const OffsetT chunk_iters_max = (total_pixels + chunk - 1) / chunk;

    for (OffsetT it = 0; it < chunk_iters_max; ++it)
    {
      const OffsetT pixel = start + it * chunk;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int u = 0; u < unroll; ++u)
      {
        const OffsetT this_pixel = pixel + u * step;
        const bool valid_pixel   = this_pixel < total_pixels;
        const OffsetT pix_off    = valid_pixel ? (this_pixel * NumChannels) : OffsetT{0};
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int ch = 0; ch < NumActiveChannels; ++ch)
        {
          int bin = -1;
          if (valid_pixel)
          {
            auto sample = d_samples[pix_off + ch];
            decode_op[ch].template BinSelect<LOAD_DEFAULT>(sample, bin, true);
            const int num_bins = num_output_bins_wrapper[ch];
            if (bin >= num_bins)
            {
              bin = -1;
            }
          }
          // Coalesce same-bin lanes into a single atomic add. All
          // lanes in the warp must call `__match_any_sync` with the
          // same mask; we use 0xffffffffu so the call is well-defined
          // even when individual lanes have invalid bins (-1).
          const unsigned int peers = __match_any_sync(0xffffffffu, static_cast<unsigned int>(bin));
          const int leader         = __ffs(static_cast<int>(peers)) - 1;
          if (bin >= 0 && static_cast<int>(lane_id) == leader)
          {
            const CounterT contribution = static_cast<CounterT>(__popc(peers));
            // Two hash functions (cuckoo-style): on a primary slot
            // collision, try a secondary slot before falling back to
            // GMEM. This roughly doubles the effective cache hit rate
            // for moderate-entropy distributions where a few bins
            // dominate but each thread is otherwise visiting random
            // bins.
            const unsigned int hash1 = static_cast<unsigned int>(bin) * 2654435761u;
            const int slot1          = static_cast<int>(hash1 & cache_mask);
            const int existing_key1  = s_cache_keys[ch][slot1];
            if (existing_key1 == bin)
            {
              // Primary hit: bump cache count.
              atomicAdd_block(&s_cache_counts[ch][slot1], contribution);
            }
            else if (existing_key1 == -1)
            {
              // Primary slot empty: try to claim it via CAS.
              const int prev = atomicCAS(&s_cache_keys[ch][slot1], -1, bin);
              if (prev == -1 || prev == bin)
              {
                atomicAdd_block(&s_cache_counts[ch][slot1], contribution);
              }
              else
              {
                // Lost the race. Try secondary slot.
                const unsigned int hash2 = static_cast<unsigned int>(bin) * 2246822519u;
                const int slot2          = static_cast<int>(hash2 & cache_mask);
                const int existing_key2  = s_cache_keys[ch][slot2];
                if (existing_key2 == bin)
                {
                  atomicAdd_block(&s_cache_counts[ch][slot2], contribution);
                }
                else if (existing_key2 == -1)
                {
                  const int prev2 = atomicCAS(&s_cache_keys[ch][slot2], -1, bin);
                  if (prev2 == -1 || prev2 == bin)
                  {
                    atomicAdd_block(&s_cache_counts[ch][slot2], contribution);
                  }
                  else
                  {
                    atomicAdd(&d_output_histograms_wrapper[ch][bin], contribution);
                  }
                }
                else
                {
                  atomicAdd(&d_output_histograms_wrapper[ch][bin], contribution);
                }
              }
            }
            else
            {
              // Primary occupied by a different bin: try the secondary slot.
              const unsigned int hash2 = static_cast<unsigned int>(bin) * 2246822519u;
              const int slot2          = static_cast<int>(hash2 & cache_mask);
              const int existing_key2  = s_cache_keys[ch][slot2];
              if (existing_key2 == bin)
              {
                atomicAdd_block(&s_cache_counts[ch][slot2], contribution);
              }
              else if (existing_key2 == -1)
              {
                const int prev2 = atomicCAS(&s_cache_keys[ch][slot2], -1, bin);
                if (prev2 == -1 || prev2 == bin)
                {
                  atomicAdd_block(&s_cache_counts[ch][slot2], contribution);
                }
                else
                {
                  atomicAdd(&d_output_histograms_wrapper[ch][bin], contribution);
                }
              }
              else
              {
                atomicAdd(&d_output_histograms_wrapper[ch][bin], contribution);
              }
            }
          }
        }
      }
    }

    // After the sweep, flush every cache slot to the global histogram.
    // Block barrier here ensures every leader's atomicAdd_block has
    // finished against the cache before we read it for the flush.
    __syncthreads();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      for (int slot = threadIdx.x; slot < cache_slots_per_channel; slot += blockDim.x)
      {
        const int key       = s_cache_keys[ch][slot];
        const CounterT cnt  = s_cache_counts[ch][slot];
        if (key >= 0 && cnt > CounterT{0})
        {
          atomicAdd(&d_output_histograms_wrapper[ch][key], cnt);
        }
      }
    }
  }
  else
  {
    // Slow path: row-strided input that is not flattenable to a single
    // linear array. No coalescing here since it's the rare path.
    for (OffsetT pixel = static_cast<OffsetT>(tid_global); pixel < total_pixels;
         pixel += static_cast<OffsetT>(total_threads))
    {
      const OffsetT row     = pixel / num_row_pixels;
      const OffsetT col     = pixel - row * num_row_pixels;
      const OffsetT pix_off = row * row_stride_samples + col * NumChannels;

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        auto sample = d_samples[pix_off + ch];
        int bin     = -1;
        decode_op[ch].template BinSelect<LOAD_DEFAULT>(sample, bin, true);
        const int num_bins = num_output_bins_wrapper[ch];
        if (bin >= 0 && bin < num_bins)
        {
          atomicAdd(&d_output_histograms_wrapper[ch][bin], CounterT{1});
        }
      }
    }
  }

  // Emit the trigger so any PDL-launched downstream kernel in the stream
  // sees a completion signal. (Cooperative launches typically do not use
  // PDL, so this is a no-op in the common case.)
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
}

//! Single-probe direct-mapped variant of the direct-atomic persistent sweep
//! kernel.
//!
//! This kernel targets the very-high-bin / very-large-input regime
//! (e.g. >=256M pixels with >16K bins, single channel) where the
//! cooperative gather-merge `DeviceHistogramSweepPersistentKernel` is
//! catastrophic: it materialises a per-block privatized histogram of size
//! `num_blocks * num_bins * sizeof(CounterT)` in GMEM (e.g. ~1.9 GB at 444
//! blocks x 1M bins x 4 B), writes it scattered, then reads it strided at a
//! tiny L2 hit rate -- DRAM-bound, while privatization buys almost nothing
//! because each output bin only receives a handful of counts (negligible
//! cross-block contention).
//!
//! Like `DeviceHistogramSweepDirectAtomicPersistentKernel`, every thread
//! reads samples directly and the warp-leader coalesces same-bin lanes with
//! `__match_any_sync`. The difference is the per-block SMEM cache policy: this
//! kernel uses a SINGLE-PROBE direct-mapped cache. Each warp-leader hashes its
//! bin to exactly one slot and:
//!   * hit  (slot key == bin)      -> `atomicAdd_block` into the slot count;
//!   * empty (slot key == -1)      -> claim it with one `atomicCAS`, then
//!                                     `atomicAdd_block` if the claim succeeds
//!                                     (or if a peer concurrently claimed the
//!                                     same bin), else fall back to GMEM;
//!   * collision (key != bin)      -> a single device-scope `atomicAdd`
//!                                     directly to the output histogram.
//! After the sweep, every claimed slot is flushed once to GMEM. There is no
//! secondary probe and no rehash chain, so the leader's hot path has far less
//! branch divergence and issues at most one cache atomic before falling
//! through to GMEM. This trades a lower cache hit rate (compared to the
//! 2-hash cuckoo cache) for a much shorter, less divergent critical section --
//! the right trade-off when bins greatly exceed the slot count and the cache
//! mostly absorbs the few genuinely hot bins of skewed/constant inputs while
//! the long uniform tail streams straight to GMEM atomics.
//!
//! Shares the cuckoo kernel's DYNAMIC-SMEM cache (the dispatch layer picks the
//! largest power-of-two `cache_slots_per_channel` that preserves occupancy);
//! the only difference from the cuckoo kernel is the single-probe policy in
//! the leader's hot path.
//!
//! CORRECTNESS: each lane's contribution is added exactly once -- either to a
//! cache slot (which is flushed to GMEM exactly once at the end) or directly
//! to GMEM on a miss. Because counts are additive and every leader's atomic is
//! independent, the result is identical regardless of how slots are claimed or
//! evicted, so the kernel is race-free for the final counts.
//!
//! Must be launched cooperatively (`cudaLaunchCooperativeKernel`) so all
//! blocks are co-resident, a precondition of `grid_group::sync()`.
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
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepDirectAtomicSingleProbePersistentKernel(
    _CCCL_GRID_CONSTANT const SampleIteratorT d_samples,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const OffsetT num_row_pixels,
    _CCCL_GRID_CONSTANT const OffsetT num_rows,
    _CCCL_GRID_CONSTANT const OffsetT row_stride_samples,
    _CCCL_GRID_CONSTANT const int cache_slots_per_channel)
{
  namespace cg = ::cooperative_groups;

  cg::grid_group grid = cg::this_grid();

  // Phase 1: zero the output histograms via a grid-wide stride loop.
  const unsigned int blocks_per_grid = gridDim.x * gridDim.y * gridDim.z;
  const unsigned int block_id        = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const unsigned int tid_global      = block_id * blockDim.x + threadIdx.x;
  const unsigned int total_threads   = blocks_per_grid * blockDim.x;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    const int channel_bins = num_output_bins_wrapper[ch];
    for (unsigned int bin = tid_global; bin < static_cast<unsigned int>(channel_bins); bin += total_threads)
    {
      d_output_histograms_wrapper[ch][bin] = 0;
    }
  }

  // Phase 2: grid-wide sync so all output-histogram zeros are visible.
  grid.sync();

  // Phase 3: direct-atomic sweep with a single-probe direct-mapped SMEM cache.
  constexpr int unroll       = 4;
  const OffsetT total_pixels = num_rows * num_row_pixels;

  // Single-probe direct-mapped cache, in DYNAMIC shared memory (same layout
  // and dispatch-chosen `cache_slots_per_channel` as the cuckoo kernel: keys
  // for all channels first, then counts). Only the probe policy differs.
  const int cache_mask = cache_slots_per_channel - 1;
  extern __shared__ unsigned char s_singleprobe_raw[];
  int* const s_cache_keys_base = reinterpret_cast<int*>(s_singleprobe_raw);
  CounterT* const s_cache_counts_base = reinterpret_cast<CounterT*>(
    s_singleprobe_raw + static_cast<size_t>(NumActiveChannels) * cache_slots_per_channel * sizeof(int));
  int* s_cache_keys[NumActiveChannels];
  CounterT* s_cache_counts[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    s_cache_keys[ch]   = s_cache_keys_base + static_cast<size_t>(ch) * cache_slots_per_channel;
    s_cache_counts[ch] = s_cache_counts_base + static_cast<size_t>(ch) * cache_slots_per_channel;
  }

  // Initialize cache: keys = -1 (empty sentinel), counts = 0.
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    for (int slot = threadIdx.x; slot < cache_slots_per_channel; slot += blockDim.x)
    {
      s_cache_keys[ch][slot]   = -1;
      s_cache_counts[ch][slot] = CounterT{0};
    }
  }
  __syncthreads();

  // Host-init path: hoist the RANGE SearchTransform's loop-invariant
  // interpolation state out of the per-sample classify. No-op for EVEN /
  // byte-sample decode ops. Used by both the num_rows==1 and the general
  // sweep below.
  PrivatizedDecodeOpT decode_op[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    decode_op[ch] = privatized_decode_op_wrapper[ch];
    decode_op[ch].PrecomputeOnDevice();
  }

  if (num_rows == 1)
  {
    const OffsetT step         = static_cast<OffsetT>(total_threads);
    const OffsetT start        = static_cast<OffsetT>(tid_global);
    const unsigned int lane_id = threadIdx.x & 0x1f;

    const OffsetT chunk           = static_cast<OffsetT>(unroll) * step;
    const OffsetT chunk_iters_max = (total_pixels + chunk - 1) / chunk;

    for (OffsetT it = 0; it < chunk_iters_max; ++it)
    {
      const OffsetT pixel = start + it * chunk;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int u = 0; u < unroll; ++u)
      {
        const OffsetT this_pixel = pixel + u * step;
        const bool valid_pixel   = this_pixel < total_pixels;
        const OffsetT pix_off    = valid_pixel ? (this_pixel * NumChannels) : OffsetT{0};
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int ch = 0; ch < NumActiveChannels; ++ch)
        {
          int bin = -1;
          if (valid_pixel)
          {
            auto sample = d_samples[pix_off + ch];
            decode_op[ch].template BinSelect<LOAD_DEFAULT>(sample, bin, true);
            const int num_bins = num_output_bins_wrapper[ch];
            if (bin >= num_bins)
            {
              bin = -1;
            }
          }
          // Coalesce same-bin lanes into a single atomic add.
          const unsigned int peers = __match_any_sync(0xffffffffu, static_cast<unsigned int>(bin));
          const int leader         = __ffs(static_cast<int>(peers)) - 1;
          if (bin >= 0 && static_cast<int>(lane_id) == leader)
          {
            const CounterT contribution = static_cast<CounterT>(__popc(peers));
            // Single direct-mapped probe: hash bin to one slot.
            const unsigned int hash = static_cast<unsigned int>(bin) * 2654435761u;
            const int slot          = static_cast<int>(hash & cache_mask);
            const int existing_key  = s_cache_keys[ch][slot];
            if (existing_key == bin)
            {
              // Hit: bump cache count (block-scope atomic, ~10x cheaper).
              atomicAdd_block(&s_cache_counts[ch][slot], contribution);
            }
            else if (existing_key == -1)
            {
              // Empty: try to claim via CAS.
              const int prev = atomicCAS(&s_cache_keys[ch][slot], -1, bin);
              if (prev == -1 || prev == bin)
              {
                atomicAdd_block(&s_cache_counts[ch][slot], contribution);
              }
              else
              {
                // Lost the claim race to a different bin: go straight to GMEM.
                atomicAdd(&d_output_histograms_wrapper[ch][bin], contribution);
              }
            }
            else
            {
              // Collision (slot owned by another bin): single GMEM atomic.
              atomicAdd(&d_output_histograms_wrapper[ch][bin], contribution);
            }
          }
        }
      }
    }

    // Flush every claimed cache slot to the global histogram.
    __syncthreads();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      for (int slot = threadIdx.x; slot < cache_slots_per_channel; slot += blockDim.x)
      {
        const int key      = s_cache_keys[ch][slot];
        const CounterT cnt = s_cache_counts[ch][slot];
        if (key >= 0 && cnt > CounterT{0})
        {
          atomicAdd(&d_output_histograms_wrapper[ch][key], cnt);
        }
      }
    }
  }
  else
  {
    // Slow path: row-strided input that is not flattenable. No coalescing.
    for (OffsetT pixel = static_cast<OffsetT>(tid_global); pixel < total_pixels;
         pixel += static_cast<OffsetT>(total_threads))
    {
      const OffsetT row     = pixel / num_row_pixels;
      const OffsetT col     = pixel - row * num_row_pixels;
      const OffsetT pix_off = row * row_stride_samples + col * NumChannels;

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        auto sample = d_samples[pix_off + ch];
        int bin     = -1;
        decode_op[ch].template BinSelect<LOAD_DEFAULT>(sample, bin, true);
        const int num_bins = num_output_bins_wrapper[ch];
        if (bin >= 0 && bin < num_bins)
        {
          atomicAdd(&d_output_histograms_wrapper[ch][bin], CounterT{1});
        }
      }
    }
  }

  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
}

//! Host-init variant of the staging histogram sweep kernel.
//! Skips StoreOutput() so a follow-on combine kernel handles cross-block reduction.
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
// minBlocks=2 hint to unlock register-bound 1-block-per-SM occupancy on
// SMEM-light staging tiers (Bins<=2048 paths use ~8 KB static SMEM).
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block), 2)
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepStagingHostInitKernel(
    _CCCL_GRID_CONSTANT const SampleIteratorT d_samples,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_privatized_bins_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<OutputDecodeOpT, NumActiveChannels> output_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const OffsetT num_row_pixels,
    _CCCL_GRID_CONSTANT const OffsetT num_rows,
    _CCCL_GRID_CONSTANT const OffsetT row_stride_samples,
    _CCCL_GRID_CONSTANT const int tiles_per_row,
    GridQueue<int> tile_queue)
{
  static constexpr histogram_policy hp = current_policy<PolicySelector>();

  using AgentHistogramPolicyT =
    AgentHistogramPolicy<hp.threads_per_block,
                         hp.pixels_per_thread,
                         hp.load_algorithm,
                         hp.load_modifier,
                         hp.rle_compress,
                         hp.mem_preference,
                         hp.work_stealing,
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

  __shared__ typename AgentHistogramT::TempStorage temp_storage;

  // Host-init path: hoist the RANGE SearchTransform's loop-invariant
  // interpolation state out of the per-sample classify. No-op for EVEN /
  // byte-sample decode ops.
  OutputDecodeOpT output_decode_op[NumActiveChannels];
  PrivatizedDecodeOpT privatized_decode_op[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int channel = 0; channel < NumActiveChannels; ++channel)
  {
    output_decode_op[channel] = output_decode_op_wrapper[channel];
    privatized_decode_op[channel] = privatized_decode_op_wrapper[channel];
    output_decode_op[channel].PrecomputeOnDevice();
    privatized_decode_op[channel].PrecomputeOnDevice();
  }

  AgentHistogramT agent(
    temp_storage,
    d_samples,
    num_output_bins_wrapper.data(),
    num_privatized_bins_wrapper.data(),
    d_output_histograms_wrapper.data(),
    d_privatized_histograms_wrapper.data(),
    output_decode_op,
    privatized_decode_op);

  agent.InitBinCounters();
  agent.ConsumeTiles(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);

  // Skip agent.StoreOutput(); the combine kernel handles cross-block reduction.
  if (agent.prefer_smem)
  {
    agent.StoreSmemToStagingSlab();
  }

  // PDL trigger MUST be after `StoreSmemToStagingSlab`: the follow-on
  // combine kernel reads from the per-block GMEM staging slabs, so it
  // cannot start until those slabs have been written by every block of
  // this kernel.
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
}

//! Histogram privatized sweep kernel that defers per-block-to-global combine.
//!
//! Same as DeviceHistogramSweepDeviceInitKernel, but skips the final StoreOutput()
//! call. The privatized per-block histograms remain in global memory at
//! `d_privatized_histograms_wrapper`, where a follow-on combine kernel reduces
//! them across blocks and writes the final output histogram. This avoids the
//! per-block `atomicAdd` to the global output bins, which is the dominant cost
//! for high-bin GMEM-privatized configurations.
template <typename PolicySelector,
          int PrivatizedSmemBins,
          int NumChannels,
          int NumActiveChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename FirstLevelArrayT,
          typename SecondLevelArrayT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT,
          bool IsEven>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
// minBlocks=2 hint mirrors DeviceHistogramSweepStagingHostInitKernel.
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block), 2)
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepStagingKernel(
    _CCCL_GRID_CONSTANT const SampleIteratorT d_samples,
    ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    ::cuda::std::array<int, NumActiveChannels> num_privatized_bins_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
    _CCCL_GRID_CONSTANT const FirstLevelArrayT first_level_array,
    _CCCL_GRID_CONSTANT const SecondLevelArrayT second_level_array,
    _CCCL_GRID_CONSTANT const OffsetT num_row_pixels,
    _CCCL_GRID_CONSTANT const OffsetT num_rows,
    _CCCL_GRID_CONSTANT const OffsetT row_stride_samples,
    _CCCL_GRID_CONSTANT const int tiles_per_row,
    _CCCL_GRID_CONSTANT const GridQueue<int> tile_queue)
{
  static constexpr histogram_policy hp = current_policy<PolicySelector>();

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

  // Hoist the RANGE SearchTransform's loop-invariant interpolation slope and
  // boundary levels out of the per-sample classify. No-op for EVEN
  // (ScaleTransform) and byte-sample (PassThruTransform) decode ops.
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int channel = 0; channel < NumActiveChannels; ++channel)
  {
    privatized_decode_op[channel].PrecomputeOnDevice();
    output_decode_op[channel].PrecomputeOnDevice();
  }

  // Thread block type for compositing input tiles
  using AgentHistogramPolicyT =
    AgentHistogramPolicy<hp.threads_per_block,
                         hp.pixels_per_thread,
                         hp.load_algorithm,
                         hp.load_modifier,
                         hp.rle_compress,
                         hp.mem_preference,
                         hp.work_stealing,
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

  // Skip agent.StoreOutput() -- combine kernel below handles per-block reduction.
  // For SMEM-privatized configurations, copy the in-block SMEM histograms out to
  // their per-block GMEM staging slabs so the combine kernel can read them.
  if (agent.prefer_smem)
  {
    agent.StoreSmemToStagingSlab();
  }

  // PDL trigger MUST be after `StoreSmemToStagingSlab`: the follow-on
  // combine kernel reads from the per-block GMEM staging slabs.
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
}

//! Host-init dynamic-SMEM variant of the staging histogram sweep kernel.
//!
//! Same per-tile accumulate as DeviceHistogramSweepStagingHostInitKernel, but the privatized
//! per-block histogram is stored in dynamic shared memory (`extern __shared__`) instead of
//! the agent's static `_TempStorage`. This lets us scale `PrivatizedSmemBins` up beyond the
//! ptxas default 48 KiB static-SMEM cap on architectures (SM90+, SM100) that support a
//! larger dynamic-SMEM region per CTA via `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`.
//!
//! The host is responsible for:
//!   - Calling `cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
//!     dyn_smem_bytes)` before launch, where `dyn_smem_bytes >= sum_ch num_privatized_bins[ch] * sizeof(CounterT)`.
//!   - Passing `dyn_smem_bytes` as the third triple-chevron parameter at launch.
//!
//! Skips StoreOutput(); a follow-on combine kernel handles cross-block reduction.
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
// Hint the compiler to fit at least 2 blocks per SM. Mirrors the FUSED variants
// (single-chunk and dual-chunk) further below; this kernel is also used by the
// host-init non-byte staging dispatch path and the dispatch sizes the grid
// using ITS sm_occupancy. Without this hint the staging-only kernel would
// remain register-bound at 1 block/SM and the dispatch would launch only
// `sm_count * 1 = 148` blocks even when the (now-occupancy-2) FUSED variant
// could host 2 * 148 = 296 blocks.
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block), 2)
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepStagingHostInitDynSmemKernel(
    _CCCL_GRID_CONSTANT const SampleIteratorT d_samples,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_privatized_bins_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<OutputDecodeOpT, NumActiveChannels> output_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const OffsetT num_row_pixels,
    _CCCL_GRID_CONSTANT const OffsetT num_rows,
    _CCCL_GRID_CONSTANT const OffsetT row_stride_samples,
    _CCCL_GRID_CONSTANT const int tiles_per_row,
    GridQueue<int> tile_queue)
{
  static constexpr histogram_policy hp = current_policy<PolicySelector>();

  using AgentHistogramPolicyT =
    AgentHistogramPolicy<hp.threads_per_block,
                         hp.pixels_per_thread,
                         hp.load_algorithm,
                         hp.load_modifier,
                         hp.rle_compress,
                         hp.mem_preference,
                         hp.work_stealing,
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
                   OffsetT,
                   /* UseDynamicSmemHistogram = */ true>;

  // Static SMEM holds only the BlockLoad union, the tile_idx, and the per-channel
  // histogram pointer array. The histogram bins themselves live in dynamic SMEM below.
  __shared__ typename AgentHistogramT::TempStorage temp_storage;

  // Dynamic SMEM allocated by the launch (third chevron parameter, must be >=
  // sum_ch num_privatized_bins[ch] * sizeof(CounterT)). Layout per channel is
  // contiguous: ch=0 occupies [0, num_privatized_bins[0]), ch=1 occupies
  // [num_privatized_bins[0], num_privatized_bins[0] + num_privatized_bins[1]), etc.
  extern __shared__ unsigned char dyn_smem_raw[];
  CounterT* dyn_smem_histograms = reinterpret_cast<CounterT*>(dyn_smem_raw);

  // Host-init path: hoist the RANGE SearchTransform's loop-invariant
  // interpolation state out of the per-sample classify. No-op for EVEN /
  // byte-sample decode ops.
  OutputDecodeOpT output_decode_op[NumActiveChannels];
  PrivatizedDecodeOpT privatized_decode_op[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int channel = 0; channel < NumActiveChannels; ++channel)
  {
    output_decode_op[channel] = output_decode_op_wrapper[channel];
    privatized_decode_op[channel] = privatized_decode_op_wrapper[channel];
    output_decode_op[channel].PrecomputeOnDevice();
    privatized_decode_op[channel].PrecomputeOnDevice();
  }

  AgentHistogramT agent(
    temp_storage,
    d_samples,
    num_output_bins_wrapper.data(),
    num_privatized_bins_wrapper.data(),
    d_output_histograms_wrapper.data(),
    d_privatized_histograms_wrapper.data(),
    output_decode_op,
    privatized_decode_op,
    dyn_smem_histograms);

  agent.InitBinCounters();
  agent.ConsumeTiles(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);

  // Skip agent.StoreOutput(); the combine kernel handles cross-block reduction.
  if (agent.prefer_smem)
  {
    agent.StoreSmemToStagingSlab();
  }

  // PDL trigger MUST be after `StoreSmemToStagingSlab`: the follow-on
  // combine kernel reads from the per-block GMEM staging slabs.
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
}

//! Host-init dynamic-SMEM, NON-staging variant of the histogram sweep kernel.
//!
//! Identical to `DeviceHistogramSweepStagingHostInitDynSmemKernel` except it
//! does NOT use per-block GMEM staging slabs and a follow-on combine kernel.
//! Instead it merges each block's dyn-SMEM privatized histogram directly into
//! the global output histogram via `agent.StoreOutput()` (per-block
//! `atomicAdd`). The host must still launch `DeviceHistogramInitKernel` first to
//! zero the output histogram (StoreOutput accumulates).
//!
//! Rationale (extends the 2K-tier finding to the dyn-SMEM tiers): for moderate
//! bin counts the staging path's overhead -- a per-block SMEM->GMEM flush, then
//! either a standalone combine launch or a cooperative grid.sync + cross-block
//! gather -- is not always amortised. When the bin count is large enough that
//! cross-block atomic contention on the output is spread thin (each output bin
//! is hit by relatively few blocks), the direct atomic merge can win: it skips
//! the GMEM round-trip of the staging slabs entirely and needs no grid.sync /
//! cooperative launch (so occupancy is governed only by the kernel itself).
//!
//! Host requirements mirror the staging dyn-SMEM kernel:
//!   - `cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem_bytes)`
//!     with `dyn_smem_bytes >= sum_ch num_privatized_bins[ch] * sizeof(CounterT)`.
//!   - Pass `dyn_smem_bytes` as the third triple-chevron parameter at launch.
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
// minBlocks=2 hint mirrors the staging dyn-SMEM kernel; the dispatch sizes the
// grid by this kernel's own sm_occupancy.
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block), 2)
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepNonStagingDynSmemKernel(
    _CCCL_GRID_CONSTANT const SampleIteratorT d_samples,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_privatized_bins_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<OutputDecodeOpT, NumActiveChannels> output_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const OffsetT num_row_pixels,
    _CCCL_GRID_CONSTANT const OffsetT num_rows,
    _CCCL_GRID_CONSTANT const OffsetT row_stride_samples,
    _CCCL_GRID_CONSTANT const int tiles_per_row,
    GridQueue<int> tile_queue)
{
  static constexpr histogram_policy hp = current_policy<PolicySelector>();

  using AgentHistogramPolicyT =
    AgentHistogramPolicy<hp.threads_per_block,
                         hp.pixels_per_thread,
                         hp.load_algorithm,
                         hp.load_modifier,
                         hp.rle_compress,
                         hp.mem_preference,
                         hp.work_stealing,
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
                   OffsetT,
                   /* UseDynamicSmemHistogram = */ true>;

  __shared__ typename AgentHistogramT::TempStorage temp_storage;

  extern __shared__ unsigned char dyn_smem_raw[];
  CounterT* dyn_smem_histograms = reinterpret_cast<CounterT*>(dyn_smem_raw);

  // Host-init path: hoist the RANGE SearchTransform's loop-invariant
  // interpolation state out of the per-sample classify. No-op for EVEN /
  // byte-sample decode ops.
  OutputDecodeOpT output_decode_op[NumActiveChannels];
  PrivatizedDecodeOpT privatized_decode_op[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int channel = 0; channel < NumActiveChannels; ++channel)
  {
    output_decode_op[channel] = output_decode_op_wrapper[channel];
    privatized_decode_op[channel] = privatized_decode_op_wrapper[channel];
    output_decode_op[channel].PrecomputeOnDevice();
    privatized_decode_op[channel].PrecomputeOnDevice();
  }

  AgentHistogramT agent(
    temp_storage,
    d_samples,
    num_output_bins_wrapper.data(),
    num_privatized_bins_wrapper.data(),
    d_output_histograms_wrapper.data(),
    d_privatized_histograms_wrapper.data(),
    output_decode_op,
    privatized_decode_op,
    dyn_smem_histograms);

  agent.InitBinCounters();
  agent.ConsumeTiles(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);

  // Direct per-block atomic merge to the global output histogram (no staging
  // slab, no combine kernel).
  agent.StoreOutput();

  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
}

//! Device-init dynamic-SMEM variant of the staging histogram sweep kernel. Mirrors
//! DeviceHistogramSweepStagingKernel but with the privatized histogram in extern __shared__
//! memory; see DeviceHistogramSweepStagingHostInitDynSmemKernel for the host-side requirements.
template <typename PolicySelector,
          int PrivatizedSmemBins,
          int NumChannels,
          int NumActiveChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename FirstLevelArrayT,
          typename SecondLevelArrayT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT,
          bool IsEven>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
// minBlocks=2 hint mirrors DeviceHistogramSweepStagingHostInitDynSmemKernel.
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block), 2)
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepStagingDynSmemKernel(
    _CCCL_GRID_CONSTANT const SampleIteratorT d_samples,
    ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    ::cuda::std::array<int, NumActiveChannels> num_privatized_bins_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
    _CCCL_GRID_CONSTANT const FirstLevelArrayT first_level_array,
    _CCCL_GRID_CONSTANT const SecondLevelArrayT second_level_array,
    _CCCL_GRID_CONSTANT const OffsetT num_row_pixels,
    _CCCL_GRID_CONSTANT const OffsetT num_rows,
    _CCCL_GRID_CONSTANT const OffsetT row_stride_samples,
    _CCCL_GRID_CONSTANT const int tiles_per_row,
    _CCCL_GRID_CONSTANT const GridQueue<int> tile_queue)
{
  static constexpr histogram_policy hp = current_policy<PolicySelector>();

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

  // Hoist the RANGE SearchTransform's loop-invariant interpolation slope and
  // boundary levels out of the per-sample classify. No-op for EVEN
  // (ScaleTransform) and byte-sample (PassThruTransform) decode ops.
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int channel = 0; channel < NumActiveChannels; ++channel)
  {
    privatized_decode_op[channel].PrecomputeOnDevice();
    output_decode_op[channel].PrecomputeOnDevice();
  }

  using AgentHistogramPolicyT =
    AgentHistogramPolicy<hp.threads_per_block,
                         hp.pixels_per_thread,
                         hp.load_algorithm,
                         hp.load_modifier,
                         hp.rle_compress,
                         hp.mem_preference,
                         hp.work_stealing,
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
                   OffsetT,
                   /* UseDynamicSmemHistogram = */ true>;

  __shared__ typename AgentHistogramT::TempStorage temp_storage;

  extern __shared__ unsigned char dyn_smem_raw[];
  CounterT* dyn_smem_histograms = reinterpret_cast<CounterT*>(dyn_smem_raw);

  AgentHistogramT agent(
    temp_storage,
    d_samples,
    num_output_bins_wrapper.data(),
    num_privatized_bins_wrapper.data(),
    d_output_histograms_wrapper.data(),
    d_privatized_histograms_wrapper.data(),
    output_decode_op,
    privatized_decode_op,
    dyn_smem_histograms);

  agent.InitBinCounters();
  agent.ConsumeTiles(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);

  // Skip agent.StoreOutput() -- combine kernel below handles per-block reduction.
  if (agent.prefer_smem)
  {
    agent.StoreSmemToStagingSlab();
  }

  // PDL trigger MUST be after `StoreSmemToStagingSlab`: the follow-on
  // combine kernel reads from the per-block GMEM staging slabs.
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
}

//! Host-init dynamic-SMEM variant of the FUSED staging+combine sweep kernel.
//!
//! This kernel fuses the staging-sweep + cross-block combine pair (previously
//! `DeviceHistogramSweepStagingHostInitDynSmemKernel` followed by
//! `DeviceHistogramCombineKernel`) into a single cooperative-launch kernel
//! that uses `cooperative_groups::this_grid().sync()` between phases. This
//! eliminates the launch overhead of the standalone combine kernel (~18us
//! visible on small-Elements configs) by avoiding a second `cudaLaunch*`
//! round-trip and the associated stream synchronization.
//!
//! The kernel must be launched cooperatively
//! (`cudaLaunchCooperativeKernel`) so that all blocks are co-resident on the
//! device, which is a precondition of `grid_group::sync()`. The dispatch
//! layer is responsible for verifying that the requested grid fits on the
//! device before selecting this kernel.
//!
//! Phases:
//!   1. Each block sweeps its share of the input via `AgentHistogram::ConsumeTiles`
//!      into per-block dyn-SMEM histograms (the AgentHistogram path with
//!      UseDynamicSmemHistogram=true).
//!   2. Each block flushes its dyn-SMEM histograms to its per-block GMEM
//!      staging slab via `agent.StoreSmemToStagingSlab()`.
//!   3. `grid.sync()` makes all per-block staging slabs visible to every
//!      block.
//!   4. Atomic-free reduce: each thread takes a slice of (channel, bin)
//!      output indices, sums the corresponding column across all blocks of
//!      the staging matrix, and writes the final value to the output
//!      histogram.
//!
//! Note that this kernel performs the output-histogram zeroing implicitly via
//! the final write-out in phase 4; the host therefore does NOT need to launch
//! `DeviceHistogramInitKernel` before this kernel for the bins covered by
//! `num_privatized_bins_wrapper` (which equals `num_output_bins_wrapper` for
//! the non-byte single-channel xlarge tier). The drain counter inside
//! `tile_queue` is reset by thread 0 of block 0 before the sweep begins, so
//! the dispatch path can also skip the standalone init kernel entirely.
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
// Hint the compiler to fit at least 2 blocks per SM. NCU showed Block Limit
// Registers = 1 for the fused staging+combine kernel (68 regs/thread × 768 threads
// = 52 KiB regs/block exceeds 32 KiB/block needed for 2 blocks/SM out of B200's
// 64 KiB regs/SM). Forcing the compiler to budget for 2 blocks reduces register
// allocation per thread and unlocks higher occupancy when SMEM fits (Bins<=8192
// staging tiers).
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block), 2)
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepStagingFusedHostInitDynSmemKernel(
    _CCCL_GRID_CONSTANT const SampleIteratorT d_samples,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_privatized_bins_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<OutputDecodeOpT, NumActiveChannels> output_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const OffsetT num_row_pixels,
    _CCCL_GRID_CONSTANT const OffsetT num_rows,
    _CCCL_GRID_CONSTANT const OffsetT row_stride_samples,
    _CCCL_GRID_CONSTANT const int tiles_per_row,
    GridQueue<int> tile_queue)
{
  static constexpr histogram_policy hp = current_policy<PolicySelector>();

  namespace cg = ::cooperative_groups;

  cg::grid_group grid = cg::this_grid();

  using AgentHistogramPolicyT =
    AgentHistogramPolicy<hp.threads_per_block,
                         hp.pixels_per_thread,
                         hp.load_algorithm,
                         hp.load_modifier,
                         hp.rle_compress,
                         hp.mem_preference,
                         hp.work_stealing,
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
                   OffsetT,
                   /* UseDynamicSmemHistogram = */ true>;

  __shared__ typename AgentHistogramT::TempStorage temp_storage;

  extern __shared__ unsigned char dyn_smem_raw[];
  CounterT* dyn_smem_histograms = reinterpret_cast<CounterT*>(dyn_smem_raw);

  // Save the per-channel base of the all-blocks staging slab BEFORE the agent
  // constructor offsets `d_privatized_histograms` by `block_id * num_privatized_bins[ch]`.
  CounterT* d_privatized_base[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    d_privatized_base[ch] = d_privatized_histograms_wrapper[ch];
  }

  // Reset the drain counter so the work-stealing path can use the queue. We do
  // this BEFORE constructing the agent so the agent's ConsumeTiles can see a
  // zero drain counter; for the non-work-stealing single-channel xlarge tier
  // the queue is unused, but resetting it is cheap and makes the kernel safe
  // for both paths.
  const unsigned int blocks_per_grid = gridDim.x * gridDim.y * gridDim.z;
  const unsigned int block_id        = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const unsigned int tid_global      = block_id * blockDim.x + threadIdx.x;
  const unsigned int total_threads   = blocks_per_grid * blockDim.x;

  if (tid_global == 0)
  {
    GridQueue<int> queue = tile_queue;
    queue.ResetDrain();
  }

  // grid.sync() so the drain reset is visible to every block before the sweep.
  grid.sync();

  // Host-init path: hoist the RANGE SearchTransform's loop-invariant
  // interpolation state out of the per-sample classify. No-op for EVEN /
  // byte-sample decode ops. (For chunked dispatch this also covers the
  // ChunkedDecodeOp wrapper, which forwards PrecomputeOnDevice to its inner
  // SearchTransform.)
  OutputDecodeOpT output_decode_op[NumActiveChannels];
  PrivatizedDecodeOpT privatized_decode_op[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    output_decode_op[ch] = output_decode_op_wrapper[ch];
    privatized_decode_op[ch] = privatized_decode_op_wrapper[ch];
    output_decode_op[ch].PrecomputeOnDevice();
    privatized_decode_op[ch].PrecomputeOnDevice();
  }

  {
    AgentHistogramT agent(
      temp_storage,
      d_samples,
      num_output_bins_wrapper.data(),
      num_privatized_bins_wrapper.data(),
      d_output_histograms_wrapper.data(),
      d_privatized_histograms_wrapper.data(),
      output_decode_op,
      privatized_decode_op,
      dyn_smem_histograms);

    agent.InitBinCounters();
    agent.ConsumeTiles(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);

    if (agent.prefer_smem)
    {
      agent.StoreSmemToStagingSlab();
    }
  }

  // Phase 3: grid-wide sync so that every block's staging slab writes are
  // visible to every block before the cross-block reduce.
  grid.sync();

  // Phase 4: atomic-free reduce across blocks. For the non-byte single-channel
  // xlarge tier `num_privatized_bins[ch] == num_output_bins[ch]` and
  // `output_decode_op` is identity (PassThruTransform), so we can directly
  // sum the staging column for each output bin.
  //
  // Two reduction strategies switch based on `num_bins` vs `total_warps`:
  //
  // 1. Warp-cooperative (when num_bins < total_warps): one warp owns one bin
  //    at a time; 32 lanes split the per-block column reduction (each lane
  //    sums ceil(blocks_per_grid / 32) strided GMEM loads in parallel) then
  //    warp-shuffle reduce. Avoids leaving 86%+ of threads idle when
  //    `total_threads >> num_bins` (e.g. Bins=2000 with 296*768=227K threads,
  //    or 148*768=113K threads at 1 block/SM).
  //
  // 2. Thread-per-bin (when num_bins >= total_warps): each thread owns one
  //    bin and sequentially sums all `blocks_per_grid` slabs. With
  //    num_bins close to total_threads, every thread has its own bin so
  //    parallelism is across bins rather than within bins; this is the
  //    original gather pattern and avoids the per-bin warp-shuffle overhead.
  {
    const unsigned int warp_id_global = tid_global >> 5; // tid_global / 32
    const unsigned int lane           = tid_global & 31u;
    const unsigned int total_warps    = total_threads >> 5;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      const int num_bins                = num_privatized_bins_wrapper[ch];
      const CounterT* __restrict__ base = d_privatized_base[ch];
      CounterT* d_out                   = d_output_histograms_wrapper[ch];
      const unsigned int num_bins_u     = static_cast<unsigned int>(num_bins);

      if (num_bins_u < total_warps)
      {
        for (unsigned int bin = warp_id_global; bin < num_bins_u; bin += total_warps)
        {
          CounterT partial = 0;
          for (unsigned int b = lane; b < blocks_per_grid; b += 32u)
          {
            partial += base[b * num_bins_u + bin];
          }
          partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 16);
          partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 8);
          partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 4);
          partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 2);
          partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 1);
          if (lane == 0)
          {
            d_out[bin] = partial;
          }
        }
      }
      else
      {
        for (unsigned int bin = tid_global; bin < num_bins_u; bin += total_threads)
        {
          CounterT total = 0;
          for (unsigned int b = 0; b < blocks_per_grid; ++b)
          {
            total += base[b * num_bins_u + bin];
          }
          d_out[bin] = total;
        }
      }
    }
  }

  // Emit the trigger so any PDL-launched downstream kernel in the stream
  // sees a completion signal. (Cooperative launches typically do not use
  // PDL, so this is a no-op in the common case.)
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
}

//! Combine kernel: reduces per-block privatized histograms across all blocks
//! into the final output histogram. Each output bin is computed by summing
//! the corresponding column of the (num_blocks x num_privatized_bins) staging
//! matrix.
//!
//! For non-byte samples (PassThruTransform output decode), num_privatized_bins
//! equals num_output_bins and the bin index is identity. For byte samples
//! (with a non-trivial output decode op) the host pre-decodes the bin mapping
//! before launch.
//!
//! Launch configuration: 256 threads/block, gridDim.x covers num_privatized_bins
//! per channel; gridDim.y is NumActiveChannels.
template <int NumActiveChannels, typename CounterT>
_CCCL_KERNEL_ATTRIBUTES void DeviceHistogramCombineKernel(
  ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
  ::cuda::std::array<const CounterT*, NumActiveChannels> d_privatized_histograms_wrapper,
  ::cuda::std::array<int, NumActiveChannels> num_privatized_bins_wrapper,
  int num_thread_blocks)
{
  // The dispatch launches this kernel with `dependent_launch=true` for PDL.
  // Without `cudaGridDependencySynchronize()` here, this kernel can start
  // running BEFORE the previous (staging-sweep) kernel has finished writing
  // its per-block staging slabs to GMEM. The sweep kernel does the staging
  // slab write in `StoreSmemToStagingSlab()` which executes AFTER the agent
  // has emitted `cudaTriggerProgrammaticLaunchCompletion()` (the agent
  // emits the trigger at the end of `ConsumeTiles`, before the per-CTA
  // SMEM->GMEM flush). The combine kernel must therefore explicitly wait
  // for the sweep kernel to fully exit (membar) before reading the staging
  // slabs. `cudaGridDependencySynchronize()` does that.
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();

  const int channel = blockIdx.y;
  if (channel >= NumActiveChannels)
  {
    return;
  }

  const int channel_bins = num_privatized_bins_wrapper[channel];
  const int bin          = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin >= channel_bins)
  {
    return;
  }

  const CounterT* __restrict__ priv = d_privatized_histograms_wrapper[channel];
  CounterT sum                      = 0;

  // Sum the same bin across all blocks. Memory layout: priv[block_idx * channel_bins + bin]
  // Stride is `channel_bins`. Since blocks read the same `bin` column, this is a strided
  // reduction; L2 cache should help amortize the strided fetches across warps.
  for (int b = 0; b < num_thread_blocks; ++b)
  {
    sum += priv[b * channel_bins + bin];
  }

  // Write final output. The init kernel zeroed d_output_histograms,
  // so a non-atomic store is safe here (bin index is unique per thread).
  d_output_histograms_wrapper[channel][bin] = sum;
}

//! Single-launch hybrid SMEM+GMEM fused staging+combine sweep kernel.
//!
//! Eliminates the 2x sample re-read of the dual-chunk kernel by handling both
//! "chunks" of the privatized-bin space in a single sweep:
//!  - Bins `[0, hybrid_split_bin)` accumulate in per-block dyn-SMEM (size
//!    hybrid_split_bin * NumActiveChannels * sizeof(CounterT) bytes).
//!  - Bins `[hybrid_split_bin, hybrid_split_bin + hybrid_secondary_size)`
//!    accumulate in a per-block per-channel GMEM staging slab via
//!    `atomicAdd_block` (CTA-scoped).
//!
//! Using the un-chunked decode op, each sample is classified ONCE; the agent
//! routes the resulting bin to either SMEM or per-block GMEM in a single pass.
//! After the sweep, this block's SMEM histogram is flushed to its primary GMEM
//! staging slab, the grid syncs, and the cross-block atomic-free reduce gathers
//! both regions into the final output histogram.
//!
//! Trade vs the dual-chunk kernel:
//!  - 1x sample reads instead of 2x (eliminates the second sweep).
//!  - 1x un-chunked classify per sample instead of 2x chunked classifies.
//!  - 1x SMEM atomic OR 1x per-block-GMEM atomic per sample (instead of 1x SMEM
//!    atomic per chunk; net same number of atomics).
//!
//! The kernel must be launched cooperatively (`cudaLaunchCooperativeKernel`)
//! so that all blocks are co-resident on the device, which is a precondition of
//! `grid_group::sync()`.
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
// __launch_bounds__(N, 1): hybrid is limited to 1 block/SM by the 56000-bin SMEM
// allocation (~224 KB/CTA on B200), so the minBlocks=1 hint lets the compiler use
// more registers per thread (no need to budget for 2 blocks/SM SMEM-fit).
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block), 1)
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSweepStagingFusedHybridSinglePassHostInitDynSmemKernel(
    _CCCL_GRID_CONSTANT const SampleIteratorT d_samples,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_smem_bins_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_gmem_bins_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_primary_staging_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_secondary_staging_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<OutputDecodeOpT, NumActiveChannels> output_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const int hybrid_split_bin,
    _CCCL_GRID_CONSTANT const int hybrid_secondary_size,
    _CCCL_GRID_CONSTANT const OffsetT num_row_pixels,
    _CCCL_GRID_CONSTANT const OffsetT num_rows,
    _CCCL_GRID_CONSTANT const OffsetT row_stride_samples,
    _CCCL_GRID_CONSTANT const int tiles_per_row,
    GridQueue<int> tile_queue)
{
  static constexpr histogram_policy hp = current_policy<PolicySelector>();

  namespace cg = ::cooperative_groups;

  cg::grid_group grid = cg::this_grid();

  using AgentHistogramPolicyT =
    AgentHistogramPolicy<hp.threads_per_block,
                         hp.pixels_per_thread,
                         hp.load_algorithm,
                         hp.load_modifier,
                         hp.rle_compress,
                         hp.mem_preference,
                         hp.work_stealing,
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
                   OffsetT,
                   /* UseDynamicSmemHistogram = */ true>;

  __shared__ typename AgentHistogramT::TempStorage temp_storage;

  extern __shared__ unsigned char dyn_smem_raw[];
  CounterT* dyn_smem_histograms = reinterpret_cast<CounterT*>(dyn_smem_raw);

  // Save the per-channel bases of the all-blocks staging slabs BEFORE the agent
  // constructor offsets them by `block_id * num_*_bins[ch]`.
  CounterT* d_primary_base[NumActiveChannels];
  CounterT* d_secondary_base[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    d_primary_base[ch]   = d_primary_staging_wrapper[ch];
    d_secondary_base[ch] = d_secondary_staging_wrapper[ch];
  }

  const unsigned int blocks_per_grid = gridDim.x * gridDim.y * gridDim.z;
  const unsigned int block_id        = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const unsigned int tid_global      = block_id * blockDim.x + threadIdx.x;
  const unsigned int total_threads   = blocks_per_grid * blockDim.x;

  // Drain-reset + grid.sync are only needed when the policy enables work stealing
  // (the drain counter is consumed by the WS even-share-then-steal loop). For
  // even-share-only policies the drain queue is never read, so we can elide both
  // the reset and the grid.sync, saving one grid-wide barrier (~5 us on B200).
  if constexpr (hp.work_stealing)
  {
    if (tid_global == 0)
    {
      GridQueue<int> queue = tile_queue;
      queue.ResetDrain();
    }
    grid.sync();
  }

  // Host-init path: hoist the RANGE SearchTransform's loop-invariant
  // interpolation state out of the per-sample classify. No-op for EVEN /
  // byte-sample decode ops.
  OutputDecodeOpT output_decode_op[NumActiveChannels];
  PrivatizedDecodeOpT privatized_decode_op[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    output_decode_op[ch] = output_decode_op_wrapper[ch];
    privatized_decode_op[ch] = privatized_decode_op_wrapper[ch];
    output_decode_op[ch].PrecomputeOnDevice();
    privatized_decode_op[ch].PrecomputeOnDevice();
  }

  // Single sweep with hybrid accumulation.
  {
    AgentHistogramT agent(
      temp_storage,
      d_samples,
      num_smem_bins_wrapper.data(),
      num_smem_bins_wrapper.data(),
      d_output_histograms_wrapper.data(),
      d_primary_staging_wrapper.data(),
      d_secondary_staging_wrapper.data(),
      output_decode_op,
      privatized_decode_op,
      dyn_smem_histograms,
      hybrid_split_bin,
      hybrid_secondary_size);

    agent.InitBinCountersHybrid();
    agent.ConsumeTilesHybrid(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);
    agent.StoreHybridSmemToStagingSlab();
  }

  // grid-wide sync: all blocks' primary+secondary staging-slab writes visible to every block.
  grid.sync();

  // Atomic-free reduce: for each channel, gather the primary slab into output[0..split)
  // and the secondary slab into output[split..split+secondary). Both are simple column-sum
  // reductions across blocks.
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    CounterT* d_out                             = d_output_histograms_wrapper[ch];
    const CounterT* __restrict__ primary_base   = d_primary_base[ch];
    const CounterT* __restrict__ secondary_base = d_secondary_base[ch];
    const unsigned int split_u                  = static_cast<unsigned int>(hybrid_split_bin);
    const unsigned int sec_u                    = static_cast<unsigned int>(hybrid_secondary_size);
    const unsigned int total_bins               = split_u + sec_u;

    for (unsigned int bin = tid_global; bin < total_bins; bin += total_threads)
    {
      CounterT total = 0;
      if (bin < split_u)
      {
        for (unsigned int b = 0; b < blocks_per_grid; ++b)
        {
          total += primary_base[b * split_u + bin];
        }
        d_out[bin] = total;
      }
      else
      {
        const unsigned int gbin = bin - split_u;
        for (unsigned int b = 0; b < blocks_per_grid; ++b)
        {
          total += secondary_base[b * sec_u + gbin];
        }
        d_out[bin] = total;
      }
    }
  }

  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
}

} // namespace detail::histogram
CUB_NAMESPACE_END
