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
#include <cuda/std/__type_traits/is_pointer.h>
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

// Cache-slot hashing for the direct-atomic SMEM cuckoo / single-probe caches.
//
// The slot index must be in [0, slots) where slots == mask + 1 is a power of
// two. Two multiplicative hash constants are used (primary / secondary) for the
// two cuckoo probes. The MODE controls how the hash maps to a slot:
//
//   CUB_HISTO_CACHE_HASH_MODE == 0 (low-bits): slot = (bin * M) & mask
//       Uses the LOW log2(slots) bits of the multiplicative product. For an odd
//       multiplier M, the low k bits of (bin * M) depend ONLY on the low k bits
//       of `bin` -- multiplicative hashing does no mixing in its low bits. Since
//       the SMEM bank is `slot & 31`, the bank is effectively the low 5 bits of
//       `bin`, so structured / clustered bin ids alias onto few banks and the key
//       loads bank-conflict on the SMEM-bound cache.
//
//   CUB_HISTO_CACHE_HASH_MODE == 1 (Fibonacci / high-bits): slot = (bin * M) >> (32 - log2(slots))
//       Uses the HIGH bits of the product, which ARE well mixed for a good
//       multiplier (classic Fibonacci hashing). The bank `slot & 31` now comes
//       from high-entropy bits, breaking the low-bit bank aliasing.
//
//   CUB_HISTO_CACHE_HASH_MODE == 2 (xor-fold): slot = ((h >> 15) ^ h) & mask
//       Folds the high bits into the low bits before masking, so the masked
//       slot (and its bank) sees mixed bits while keeping a cheap `& mask`.
//
// `cache_slot_log2` is log2(slots) == popcount(mask) == 32 - clz(mask) for a
// power-of-two `slots`. We pass it explicitly (computed once per launch) to keep
// the hot path free of clz.
#ifndef CUB_HISTO_CACHE_HASH_MODE
// Default to Fibonacci high-bits hashing (mode 1). The low-bits `& mask` (mode 0)
// takes the low log2(slots) bits of the multiplicative product, which for an odd
// multiplier depend ONLY on the low bits of `bin`; the SMEM bank (slot & 31) is
// therefore the low 5 bits of `bin`, so clustered / skewed bin distributions alias
// onto few banks and bank-conflict on the SMEM-bound multi-channel cache. Mode 1
// uses the well-mixed HIGH bits, breaking that aliasing.
#  define CUB_HISTO_CACHE_HASH_MODE 1
#endif

// Set-associativity for the SINGLE-PROBE direct-atomic SMEM cache.
//
// The single-probe cache (the `single_probe_cache` probe op of
// DeviceHistogramDirectKernel) is direct-mapped: each bin hashes to exactly
// ONE slot, and a slot collision (slot owned by another bin) is an IMMEDIATE GMEM
// spill -- no second chance. That
// single-slot conflict miss is the residual loss once Fibonacci hashing and
// vectorized loads are in place.
//
// CUB_HISTO_SINGLE_PROBE_WAYS sets the SET ASSOCIATIVITY of that cache WITHOUT
// changing its SMEM footprint: the same `cache_slots_per_channel` budget is
// reinterpreted as `slots / WAYS` SETS of `WAYS` contiguous slots. A bin hashes to
// a set; the leader probes all WAYS slots of the set before spilling. The ways of a
// set are ADJACENT in SMEM (base..base+WAYS-1) so the WAYS key reads land in one or
// two banks (unlike the cuckoo kernel's two INDEPENDENT random hashes, which scatter
// across banks) -- like cuckoo's two candidate slots but without full eviction and
// with bank locality. WAYS must be a power of two and must divide the (power-of-two)
// slot count.
//
//   WAYS == 1 (default): direct-mapped, BYTE-IDENTICAL to an unassociative cache.
//   WAYS == 2: 2-way set-associative (a colliding bin gets a fallback way before
//              spilling, roughly halving conflict misses).
//   WAYS == 4: 4-way (more ways = fewer conflict misses but more probes/set).
#ifndef CUB_HISTO_SINGLE_PROBE_WAYS
#  define CUB_HISTO_SINGLE_PROBE_WAYS 1
#endif

_CCCL_DEVICE _CCCL_FORCEINLINE int
cache_slot_from_hash(unsigned int product, int cache_mask, int cache_slot_log2)
{
#if CUB_HISTO_CACHE_HASH_MODE == 1
  // High-bits (Fibonacci): shift the well-mixed top bits down to the slot range.
  (void) cache_mask;
  return static_cast<int>(product >> (32 - cache_slot_log2));
#elif CUB_HISTO_CACHE_HASH_MODE == 2
  // XOR-fold high bits into low, then mask.
  (void) cache_slot_log2;
  return static_cast<int>(((product >> 15) ^ product) & static_cast<unsigned int>(cache_mask));
#else
  // Historical: low bits of the multiplicative product.
  (void) cache_slot_log2;
  return static_cast<int>(product & static_cast<unsigned int>(cache_mask));
#endif
}

//! @brief Return the underlying native sample pointer for the direct-atomic
//! sweep kernels' VECTORIZED multi-channel load fast path, or `nullptr` when
//! the input iterator is not a plain pointer.
//!
//! The multi-channel direct-atomic sweep loads each pixel's `NumChannels`
//! interleaved samples. The scalar form issues `NumChannels` separate global
//! loads per pixel; because consecutive threads read consecutive pixels and a
//! pixel spans `NumChannels` samples, that pattern leaves a gap every
//! `NumChannels`-th sample (e.g. the unhistogrammed alpha lane of an RGBA
//! image) and under-fills global-load transactions. Issuing instead ONE wide
//! `CubVector<SampleT, NumChannels>` (e.g. `int4`) load per pixel reads the
//! whole pixel contiguously, so the warp's loads are perfectly packed and the
//! global-load transaction count drops by up to `NumChannels`x.
//!
//! This only works when the samples live behind a real pointer (the common
//! `DeviceHistogram::MultiHistogram*` case) and that pointer is suitably
//! aligned for the vector type. For fancy iterators we return a typed null
//! pointer; the caller then keeps the per-channel scalar path. The return type
//! is always `const SampleValueT*` so the kernel's `auto*` binding and its
//! `nullptr` guard are well-formed for every iterator instantiation.
template <typename SampleValueT, typename SampleIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE const SampleValueT* SampleNativePointer(SampleIteratorT itr)
{
  if constexpr (::cuda::std::is_pointer_v<SampleIteratorT>)
  {
    return itr; // raw pointer: already native.
  }
  else
  {
    // CacheModifiedInputIterator (and anything else exposing a native `.ptr`
    // member) fall through to the agent's NativePointer overloads; everything
    // else yields nullptr and the scalar fallback is used. NativePointer
    // returns `nullptr` (nullptr_t) for the generic case, which converts to a
    // null `const SampleValueT*`.
    return NativePointer(itr);
  }
  _CCCL_UNREACHABLE();
}

//! @brief Atomic-free gather-merge of a per-block GMEM-privatized slab into the
//! output histogram (one channel, one contiguous bin range).
//!
//! Every block holds its own private copy of `count` consecutive output bins at
//! `slab_base + block * slab_stride`. This sums those copies across all blocks
//! into `out[out_offset + i]` with plain reads + one write per bin, turning the
//! `num_blocks * count` cross-block `atomicAdd`s the privatized path would
//! otherwise need into contention-free loads. A grid-wide barrier MUST have been
//! issued by the caller before this runs (so every block's slab writes are
//! visible). Each thread of the cooperative grid owns a grid-strided slice of the
//! `count` output bins.
//!
//! This is the single source for the column-sum reduce shared by the privatized
//! sweep kernels (`DeviceHistogramGmemPrivatizedKernel` -- the merged hybrid /
//! gather kernel -- and the direct kernel's private-spill Phase 4); previously the
//! same loop was open-coded in the GMEM-priv gather kernel and the hybrid kernel.
//!
//! @param out          Output histogram base for this channel.
//! @param out_offset   First output bin this gather writes (0 for the primary
//!                     region; `split` for the hybrid secondary region).
//! @param slab_base    Base of the all-blocks private slab for this channel.
//! @param slab_stride  Per-block stride within the slab (bins each block owns).
//! @param count        Number of bins to gather (== slab_stride for the simple
//!                     case; the hybrid secondary size for the tail region).
//! @param blocks_per_grid Number of co-resident blocks.
//! @param tid_global   Global thread id within the cooperative grid.
//! @param total_threads Total threads in the cooperative grid.
template <typename CounterT>
_CCCL_DEVICE _CCCL_FORCEINLINE void gather_privatized_slab(
  CounterT* out,
  unsigned int out_offset,
  const CounterT* slab_base,
  unsigned int slab_stride,
  unsigned int count,
  unsigned int blocks_per_grid,
  unsigned int tid_global,
  unsigned int total_threads)
{
  for (unsigned int i = tid_global; i < count; i += total_threads)
  {
    CounterT total = 0;
    for (unsigned int b = 0; b < blocks_per_grid; ++b)
    {
      total += slab_base[static_cast<size_t>(b) * slab_stride + i];
    }
    out[out_offset + i] = total;
  }
}

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
    // Compile-time RANGE marker, resolved at instantiation (no runtime branch).
    // The direct-atomic kernels read this to specialize behavior that only helps
    // the RANGE (SearchTransform) classify, e.g. the per-thread bracket cache.
    static constexpr bool is_range_transform = true;

    //! @brief Per-thread most-recently-used (MRU) bin-bracket cache.
    //!
    //! Carries the last successfully-resolved bin and its two boundary level
    //! values across consecutive `BinSelect` calls so a new sample that falls in
    //! the same `[lo, hi)` bracket is classified with ZERO level-array loads (a
    //! handful of register compares), skipping the interpolated first-guess, the
    //! clamp, and -- crucially -- both verify loads on the dependent
    //! `IMAD.WIDE -> LDG` level-load chain that binds the latency-bound RANGE
    //! classify. Low-entropy inputs (constant or heavily-skewed samples) have high
    //! consecutive-sample locality, so the bracket hits dominate. A `bin < 0`
    //! sentinel marks the cache empty.
    //! This is per-thread mutable state, so it is only sound on a per-thread
    //! `SearchTransform` copy (the direct-atomic cuckoo/single-probe kernels'
    //! `decode_op[ch]`), never on the shared `__grid_constant__` decode op that
    //! the SMEM-privatized agent path reads through a const pointer.
    struct BracketCacheT
    {
      LevelT lo; // cached d_levels[bin]
      LevelT hi; // cached d_levels[bin + 1]
      int bin = -1; // cached bin; < 0 means empty
    };

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

    // Three-point (piecewise-linear) interpolation state, populated by
    // PrecomputeOnDevice alongside the single-secant fields above. Splitting
    // the [first,last] range at the midpoint level d_levels[mid] and
    // interpolating on whichever half the sample falls in (a) halves the
    // magnitude of `delta` fed to the lossy 32-bit float guess -- so the
    // first-guess lands closer to the true bin and the verify-or-1-step ladder
    // converges without reaching UpperBound -- and (b) captures large-scale
    // non-uniformity (a slope change between the two halves) that a single
    // first->last secant cannot. `m_mid_bin` is the bin index at the split.
    LevelT m_mid; // cached d_levels[mid_bin]
    float m_inv_scale_lo; // mid_bin / (float)(mid - first)
    float m_inv_scale_hi; // (num_bins - mid_bin) / (float)(last - mid)
    int m_mid_bin; // split bin index (num_bins / 2)

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

      // Three-point split at the midpoint bin. Read d_levels[mid] and derive
      // the two half-slopes. If either half is degenerate (non-increasing),
      // fall back to the single-secant guess by setting m_mid_bin = 0, which
      // BinSelect treats as "no split".
      m_mid_bin      = 0;
      m_inv_scale_lo = m_inv_scale;
      m_inv_scale_hi = m_inv_scale;
      m_mid          = first;
      const int mid_bin = num_bins >> 1;
      if (mid_bin > 0 && mid_bin < num_bins)
      {
        const LevelT mid = wrapped_levels[mid_bin];
        if ((first < mid) && (mid < last))
        {
          m_mid          = mid;
          m_mid_bin      = mid_bin;
          m_inv_scale_lo = static_cast<float>(mid_bin) / static_cast<float>(mid - first);
          m_inv_scale_hi = static_cast<float>(num_bins - mid_bin) / static_cast<float>(last - mid);
        }
      }
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
        // Three-point piecewise-linear first guess: interpolate on whichever
        // half of [first, last] the sample falls in (split at the cached
        // midpoint level m_mid / m_mid_bin). Using a local slope and a smaller
        // delta magnitude lands the guess closer to the true bin than a single
        // first->last secant, so the verify-or-1-step ladder converges without
        // reaching the UpperBound binary search. m_mid_bin == 0 means the split
        // was degenerate, so we use the single-secant guess.
        if (m_mid_bin > 0)
        {
          if (s < m_mid)
          {
            guess = static_cast<int>(static_cast<float>(delta) * m_inv_scale_lo);
          }
          else
          {
            const auto delta_hi = (s - m_mid);
            guess               = m_mid_bin + static_cast<int>(static_cast<float>(delta_hi) * m_inv_scale_hi);
          }
        }
        else
        {
          guess = static_cast<int>(static_cast<float>(delta) * m_inv_scale);
        }
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

    //! @brief MRU-bracket-cached `BinSelect`.
    //!
    //! Same contract and result as the plain `BinSelect` above, but threads a
    //! per-thread `BracketCacheT` across calls to exploit consecutive-sample
    //! temporal locality. The fast path tests the cached `[lo, hi)` bracket with
    //! register compares only -- on a hit it returns the cached bin without ANY
    //! level-array load, cutting the dependent `IMAD.WIDE -> LDG` chain that
    //! binds the high-bin RANGE classify. On a miss it runs the identical
    //! interpolated-guess / verify / 1-step / `UpperBound` ladder as the plain
    //! path (so correctness, including the non-uniform-level fallback, is
    //! unchanged) and then records the resolved bracket -- reusing the bracket
    //! levels the ladder already loaded in the common verify/1-step cases, and
    //! reloading only on the rare `UpperBound` fallback.
    template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void
    BinSelect(_SampleT sample, int& bin, bool valid, BracketCacheT& mru) const
    {
      using WrappedLevelIteratorT =
        ::cuda::std::_If<::cuda::std::is_pointer_v<LevelIteratorT>,
                         CacheModifiedInputIterator<LOAD_MODIFIER, LevelT, OffsetT>,
                         LevelIteratorT>;

      const int num_bins = num_output_levels - 1;
      if (!valid)
      {
        return;
      }

      const LevelT s = static_cast<LevelT>(sample);

      // Fast path: the cached bracket holds the answer with no level loads.
      // `mru.bin >= 0` guarantees the bracket is populated and in-range.
      if (mru.bin >= 0 && !(s < mru.lo) && (s < mru.hi))
      {
        bin = mru.bin;
        return;
      }

      // Tiny bin counts: the interpolation/bracket machinery is not worth it.
      if (num_bins < 4)
      {
        WrappedLevelIteratorT wrapped_levels(d_levels);
        bin = UpperBound(wrapped_levels, num_output_levels, s) - 1;
        if (bin >= num_bins)
        {
          bin = -1;
        }
        return;
      }

      WrappedLevelIteratorT wrapped_levels(d_levels);

      const LevelT first_level = m_have_precompute ? m_first : wrapped_levels[0];
      const LevelT last_level  = m_have_precompute ? m_last : wrapped_levels[num_bins];

      if (!(first_level < last_level))
      {
        bin = UpperBound(wrapped_levels, num_output_levels, s) - 1;
        if (bin >= num_bins)
        {
          bin = -1;
        }
        return;
      }

      // Out-of-range samples map to bin -1 (and do not update the cache).
      if (s < first_level || !(s < last_level))
      {
        bin = -1;
        return;
      }

      // Identical first-guess ladder to the plain BinSelect above: on a cache
      // MISS we reproduce the same three-point piecewise-linear first guess so
      // miss-heavy inputs (high-entropy samples, and the irregular-level
      // fallback) converge exactly as the uncached path does. Only the hit fast
      // path and the cache writebacks differ.
      const auto delta = (s - first_level);
      int guess;
      if (m_have_precompute)
      {
        if (m_mid_bin > 0)
        {
          if (s < m_mid)
          {
            guess = static_cast<int>(static_cast<float>(delta) * m_inv_scale_lo);
          }
          else
          {
            const auto delta_hi = (s - m_mid);
            guess               = m_mid_bin + static_cast<int>(static_cast<float>(delta_hi) * m_inv_scale_hi);
          }
        }
        else
        {
          guess = static_cast<int>(static_cast<float>(delta) * m_inv_scale);
        }
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

      const LevelT lvl_lo = wrapped_levels[guess];
      const LevelT lvl_hi = wrapped_levels[guess + 1];

      if (!(s < lvl_lo) && (s < lvl_hi))
      {
        bin     = guess;
        mru.lo  = lvl_lo;
        mru.hi  = lvl_hi;
        mru.bin = guess;
        return;
      }

      // One-step linear correction.
      if (s < lvl_lo)
      {
        const int g2 = guess - 1;
        if (g2 >= 0)
        {
          const LevelT lvl2_lo = wrapped_levels[g2];
          if (!(s < lvl2_lo))
          {
            bin     = g2;
            mru.lo  = lvl2_lo;
            mru.hi  = lvl_lo; // lvl2_hi == lvl_lo (already loaded)
            mru.bin = g2;
            return;
          }
        }
      }
      else
      {
        const int g2 = guess + 1;
        if (g2 <= num_bins - 1)
        {
          const LevelT lvl2_hi = wrapped_levels[g2 + 1];
          if (s < lvl2_hi)
          {
            bin     = g2;
            mru.lo  = lvl_hi; // lvl2_lo == lvl_hi (already loaded)
            mru.hi  = lvl2_hi;
            mru.bin = g2;
            return;
          }
        }
      }

      // Fall back to binary search for irregular level distributions. This is
      // the rare path, so the two extra bracket loads needed to refresh the MRU
      // cache are amortized; they keep subsequent in-bracket samples on the
      // zero-load fast path.
      bin = UpperBound(wrapped_levels, num_output_levels, s) - 1;
      if (bin >= num_bins)
      {
        bin = -1;
        return;
      }
      if (bin >= 0)
      {
        mru.lo  = wrapped_levels[bin];
        mru.hi  = wrapped_levels[bin + 1];
        mru.bin = bin;
      }
    }
  };

  // Scales samples to evenly-spaced bins
  struct ScaleTransform
  {
    // Compile-time RANGE marker (false: this is the EVEN transform). See
    // SearchTransform::is_range_transform.
    static constexpr bool is_range_transform = false;

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
        // Double-precision reciprocal slope `bins / range`. For narrow integer
        // CommonT (<= 32-bit) the per-sample classify can compute the bin as a
        // single `(double)(sample - min) * recip_f64` multiply instead of the
        // 64-bit magic multiply-high + funnel-shift integer-divide sequence in
        // `range_divider.Divide`. This trades the integer divide (the top ALU
        // consumer on the SM-compute-bound EVEN high-bin classify) for one DMUL.
        // See `ComputeBin`'s `kUseFloat64Reciprocal` fast path for the
        // correctness argument (it reproduces the IEEE-754 double reference
        // `(double)(sample - min) * (bins / range)` bit-for-bit). Unused (and
        // uninitialised-safe via Init) for wider integer / custom CommonT, which
        // keep the exact integer divide.
        double recip_f64;
        // True iff bins == range, a common case (e.g. uniform even-spaced bins
        // where one bin == one sample value). When set, ComputeBin
        // short-circuits to `sample - min_level` and skips both the multiply by
        // `bins` and the divide-by-range.
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
      // Double-precision reciprocal slope for the narrow-integer fast path in
      // ComputeBin. Formed as `(double)num_bins / (double)(upper - lower)`, the
      // same way the IEEE-754 double reference forms its scale, so that path
      // reproduces the reference bin-for-bin. Harmless to compute for wide
      // CommonT (it is simply unused there); guards against div-by-zero
      // degenerate ranges.
      result.fraction.recip_f64 =
        (result.fraction.range != FractionStorageT{0})
          ? (static_cast<double>(result.fraction.bins) / static_cast<double>(result.fraction.range))
          : 0.0;
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
    // Whether the narrow-integer double-reciprocal fast path is exact for this
    // CommonT. It requires the non-negative difference `sample - min_level`
    // (which fits in the unsigned counterpart of CommonT) to be exactly
    // representable as a `double`, i.e. CommonT no wider than 32 bits: then the
    // bin reduces to a single `(double)diff * (bins/range)` multiply that
    // reproduces the IEEE-754 double reference `(double)(s - lo) * scale`
    // bit-for-bit. Wider integer CommonT (e.g. int64_t) keeps the exact
    // 64-bit magic integer divide, since a 64-bit difference is not exactly
    // representable in double and the floor could differ by one.
    static constexpr bool kUseFloat64Reciprocal = (sizeof(CommonT) <= 4);

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
      // Narrow-integer EVEN classify: compute the bin with one double FMA-class
      // multiply instead of the multi-instruction 64-bit magic integer divide.
      // `diff` (< 2^32 for <=32-bit CommonT) is exact in double, and
      // `recip_f64 == (double)bins / (double)range` is formed exactly as the
      // reference forms its scale, so `(int)((double)diff * recip_f64)` equals
      // the reference bin-for-bin. The integer divide is the top ALU consumer on
      // the SM-compute-bound high-bin EVEN classify; this collapses it to an
      // I2F + DMUL + F2I, cutting the per-sample instruction count substantially.
      if constexpr (kUseFloat64Reciprocal)
      {
        return static_cast<int>(static_cast<double>(diff) * scale.fraction.recip_f64);
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

    // Empty MRU cache type so direct-atomic kernels can declare a uniform
    // per-channel cache array regardless of transform; EVEN classify is pure
    // register arithmetic with no level loads, so there is nothing to cache.
    struct BracketCacheT
    {};

    // MRU-cache overload for call-site uniformity: ignores the cache and
    // forwards to the plain `BinSelect` (no level loads to elide for EVEN).
    template <CacheLoadModifier LOAD_MODIFIER>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(SampleT sample, int& bin, bool valid, BracketCacheT&) const
    {
      this->template BinSelect<LOAD_MODIFIER>(sample, bin, valid);
    }
  };

  // Pass-through bin transform operator
  struct PassThruTransform
  {
    // Compile-time RANGE marker (false: byte-sample pass-through is not RANGE).
    // See SearchTransform::is_range_transform.
    static constexpr bool is_range_transform = false;

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

    // Empty MRU cache type + ignoring overload for call-site uniformity with
    // SearchTransform (byte-sample pass-through has no level loads to elide).
    struct BracketCacheT
    {};

    template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(_SampleT sample, int& bin, bool valid, BracketCacheT&) const
    {
      this->template BinSelect<LOAD_MODIFIER>(sample, bin, valid);
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
// Request a minimum of 2 resident blocks/SM for the WIDE (>=512-thread) launch
// shapes only. On the wide multi-channel policies the SMEM-privatized sweep is
// register-limited to 1 block/SM, and the sweep is bound by shared-memory
// atomicAdd contention; admitting a 2nd resident CTA doubles the warps available
// to hide that latency.
//
// The hint must NOT touch the narrow single-channel policies (the 384-thread
// fallback used for both EVEN and RANGE). There the register cap is loose, so the
// bound changes no occupancy but does perturb ptxas codegen, and the
// latency-bound SearchTransform path is hurt by the reschedule rather than helped
// (it is classify-latency bound, not occupancy bound). Since EVEN and RANGE share
// the same 384-thread fallback policy struct they cannot be split on a policy
// field, so gate the hint on threads_per_block >= 512: it applies to the wide
// multi-channel policies and falls back to minBlocks=0 for the narrow fallback.
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block),
                  (current_policy<PolicySelector>().threads_per_block >= 512) ? 2 : 0)
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSmemPrivatizedKernel(
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
  // register pressure and cost occupancy. Keep the lean path: use the
  // grid-constant decode ops directly (read from constant memory) without the
  // device-side precompute. The other (higher-bin) sweep kernels, where the
  // classify dominates, still precompute.
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
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSmemPrivatizedDeviceInitKernel(
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
//! This is the host-init variant: it mirrors `DeviceHistogramSmemPrivatizedKernel`'s
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
//! Unified GMEM-privatized histogram sweep kernel — the design doc's
//! `GmemPrivatizedKernel<NoCache, smem_split>`, merging what used to be two
//! separate kernels:
//!
//!   * `smem_split == 0`  → pure GMEM privatization. Every block privatizes the
//!     WHOLE histogram in a per-block GMEM slab; a grid.sync + Phase-4
//!     `gather_privatized_slab` merges. (Was `DeviceHistogramGmemPrivGatherKernel`.)
//!   * `smem_split  > 0`  → hybrid SMEM+GMEM. Bins `[0, smem_split)` accumulate in
//!     per-block dynamic SMEM, the tail `[smem_split, smem_split+secondary)` in a
//!     per-block GMEM secondary slab; both flush to staging slabs, then a grid.sync
//!     + two `gather_privatized_slab` calls merge primary and secondary regions.
//!     (Was `DeviceHistogramHybridSinglePassKernel`.)
//!
//! The two paths share one name, the `UseDynamicSmemHistogram` AgentHistogram, and
//! the gather helper; `smem_split` (a runtime grid-constant) selects the body. The
//! `HybridSplit` non-type template parameter mirrors `smem_split>0` at compile time
//! so the dead path is pruned and the launch bound can pin the hybrid member to 1
//! block/SM (its ~192 KB dyn-SMEM allocation forbids a 2nd resident CTA) while the
//! pure-gather member keeps the looser `minBlocks=0` hint. `smem_split` must be 0
//! iff `HybridSplit==false`.
//!
//! Must be launched cooperatively (`cudaLaunchCooperativeKernel`): all blocks must
//! be co-resident for `grid_group::sync()`. The dispatch layer verifies the grid
//! fits before selecting this kernel.
//!
//! Args used by BOTH paths: `d_samples`, `num_output_bins_wrapper`,
//! `d_output_histograms_wrapper`, decode ops, input geometry, `tile_queue`.
//! Pure-gather (`HybridSplit==false`) additionally uses
//! `num_privatized_bins_wrapper`, `d_privatized_histograms_wrapper`,
//! `max_num_output_bins`. Hybrid (`HybridSplit==true`) additionally uses
//! `d_secondary_histograms_wrapper`, `smem_split`, `secondary_size` (and treats
//! `d_privatized_histograms_wrapper` as the PRIMARY staging slab).
//! Unified GMEM-privatized histogram sweep kernel — the design doc's
//! `GmemPrivatizedKernel<NoCache, smem_split>`. ONE kernel; the `HybridSplit`
//! non-type template parameter (mirroring `smem_split > 0`) selects the body via
//! `if constexpr`, so each instantiation contains only its own path:
//!
//!   * `HybridSplit == false` (`smem_split == 0`)  → pure GMEM privatization. Every
//!     block privatizes the WHOLE histogram in a per-block GMEM slab; grid.sync +
//!     the shared `gather_privatized_slab` helper merges. (Was the standalone
//!     gather kernel.)
//!   * `HybridSplit == true`  (`smem_split  > 0`)  → hybrid SMEM+GMEM. Bins
//!     `[0, smem_split)` accumulate in per-block dynamic SMEM, the tail
//!     `[smem_split, smem_split + secondary_size)` in a per-block GMEM secondary
//!     slab; both flush to staging slabs, then grid.sync + a fused primary+secondary
//!     reduce merges into the output. (Was the standalone hybrid kernel.) The fused
//!     single-loop reduce is kept verbatim here rather than calling
//!     `gather_privatized_slab` twice: the two-call form measured a ~5% regression
//!     on this hot path (B200, even/65536/concentrated), so it stays open-coded.
//!
//! The launch bound pins the hybrid instantiation to 1 block/SM (its ~192 KB
//! dyn-SMEM allocation forbids a 2nd resident CTA) while the pure-gather
//! instantiation keeps the looser `minBlocks=0` hint. Each `HybridSplit` value is
//! a distinct `__global__`, so the dead branch is pruned and never costs the other.
//!
//! Must be launched cooperatively (`cudaLaunchCooperativeKernel`): all blocks must
//! be co-resident for `grid_group::sync()`. The dispatch layer verifies the grid
//! fits before selecting this kernel.
//!
//! Args used by BOTH paths: `d_samples`, `num_output_bins_wrapper`,
//! `d_output_histograms_wrapper`, decode ops, input geometry, `tile_queue`.
//! Pure-gather additionally uses `num_privatized_bins_wrapper`,
//! `d_privatized_histograms_wrapper`, `max_num_output_bins`. Hybrid additionally
//! uses `d_secondary_histograms_wrapper`, `smem_split`, `secondary_size` (and
//! treats `num_output_bins_wrapper` as the split-sized primary bin count and
//! `d_privatized_histograms_wrapper` as the PRIMARY staging slab).
template <typename PolicySelector,
          int PrivatizedSmemBins,
          int NumChannels,
          int NumActiveChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT,
          bool HybridSplit = false>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block), HybridSplit ? 1 : 0)
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramGmemPrivatizedKernel(
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
    _CCCL_GRID_CONSTANT const int max_num_output_bins,
    // Hybrid-only (HybridSplit==true): per-block SECONDARY (tail) staging slab,
    // the SMEM split point, and the secondary region size. Unused for the
    // pure-gather instantiation (defaulted so its launch arg array is unchanged).
    ::cuda::std::array<CounterT*, NumActiveChannels> d_secondary_histograms_wrapper = {},
    _CCCL_GRID_CONSTANT const int smem_split                                        = 0,
    _CCCL_GRID_CONSTANT const int secondary_size                                    = 0)
{
  static constexpr histogram_policy hp = current_policy<PolicySelector>();

  namespace cg = ::cooperative_groups;
  cg::grid_group grid = cg::this_grid();

  const unsigned int blocks_per_grid = gridDim.x * gridDim.y * gridDim.z;
  const unsigned int block_id        = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const unsigned int tid_global      = block_id * blockDim.x + threadIdx.x;
  const unsigned int total_threads   = blocks_per_grid * blockDim.x;

  using AgentHistogramPolicyT =
    AgentHistogramPolicy<hp.threads_per_block,
                         hp.pixels_per_thread,
                         hp.load_algorithm,
                         hp.load_modifier,
                         hp.rle_compress,
                         hp.mem_preference,
                         hp.work_stealing,
                         hp.vec_size>;

  if constexpr (!HybridSplit)
  {
    // =================== smem_split == 0 : pure-gather path ===================
    (void) d_secondary_histograms_wrapper;
    (void) smem_split;
    (void) secondary_size;

    // Phase 1: zero the output histograms + reset the work-stealing drain counter.
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
      GridQueue<int> queue = tile_queue;
      queue.ResetDrain();
    }

    // Phase 2: make the zeros + drain reset visible to every block.
    grid.sync();

    // Phase 3: AgentHistogram sweep. For the GMEM-privatized path each block's
    // privatized histogram lives at `d_privatized_histograms[ch] + block_id * num_privatized_bins[ch]`.
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

    // Save the per-channel all-blocks privatized base before the agent ctor
    // offsets it by block_id, for the gather merge.
    CounterT* d_privatized_base[NumActiveChannels];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      d_privatized_base[ch] = d_privatized_histograms_wrapper[ch];
    }

    OutputDecodeOpT output_decode_op[NumActiveChannels];
    PrivatizedDecodeOpT privatized_decode_op[NumActiveChannels];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      output_decode_op[ch]     = output_decode_op_wrapper[ch];
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

      agent.InitBinCounters();
      agent.ConsumeTiles(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);

      if constexpr (PrivatizedSmemBins != 0)
      {
        agent.StoreOutput();
      }
    }

    if constexpr (PrivatizedSmemBins == 0)
    {
      // Grid-wide barrier so every block's ConsumeTiles writes are visible, then
      // Phase 4 gather-merge via the shared helper (turns num_blocks * num_bins
      // contended atomics into plain reads + one write per bin).
      grid.sync();
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ch = 0; ch < NumActiveChannels; ++ch)
      {
        const unsigned int num_bins_u = static_cast<unsigned int>(num_privatized_bins_wrapper[ch]);
        gather_privatized_slab<CounterT>(
          d_output_histograms_wrapper[ch],
          /*out_offset=*/0u,
          d_privatized_base[ch],
          /*slab_stride=*/num_bins_u,
          /*count=*/num_bins_u,
          blocks_per_grid,
          tid_global,
          total_threads);
      }
    }

    _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
  }
  else
  {
    // ===================== smem_split > 0 : hybrid path =====================
    // Here `num_output_bins_wrapper` carries the split-sized primary bin count,
    // `d_privatized_histograms_wrapper` is the PRIMARY staging slab, and
    // `d_secondary_histograms_wrapper` is the secondary (tail) staging slab.
    (void) num_privatized_bins_wrapper;
    (void) max_num_output_bins;

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

    CounterT* d_primary_base[NumActiveChannels];
    CounterT* d_secondary_base[NumActiveChannels];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      d_primary_base[ch]   = d_privatized_histograms_wrapper[ch];
      d_secondary_base[ch] = d_secondary_histograms_wrapper[ch];
    }

    // Drain-reset + grid.sync only when work stealing is enabled.
    if constexpr (hp.work_stealing)
    {
      if (tid_global == 0)
      {
        GridQueue<int> queue = tile_queue;
        queue.ResetDrain();
      }
      grid.sync();
    }

    OutputDecodeOpT output_decode_op[NumActiveChannels];
    PrivatizedDecodeOpT privatized_decode_op[NumActiveChannels];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      output_decode_op[ch]     = output_decode_op_wrapper[ch];
      privatized_decode_op[ch] = privatized_decode_op_wrapper[ch];
      output_decode_op[ch].PrecomputeOnDevice();
      privatized_decode_op[ch].PrecomputeOnDevice();
    }

    // Single sweep with hybrid accumulation.
    {
      AgentHistogramT agent(
        temp_storage,
        d_samples,
        num_output_bins_wrapper.data(),
        num_output_bins_wrapper.data(),
        d_output_histograms_wrapper.data(),
        d_privatized_histograms_wrapper.data(),
        d_secondary_histograms_wrapper.data(),
        output_decode_op,
        privatized_decode_op,
        dyn_smem_histograms,
        smem_split,
        secondary_size);

      agent.InitBinCountersHybrid();
      agent.ConsumeTilesHybrid(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);
      agent.StoreHybridSmemToStagingSlab();
    }

    // grid-wide sync: all blocks' primary+secondary staging-slab writes visible.
    grid.sync();

    // Atomic-free reduce: primary slab -> output[0..split), secondary slab ->
    // output[split..split+secondary). Fused single-loop form (NOT the two-call
    // gather_privatized_slab helper, which measured ~5% slower on this hot path).
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      CounterT* d_out                             = d_output_histograms_wrapper[ch];
      const CounterT* __restrict__ primary_base   = d_primary_base[ch];
      const CounterT* __restrict__ secondary_base = d_secondary_base[ch];
      const unsigned int split_u                  = static_cast<unsigned int>(smem_split);
      const unsigned int sec_u                    = static_cast<unsigned int>(secondary_size);
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
}

//! Per-block SMEM cache update operators for the direct-atomic histogram kernel.
//! `DeviceHistogramDirectKernel` is templated on one of these and calls
//! `ProbeOp::apply(...)` for each (bin, contribution); the cache policy is thus a
//! pluggable op rather than a branch in the kernel. Each operator receives this
//! channel's key array, this warp's replica count array, this channel's GMEM
//! output, the bin and its (warp-coalesced) increment, and the hash parameters.
//! A cache hit/claim bumps a block-scope count; a miss spills one device-scope
//! atomic to `output[bin]`.

//! Spill functors: where a cache MISS (or flush) deposits its contribution. The
//! probe ops below are templated on one of these so the cache strategy (cuckoo /
//! single-probe) is orthogonal to the spill target. Two instantiations:
//!
//!   output_atomic_spill  -- device-scope atomicAdd straight to the SHARED output
//!     histogram. This is the "direct atomic" commit; a hot-bin spill contends
//!     across every block. Used by DeviceHistogramDirectKernel.
//!
//!   private_block_spill  -- block-scope atomicAdd_block into THIS block's PRIVATE
//!     GMEM histogram (a per-block slab). Uncontended across blocks; a later
//!     grid-sync + atomic-free gather merges the slabs into the output. Used by
//!     the privatized-spill variant of the same kernel.
//!
//! `target` is the per-channel base the kernel hands the probe (the shared output
//! for output_atomic_spill, this block's slab base for private_block_spill), so the
//! probe body is identical for both; only the atomic scope differs.
struct output_atomic_spill
{
  template <typename CounterT>
  static _CCCL_DEVICE _CCCL_FORCEINLINE void spill(CounterT* target, int bin, CounterT contribution)
  {
    atomicAdd(&target[bin], contribution);
  }
};

struct private_block_spill
{
  template <typename CounterT>
  static _CCCL_DEVICE _CCCL_FORCEINLINE void spill(CounterT* target, int bin, CounterT contribution)
  {
    atomicAdd_block(&target[bin], contribution);
  }
};

//! 2-hash cuckoo cache update: on a primary-slot collision, try a secondary slot
//! before spilling. `DisableSecondProbe` compiles the secondary probe out for the
//! very-high-bin tier (hit rate ~0, so the second key read is pure waste).
//!
//! `SpillOp` (defaulted to output_atomic_spill, keeping every existing caller
//! byte-identical) selects where a miss deposits its contribution: the shared
//! output (device-scope) or this block's private slab (block-scope). `output` is
//! whichever per-channel base the kernel passes for that spill policy.
template <bool DisableSecondProbe = false>
struct cuckoo_cache_probe
{
  template <typename CounterT, typename SpillOp = output_atomic_spill>
  static _CCCL_DEVICE _CCCL_FORCEINLINE void
  apply(int* keys, CounterT* counts, CounterT* output, int bin, CounterT contribution, int cache_mask, int cache_slot_log2)
  {
    const unsigned int hash1 = static_cast<unsigned int>(bin) * 2654435761u;
    const int slot1          = cache_slot_from_hash(hash1, cache_mask, cache_slot_log2);
    const int existing_key1  = keys[slot1];
    if (existing_key1 == bin)
    {
      // Primary hit: bump cache count.
      atomicAdd_block(&counts[slot1], contribution);
    }
    else if (existing_key1 == -1)
    {
      // Primary slot empty: try to claim it via CAS.
      const int prev = atomicCAS(&keys[slot1], -1, bin);
      if (prev == -1 || prev == bin)
      {
        atomicAdd_block(&counts[slot1], contribution);
      }
      else if constexpr (DisableSecondProbe)
      {
        // High-bin tier: no secondary probe -- spill.
        SpillOp::spill(output, bin, contribution);
      }
      else
      {
        // Lost the race. Try secondary slot.
        const unsigned int hash2 = static_cast<unsigned int>(bin) * 2246822519u;
        const int slot2          = static_cast<int>(hash2 & cache_mask);
        const int existing_key2  = keys[slot2];
        if (existing_key2 == bin)
        {
          atomicAdd_block(&counts[slot2], contribution);
        }
        else if (existing_key2 == -1)
        {
          const int prev2 = atomicCAS(&keys[slot2], -1, bin);
          if (prev2 == -1 || prev2 == bin)
          {
            atomicAdd_block(&counts[slot2], contribution);
          }
          else
          {
            SpillOp::spill(output, bin, contribution);
          }
        }
        else
        {
          SpillOp::spill(output, bin, contribution);
        }
      }
    }
    else if constexpr (DisableSecondProbe)
    {
      // High-bin tier: primary occupied by a different bin -> spill without a
      // secondary probe.
      SpillOp::spill(output, bin, contribution);
    }
    else
    {
      // Primary occupied by a different bin: try the secondary slot.
      const unsigned int hash2 = static_cast<unsigned int>(bin) * 2246822519u;
      const int slot2          = cache_slot_from_hash(hash2, cache_mask, cache_slot_log2);
      const int existing_key2  = keys[slot2];
      if (existing_key2 == bin)
      {
        atomicAdd_block(&counts[slot2], contribution);
      }
      else if (existing_key2 == -1)
      {
        const int prev2 = atomicCAS(&keys[slot2], -1, bin);
        if (prev2 == -1 || prev2 == bin)
        {
          atomicAdd_block(&counts[slot2], contribution);
        }
        else
        {
          SpillOp::spill(output, bin, contribution);
        }
      }
      else
      {
        SpillOp::spill(output, bin, contribution);
      }
    }
  }
};

//! Single-probe direct-mapped cache update (or `CUB_HISTO_SINGLE_PROBE_WAYS`-way
//! set-associative): leaner critical section than cuckoo; wins at very high bin
//! counts where the cache holds almost nothing.
struct single_probe_cache
{
  template <typename CounterT, typename SpillOp = output_atomic_spill>
  static _CCCL_DEVICE _CCCL_FORCEINLINE void
  apply(int* keys, CounterT* counts, CounterT* output, int bin, CounterT contribution, int cache_mask, int cache_slot_log2)
  {
    const unsigned int hash = static_cast<unsigned int>(bin) * 2654435761u;
    if constexpr (CUB_HISTO_SINGLE_PROBE_WAYS == 1)
    {
      // Single direct-mapped probe: hash bin to one slot.
      const int slot         = cache_slot_from_hash(hash, cache_mask, cache_slot_log2);
      const int existing_key = keys[slot];
      if (existing_key == bin)
      {
        // Hit: bump cache count (block-scope atomic, ~10x cheaper).
        atomicAdd_block(&counts[slot], contribution);
      }
      else if (existing_key == -1)
      {
        // Empty: try to claim via CAS.
        const int prev = atomicCAS(&keys[slot], -1, bin);
        if (prev == -1 || prev == bin)
        {
          atomicAdd_block(&counts[slot], contribution);
        }
        else
        {
          // Lost the claim race to a different bin: spill.
          SpillOp::spill(output, bin, contribution);
        }
      }
      else
      {
        // Collision (slot owned by another bin): single spill.
        SpillOp::spill(output, bin, contribution);
      }
    }
    else
    {
      // WAYS-way set-associative probe. Hash `bin` to a SET of
      // `CUB_HISTO_SINGLE_PROBE_WAYS` ADJACENT slots [base, base+WAYS); the set
      // budget is `cache_slots_per_channel / WAYS` sets. Probe the ways in two
      // passes so a colliding bin gets a fallback slot before spilling:
      //   pass 1: any way already holding `bin` -> hit (bump its count);
      //   pass 2: first EMPTY way -> CAS-claim it;
      //   neither: every way is owned by a different live bin -> GMEM spill.
      // Adjacent ways keep the WAYS key reads in one/two SMEM banks (vs the
      // cuckoo cache's two independent random-hash slots).
      constexpr int kWaysLog2 = (CUB_HISTO_SINGLE_PROBE_WAYS == 4) ? 2 : 1;
      const int set           = cache_slot_from_hash(hash, cache_mask >> kWaysLog2, cache_slot_log2 - kWaysLog2);
      const int base          = set << kWaysLog2;
      bool done               = false;
      // Pass 1: existing-key hit on any way.
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int w = 0; w < CUB_HISTO_SINGLE_PROBE_WAYS; ++w)
      {
        if (!done && keys[base + w] == bin)
        {
          atomicAdd_block(&counts[base + w], contribution);
          done = true;
        }
      }
      // Pass 2: claim the first empty way (CAS); a lost race that resolves to
      // `bin` is also a hit, a lost race to another bin moves to the next way.
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int w = 0; w < CUB_HISTO_SINGLE_PROBE_WAYS; ++w)
      {
        if (!done && keys[base + w] == -1)
        {
          const int prev = atomicCAS(&keys[base + w], -1, bin);
          if (prev == -1 || prev == bin)
          {
            atomicAdd_block(&counts[base + w], contribution);
            done = true;
          }
        }
      }
      if (!done)
      {
        // All ways occupied by other live bins (or every empty way lost its CAS
        // race): single spill. Correctness: contribution added once.
        SpillOp::spill(output, bin, contribution);
      }
    }
  }
};

//! No-cache combiner: the SMEM key-probe cache is disabled entirely. Every
//! (warp-coalesced) bin spills straight through `SpillOp` -- to the shared output
//! (`output_atomic_spill`) for the `Direct` kernel's `NoCache` combiner, or to
//! this block's private slab (`private_block_spill`) for the `GmemPrivatized`
//! kernel's `NoCache` combiner. The `keys` / `counts` / hash arguments are
//! ignored, so a `NoCache` kernel that passes a zero-sized cache allocates no
//! SMEM cache. This is the third member of `Combiner ∈ {NoCache, Cuckoo,
//! SingleProbe}` from the design doc: it isolates the combiner's contribution
//! (no on-chip combining) and is the building block of `direct_nocache` and of
//! the `GmemPrivatized<NoCache>` family (= `gmem_priv_gather` / `hybrid`).
struct no_cache_probe
{
  template <typename CounterT, typename SpillOp = output_atomic_spill>
  static _CCCL_DEVICE _CCCL_FORCEINLINE void
  apply(int* keys, CounterT* counts, CounterT* output, int bin, CounterT contribution, int cache_mask, int cache_slot_log2)
  {
    (void) keys;
    (void) counts;
    (void) cache_mask;
    (void) cache_slot_log2;
    SpillOp::spill(output, bin, contribution);
  }
};

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
//! Unlike `DeviceHistogramGmemPrivatizedKernel`, this kernel does not
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
          typename OffsetT,
          typename ProbeOp = cuckoo_cache_probe<>,
          typename SpillOp = output_atomic_spill>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().direct_atomic_threads()))
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramDirectKernel(
    _CCCL_GRID_CONSTANT const SampleIteratorT d_samples,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<int, NumActiveChannels> num_output_bins_wrapper,
    ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms_wrapper,
    _CCCL_GRID_CONSTANT const ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op_wrapper,
    _CCCL_GRID_CONSTANT const OffsetT num_row_pixels,
    _CCCL_GRID_CONSTANT const OffsetT num_rows,
    _CCCL_GRID_CONSTANT const OffsetT row_stride_samples,
    _CCCL_GRID_CONSTANT const int cache_slots_per_channel,
    // Private-spill variant only (SpillOp == private_block_spill): per-channel
    // base of the ALL-BLOCKS private histogram slab; this block owns the slice
    // [block_id * num_bins, (block_id+1) * num_bins). Unused (and may be empty
    // null pointers) for the default output-spill variant.
    ::cuda::std::array<CounterT*, NumActiveChannels> d_private_histograms_wrapper = {})
{
  namespace cg = ::cooperative_groups;

  cg::grid_group grid = cg::this_grid();

  // Compile-time spill policy. With private_block_spill, cache misses (and the
  // post-sweep flush) deposit block-scope into THIS block's private GMEM slab
  // rather than device-scope into the shared output; a Phase-4 atomic-free
  // gather then sums the per-block slabs into the output. Boundedness note: a
  // cache miss can hit ANY bin, so each block's slab must be full-size
  // (num_bins), giving a num_blocks * num_bins footprint (see design doc).
  constexpr bool kPrivateSpill = ::cuda::std::is_same_v<SpillOp, private_block_spill>;

  // ---------------------------------------------------------------------
  // Phase 1: zero the spill destination via a grid-wide stride loop. For the
  // output-spill variant that is the shared output histogram; for the
  // private-spill variant it is this grid's per-block private slabs (the
  // output is written, not accumulated, by the Phase-4 gather, so it needs no
  // pre-zero there).
  // ---------------------------------------------------------------------
  const unsigned int blocks_per_grid = gridDim.x * gridDim.y * gridDim.z;
  const unsigned int block_id        = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const unsigned int tid_global      = block_id * blockDim.x + threadIdx.x;
  const unsigned int total_threads   = blocks_per_grid * blockDim.x;

  // Per-channel spill target for THIS block: the shared output, or this block's
  // private slab slice (base + block_id * num_bins). Hoisted once; the hot-path
  // probe and the flush both spill through it.
  CounterT* spill_target[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    if constexpr (kPrivateSpill)
    {
      spill_target[ch] =
        d_private_histograms_wrapper[ch] + static_cast<size_t>(block_id) * num_output_bins_wrapper[ch];
    }
    else
    {
      spill_target[ch] = d_output_histograms_wrapper[ch];
    }
  }

  if constexpr (kPrivateSpill)
  {
    // Zero every block's own private slab (num_blocks * num_bins words total,
    // grid-strided). The shared output is written by the gather, not accumulated.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      const size_t slab_words = static_cast<size_t>(blocks_per_grid) * num_output_bins_wrapper[ch];
      for (size_t i = tid_global; i < slab_words; i += total_threads)
      {
        d_private_histograms_wrapper[ch][i] = 0;
      }
    }
  }
  else
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      const int channel_bins = num_output_bins_wrapper[ch];
      for (unsigned int bin = tid_global; bin < static_cast<unsigned int>(channel_bins); bin += total_threads)
      {
        d_output_histograms_wrapper[ch][bin] = 0;
      }
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
  // path: with one atomic per iteration the kernel is bottlenecked on
  // L1TEX scoreboard dependencies (the atomic-latency stall dominates CPI).
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
  // SMEM size). On the high-bin path the kernel is bound by scattered
  // GMEM-atomic spills, and the spill rate is fixed by how many distinct hot
  // bins fit in cache -- neither packing the slot nor changing the probe count
  // moves it. Growing the slot count is the one lever that raises the hit rate
  // and pulls spills off the contended global histogram onto cheap block-scope
  // SMEM atomics.
  //
  // `cache_slots_per_channel` is a runtime power of two (mask = slots-1).
  // The extern __shared__ region holds, per channel, a key array (int)
  // followed by a count array (CounterT): keys for all channels first,
  // then counts for all channels. Two multiplicative-hash probes are
  // retained because the secondary slot raises the hit rate on skewed
  // distributions at moderate bin counts. Keys are write-once / immutable
  // after a CAS claim, so the cache is race-free (no hit-vs-evict window).
  //
  // SMEM-atomic SERIALIZATION relief: the warp-leader's `atomicAdd_block` into a
  // hot slot's count serialises across EVERY warp of the block that owns that
  // bin. We split the count array into `kCountReplicas` independent replicas and
  // route warp `w` to replica `w % kCountReplicas`, so at most
  // ~ceil(num_warps/kCountReplicas) warps ever contend a given (replica, slot)
  // word -- an R-fold reduction in the SMEM-atomic serialization on hot bins.
  // Keys stay SHARED (one slot->bin assignment for the whole block); only the
  // COUNT is replicated. Updates stay `atomicAdd_block` so warps sharing a
  // replica are race-free by construction. At flush we sum the R replicas of
  // each claimed slot and issue one GMEM atomic. Layout in dynamic SMEM: keys
  // for all channels first (`NumActiveChannels * slots` ints), then replicated
  // counts (`NumActiveChannels * kCountReplicas * slots` CounterT). Dispatch
  // sizes the dynamic SMEM with the same replica factor.
  //
  // R is gated PER NUM_ACTIVE_CHANNELS (compile-time, via cache_tuning::replicas
  // so the kernel and the host sizer share one definition): multi-channel uses
  // R>1 because several channels share one block's cache and hot-slot atomic
  // serialization dominates -- and at the high bin counts that reach this kernel
  // the cache hit rate is near zero, so slot capacity is dead weight that is
  // freely traded for more independent count replicas. Single-channel uses R=1:
  // it is occupancy/key-read-bound rather than count-serialization-bound, so a
  // larger per-slot footprint would only halve the grid and regress it. At R=1
  // the replica index is always 0 and the count layout is byte-for-byte
  // identical to an unreplicated cache.
  constexpr int kCountReplicas = cache_tuning::replicas(NumActiveChannels);
  constexpr bool kWarpCoalesce = current_policy<PolicySelector>().warp_coalesce;
  const int cache_mask  = cache_slots_per_channel - 1;
  // log2(slots) for the high-bits hash mode; slots is a power of two so this is
  // popcount(mask) == 32 - clz(mask). Computed once; the hot path is clz-free.
  const int cache_slot_log2 = 32 - __clz(cache_mask);
  const size_t slots_sz = static_cast<size_t>(cache_slots_per_channel);
  const int replica     = static_cast<int>((threadIdx.x >> 5)) % kCountReplicas;
  extern __shared__ unsigned char s_bin_cache_raw[];
  int* const s_cache_keys_base       = reinterpret_cast<int*>(s_bin_cache_raw);
  CounterT* const s_cache_counts_base = reinterpret_cast<CounterT*>(
    s_bin_cache_raw + static_cast<size_t>(NumActiveChannels) * slots_sz * sizeof(int));
  // Per-channel shared key base and this warp's replica count base.
  int* s_cache_keys[NumActiveChannels];
  CounterT* s_cache_counts[NumActiveChannels];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    s_cache_keys[ch]   = s_cache_keys_base + static_cast<size_t>(ch) * slots_sz;
    s_cache_counts[ch] =
      s_cache_counts_base + (static_cast<size_t>(ch) * kCountReplicas + replica) * slots_sz;
  }

  // Initialize cache: keys = -1 (empty sentinel), all replica counts = 0.
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ch = 0; ch < NumActiveChannels; ++ch)
  {
    for (int slot = threadIdx.x; slot < cache_slots_per_channel; slot += blockDim.x)
    {
      s_cache_keys[ch][slot] = -1;
    }
  }
  {
    const size_t total_count_words = static_cast<size_t>(NumActiveChannels) * kCountReplicas * slots_sz;
    for (size_t i = threadIdx.x; i < total_count_words; i += blockDim.x)
    {
      s_cache_counts_base[i] = CounterT{0};
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

  // Per-thread, per-channel MRU bin-bracket cache (empty for non-RANGE
  // transforms). Exploits consecutive-sample temporal locality so an in-bracket
  // RANGE sample classifies with no level-array loads; see BracketCacheT.
  //
  // GATED TO SINGLE-CHANNEL: this direct-atomic kernel runs at its register /
  // occupancy ceiling. One per-thread bracket cache (single-channel, mru[1])
  // fits in registers and speeds the RANGE classify. But NumActiveChannels
  // caches (multi-channel, mru[NumActiveChannels]) do NOT fit: holding the
  // occupancy tier forces the cache to SPILL to local memory, and the per-sample
  // LMEM round-trips cost more than the L1-cached level loads they would replace.
  // So multi-channel keeps the plain (cacheless) BinSelect. `is_range_transform`
  // guards EVEN (no level loads to cache); NumActiveChannels == 1 guards the
  // spill.
  constexpr bool kUseMruCache = (NumActiveChannels == 1) && PrivatizedDecodeOpT::is_range_transform;
  typename PrivatizedDecodeOpT::BracketCacheT mru[NumActiveChannels];

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

    // Sample type produced by the input iterator (used to stage prefetched
    // loads in registers across the unrolled chunk).
    using SampleValueT = it_value_t<SampleIteratorT>;

    // Single-source the leader's warp-coalesce + cache update so the pipelined
    // (single-channel) and interleaved (multi-channel) loops below share
    // identical probe logic. The cache update itself is the pluggable `ProbeOp`
    // (cuckoo_cache_probe<> or single_probe_cache), called per (bin,
    // contribution); the kernel holds no probe-specific branches.
    auto probe = [&](int ch, int bin) {
      // The cache update for one (bin, contribution). Factored so the
      // warp_coalesce knob can drive it once per peer group (leader) or once per
      // valid lane without duplicating the body.
      auto update = [&](int bin, CounterT contribution) {
        ProbeOp::template apply<CounterT, SpillOp>(
          s_cache_keys[ch], s_cache_counts[ch], spill_target[ch], bin, contribution, cache_mask, cache_slot_log2);
      };

      // Warp-coalesce same-bin lanes into one cache update (gated by the
      // warp_coalesce policy knob; on by default). When on, the lowest matching
      // lane applies the popcount-summed contribution; when off, every valid
      // lane applies 1. The inline form below keeps the on-path codegen
      // byte-identical to the hand-tuned kernel (this kernel is register-pinned).
      if constexpr (kWarpCoalesce)
      {
        const unsigned int peers = __match_any_sync(0xffffffffu, static_cast<unsigned int>(bin));
        const int leader         = __ffs(static_cast<int>(peers)) - 1;
        if (bin >= 0 && static_cast<int>(lane_id) == leader)
        {
          update(bin, static_cast<CounterT>(__popc(peers)));
        }
      }
      else if (bin >= 0)
      {
        update(bin, CounterT{1});
      }
    };

    if constexpr (NumActiveChannels == 1)
    {
      // Single-channel: SOFTWARE-PIPELINE the unrolled chunk. Issue all
      // `unroll` independent global loads up front (Phase A) so the long
      // global-load latency overlaps instead of serialising a
      // load -> classify -> match -> SMEM-read -> atomic chain per pixel;
      // then classify (Phase B); then coalesce/probe/atomic (Phase C). The
      // staged register array is only `unroll` samples deep here
      // (NumActiveChannels==1), so the extra register footprint is small and
      // occupancy is preserved. Primary latency-hiding lever for this
      // issue-bound / eligible-warp-scarce kernel.
      for (OffsetT it = 0; it < chunk_iters_max; ++it)
      {
        const OffsetT pixel = start + it * chunk;

        // Phase A: issue all of this chunk's loads up front.
        SampleValueT staged[unroll];
        bool valid[unroll];
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int u = 0; u < unroll; ++u)
        {
          const OffsetT this_pixel = pixel + u * step;
          valid[u]                 = this_pixel < total_pixels;
          const OffsetT pix_off    = valid[u] ? (this_pixel * NumChannels) : OffsetT{0};
          staged[u]                = d_samples[pix_off];
        }

        // Phase B: classify all staged samples into bins.
        int bins[unroll];
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int u = 0; u < unroll; ++u)
        {
          int bin = -1;
          if (valid[u])
          {
            if constexpr (kUseMruCache)
            {
              decode_op[0].template BinSelect<LOAD_DEFAULT>(staged[u], bin, true, mru[0]);
            }
            else
            {
              decode_op[0].template BinSelect<LOAD_DEFAULT>(staged[u], bin, true);
            }
            const int num_bins = num_output_bins_wrapper[0];
            if (bin >= num_bins)
            {
              bin = -1;
            }
          }
          bins[u] = bin;
        }

        // Phase C: warp-coalesce, probe the SMEM cache, atomic-update.
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int u = 0; u < unroll; ++u)
        {
          probe(0, bins[u]);
        }
      }
    }
    else
    {
      // Multi-channel: VECTORIZE the per-pixel load. The scalar form issues
      // `NumActiveChannels` separate global loads per pixel; consecutive
      // threads read consecutive pixels, so with `NumChannels` samples per
      // pixel the access leaves a gap every `NumChannels`-th sample (the
      // unhistogrammed alpha lane of an RGBA image) and under-fills the
      // global-load transactions. Instead we issue ONE wide
      // `CubVector<SampleT, NumChannels>` (e.g. `int4`) load per pixel that
      // reads the whole pixel contiguously, then classify the active lanes
      // straight out of registers -- the warp's loads are now perfectly
      // packed, cutting the multi-channel global-load transaction count by up
      // to `NumChannels`x. The vector path requires a native, suitably-aligned
      // sample pointer; otherwise we fall back to the per-channel scalar loop.
      using PixelT            = typename CubVector<SampleValueT, NumChannels>::Type;
      const auto* native_base = SampleNativePointer<SampleValueT>(d_samples);
      const bool vectorizable = (native_base != nullptr)
                             && ((reinterpret_cast<size_t>(native_base) & (alignof(PixelT) - 1)) == 0);
      if (vectorizable)
      {
        const PixelT* const pixels = reinterpret_cast<const PixelT*>(native_base);
        for (OffsetT it = 0; it < chunk_iters_max; ++it)
        {
          const OffsetT pixel = start + it * chunk;
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int u = 0; u < unroll; ++u)
          {
            const OffsetT this_pixel = pixel + u * step;
            const bool valid_pixel   = this_pixel < total_pixels;
            // Load the whole pixel (all NumChannels samples) in one transaction;
            // index pixel 0 for past-the-end lanes (discarded via bin = -1).
            const PixelT pix = pixels[valid_pixel ? this_pixel : OffsetT{0}];
            const SampleValueT* const lanes = reinterpret_cast<const SampleValueT*>(&pix);

            // CHANNEL-LEVEL PARALLELISM: classify ALL C channels first, THEN
            // probe all C. The per-channel RANGE SearchTransform classify is
            // bound by the dependent `wrapped_levels[guess]` LDG latency, and
            // occupancy is already exhausted, so the only remaining way to hide
            // that latency is more INDEPENDENT loads in flight. An interleaved
            // `for ch { BinSelect; probe_cuckoo }` puts `probe_cuckoo`'s
            // `__match_any_sync` (a warp convergence point) BETWEEN consecutive
            // channels, which prevents the compiler from issuing channel ch+1's
            // level loads before channel ch's classify retires -- serialising the
            // C otherwise-independent dependent-LDG chains per pixel. Splitting
            // the loop so every channel's `BinSelect` runs back-to-back (no
            // intervening warp sync) lets the C chains OVERLAP (C-way
            // memory-level parallelism), then the probe phase coalesces+atomics
            // each bin.
            int bins[NumActiveChannels];
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int ch = 0; ch < NumActiveChannels; ++ch)
            {
              int bin = -1;
              if (valid_pixel)
              {
                if constexpr (kUseMruCache)
                {
                  decode_op[ch].template BinSelect<LOAD_DEFAULT>(lanes[ch], bin, true, mru[ch]);
                }
                else
                {
                  decode_op[ch].template BinSelect<LOAD_DEFAULT>(lanes[ch], bin, true);
                }
                const int num_bins = num_output_bins_wrapper[ch];
                if (bin >= num_bins)
                {
                  bin = -1;
                }
              }
              bins[ch] = bin;
            }
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int ch = 0; ch < NumActiveChannels; ++ch)
            {
              probe(ch, bins[ch]);
            }
          }
        }
      }
      else
      {
        // Scalar fallback: per-channel global loads (non-native / unaligned
        // pointer). Same CHANNEL-LEVEL PARALLELISM split as the vectorized path:
        // classify all C channels back-to-back (overlapping the C independent
        // dependent-LDG SearchTransform chains, no intervening `__match_any_sync`
        // from probe_cuckoo), then probe all C.
        for (OffsetT it = 0; it < chunk_iters_max; ++it)
        {
          const OffsetT pixel = start + it * chunk;
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int u = 0; u < unroll; ++u)
          {
            const OffsetT this_pixel = pixel + u * step;
            const bool valid_pixel   = this_pixel < total_pixels;
            const OffsetT pix_off    = valid_pixel ? (this_pixel * NumChannels) : OffsetT{0};
            int bins[NumActiveChannels];
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int ch = 0; ch < NumActiveChannels; ++ch)
            {
              int bin = -1;
              if (valid_pixel)
              {
                auto sample = d_samples[pix_off + ch];
                if constexpr (kUseMruCache)
                {
                  decode_op[ch].template BinSelect<LOAD_DEFAULT>(sample, bin, true, mru[ch]);
                }
                else
                {
                  decode_op[ch].template BinSelect<LOAD_DEFAULT>(sample, bin, true);
                }
                const int num_bins = num_output_bins_wrapper[ch];
                if (bin >= num_bins)
                {
                  bin = -1;
                }
              }
              bins[ch] = bin;
            }
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int ch = 0; ch < NumActiveChannels; ++ch)
            {
              probe(ch, bins[ch]);
            }
          }
        }
      }
    }

    // After the sweep, flush every cache slot to the spill target (shared output
    // for output-spill; this block's private slab for private-spill). Each slot's
    // total is the sum of its `kCountReplicas` replicas. The block barrier ensures
    // every leader's atomicAdd_block has finished before the flush reads the
    // replicas. For the private-spill variant the flush is block-scope and
    // uncontended (each block owns its slab slice), so the SpillOp scope applies
    // here too.
    __syncthreads();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      CounterT* const ch_counts_base = s_cache_counts_base + static_cast<size_t>(ch) * kCountReplicas * slots_sz;
      for (int slot = threadIdx.x; slot < cache_slots_per_channel; slot += blockDim.x)
      {
        const int key = s_cache_keys[ch][slot];
        if (key >= 0)
        {
          CounterT cnt = CounterT{0};
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int r = 0; r < kCountReplicas; ++r)
          {
            cnt += ch_counts_base[static_cast<size_t>(r) * slots_sz + slot];
          }
          if (cnt > CounterT{0})
          {
            SpillOp::spill(spill_target[ch], key, cnt);
          }
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
        if constexpr (kUseMruCache)
        {
          decode_op[ch].template BinSelect<LOAD_DEFAULT>(sample, bin, true, mru[ch]);
        }
        else
        {
          decode_op[ch].template BinSelect<LOAD_DEFAULT>(sample, bin, true);
        }
        const int num_bins = num_output_bins_wrapper[ch];
        if (bin >= 0 && bin < num_bins)
        {
          SpillOp::spill(spill_target[ch], bin, CounterT{1});
        }
      }
    }
  }

  // ---------------------------------------------------------------------
  // Phase 4 (private-spill variant only): grid-sync so every block's slab
  // writes are visible, then an atomic-free gather sums the per-block slabs
  // into the shared output. Mirrors DeviceHistogramGmemPrivatizedKernel's
  // gather: each thread owns a slice of OUTPUT bins and column-sums across
  // blocks, turning num_blocks * num_bins contended atomics into plain
  // reads + one write per bin.
  // ---------------------------------------------------------------------
  if constexpr (kPrivateSpill)
  {
    grid.sync();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ch = 0; ch < NumActiveChannels; ++ch)
    {
      const unsigned int num_bins_u = static_cast<unsigned int>(num_output_bins_wrapper[ch]);
      gather_privatized_slab<CounterT>(
        d_output_histograms_wrapper[ch],
        /*out_offset=*/0u,
        d_private_histograms_wrapper[ch],
        /*slab_stride=*/num_bins_u,
        /*count=*/num_bins_u,
        blocks_per_grid,
        tid_global,
        total_threads);
    }
  }

  // Emit the trigger so any PDL-launched downstream kernel in the stream
  // sees a completion signal. (Cooperative launches typically do not use
  // PDL, so this is a no-op in the common case.)
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
}

//! Dynamic-SMEM privatized histogram sweep kernel (host-init).
//!
//! Holds the per-block privatized histogram in extern __shared__ sized at
//! launch, so one kernel serves the whole 256 < bins <= max_dynamic_smem_bins
//! range (16384 bins x 4 B = 64 KB exceeds the ptxas 48 KB static cap). Each
//! block merges its histogram directly into the global output via
//! `agent.StoreOutput()` (per-block `atomicAdd`); the host launches
//! `DeviceHistogramInitKernel` first to zero the output (StoreOutput
//! accumulates). A direct merge beats a GMEM staging round-trip here because at
//! these bin counts cross-block contention on the output is spread over enough
//! distinct bins that few blocks collide on any one.
//!
//! Host requirements:
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
// minBlocks=2 hint; the dispatch sizes the grid by this kernel's own sm_occupancy.
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block), 2)
  _CCCL_KERNEL_ATTRIBUTES void DeviceHistogramSmemPrivatizedDynamicKernel(
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

} // namespace detail::histogram
CUB_NAMESPACE_END
