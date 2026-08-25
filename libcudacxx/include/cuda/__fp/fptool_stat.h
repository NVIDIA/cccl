//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPTOOL_STAT_H
#define _CUDA___FP_FPTOOL_STAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! @file fptool_stat.h
//! @brief fpmp2_stat - a statistics-collecting drop-in replacement for fpmp2
//!
//! `fpmp2_stat<_FpType, _TypeAcc>` holds an `fpmp2<_FpType, _TypeAcc>` and mirrors its
//! arithmetic API. Results are bit-identical to the wrapped type: the wrapper only
//! observes them, so swapping `fp32mp2` for `fp32mp2_stat` never changes what a program
//! computes.
//!
//! ## Quick Start
//!
//! ```cpp
//! #include <cuda/fptool>
//!
//! using namespace cuda::experimental;
//!
//! using Real = fp32mp2_stat; // instead of fp32mp2
//!
//! cuda::stream_ref stream = ...;
//!
//! fpmp2_stat_reset_device_data(stream);          // clear the counters
//! my_kernel<<<blocks, threads, 0, stream.get()>>>(); // run the region of interest
//!
//! const fpmp2_stat_data stats = fpmp2_stat_read_device_data(stream);
//! printf("%llu adds, %llu muls\n", stats.add_count, stats.mul_count);
//! ```
//!
//! ## What Is Collected
//!
//! Each binary `+`, `-`, `*`, `/` - including the compound assignments, `++`, `--`, the
//! mixed value/scalar overloads and `atomicAdd`/`atomicSub` - increments its own counter,
//! classifies the operation as a whole through four event counters, and summarizes its two
//! operands and its result into `fpmp2_stat_data::arg[0]`, `arg[1]` and `result`.
//!
//! Each of the three slots receives exactly one value per counted operation, so
//! `ops_count` is the total to divide by when turning a count into a share, as in
//! `zero_lo_count / ops_count`.
//!
//! `sqrt`, `rsqrt`, `fma`, `mad`, `renormalize` and the math functions are not counted:
//! they are composites whose internal operations would swamp the counters. `arg[2]` is
//! reserved for a future ternary operation.
//!
//! ## Metrics
//!
//! The record is one `fpmp2_stat_data`: nine counters describing operations, and four
//! `fpmp2_stat_value` summaries describing the values that passed through the slots. The
//! fields carry a one-line description each; what they mean and what they are good for is
//! here.
//!
//! ### Operation Counters
//!
//! | Counter     | Incremented by               |
//! |-------------|------------------------------|
//! | `ops_count` | any of the four below        |
//! | `add_count` | `+`, `+=`, `++`, `atomicAdd` |
//! | `sub_count` | `-`, `-=`, `--`, `atomicSub` |
//! | `mul_count` | `*`, `*=`                    |
//! | `div_count` | `/`, `/=`                    |
//!
//! Together they are the operation mix of the instrumented region. Most kernels have a
//! closed form for it, and checking that prediction against these four is the cheapest
//! confirmation available that the region is the intended one, that nothing was optimized
//! away and that nothing outside it contributed. The division share is where both a
//! performance and an accuracy review start, division being the most expensive and least
//! accurate pair operation, and the subtraction count is what shows that a compensated
//! algorithm is really computing its corrections.
//!
//! ### Numerical Events
//!
//! These four classify the operation as a whole rather than its values, which takes the
//! operands and the result together. They apply where both operands were finite and
//! non-zero, so that a degenerate result cannot simply be an operand passing through:
//!
//! | Counter                | Operation | Result                                   |
//! |------------------------|-----------|------------------------------------------|
//! | `full_cancel_count`    | `+` `-`   | exact zero, i.e. the operands cancelled  |
//! | `partial_cancel_count` | `+` `-`   | more than half the significand cancelled |
//! | `underflow_count`      | `*` `/`   | exact zero, i.e. too small to represent  |
//! | `overflow_count`       | any       | non-finite                               |
//!
//! The split between cancellation and underflow needs no heuristic, because the operation
//! decides it: the difference of two distinct values is a non-zero multiple of the smaller
//! one's ulp and so never rounds to zero, which makes an additive zero proof that the
//! operands were equal and opposite, while multiplication and division have nothing to
//! cancel at all. A cancelling result is therefore exact, which makes `full_cancel_count` a
//! report of where the significance went rather than of an error. Complete underflow is the
//! opposite case, the result vanishing entirely; where a result survives but its precision
//! degrades - gradual underflow - `fpmp2_stat_value::denorm_count` is what reports it.
//!
//! Partial cancellation is measured as `max(ilogb of the operands) - ilogb of the result`,
//! the number of leading bits that cancelled, against a threshold of half the significand:
//! 23 bits of 46 for `fp32mp2`, 52 of 104 for `fp64mp2`. It is invisible in
//! `full_cancel_count`, the result not being zero. What it measures is the drop in
//! magnitude, which is not the same as bits gone from the representation: a cancellation
//! this deep usually leaves fewer surviving bits than a single limb holds, so the result
//! comes back with a zero `lo` and the pair really has lost half its width, but one bit past
//! the threshold the tail can still need the second limb, and then the pair keeps its full
//! width relative to its new, smaller magnitude. These are the prime suspects for
//! reformulation or for compensated summation.
//!
//! `overflow_count` counts NaN as well as infinity, because an `fpmp2` overflow usually
//! produces a NaN: the `hi` limb goes infinite and the tail is then computed as `inf - inf`.
//! That is unambiguous, since ordinary arithmetic on finite non-zero operands can produce
//! neither. Division by zero is not counted, one of its operands being zero.
//!
//! Read the cancellation counters as occurrences, not as damage. The subtraction itself is
//! exact, so a cancelling operation loses no accuracy of its own; what makes cancellation
//! harmful is rounding already present in the operands, which is not visible from one
//! operation. Algorithms that cancel deliberately - compensated summation above all, whose
//! whole purpose is to extract a small residual from two nearly equal values - therefore run
//! up large counts while being at their most accurate.
//!
//! ### Per-Slot Value Statistics
//!
//! One `fpmp2_stat_value` per slot, fed exactly once per counted operation:
//!
//! | Field                            | Records                                           |
//! |----------------------------------|---------------------------------------------------|
//! | `min_exp`, `max_exp`             | exponent range of the leading limb                |
//! | `zero_count`                     | both limbs zero                                   |
//! | `zero_lo_count`                  | non-zero `hi` and zero `lo`, no extra precision   |
//! | `inf_count`, `nan_count`         | an infinite, resp. NaN, limb                      |
//! | `infnan_count`                   | limbs infinite with opposite signs, so a NaN pair |
//! | `denorm_count`                   | a subnormal limb                                  |
//! | `overlap_count`                  | the limbs overlap, i.e. the gap is negative       |
//! | `invert_count`                   | the limbs are inverted, `abs(lo) > abs(hi)`       |
//! | `min_hi_lo_gap`, `max_hi_lo_gap` | range of the gap between the limbs                |
//! | `min_hi_lo_gap_sample_hi`, `_lo` | the limbs of a value that lowered the gap minimum |
//!
//! Every count is per value rather than per limb. A range says what the worst case was and
//! never how often it happened, so it is read together with the counter beside it:
//! `denorm_count` for the bottom of the exponent range, `overlap_count` and `invert_count`
//! for the bottom of the gap range. Both ranges start out empty, i.e. `min > max`, which is
//! how they say that nothing has been sampled yet.
//!
//! The **exponent range** is unbiased, as `ilogb` reports it, and is taken from the limb
//! that leads the pair: `hi` for anything a renormalizing accuracy level produces, `lo` for
//! an unnormalized pair led by `lo`, including one whose `hi` is zero. It is the dynamic
//! range the data actually exercises - how much headroom is left before overflow, and
//! whether a narrower format would hold it. `fp32mp2` inherits the exponent range of
//! `float`, which is the usual surprise when porting from `double`.
//!
//! **`zero_lo_count`** is the best single indicator of whether a pair type is earning its
//! cost: a high share means the second limb is idle for most of the data, and a narrower or
//! cheaper type may do the same job.
//!
//! **`inf_count`, `nan_count` and `infnan_count`** are the corruption detectors, and the
//! place to look once `overflow_count` has fired. The third is a pair whose two limbs are
//! infinities of opposite sign: reading such a pair adds them, so the value as a whole is a
//! NaN, and it is a specific pathological encoding that is easy to produce and hard to spot.
//! `zero_count`, by contrast, is unremarkable - it says how many of the operations were
//! trivial.
//!
//! **`denorm_count`** says the computation reached the bottom of the exponent range, where
//! precision degrades gradually. `lo` normally sits `digits` binades below `hi` and cannot
//! stay there once it is subnormal, so it reaches the limit long before `hi` does, and this
//! counter usually reports a pair that lost its tail rather than a subnormal result. It is
//! also a domain check: the `low` and `mid` accuracy levels support the normal range only,
//! so any subnormal means such a configuration is being used outside its domain.
//!
//! **`overlap_count` and `invert_count`** are about how the two limbs are placed with
//! respect to each other, not about how small they became. Overlap is the frequency behind
//! `min_hi_lo_gap`, and the two answer different questions: a minimum of -3 could be one
//! value in a billion or every second one, which call for opposite responses. Only the
//! `low` accuracy level, which skips renormalization, produces either case through
//! arithmetic, so this is what says whether such a configuration needs `renormalize` after
//! all; the two-limb constructor can produce them at any level.
//!
//! An **inverted** pair is the worst thing a pair can be, and the one number here that
//! should always be zero: the tail no longer describes a correction to the head but
//! outweighs it, so anything that reads the pair through its `hi` limb - a comparison, a
//! conversion, a sign test, a branch on magnitude - reaches the wrong answer, and every
//! later operation inherits it. A pair whose `hi` is zero while `lo` is not is the extreme
//! of this, and `invert_count` is the only counter that names it, its gap being deliberately
//! not sampled. Every inverted pair with two non-zero limbs is also an overlap, inverting
//! implying a gap of at most `-digits`; the reverse does not hold, an overlap of a few bits
//! being untidy but still describing the value correctly, which is why the two are worth
//! reading separately. Equal magnitudes are not counted, `abs(lo) > abs(hi)` being strict:
//! such a pair is degenerate in its own way - opposite signs make it an exact zero written
//! in two non-zero limbs - and shows up at the bottom of the gap range instead.
//!
//! ### Reading the Limb Gap
//!
//! The gap is `exp(hi) - exp(lo) - digits`, the raw exponent difference with the mantissa
//! width of the base type taken out, so it says how far the pair is from being tightly
//! normalized rather than how wide the format is:
//!
//! | Gap                   | Meaning                                                     |
//! |-----------------------|-------------------------------------------------------------|
//! | `0` or `1`            | normalized: no bits wasted, none overlapping                |
//! | negative              | the limbs overlap, so fewer significant bits than they hold |
//! | much greater than `1` | `lo` carries almost nothing, that many bits held in reserve |
//!
//! A normalized `lo` is at most half an ulp of `hi`, which puts its exponent exactly
//! `digits` places below, hence two values rather than one for a normalized pair: `0` is the
//! tie, `abs(lo)` exactly half an ulp of `hi`, which needs a set bit exactly `digits` places
//! below the leading one and so shows up for values with few significant bits; `1` is
//! anything strictly below, which is what an inexact division or a value with a dense
//! significand gives.
//!
//! Only pairs with two non-zero limbs are sampled. A gap places one limb against the other,
//! so a value with either limb zero has none to report: `zero_lo_count` covers the ordinary
//! case of a missing tail, and a pair whose `hi` alone is zero - which needs an unnormalized
//! value - is measured by `lo` in the exponent range instead. Subnormal limbs are measured
//! by their leading significant bit, as `ilogb` would report them, so a subnormal `lo` does
//! not fake an overlap; it does mean the pair lost part of its tail, which `denorm_count`
//! reports.
//!
//! ### Which Metric Answers Which Question
//!
//! | Question                             | Metrics that answer it                         |
//! |--------------------------------------|------------------------------------------------|
//! | Is the pair type earning its cost?   | `zero_lo_count`, `max_hi_lo_gap`               |
//! | Am I measuring the intended region?  | the four operation counters                    |
//! | Is `low` accuracy safe on this data? | `overlap_count`, `invert_count`, gap minimum   |
//! | Where does the precision go?         | the two cancellation counters                  |
//! | Is the dynamic range near an edge?   | exponent range, `denorm_count`, over/underflow |
//! | Is anything silently corrupt?        | `invert_count`, `nan_count`, `infnan_count`    |
//!
//! ## Collection Is Device-Only
//!
//! The counters live in device memory and are updated with atomics, so only work that
//! runs on the GPU is observed. The very same code compiles and runs on the host, where
//! the wrapper is a transparent pass-through and no counters are gathered.
//!
//! | Function                                   | Description                             |
//! |--------------------------------------------|-----------------------------------------|
//! | `fpmp2_stat_reset_device_data(stream_ref)` | Clear counters, arm the range sentinels |
//! | `fpmp2_stat_read_device_data(stream_ref)`  | Return the record, read on that stream  |
//!
//! The record is per-device state, so both take the stream that says which device to touch
//! and when: the clear is enqueued on it, and the read waits on it before handing the
//! record back. Both throw `cuda::cuda_error` if the copy fails, rather than leaving a
//! status to be checked, and both are host-only, so they do not exist under NVRTC, whose
//! translation unit has no host side.
//!
//! The record lives in one program-wide copy shared by all translation units. On the
//! device that sharing needs relocatable device code (`-rdc=true`); in whole-program
//! mode each translation unit gets its own device copy, and a reset or read then only
//! sees the copy belonging to its own translation unit.
//!
//! @note The limb gap sample (`min_hi_lo_gap_sample_hi` / `_lo`) is best-effort
//! under concurrency: a thread with a larger gap never overwrites it, but two threads
//! lowering the minimum at once may leave the sample of either one.
//! @note Instrumentation costs a handful of atomics per operation, so a `_stat` type is
//! meant for analysis runs rather than production ones.

#include <cuda/__fp/fpmp.h>
#include <cuda/__fp/fpmp_limits.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)
// The host side of the record: a stream-ordered copy to and from the device global
#  include <cuda/__memory/get_device_address.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/stream_ref.h>
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// === collected record ===

//! @brief Summary of the fpmp2 values that passed through one operand or result slot
//!
//! One of these is fed exactly once per counted operation. What the fields mean, and what
//! each is good for, is in the metrics reference in this file's documentation; the ranges
//! start out empty, i.e. `min > max`, to say that nothing has been sampled yet, and
//! `fpmp2_stat_reset_device_data` arms them.
//!
//! The field types are the ones the device-side atomics take, which is also what makes
//! them the portable choice here: `unsigned long long int` is what `atomicAdd` accepts
//! and what `%llu` prints, and it is at least 64 bits by the standard and exactly 64 on
//! every platform CUDA supports. (`uint64_t` would not do: it is `unsigned long` on LP64
//! platforms, a distinct type for which no `atomicAdd` overload exists.) The exponents
//! are `int`, the type of `ilogb`, of `numeric_limits<>::min_exponent` and of
//! `atomicMin`/`atomicMax`, and identical to `int32_t` wherever CCCL builds.
struct fpmp2_stat_value
{
  //! @brief Largest exponent of the leading limb, `numeric_limits<int>::min()` until sampled
  int max_exp;
  //! @brief Smallest exponent of the leading limb, `numeric_limits<int>::max()` until sampled
  int min_exp;
  //! @brief Values whose `hi` and `lo` limbs were both zero
  unsigned long long int zero_count;
  //! @brief Finite values with a non-zero `hi` and a zero `lo`, i.e. no extra precision
  unsigned long long int zero_lo_count;
  //! @brief Values with an infinite `hi` or `lo`, counted once per value
  unsigned long long int inf_count;
  //! @brief Values with a NaN `hi` or `lo`, counted once per value
  unsigned long long int nan_count;
  //! @brief Values whose limbs were infinities of opposite signs, whose sum is a NaN
  unsigned long long int infnan_count;
  //! @brief Values with a subnormal `hi` or `lo`, counted once per value
  unsigned long long int denorm_count;
  //! @brief Values whose limbs overlapped, i.e. whose gap was negative
  unsigned long long int overlap_count;
  //! @brief Values whose limbs were inverted, i.e. `abs(lo) > abs(hi)`
  unsigned long long int invert_count;
  //! @brief Largest limb gap seen, `numeric_limits<int>::min()` until a value is sampled
  int max_hi_lo_gap;
  //! @brief Smallest limb gap seen, `numeric_limits<int>::max()` until a value is sampled
  int min_hi_lo_gap;
  //! @brief `hi` limb of a value that lowered `min_hi_lo_gap`, for inspection
  double min_hi_lo_gap_sample_hi;
  //! @brief `lo` limb of the same value
  double min_hi_lo_gap_sample_lo;
};

//! @brief The record a `_stat` type fills in on the device
//!
//! Counters are shared by every `fpmp2_stat` instantiation, so a program that uses more
//! than one of them sees their operations summed. What the fields mean, and what each is
//! good for, is in the metrics reference in this file's documentation.
struct fpmp2_stat_data
{
  //! @brief Instrumented binary operations, the sum of the four counters below
  unsigned long long int ops_count;
  //! @brief Instrumented additions, including `+=`, `++` and `atomicAdd`
  unsigned long long int add_count;
  //! @brief Instrumented subtractions, including `-=`, `--` and `atomicSub`
  unsigned long long int sub_count;
  //! @brief Instrumented multiplications, including `*=`
  unsigned long long int mul_count;
  //! @brief Instrumented divisions, including `/=`
  unsigned long long int div_count;
  //! @brief Additive operations that cancelled completely, i.e. returned an exact zero
  unsigned long long int full_cancel_count;
  //! @brief Additive operations that lost more than half the significand to cancellation
  //! without reaching zero
  unsigned long long int partial_cancel_count;
  //! @brief Multiplications and divisions whose non-zero operands produced an exact zero
  unsigned long long int underflow_count;
  //! @brief Operations whose finite non-zero operands produced a non-finite result
  unsigned long long int overflow_count;
  //! @brief Operand summaries: `arg[0]` and `arg[1]` are the binary operands, `arg[2]`
  //! is reserved for a future ternary operation
  fpmp2_stat_value arg[3];
  //! @brief Result summary
  fpmp2_stat_value result;
};

// Which counter an instrumented operation bumps. A template argument rather than a
// runtime one, so the increment folds down to a single atomic.
enum class __fpmp2_stat_binop
{
  __add,
  __sub,
  __mul,
  __div,
};

// A slot with both ranges armed: empty ranges that the first sample replaces.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr fpmp2_stat_value __fpmp2_stat_cleared_value() noexcept
{
  fpmp2_stat_value __slot{};
  __slot.max_exp       = ::cuda::std::numeric_limits<int>::min();
  __slot.min_exp       = ::cuda::std::numeric_limits<int>::max();
  __slot.max_hi_lo_gap = ::cuda::std::numeric_limits<int>::min();
  __slot.min_hi_lo_gap = ::cuda::std::numeric_limits<int>::max();
  return __slot;
}

// A zeroed record with every slot armed, used both as the initial value of the device
// record and as the source of a reset.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr fpmp2_stat_data __fpmp2_stat_cleared_data() noexcept
{
  fpmp2_stat_data __data{};
  for (int __i = 0; __i < 3; ++__i)
  {
    __data.arg[__i] = __fpmp2_stat_cleared_value();
  }
  __data.result = __fpmp2_stat_cleared_value();
  return __data;
}

#if _CCCL_CUDA_COMPILATION()
// The record every instrumented operation updates.
//
// A variable template rather than a plain variable on purpose: a variable template has
// vague linkage, so all translation units share one copy, while a plain
// `inline _CCCL_DEVICE` variable is rejected by nvcc outside relocatable device code.
// The dummy parameter exists only to make it a template; the counters are deliberately
// shared by all instantiations of fpmp2_stat.
template <class _Void = void>
_CCCL_DEVICE fpmp2_stat_data __fpmp2_stat_device_data = __fpmp2_stat_cleared_data();
#endif // _CCCL_CUDA_COMPILATION()

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

//! @brief Clear the device record and arm its range sentinels
//!
//! Call it before the region of interest. The clear is enqueued on `__stream`, so counting
//! starts with the next kernel that runs there and no synchronization is needed. The record
//! is per-device state, and the device it is cleared on is the stream's.
//!
//! @param __stream Stream to order the clear against, and whose device to clear on
//! @throws cuda::cuda_error if the copy cannot be enqueued
_CCCL_HOST_API inline void fpmp2_stat_reset_device_data(::cuda::stream_ref __stream)
{
  // A pageable host-to-device copy consumes its source before returning, so the cleared
  // record does not have to outlive the call.
  const fpmp2_stat_data __cleared = __fpmp2_stat_cleared_data();
  fpmp2_stat_data* __data_ptr     = ::cuda::get_device_address(__fpmp2_stat_device_data<>, __stream.device());
  _CCCL_TRY_CUDA_API(
    ::cudaMemcpyAsync,
    "failed to clear the fpmp2_stat device record",
    __data_ptr,
    &__cleared,
    sizeof(fpmp2_stat_data),
    ::cudaMemcpyHostToDevice,
    __stream.get());
}

//! @brief Copy the device record to the host
//!
//! Waits on `__stream`, so the kernels enqueued there have finished and their counts are
//! included. Work on other streams has to be synchronized by the caller.
//!
//! @param __stream Stream to read the record through, and whose device to read from
//! @return The record as of the end of that stream's work
//! @throws cuda::cuda_error if the copy fails
[[nodiscard]] _CCCL_HOST_API inline fpmp2_stat_data fpmp2_stat_read_device_data(::cuda::stream_ref __stream)
{
  fpmp2_stat_data __dst{};
  const fpmp2_stat_data* __data_ptr = ::cuda::get_device_address(__fpmp2_stat_device_data<>, __stream.device());
  _CCCL_TRY_CUDA_API(
    ::cudaMemcpyAsync,
    "failed to read the fpmp2_stat device record",
    &__dst,
    __data_ptr,
    sizeof(fpmp2_stat_data),
    ::cudaMemcpyDeviceToHost,
    __stream.get());
  __stream.sync();
  return __dst;
}

#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

// === value inspection ===

// The IEEE-754 fields of one limb, as the accumulation below needs them.
struct __fpmp2_stat_parts
{
  // Unbiased exponent, i.e. the encoded field minus the bias. Zero and subnormals share
  // the lowest value, -bias, and infinity and NaN the highest one. Use it to classify a
  // limb, not to measure it.
  int __exp;
  // What ilogb would report: the position of the leading significant bit. Identical to
  // __exp for normal values, but a subnormal's encoded field is pinned at its minimum and
  // says nothing about the magnitude, so the leading mantissa bit has to be located.
  // Meaningless for zero, infinity and NaN, which the callers exclude.
  int __exp_ilogb;
  // The encoding with the sign bit cleared, widened to the largest limb type. IEEE-754 lays
  // out exponent above mantissa, so for same-typed limbs this orders magnitudes exactly as a
  // floating-point comparison of absolute values would, subnormals included, and answers
  // "which limb is larger" in one integer compare. Not meaningful for NaN, which the callers
  // exclude.
  ::cuda::std::uint64_t __mag;
  bool __mant_is_zero;
  bool __exp_is_max;
  bool __sign;
};

// Splits a limb without touching the FPU, so the same code serves float and double.
// The two constants come from numeric_limits: digits is the mantissa size including the
// implicit bit, and max_exponent is the bias plus one.
template <class _FpType>
[[nodiscard]] _CCCL_HOST_DEVICE_API __fpmp2_stat_parts __fpmp2_stat_split(_FpType __x) noexcept
{
  using _UInt = ::cuda::std::
    conditional_t<sizeof(_FpType) == sizeof(::cuda::std::uint32_t), ::cuda::std::uint32_t, ::cuda::std::uint64_t>;

  constexpr int __mant_size = ::cuda::std::numeric_limits<_FpType>::digits - 1;
  constexpr int __bias      = ::cuda::std::numeric_limits<_FpType>::max_exponent - 1;
  constexpr int __exp_max   = 2 * __bias + 1;

  const _UInt __bits = ::cuda::std::bit_cast<_UInt>(__x);
  const int __exp    = static_cast<int>((__bits >> __mant_size) & static_cast<_UInt>(__exp_max));
  const _UInt __mant = __bits & ((_UInt{1} << __mant_size) - _UInt{1});

  // A subnormal is `mant * 2^(1 - bias - mant_size)`, so its leading bit sits at
  // `msb(mant) + 1 - bias - mant_size`. The count is computed unconditionally, which is one
  // instruction, rather than behind a branch that would practically never be taken.
  const int __msb        = static_cast<int>(8 * sizeof(_UInt)) - 1 - ::cuda::std::countl_zero(__mant);
  const bool __is_denorm = __exp == 0 && __mant != _UInt{0};
  const int __exp_ilogb  = __is_denorm ? (__msb + 1 - __bias - __mant_size) : (__exp - __bias);

  constexpr _UInt __sign_mask = _UInt{1} << (8 * sizeof(_UInt) - 1);

  return {__exp - __bias,
          __exp_ilogb,
          static_cast<::cuda::std::uint64_t>(__bits & static_cast<_UInt>(~__sign_mask)),
          __mant == _UInt{0},
          __exp == __exp_max,
          (__bits & __sign_mask) != _UInt{0}};
}

#if _CCCL_CUDA_COMPILATION()
// What one accumulated value was, for the caller to classify the operation as a whole. The
// slot summaries answer questions about values; these three answer questions about an
// operation, which needs its operands and its result together.
struct __fpmp2_stat_summary
{
  // ilogb of the pair, i.e. of whichever limb leads it. That is `hi` for any pair a
  // renormalizing accuracy level can produce, but an unnormalized one may be led by `lo`,
  // and `hi` may even be zero while the pair is not. Meaningful only for a finite non-zero
  // value.
  int __exp;
  bool __is_zero;
  bool __is_finite;
};

// Folds one fpmp2 value into a slot. Device-only: the updates are atomic because every
// thread of a grid writes the same slot.
template <class _FpType>
_CCCL_DEVICE_API inline __fpmp2_stat_summary
__fpmp2_stat_accumulate(fpmp2_stat_value* __slot, _FpType __hi, _FpType __lo) noexcept
{
  const __fpmp2_stat_parts __p_hi = __fpmp2_stat_split(__hi);
  const __fpmp2_stat_parts __p_lo = __fpmp2_stat_split(__lo);

  constexpr int __exp_min = 1 - ::cuda::std::numeric_limits<_FpType>::max_exponent;

  const bool __hi_is_zero = __p_hi.__exp == __exp_min && __p_hi.__mant_is_zero;
  const bool __lo_is_zero = __p_lo.__exp == __exp_min && __p_lo.__mant_is_zero;
  const bool __is_zero    = __hi_is_zero && __lo_is_zero;
  const bool __is_nan     = (__p_hi.__exp_is_max && !__p_hi.__mant_is_zero) //
                         || (__p_lo.__exp_is_max && !__p_lo.__mant_is_zero);
  const bool __is_inf     = (__p_hi.__exp_is_max && __p_hi.__mant_is_zero) //
                         || (__p_lo.__exp_is_max && __p_lo.__mant_is_zero);
  const bool __is_finite  = !__p_hi.__exp_is_max && !__p_lo.__exp_is_max;
  // A subnormal is the minimum exponent field with a non-zero mantissa, which is what the
  // zero tests above rule out. Either limb counts.
  const bool __is_denorm = (__p_hi.__exp == __exp_min && !__p_hi.__mant_is_zero) //
                        || (__p_lo.__exp == __exp_min && !__p_lo.__mant_is_zero);

  if (__is_nan)
  {
    ::atomicAdd(&__slot->nan_count, 1ull);
  }
  if (__is_inf)
  {
    ::atomicAdd(&__slot->inf_count, 1ull);
  }
  // Infinities of opposite signs: the value the pair stands for is a NaN, which usually
  // means an overflow the algorithm did not expect.
  if (__p_hi.__exp_is_max && __p_hi.__mant_is_zero && __p_lo.__exp_is_max && __p_lo.__mant_is_zero
      && __p_hi.__sign != __p_lo.__sign)
  {
    ::atomicAdd(&__slot->infnan_count, 1ull);
  }
  if (__is_denorm)
  {
    ::atomicAdd(&__slot->denorm_count, 1ull);
  }
  if (__is_zero)
  {
    ::atomicAdd(&__slot->zero_count, 1ull);
  }
  if (__is_finite && !__hi_is_zero && __lo_is_zero)
  {
    ::atomicAdd(&__slot->zero_lo_count, 1ull);
  }

  // The exponent of the pair is the one of the limb that leads it. `hi` leads any pair the
  // renormalizing levels produce, but `low` skips renormalization and the two-limb
  // constructor accepts anything, so `lo` may lead - and `hi` may be zero while the pair is
  // not. `__exp_ilogb` of a zero limb is the pinned minimum, which says nothing about a
  // magnitude, so it must not be allowed to stand in for one.
  int __pair_exp = __p_hi.__exp_ilogb;
  if (__hi_is_zero || (!__lo_is_zero && __p_lo.__exp_ilogb > __p_hi.__exp_ilogb))
  {
    __pair_exp = __p_lo.__exp_ilogb;
  }

  if (__is_finite && !__is_zero)
  {
    ::atomicMax(&__slot->max_exp, __pair_exp);
    ::atomicMin(&__slot->min_exp, __pair_exp);

    // A tail heavier than the head it corrects, which includes a zero `hi` carrying a
    // non-zero `lo`. One integer compare of the sign-cleared encodings, no branch needed to
    // exclude a zero `lo`: its magnitude is zero and cannot exceed anything.
    if (__p_lo.__mag > __p_hi.__mag)
    {
      ::atomicAdd(&__slot->invert_count, 1ull);
    }

    // The gap describes how the two limbs are placed relative to each other, which needs
    // both of them: with `hi` zero there is no leading limb to measure a tail against, and
    // the pinned exponent of that zero would fake an arbitrarily deep overlap.
    if (!__hi_is_zero && !__lo_is_zero)
    {
      // Measured against a tightly normalized pair rather than as a raw exponent
      // difference: a normalized `lo` is at most half an ulp of `hi`, which puts its
      // exponent `digits` places below, so subtracting `digits` makes 0 the tight case and
      // a negative value an overlap. See the metrics reference in the file documentation.
      constexpr int __digits = ::cuda::std::numeric_limits<_FpType>::digits;

      const int __gap      = __p_hi.__exp_ilogb - __p_lo.__exp_ilogb - __digits;
      const int __prev_min = ::atomicMin(&__slot->min_hi_lo_gap, __gap);
      ::atomicMax(&__slot->max_hi_lo_gap, __gap);

      if (__gap < 0)
      {
        ::atomicAdd(&__slot->overlap_count, 1ull);
      }

      // Best-effort sample of the tightest pair seen, see the note in the file documentation.
      if (__gap < __prev_min)
      {
        __slot->min_hi_lo_gap_sample_hi = static_cast<double>(__hi);
        __slot->min_hi_lo_gap_sample_lo = static_cast<double>(__lo);
      }
    }
  }

  return {__pair_exp, __is_zero, __is_finite};
}

// Records one instrumented binary operation: the counters plus the three value slots.
template <__fpmp2_stat_binop _Kind, class _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_DEVICE_API inline void __fpmp2_stat_note_binop(
  const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y, const fpmp2<_FpType, _TypeAcc>& __r) noexcept
{
  fpmp2_stat_data& __data = __fpmp2_stat_device_data<>;

  ::atomicAdd(&__data.ops_count, 1ull);
  if constexpr (_Kind == __fpmp2_stat_binop::__add)
  {
    ::atomicAdd(&__data.add_count, 1ull);
  }
  else if constexpr (_Kind == __fpmp2_stat_binop::__sub)
  {
    ::atomicAdd(&__data.sub_count, 1ull);
  }
  else if constexpr (_Kind == __fpmp2_stat_binop::__mul)
  {
    ::atomicAdd(&__data.mul_count, 1ull);
  }
  else
  {
    ::atomicAdd(&__data.div_count, 1ull);
  }

  const __fpmp2_stat_summary __s_x = __fpmp2_stat_accumulate(&__data.arg[0], __x.hi(), __x.lo());
  const __fpmp2_stat_summary __s_y = __fpmp2_stat_accumulate(&__data.arg[1], __y.hi(), __y.lo());
  const __fpmp2_stat_summary __s_r = __fpmp2_stat_accumulate(&__data.result, __r.hi(), __r.lo());

  // Cancellation, underflow and overflow are only meaningful where both operands were
  // ordinary values: otherwise a zero or a non-finite result may just be an operand passing
  // through. The guard also excludes division by zero, whose divisor is zero.
  if (__s_x.__is_finite && !__s_x.__is_zero && __s_y.__is_finite && !__s_y.__is_zero)
  {
    constexpr bool __is_additive = _Kind == __fpmp2_stat_binop::__add || _Kind == __fpmp2_stat_binop::__sub;

    if (!__s_r.__is_finite)
    {
      ::atomicAdd(&__data.overflow_count, 1ull);
    }
    else if (__s_r.__is_zero)
    {
      // Only an additive operation can reach zero by cancelling; a multiplicative one has
      // nothing to cancel, so its zero is an underflow. See the metrics reference in the
      // file documentation.
      ::atomicAdd(__is_additive ? &__data.full_cancel_count : &__data.underflow_count, 1ull);
    }
    else if (__is_additive)
    {
      // How far the result fell below its operands, in binades. Only an effective subtraction
      // can push it below both, so no sign test is needed: a same-sign addition cannot make
      // this positive.
      //
      // Magnitudes are all this looks at - the exponent of each value's leading limb, never
      // what the limbs contain. That is the whole of the measurement: the operation itself is
      // exact, and what a cancellation costs is the accuracy the operands brought into it,
      // which is relative to a magnitude that just dropped by this many bits.
      //
      // Crossing half the pair's significand is therefore the point where a double-word value
      // stops being worth more than a single limb. It is also where the survivors stop needing
      // the second limb, so a faithful operation returns a zero `lo` past this threshold and
      // keeps a non-zero one below it. That is an observation about the arithmetic, not
      // something the test relies on: a malformed pair - garbage in `lo`, or limbs that
      // overlap - cannot change the count, and is reported by `overlap_count`, `denorm_count`
      // and the non-finite counters of its slot instead.
      constexpr int __threshold = ::cuda::std::numeric_limits<fpmp2<_FpType, _TypeAcc>>::digits / 2;

      const int __larger = (__s_x.__exp > __s_y.__exp) ? __s_x.__exp : __s_y.__exp;
      if (__larger - __s_r.__exp > __threshold)
      {
        ::atomicAdd(&__data.partial_cancel_count, 1ull);
      }
    }
  }
}
#endif // _CCCL_CUDA_COMPILATION()

// === main class definition ===

//! @brief Statistics-collecting drop-in replacement for `fpmp2`
//!
//! Wraps an `fpmp2<_FpType, _TypeAcc>` and mirrors its arithmetic, comparison and
//! conversion API. Every instrumented operation is computed by the wrapped type first
//! and only then observed, so results are bit-identical to the plain type.
//!
//! ## Memory Layout
//! Same size and alignment as the wrapped `fpmp2`, and trivially copyable, so arrays
//! and kernel arguments can be reinterpreted between the two.
//!
//! ## Interoperability
//! Conversion to and from the wrapped `fpmp2` of the same accuracy is implicit, and
//! mixing the two in one expression yields an instrumented result. Accuracy levels do
//! not mix implicitly, exactly as they do not for `fpmp2` itself: convert explicitly,
//! e.g. `fp32mp2_stat_low(x)`.
//!
//! @tparam _FpType Limb type: `float` for double-float, `double` for double-double
//! @tparam _TypeAcc Arithmetic accuracy level, see `fpmp2_accuracy`
//!
//! @note Statistics are collected on the device only; on the host the wrapper is a
//! transparent pass-through.
template <class _FpType, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
class alignas(alignof(fpmp2<_FpType, _TypeAcc>)) fpmp2_stat
{
public:
  //! @brief The wrapped type, whose results this type reproduces exactly
  using base_type = fpmp2<_FpType, _TypeAcc>;

private:
  base_type __stat_v_;

  // Instrumentation hook. Compiled out on the host, where there is no record to update,
  // which leaves the operands unread in a host-only translation unit.
  template <__fpmp2_stat_binop _Kind>
  _CCCL_HOST_DEVICE_API static void
  __trace([[maybe_unused]] const base_type& __x,
          [[maybe_unused]] const base_type& __y,
          [[maybe_unused]] const base_type& __r) noexcept
  {
    NV_IF_TARGET(NV_IS_DEVICE, (__fpmp2_stat_note_binop<_Kind>(__x, __y, __r);))
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr fpmp2_stat __from_base(const base_type& __v) noexcept
  {
    return fpmp2_stat(__v);
  }

public:
  //! @brief Read the high limb
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _FpType hi() const noexcept
  {
    return __stat_v_.hi();
  }
  //! @brief Read the low limb
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _FpType lo() const noexcept
  {
    return __stat_v_.lo();
  }
  //! @brief Read the high limb of a volatile object
  [[nodiscard]] _CCCL_HOST_DEVICE_API _FpType hi() const volatile noexcept
  {
    return __stat_v_.hi();
  }
  //! @brief Read the low limb of a volatile object
  [[nodiscard]] _CCCL_HOST_DEVICE_API _FpType lo() const volatile noexcept
  {
    return __stat_v_.lo();
  }

  //! @brief Access the wrapped value, e.g. to hand it to an API taking `fpmp2`
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const base_type& as_fpmp2() const noexcept
  {
    return __stat_v_;
  }
  //! @brief Access the wrapped value for modification, which is not instrumented
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr base_type& as_fpmp2() noexcept
  {
    return __stat_v_;
  }

  // === constructors ===

  //! @brief Default constructor, leaving the value uninitialized like `fpmp2`
  _CCCL_HIDE_FROM_ABI fpmp2_stat() = default;

  //! @brief Construct from the two limbs directly
  _CCCL_HOST_DEVICE_API constexpr fpmp2_stat(_FpType __hi, _FpType __lo) noexcept
      : __stat_v_{__hi, __lo}
  {}

  //! @brief Copy constructor, defaulted so the type stays trivially copyable
  //! @note NVCC implicitly makes defaulted special members __host__ __device__
  _CCCL_HIDE_FROM_ABI fpmp2_stat(const fpmp2_stat&) = default;
  //! @brief Copy assignment, defaulted so the type stays trivially copyable
  _CCCL_HIDE_FROM_ABI constexpr fpmp2_stat& operator=(const fpmp2_stat&) = default;

  // Volatile support, mirroring fpmp2: storage only, i.e. load, store and a
  // limb-preserving round-trip. Each overload is wrapped in a dummy template so that
  // the C++ standard does not consider it a copy constructor or copy assignment
  // operator, which preserves trivial copyability.

  //! @brief Copy constructor from a volatile object
  template <class _Dummy = void>
  _CCCL_HOST_DEVICE_API fpmp2_stat(const volatile fpmp2_stat& __other) noexcept
      : __stat_v_{__other.hi(), __other.lo()}
  {}

  //! @brief Assignment to a volatile object
  //! @note Returns void to avoid the C++20 deprecation of a volatile return type
  template <class _Dummy = void>
  _CCCL_HOST_DEVICE_API void operator=(const fpmp2_stat& __other) volatile noexcept
  {
    __stat_v_ = __other.__stat_v_;
  }

  //! @brief Assignment from a volatile object
  template <class _Dummy = void>
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator=(const volatile fpmp2_stat& __other) noexcept
  {
    __stat_v_ = base_type{__other.hi(), __other.lo()};
    return *this;
  }

  //! @brief Implicit conversion from the wrapped type, the symmetric counterpart of
  //! `operator base_type()`
  _CCCL_HOST_DEVICE_API constexpr fpmp2_stat(const base_type& __other) noexcept
      : __stat_v_{__other}
  {}

  //! @brief Explicit conversion from another accuracy level, which copies the limbs
  //! without renormalizing, as the `fpmp2` counterpart does
  _CCCL_TEMPLATE(fpmp2_accuracy _TypeAcc2)
  _CCCL_REQUIRES((_TypeAcc2 != _TypeAcc))
  _CCCL_HOST_DEVICE_API constexpr explicit fpmp2_stat(const fpmp2<_FpType, _TypeAcc2>& __other) noexcept
      : __stat_v_{base_type{__other}}
  {}

  //! @brief Explicit conversion from a `_stat` type of another accuracy level
  _CCCL_TEMPLATE(fpmp2_accuracy _TypeAcc2)
  _CCCL_REQUIRES((_TypeAcc2 != _TypeAcc))
  _CCCL_HOST_DEVICE_API constexpr explicit fpmp2_stat(const fpmp2_stat<_FpType, _TypeAcc2>& __other) noexcept
      : __stat_v_{base_type{__other.as_fpmp2()}}
  {}

  //! @brief Implicit conversion from a single limb, leaving `lo` zero
  _CCCL_HOST_DEVICE_API constexpr fpmp2_stat(_FpType __f) noexcept
      : __stat_v_{__f}
  {}

  //! @brief Construct a double-float from a double, splitting it into the two limbs
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  _CCCL_HOST_DEVICE_API constexpr _CCCL_FPMP_EXPLICIT fpmp2_stat(double __d) noexcept
      : __stat_v_{__d}
  {}

#if _CCCL_FPMP_FP128_ENABLE == 1
  //! @brief Construct a double-double from a binary128 value
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  _CCCL_FPMP_FP128_API constexpr _CCCL_FPMP_EXPLICIT fpmp2_stat(__fpmp_fp128 __d) noexcept
      : __stat_v_{__d}
  {}
  //! @brief Explicit conversion of a double-double to binary128
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  [[nodiscard]] _CCCL_FPMP_FP128_API explicit operator __fpmp_fp128() const noexcept
  {
    return static_cast<__fpmp_fp128>(__stat_v_);
  }
  // A double-float has no binary128 interchange in either direction, deleted rather
  // than absent for the same reason as in fpmp2: the diagnostic then names the rule.
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  _CCCL_FPMP_FP128_API _CCCL_FPMP_EXPLICIT fpmp2_stat(__fpmp_fp128) = delete;
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  _CCCL_FPMP_FP128_API explicit operator __fpmp_fp128() const = delete;
#endif // _CCCL_FPMP_FP128_ENABLE == 1

  //! @brief Construct from any standard integer type
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_HOST_DEVICE_API _CCCL_FPMP_EXPLICIT fpmp2_stat(_Tp __i) noexcept
      : __stat_v_{__i}
  {}

  //! @brief Construct from `bool` or a character type, which `__cccl_is_integer_v`
  //! excludes but `double` accepts
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND(!::cuda::std::__cccl_is_integer_v<_Tp>))
  _CCCL_HOST_DEVICE_API _CCCL_FPMP_EXPLICIT fpmp2_stat(_Tp __i) noexcept
      : __stat_v_{__i}
  {}

#if _CCCL_HAS_INT128()
  // Deleted for the same reason as in fpmp2: a 128-bit integer would silently truncate.
  _CCCL_HOST_DEVICE_API _CCCL_FPMP_EXPLICIT fpmp2_stat(__int128_t)  = delete;
  _CCCL_HOST_DEVICE_API _CCCL_FPMP_EXPLICIT fpmp2_stat(__uint128_t) = delete;
#endif // _CCCL_HAS_INT128()

  // === conversions out ===

  //! @brief Implicit conversion to the wrapped type
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr operator base_type() const noexcept
  {
    return __stat_v_;
  }

  //! @brief Conversion to the wrapped type from a volatile object
  [[nodiscard]] _CCCL_HOST_DEVICE_API operator base_type() const volatile noexcept
  {
    return base_type{hi(), lo()};
  }

  //! @brief Explicit conversion to the wrapped type of another accuracy level
  _CCCL_TEMPLATE(fpmp2_accuracy _TypeAcc2)
  _CCCL_REQUIRES((_TypeAcc2 != _TypeAcc))
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit constexpr operator fpmp2<_FpType, _TypeAcc2>() const noexcept
  {
    return fpmp2<_FpType, _TypeAcc2>{__stat_v_};
  }

  // Conversion to double follows the wrapped type: implicit out of a double-float,
  // where the pair sums exactly, explicit out of a double-double, where the low limb is
  // dropped. See the corresponding comment in fpmp.h.
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API operator double() const noexcept
  {
    return static_cast<double>(__stat_v_);
  }
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API operator double() const volatile noexcept
  {
    return static_cast<double>(base_type{hi(), lo()});
  }
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator double() const noexcept
  {
    return static_cast<double>(__stat_v_);
  }
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator double() const volatile noexcept
  {
    return static_cast<double>(base_type{hi(), lo()});
  }

  //! @brief Explicit conversion to float
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator float() const noexcept
  {
    return static_cast<float>(__stat_v_);
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator float() const volatile noexcept
  {
    return static_cast<float>(base_type{hi(), lo()});
  }

  //! @brief Explicit conversion to any standard integer type
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator _Tp() const noexcept
  {
    return static_cast<_Tp>(__stat_v_);
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator _Tp() const volatile noexcept
  {
    return static_cast<_Tp>(base_type{hi(), lo()});
  }

#if _CCCL_HAS_INT128()
  _CCCL_HOST_DEVICE_API explicit operator __int128_t() const           = delete;
  _CCCL_HOST_DEVICE_API explicit operator __uint128_t() const          = delete;
  _CCCL_HOST_DEVICE_API explicit operator __int128_t() const volatile  = delete;
  _CCCL_HOST_DEVICE_API explicit operator __uint128_t() const volatile = delete;
#endif // _CCCL_HAS_INT128()

  // === arithmetic ===

  //! @brief Renormalize the pair, which is not an instrumented operation
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat renormalize(const fpmp2_stat& __x) noexcept
  {
    return __from_base(renormalize(__x.__stat_v_));
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator+(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    const base_type __r = __x.__stat_v_ + __y.__stat_v_;
    __trace<__fpmp2_stat_binop::__add>(__x.__stat_v_, __y.__stat_v_, __r);
    return __from_base(__r);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator-(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    const base_type __r = __x.__stat_v_ - __y.__stat_v_;
    __trace<__fpmp2_stat_binop::__sub>(__x.__stat_v_, __y.__stat_v_, __r);
    return __from_base(__r);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator*(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    const base_type __r = __x.__stat_v_ * __y.__stat_v_;
    __trace<__fpmp2_stat_binop::__mul>(__x.__stat_v_, __y.__stat_v_, __r);
    return __from_base(__r);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator/(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    const base_type __r = __x.__stat_v_ / __y.__stat_v_;
    __trace<__fpmp2_stat_binop::__div>(__x.__stat_v_, __y.__stat_v_, __r);
    return __from_base(__r);
  }

  //! @brief Negation, which is exact and therefore not instrumented
  [[nodiscard]] _CCCL_HOST_DEVICE_API fpmp2_stat operator-() const noexcept
  {
    return __from_base(-__stat_v_);
  }

  // Mixing the wrapped type into an expression. Without these the operand would have to
  // convert - either way round - and the two conversions would be equally good, so the
  // call would be ambiguous. Being exact matches, they also keep the result
  // instrumented.
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator+(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x + __from_base(__y);
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator+(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __from_base(__x) + __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator-(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x - __from_base(__y);
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator-(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __from_base(__x) - __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator*(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x * __from_base(__y);
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator*(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __from_base(__x) * __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator/(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x / __from_base(__y);
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator/(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __from_base(__x) / __y;
  }

  // === compound assignment ===

  _CCCL_HOST_DEVICE_API fpmp2_stat& operator+=(const fpmp2_stat& __other) noexcept
  {
    const base_type __x = __stat_v_;
    __stat_v_ += __other.__stat_v_;
    __trace<__fpmp2_stat_binop::__add>(__x, __other.__stat_v_, __stat_v_);
    return *this;
  }
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator-=(const fpmp2_stat& __other) noexcept
  {
    const base_type __x = __stat_v_;
    __stat_v_ -= __other.__stat_v_;
    __trace<__fpmp2_stat_binop::__sub>(__x, __other.__stat_v_, __stat_v_);
    return *this;
  }
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator*=(const fpmp2_stat& __other) noexcept
  {
    const base_type __x = __stat_v_;
    __stat_v_ *= __other.__stat_v_;
    __trace<__fpmp2_stat_binop::__mul>(__x, __other.__stat_v_, __stat_v_);
    return *this;
  }
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator/=(const fpmp2_stat& __other) noexcept
  {
    const base_type __x = __stat_v_;
    __stat_v_ /= __other.__stat_v_;
    __trace<__fpmp2_stat_binop::__div>(__x, __other.__stat_v_, __stat_v_);
    return *this;
  }

  //! @brief Add a single limb, mirroring the `fpmp2` overload that skips building a pair
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator+=(const _FpType __c) noexcept
  {
    const base_type __x = __stat_v_;
    __stat_v_ += __c;
    __trace<__fpmp2_stat_binop::__add>(__x, base_type{__c}, __stat_v_);
    return *this;
  }
  //! @brief Subtract a single limb
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator-=(const _FpType __c) noexcept
  {
    const base_type __x = __stat_v_;
    __stat_v_ -= __c;
    __trace<__fpmp2_stat_binop::__sub>(__x, base_type{__c}, __stat_v_);
    return *this;
  }

  _CCCL_HOST_DEVICE_API fpmp2_stat& operator++() noexcept
  {
    return *this += _FpType(1);
  }
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator--() noexcept
  {
    return *this -= _FpType(1);
  }
  _CCCL_HOST_DEVICE_API fpmp2_stat operator++(int) noexcept
  {
    const fpmp2_stat __old = *this;
    ++(*this);
    return __old;
  }
  _CCCL_HOST_DEVICE_API fpmp2_stat operator--(int) noexcept
  {
    const fpmp2_stat __old = *this;
    --(*this);
    return __old;
  }

  // === comparisons ===

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator==(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    return __x.__stat_v_ == __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator!=(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    return __x.__stat_v_ != __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    return __x.__stat_v_ < __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    return __x.__stat_v_ > __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<=(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    return __x.__stat_v_ <= __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>=(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    return __x.__stat_v_ >= __y.__stat_v_;
  }

  // Comparing against the wrapped type, for the same ambiguity reason as the arithmetic
  // overloads above.
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator==(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x.__stat_v_ == __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator==(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __x == __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator!=(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x.__stat_v_ != __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator!=(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __x != __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x.__stat_v_ < __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __x < __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x.__stat_v_ > __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __x > __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<=(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x.__stat_v_ <= __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<=(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __x <= __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>=(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x.__stat_v_ >= __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>=(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __x >= __y.__stat_v_;
  }

  // === mixed arithmetic with built-in scalars ===
  // Same shape as the fpmp2 overloads: the scalar is promoted to the pair type, so
  // `2.0f * x` and `x / 3` behave as they would for the wrapped type, instrumented.

  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator+(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x) + fpmp2_stat(__y);
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator-(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x) - fpmp2_stat(__y);
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator*(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x) * fpmp2_stat(__y);
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator/(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x) / fpmp2_stat(__y);
  }

  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator==(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x).__stat_v_ == fpmp2_stat(__y).__stat_v_;
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator!=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x).__stat_v_ != fpmp2_stat(__y).__stat_v_;
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x).__stat_v_ < fpmp2_stat(__y).__stat_v_;
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x).__stat_v_ > fpmp2_stat(__y).__stat_v_;
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x).__stat_v_ <= fpmp2_stat(__y).__stat_v_;
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x).__stat_v_ >= fpmp2_stat(__y).__stat_v_;
  }
};

// Trait: detect any specialization of fpmp2_stat<_FpType, _TypeAcc>, the counterpart of
// __fpmp_is_fpmp2_v.
template <class _Tp>
inline constexpr bool __fpmp_is_fpmp2_stat_v = false;
template <class _FpType, fpmp2_accuracy _TypeAcc>
inline constexpr bool __fpmp_is_fpmp2_stat_v<fpmp2_stat<_FpType, _TypeAcc>> = true;

// === math free functions mirroring the fpmp2 ones ===
// None of these are instrumented: they are composites, and counting the operations
// inside them would drown the counters of the surrounding algorithm.

//! @brief Square root
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
sqrt(const fpmp2_stat<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(sqrt(__x.as_fpmp2()));
}

//! @brief Reciprocal square root
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
rsqrt(const fpmp2_stat<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(rsqrt(__x.as_fpmp2()));
}

//! @brief Fused multiply-add
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
fma(const fpmp2_stat<_FpType, _TypeAcc>& __x,
    const fpmp2_stat<_FpType, _TypeAcc>& __y,
    const fpmp2_stat<_FpType, _TypeAcc>& __z) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(fma(__x.as_fpmp2(), __y.as_fpmp2(), __z.as_fpmp2()));
}

//! @brief Multiply-add
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
mad(const fpmp2_stat<_FpType, _TypeAcc>& __x,
    const fpmp2_stat<_FpType, _TypeAcc>& __y,
    const fpmp2_stat<_FpType, _TypeAcc>& __z) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(mad(__x.as_fpmp2(), __y.as_fpmp2(), __z.as_fpmp2()));
}

//! @brief Fused multiply-add with built-in scalars mixed in
_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(
  ((__fpmp_is_fpmp2_stat_v<_T1> || __fpmp_is_fpmp2_stat_v<_T2> || __fpmp_is_fpmp2_stat_v<_T3>)
   && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>) ))
[[nodiscard]] _CCCL_HOST_DEVICE_API inline auto fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using __stat = ::cuda::std::
    conditional_t<__fpmp_is_fpmp2_stat_v<_T1>, _T1, ::cuda::std::conditional_t<__fpmp_is_fpmp2_stat_v<_T2>, _T2, _T3>>;
  return fma(__stat(__x), __stat(__y), __stat(__z));
}

//! @brief Multiply-add with built-in scalars mixed in
_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(
  ((__fpmp_is_fpmp2_stat_v<_T1> || __fpmp_is_fpmp2_stat_v<_T2> || __fpmp_is_fpmp2_stat_v<_T3>)
   && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>) ))
[[nodiscard]] _CCCL_HOST_DEVICE_API inline auto mad(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using __stat = ::cuda::std::
    conditional_t<__fpmp_is_fpmp2_stat_v<_T1>, _T1, ::cuda::std::conditional_t<__fpmp_is_fpmp2_stat_v<_T2>, _T2, _T3>>;
  return mad(__stat(__x), __stat(__y), __stat(__z));
}

#if _CCCL_CUDA_COMPILATION()
// === atomics ===
// Atomic accumulation into a shared or global value, mirroring the fpmp2 overloads and
// returning the old value as they do. These are instrumented, since they perform the
// arithmetic the counters are about, with one caveat: the result summary is the sum
// recomputed from the returned old value rather than one observed inside the atomic, so
// a value that another thread's update changed in between is summarized as this thread
// computed it.

template <class _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
atomicAdd(fpmp2_stat<_FpType, _TypeAcc>* __address, const fpmp2_stat<_FpType, _TypeAcc>& __val) noexcept
{
  const fpmp2<_FpType, _TypeAcc> __old = atomicAdd(&__address->as_fpmp2(), __val.as_fpmp2());
  __fpmp2_stat_note_binop<__fpmp2_stat_binop::__add>(__old, __val.as_fpmp2(), __old + __val.as_fpmp2());
  return fpmp2_stat<_FpType, _TypeAcc>(__old);
}

template <class _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
atomicSub(fpmp2_stat<_FpType, _TypeAcc>* __address, const fpmp2_stat<_FpType, _TypeAcc>& __val) noexcept
{
  const fpmp2<_FpType, _TypeAcc> __old = atomicSub(&__address->as_fpmp2(), __val.as_fpmp2());
  __fpmp2_stat_note_binop<__fpmp2_stat_binop::__sub>(__old, __val.as_fpmp2(), __old - __val.as_fpmp2());
  return fpmp2_stat<_FpType, _TypeAcc>(__old);
}

// === warp shuffles ===
// Overloads of CUDA's __shfl_sync family, mirroring the fpmp2 ones so that a kernel
// written against the wrapped type keeps compiling after the swap. Thread-cooperation
// primitives, not arithmetic, so they are not instrumented.

template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
__shfl_sync(unsigned mask, const fpmp2_stat<_FpType, _TypeAcc>& var, int srcLane, int width = warpSize) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(__shfl_sync(mask, var.as_fpmp2(), srcLane, width));
}

template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
__shfl_xor_sync(unsigned mask, const fpmp2_stat<_FpType, _TypeAcc>& var, int laneMask, int width = warpSize) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(__shfl_xor_sync(mask, var.as_fpmp2(), laneMask, width));
}

template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc> __shfl_down_sync(
  unsigned mask, const fpmp2_stat<_FpType, _TypeAcc>& var, unsigned int delta, int width = warpSize) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(__shfl_down_sync(mask, var.as_fpmp2(), delta, width));
}

template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc> __shfl_up_sync(
  unsigned mask, const fpmp2_stat<_FpType, _TypeAcc>& var, unsigned int delta, int width = warpSize) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(__shfl_up_sync(mask, var.as_fpmp2(), delta, width));
}
#endif // _CCCL_CUDA_COMPILATION()

// === type aliases ===

//! @brief Instrumented double-float, default accuracy
using fp32mp2_stat = fpmp2_stat<float, fpmp2_accuracy::def>;
//! @brief Instrumented double-float, `low` accuracy
using fp32mp2_stat_low = fpmp2_stat<float, fpmp2_accuracy::low>;
//! @brief Instrumented double-float, `mid` accuracy
using fp32mp2_stat_mid = fpmp2_stat<float, fpmp2_accuracy::mid>;
//! @brief Instrumented double-float, `high` accuracy
using fp32mp2_stat_high = fpmp2_stat<float, fpmp2_accuracy::high>;

//! @brief Instrumented double-double, default accuracy
using fp64mp2_stat = fpmp2_stat<double, fpmp2_accuracy::def>;
//! @brief Instrumented double-double, `low` accuracy
using fp64mp2_stat_low = fpmp2_stat<double, fpmp2_accuracy::low>;
//! @brief Instrumented double-double, `mid` accuracy
using fp64mp2_stat_mid = fpmp2_stat<double, fpmp2_accuracy::mid>;
//! @brief Instrumented double-double, `high` accuracy
using fp64mp2_stat_high = fpmp2_stat<double, fpmp2_accuracy::high>;

// The drop-in promise in memory: same footprint as the wrapped type, and copyable with
// a plain memcpy, so buffers can be reinterpreted between the two.
static_assert(sizeof(fp32mp2_stat) == sizeof(fp32mp2) && alignof(fp32mp2_stat) == alignof(fp32mp2));
static_assert(sizeof(fp64mp2_stat) == sizeof(fp64mp2) && alignof(fp64mp2_stat) == alignof(fp64mp2));
static_assert(::cuda::std::is_trivially_copyable_v<fp32mp2_stat>);
static_assert(::cuda::std::is_trivially_copyable_v<fp64mp2_stat>);
} // namespace cuda::experimental

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//! @brief numeric_limits for the instrumented types
//!
//! Inherits every characteristic from the wrapped type's specialization and only
//! rewraps the values it hands out, so `numeric_limits<fp32mp2_stat>::epsilon()` reports
//! the same number as `numeric_limits<fp32mp2>::epsilon()`.
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
class numeric_limits<::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>>
    : public numeric_limits<::cuda::experimental::fpmp2<_FpType, _TypeAcc>>
{
private:
  using __base = numeric_limits<::cuda::experimental::fpmp2<_FpType, _TypeAcc>>;

public:
  using type = ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>;

  _CCCL_HOST_DEVICE_API static constexpr type min() noexcept
  {
    return type(__base::min());
  }
  _CCCL_HOST_DEVICE_API static constexpr type max() noexcept
  {
    return type(__base::max());
  }
  _CCCL_HOST_DEVICE_API static constexpr type lowest() noexcept
  {
    return type(__base::lowest());
  }
  _CCCL_HOST_DEVICE_API static constexpr type epsilon() noexcept
  {
    return type(__base::epsilon());
  }
  _CCCL_HOST_DEVICE_API static constexpr type round_error() noexcept
  {
    return type(__base::round_error());
  }
  _CCCL_HOST_DEVICE_API static constexpr type infinity() noexcept
  {
    return type(__base::infinity());
  }
  _CCCL_HOST_DEVICE_API static constexpr type quiet_NaN() noexcept
  {
    return type(__base::quiet_NaN());
  }
  _CCCL_HOST_DEVICE_API static constexpr type signaling_NaN() noexcept
  {
    return type(__base::signaling_NaN());
  }
  _CCCL_HOST_DEVICE_API static constexpr type denorm_min() noexcept
  {
    return type(__base::denorm_min());
  }
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPTOOL_STAT_H
