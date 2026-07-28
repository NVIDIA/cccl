//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_IMPL_ATOMIC_H
#define _CUDA___FP_FPMP_IMPL_ATOMIC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_impl_atomic.h - fpmp2 atomic operations (CUDA device only)
    ==================================================================================================
    Per-operation implementation core split out of <cuda/__fp/fpmp_impl.h>. It carries the
    atomic operations (CUDA device only)
    for the fpmp2 double-word type, for both the header-only (inline) mode and the library
    (_CCCL_FPMP_USE_LIB) mode. All shared macros, the fp128 vocabulary type, and the __fpmp_*
    error-free-transform primitives live in <cuda/__fp/fpmp_impl.h>, which this header includes.
*/

#include <cuda/__atomic/atomic.h> // dd atomics use cuda::atomic_ref for the 128-bit compare-exchange
#include <cuda/__fp/fpmp_impl.h>
#include <cuda/__fp/fpmp_impl_muladd.h> // dd atomics reuse __fpmp2_high_add (muladd family)

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if !(defined _CCCL_FPMP_USE_LIB)
/*
 * --------------------------------------------------------------------
 * Atomic operations (CUDA device only)
 * --------------------------------------------------------------------
 */
#  ifdef __CUDACC__
/*
 * --------------------------------------------------------------------
 * Atomic operations - Primary template declarations
 * --------------------------------------------------------------------
 */
// Primary template declarations (specialized for float and double below)
template <typename _FpType>
_CCCL_FPMP_CORE_DEVICE_API inline void __fpmp2_atomicAdd(
  _FpType* __address_hi,
  _FpType* __address_lo,
  const _FpType __addition_hi,
  const _FpType __addition_lo,
  _FpType* __old_hi,
  _FpType* __old_lo) noexcept;

template <typename _FpType>
_CCCL_FPMP_CORE_DEVICE_API inline void __fpmp2_atomicSub(
  _FpType* __address_hi,
  _FpType* __address_lo,
  const _FpType __val_hi,
  const _FpType __val_lo,
  _FpType* __old_hi,
  _FpType* __old_lo) noexcept;

/*
 * --------------------------------------------------------------------
 * Atomic operations - Float (fp32) specializations
 * --------------------------------------------------------------------
 */
// atomicAdd for float (fp32mp2): Uses 64-bit atomicCAS
// Two floats = 64 bits fits in unsigned long long int
// Returns the old value before the addition
template <>
_CCCL_FPMP_CORE_DEVICE_API inline void __fpmp2_atomicAdd<float>(
  float* __address_hi,
  float* __address_lo,
  const float __addition_hi,
  const float __addition_lo,
  float* __old_hi,
  float* __old_lo) noexcept
{
  // Treat the two floats as a single 64-bit value for atomic operations
  // The address must be 8-byte aligned (guaranteed by alignas(2*alignof(float)) in the class)
  static_assert(sizeof(float) * 2 == sizeof(unsigned long long int),
                "Two floats must equal one unsigned long long int");

  unsigned long long int* __address_as_ull = reinterpret_cast<unsigned long long int*>(__address_hi);
  unsigned long long int __old             = *__address_as_ull;
  unsigned long long int __assumed;

  // Use the atomicCAS loop with retries to ensure atomicity
  do
  {
    __assumed = __old;

    // Extract old values from the 64-bit integer
    uint32_t __old_hi_bits = static_cast<uint32_t>(__assumed & 0xFFFFFFFFULL);
    uint32_t __old_lo_bits = static_cast<uint32_t>((__assumed >> 32) & 0xFFFFFFFFULL);
    float __old_hi_val     = ::cuda::std::bit_cast<float>(__old_hi_bits);
    float __old_lo_val     = ::cuda::std::bit_cast<float>(__old_lo_bits);

    // Perform addition based on method
    float __new_hi;
    float __new_lo;
    __fpmp2_high_add(__old_hi_val, __old_lo_val, __addition_hi, __addition_lo, &__new_hi, &__new_lo);

    // Pack new values into a 64-bit integer
    uint32_t __new_hi_bits = ::cuda::std::bit_cast<uint32_t>(__new_hi);
    uint32_t __new_lo_bits = ::cuda::std::bit_cast<uint32_t>(__new_lo);
    unsigned long long int __new_ull =
      static_cast<unsigned long long int>(__new_hi_bits) | (static_cast<unsigned long long int>(__new_lo_bits) << 32);

    __old = atomicCAS(__address_as_ull, __assumed, __new_ull);
  } while (__assumed != __old);

  // Return old value - extract from the final 'old' value
  uint32_t __old_hi_bits = static_cast<uint32_t>(__old & 0xFFFFFFFFULL);
  uint32_t __old_lo_bits = static_cast<uint32_t>((__old >> 32) & 0xFFFFFFFFULL);
  *__old_hi              = ::cuda::std::bit_cast<float>(__old_hi_bits);
  *__old_lo              = ::cuda::std::bit_cast<float>(__old_lo_bits);
}

// atomicSub for float: Uses negation and atomicAdd
template <>
_CCCL_FPMP_CORE_DEVICE_API inline void __fpmp2_atomicSub<float>(
  float* __address_hi,
  float* __address_lo,
  const float __val_hi,
  const float __val_lo,
  float* __old_hi,
  float* __old_lo) noexcept
{
  // Negate the value and call atomicAdd with the same method
  __fpmp2_atomicAdd<float>(__address_hi, __address_lo, -__val_hi, -__val_lo, __old_hi, __old_lo);
}

/*
 * --------------------------------------------------------------------
 * Atomic operations - Double (fp64) specializations
 * --------------------------------------------------------------------
 */
// Declared but never defined, so that calling the double-double atomics on a target
// without a 128-bit compare-exchange is a link-time error naming the requirement rather
// than a silent no-op. Only callers pay: this body is discarded when nobody uses it.
// Mirrors __atomic_cas_128b_unsupported_before_SM_90 in <cuda/std/atomic>.
extern "C" _CCCL_DEVICE void __fpmp2_dd_atomic_requires_SM_90_and_ptx_isa_840();

// Bit image of a double-double, laid out like the class (hi first), so that a whole
// fp64mp2 can be exchanged as a single 128-bit word.
struct alignas(16) __fpmp2_dd_bits
{
  uint64_t __hi;
  uint64_t __lo;
};

// atomicAdd for double (fp64mp2): Uses a 128-bit compare-exchange
// Two doubles = 128 bits requires sm_90+ (Hopper architecture) and PTX ISA 8.4+
// Returns the old value before the addition
template <>
_CCCL_FPMP_CORE_DEVICE_API inline void __fpmp2_atomicAdd<double>(
  double* __address_hi,
  double* __address_lo,
  const double __addition_hi,
  const double __addition_lo,
  double* __old_hi,
  double* __old_lo) noexcept
{
#    if __CUDA_ARCH__ >= 900 && __cccl_ptx_isa >= 840

  // Treat the two doubles as a single 128-bit value for atomic operations
  // The address must be 16-byte aligned for 128-bit atomics
  static_assert(sizeof(double) * 2 == sizeof(__fpmp2_dd_bits), "Two doubles must equal one 128-bit word");

  // The 128-bit CAS goes through cuda::atomic_ref rather than atomicCAS: the ulonglong2
  // overload of atomicCAS lives in CUDA's sm_90_rt.hpp, which clang-CUDA never includes.
  ::cuda::atomic_ref<__fpmp2_dd_bits, ::cuda::thread_scope_device> __address_as_bits{
    *reinterpret_cast<__fpmp2_dd_bits*>(__address_hi)};

  __fpmp2_dd_bits __assumed = __address_as_bits.load(::cuda::std::memory_order_relaxed);
  __fpmp2_dd_bits __new_bits;

  // Use the compare-exchange loop with retries to ensure atomicity
  do
  {
    // Extract old values from the 128-bit structure
    double __old_hi_val = ::cuda::std::bit_cast<double>(__assumed.__hi);
    double __old_lo_val = ::cuda::std::bit_cast<double>(__assumed.__lo);

    // Perform addition based on method
    double __new_hi;
    double __new_lo;
    __fpmp2_high_add(__old_hi_val, __old_lo_val, __addition_hi, __addition_lo, &__new_hi, &__new_lo);

    // Pack new values into a 128-bit structure
    __new_bits.__hi = ::cuda::std::bit_cast<uint64_t>(__new_hi);
    __new_bits.__lo = ::cuda::std::bit_cast<uint64_t>(__new_lo);

    // On failure __assumed receives the current value, so the retry re-adds to it
  } while (!__address_as_bits.compare_exchange_weak(__assumed, __new_bits, ::cuda::std::memory_order_relaxed));

  // Return old value - __assumed holds the value that the successful exchange replaced
  *__old_hi = ::cuda::std::bit_cast<double>(__assumed.__hi);
  *__old_lo = ::cuda::std::bit_cast<double>(__assumed.__lo);
#    else
  // The 128-bit compare-exchange requires sm_90+ (Hopper architecture) and PTX ISA 8.4+
  (void) __address_hi;
  (void) __address_lo;
  (void) __addition_hi;
  (void) __addition_lo;
  (void) __old_hi;
  (void) __old_lo;
  __fpmp2_dd_atomic_requires_SM_90_and_ptx_isa_840();
#    endif
}

// atomicSub for double: Uses negation and atomicAdd
template <>
_CCCL_FPMP_CORE_DEVICE_API inline void __fpmp2_atomicSub<double>(
  double* __address_hi,
  double* __address_lo,
  const double __val_hi,
  const double __val_lo,
  double* __old_hi,
  double* __old_lo) noexcept
{
  // Negate the value and call atomicAdd with the same method
  __fpmp2_atomicAdd<double>(__address_hi, __address_lo, -__val_hi, -__val_lo, __old_hi, __old_lo);
}

#  endif // __CUDACC__
#else // _CCCL_FPMP_USE_LIB

// -- fp32 (single precision) built-in declarations --
#  ifdef __CUDACC__
_CCCL_FPMP_BUILTIN_DEVICE_DECL void __fp32mp2_atomicAdd(
  float* __address_hi,
  float* __address_lo,
  const float __addition_hi,
  const float __addition_lo,
  float* __old_hi,
  float* __old_lo) noexcept;
_CCCL_FPMP_BUILTIN_DEVICE_DECL void __fp32mp2_atomicSub(
  float* __address_hi,
  float* __address_lo,
  const float __val_hi,
  const float __val_lo,
  float* __old_hi,
  float* __old_lo) noexcept;
#  endif // __CUDACC__

// -- fp64 (double precision) built-in declarations --
#  ifdef __CUDACC__
_CCCL_FPMP_BUILTIN_DEVICE_DECL void __fp64mp2_atomicAdd(
  double* __address_hi,
  double* __address_lo,
  const double __addition_hi,
  const double __addition_lo,
  double* __old_hi,
  double* __old_lo) noexcept;
_CCCL_FPMP_BUILTIN_DEVICE_DECL void __fp64mp2_atomicSub(
  double* __address_hi,
  double* __address_lo,
  const double __val_hi,
  const double __val_lo,
  double* __old_hi,
  double* __old_lo) noexcept;
#  endif // __CUDACC__

// -- type-generic template declarations (dispatch to fp32/fp64) --
#  ifdef __CUDACC__
template <typename _Tp>
_CCCL_FPMP_CORE_DEVICE_API inline void __fpmp2_atomicAdd(
  _Tp* __address_hi,
  _Tp* __address_lo,
  const _Tp __addition_hi,
  const _Tp __addition_lo,
  _Tp* __old_hi,
  _Tp* __old_lo) noexcept;
template <typename _Tp>
_CCCL_FPMP_CORE_DEVICE_API inline void __fpmp2_atomicSub(
  _Tp* __address_hi, _Tp* __address_lo, const _Tp __val_hi, const _Tp __val_lo, _Tp* __old_hi, _Tp* __old_lo) noexcept;
#  endif

// -- fp32 template specializations --
#  ifdef __CUDACC__
template <>
_CCCL_FPMP_CORE_DEVICE_API inline void __fpmp2_atomicAdd<float>(
  float* __address_hi,
  float* __address_lo,
  const float __addition_hi,
  const float __addition_lo,
  float* __old_hi,
  float* __old_lo) noexcept
{
  __fp32mp2_atomicAdd(__address_hi, __address_lo, __addition_hi, __addition_lo, __old_hi, __old_lo);
}
template <>
_CCCL_FPMP_CORE_DEVICE_API inline void __fpmp2_atomicSub<float>(
  float* __address_hi,
  float* __address_lo,
  const float __val_hi,
  const float __val_lo,
  float* __old_hi,
  float* __old_lo) noexcept
{
  __fp32mp2_atomicSub(__address_hi, __address_lo, __val_hi, __val_lo, __old_hi, __old_lo);
}
#  endif // __CUDACC__

// -- fp64 template specializations --
#  ifdef __CUDACC__
template <>
_CCCL_FPMP_CORE_DEVICE_API inline void __fpmp2_atomicAdd<double>(
  double* __address_hi,
  double* __address_lo,
  const double __addition_hi,
  const double __addition_lo,
  double* __old_hi,
  double* __old_lo) noexcept
{
  __fp64mp2_atomicAdd(__address_hi, __address_lo, __addition_hi, __addition_lo, __old_hi, __old_lo);
}
template <>
_CCCL_FPMP_CORE_DEVICE_API inline void __fpmp2_atomicSub<double>(
  double* __address_hi,
  double* __address_lo,
  const double __val_hi,
  const double __val_lo,
  double* __old_hi,
  double* __old_lo) noexcept
{
  __fp64mp2_atomicSub(__address_hi, __address_lo, __val_hi, __val_lo, __old_hi, __old_lo);
}
#  endif // __CUDACC__

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_IMPL_ATOMIC_H
