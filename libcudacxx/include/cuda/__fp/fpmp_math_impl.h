//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_H
#define _CUDA___FP_FPMP_MATH_IMPL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl.h - shared internals for the fpmp2 math families
    ==================================================================================================
    Common building blocks for the transcendental / special math functions on the
    fpmp2 types: polynomial evaluation, error-free transforms, constants, argument
    reduction, the fp32mp2 kernel helpers, the placeholder factory macros, and the
    fp128 (fp64mp2 <double>) fallback scaffolding + macros.

    Included by <cuda/__fp/fpmp_math.h> (header mode) and by each
    fpmp_math_impl_<family>.h header. Cross-family kernel dependencies are
    satisfied by each family header #including the sibling families it calls
    (e.g. pow/trig/hyperbolic include exp), so no central forward declarations
    are needed and every family header is self-contained.
*/

#include <cuda/__fp/fpmp.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

/*
// fp128 math functions fallback to system implementation enabling
//
// The third of the fp128 knobs, after _CCCL_FPMP_FP128_ENABLE and
// _CCCL_FPMP_FP128_DEVICE_OPS in <cuda/__fp/fpmp_impl.h>. Those two decide what the fpmp2
// class looks like; this one only selects the bodies of the fp64mp2 math functions, which
// is why it lives here rather than with them.
//
// Being about bodies, it is decided per pass. The device pass takes the quad path only
// where fp128 arithmetic is device-callable, since those bodies go through
// __fpmp2_to_quad; the host pass of a CUDA compilation stays on the double path, so that
// a .cu file does not silently acquire a libquadmath dependency its host-only counterpart
// never had. Outside CUDA the host decides alone, but only if it has a binary128 math
// backend to decide in favour of: the __float128 type alone is not enough, the bodies also
// need the *q or *l entry points behind it, which an x86_64 clang host without <quadmath.h>
// does not have.
//
// The two passes can therefore disagree on fp64mp2 accuracy. That is the price of not
// forcing -lquadmath on every translation unit built for an architecture with device
// fp128. A program that needs both halves on the quad path says so with the public knob
// CCCL_FPMP_FP128_MATH_FALLBACK from <cuda/__fp/fpmp_common.h>, which maps onto this
// macro and is documented there:
//
//   nvcc -arch=sm_100 -DCCCL_FPMP_FP128_MATH_FALLBACK=1 app.cu -lquadmath
//
// Below sm_100 that also takes _CCCL_FPMP_FP128_DEVICE_OPS=1 and a toolchain that emits
// fp128 there (nvcc -nvvm-version=nvvm-latest); asking for the quad path alone on such a
// target makes the device pass fail to compile, since these bodies then need fp128
// arithmetic the architecture does not have.
//
// In library mode this has to match between the library build and its consumers, like the
// other fp128 knobs.
*/
#ifndef _CCCL_FPMP_FP128_MATH_FALLBACK
#  if (_CCCL_FPMP_FP128_ENABLE == 1)                                                                     \
    && ((!_CCCL_CUDA_COMPILATION()                                                                       \
         && ((_CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH == 1) || (_CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128 == 1))) \
        || (_CCCL_DEVICE_COMPILATION() && (_CCCL_FPMP_FP128_DEVICE_OPS == 1)))
#    define _CCCL_FPMP_FP128_MATH_FALLBACK 1
#  else
#    define _CCCL_FPMP_FP128_MATH_FALLBACK 0
#  endif
#endif

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * Polynomial evaluation helpers
 */

/*********************************************************************
 * Mixed-precision (split-M) Horner polynomial evaluation
 * (internal building block -- namespace `fpmp`)
 *
 *   p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[N-1]*x^(N-1)
 *
 * with x given as fpmp2<FpType, met> and the coefficient
 * table `c[]` packed as fpmp2<FpType, met> in ascending order
 * of degree (c[0] = constant term, c[N-1] = leading coefficient).
 *
 * The template parameter M controls the precision split:
 *
 *   - The M HIGHEST-degree coefficients
 *         c[N-1], c[N-2], ..., c[N-M]
 *     are treated as plain FpType constants. Their `.lo()` parts
 *     are assumed to be zero (which is the natural state when the
 *     coefficient is built from a single FpType literal via the
 *     implicit `fpmp2(FpType)` ctor; using a __ffloat literal
 *     whose `.lo()` happens to be zero -- e.g. for layout
 *     consistency -- works just as well, the `.lo()` is simply
 *     ignored in this phase). The leading M iterations run in
 *     pure FpType arithmetic:
 *         v_f = v_f * x.hi() + c[k].hi()
 *     using the "op" form r = a*b + c (no fma/mad), matching the
 *     pattern used by the hand-written math kernels (e.g. erfc).
 *
 *   - The remaining N - M LOWER-degree coefficients
 *         c[N-M-1], c[N-M-2], ..., c[0]
 *     are evaluated in full ff (float-float) arithmetic. The
 *     transition step
 *         v = v_f * x.hi() + c[N-M-1]
 *     mirrors the float*float + ff layout used in `__fpmp2_erfc`
 *     (the float product gets promoted to ff via the mixed-arithmetic
 *     operator, then the remaining iterations are plain ff Horner).
 *
 * Special cases:
 *   - M == 0     : pure ff Horner (no FpType phase).
 *   - M == N     : pure FpType Horner; the FpType accumulator is
 *                  promoted to __ff_t (lo == 0) at the return point.
 *
 * Use this routine as the "B" side of an A/B switch against
 * `poly_horner_comp`; the call site is identical:
 *     __ffloat v = __fpmp_poly_horner_mixed<M>(x, c);  // mixed standard
 *     __ffloat v = __fpmp_poly_horner_comp    (x, c);  // compensated
 * or dispatch via `__fpmp_poly_eval<strategy, M>(x, c)` below.
 *
 * Template params:
 *   M      : number of high-degree coefficients to evaluate in
 *            plain FpType arithmetic (0 <= M <= N). Must be
 *            supplied explicitly at the call site.
 *   N      : number of coefficients (= array length, deduced).
 *            Polynomial degree is N - 1.
 *   FpType : float or double (deduced from arguments).
 *   met    : fpmp arithmetic accuracy level (deduced from arguments).
 *********************************************************************/
template <int _Mp, int _Np, typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
__fpmp_poly_horner_mixed(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc> (&__c)[_Np]) noexcept
{
  static_assert(_Np >= 2, "poly_horner_mixed requires at least 2 coefficients (degree >= 1)");
  static_assert(_Mp >= 0, "poly_horner_mixed: M must be non-negative");
  static_assert(_Mp <= _Np, "poly_horner_mixed: M must not exceed N");

  using __ff_t = fpmp2<_FpType, _TypeAcc>;

  if constexpr (_Mp == 0)
  {
    // Pure ff Horner -- no FpType phase.
    __ff_t __v = __c[_Np - 1];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __k = _Np - 2; __k >= 0; --__k)
    {
      __v = __v * __x + __c[__k];
    }
    return __v;
  }
  else
  {
    // FpType phase: M iterations consuming c[N-1] ... c[N-M].
    const _FpType __xh = __x.hi();
    _FpType __v_f      = __c[_Np - 1].hi();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __k = _Np - 2; __k >= _Np - _Mp; --__k)
    {
      __v_f = __v_f * __xh + __c[__k].hi();
    }

    if constexpr (_Mp == _Np)
    {
      // No ff phase at all -- promote the FpType result.
      return __ff_t(__v_f);
    }
    else
    {
      // Transition step: (float * float) + ff -> ff
      // (the mixed-type operator+ promotes the FpType product
      // to __ff_t with .lo() == 0 before adding c[N-M-1].)
      __ff_t __v = __v_f * __xh + __c[_Np - _Mp - 1];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __k = _Np - _Mp - 2; __k >= 0; --__k)
      {
        __v = __v * __x + __c[__k];
      }
      return __v;
    }
  }
} // poly_horner_mixed

/*********************************************************************
 * Compensated Horner polynomial evaluation for fp32mp2 / fp64mp2
 * (internal building block -- namespace `fpmp`)
 *
 *   p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[N-1]*x^(N-1)
 *
 * with x and c[k] given as fpmp2<FpType, met> (ascending order
 * of degree, c[0] = constant term, c[N-1] = leading coefficient).
 * The bulk of the work runs in single-precision FpType arithmetic
 * with an error-tracking ("compensated") Horner inner loop, then two
 * correction sweeps fold in the c[k].lo terms and the x.lo * p'(x.hi)
 * cross term.
 *
 * Layout:
 *   Phase 0:  (optional)  plain FpType Horner over the M highest-degree
 *             coefficients c[N-1].hi() ... c[N-M].hi() -- error not
 *             tracked. Use M > 0 to skip compensation on iterations
 *             whose values are small enough that the rounding error
 *             they would contribute is dominated by the polynomial
 *             truncation noise; mirrors the M-split of
 *             `poly_horner_mixed<M>`. Top M coefficients are required
 *             to have c[k].lo() == 0 (the natural state for plain
 *             FpType literals built via `fpmp2(FpType)` ctor).
 *   Phase 1:  compensated Horner over the remaining (N-M) coefficients
 *             c[N-M-1] ... c[0] -- running FpType acc + FpType err such
 *             that acc + err equals Sum_{k<=N-M-1} c[k].hi * x.hi^k + acc0
 *             (where acc0 is Phase 0's output) to ~2*FpType precision
 *             (Graillat-Langlois-Louvet, "Compensated Horner", 2005).
 *   Phase 2a: + Sum_{k<=N-M-1} c[k].lo * x.hi^k   (plain FpType Horner;
 *             top M iterations are skipped, their .lo() == 0).
 *   Phase 2b: + x.lo * p'(x.hi)                (plain FpType Horner
 *             over the full derivative -- all N-1 terms regardless
 *             of M, because the high-degree derivative terms carry
 *             the x.lo correction signal, not rounding noise).
 *   Phase 3:  fast_two_sum(acc, err+corr) -> (hi, lo) __ffloat
 *
 * Coefficients with c[k].lo == 0 (e.g. those built from a pure
 * FpType constant via the implicit `fpmp2(FpType)` ctor)
 * fold cleanly: their Phase 2a iterations are no-ops, and Phase 2b's
 * `(FpType)k * c[k].hi()` constant evaluates at compile time inside
 * the unrolled loop. This makes the helper a uniform way to express
 * mixed-precision polynomials.
 *
 * Special cases:
 *   - M == 0 : full compensated Horner over all N coefficients
 *              (bit-identical to the un-split implementation).
 *   - M == N : pure plain FpType Horner over all N coefficients
 *              with the x.lo * p'(x.hi) cross-term correction --
 *              cheaper than full compensated, more accurate than
 *              `poly_horner_mixed<N>` (which drops the cross term).
 *
 * Template params:
 *   M      : number of HIGH-degree coefficients to evaluate in
 *            plain FpType (no error tracking).  0 <= M <= N.
 *            Defaults to 0 (= full compensated Horner).
 *   N      : number of coefficients (= array length, polynomial
 *            degree is N-1). Deduced from the coefficient array.
 *   FpType : float or double (deduced from arguments)
 *   met    : fpmp arithmetic accuracy level (deduced from arguments)
 *********************************************************************/
template <int _Mp = 0, int _Np, typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
__fpmp_poly_horner_comp(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc> (&__c)[_Np]) noexcept
{
  static_assert(_Np >= 2, "poly_horner_comp requires at least 2 coefficients (degree >= 1)");
  static_assert(_Mp >= 0, "poly_horner_comp: M must be non-negative");
  static_assert(_Mp <= _Np, "poly_horner_comp: M must not exceed N");

  const _FpType __xh = __x.hi();
  const _FpType __xl = __x.lo();

  // === Phase 0: M-1 plain FpType Horner steps (no error tracking) ===
  _FpType __acc = __c[_Np - 1].hi();
  if constexpr (_Mp >= 2)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __k = _Np - 2; __k >= _Np - _Mp; --__k)
    {
      __acc = __acc * __xh + __c[__k].hi();
    }
  }

  // === Phase 1: N-M compensated Horner steps ===
  _FpType __err = static_cast<_FpType>(0);
  if constexpr (_Mp < _Np)
  {
    // For M == 0 the init handled c[N-1], so compensated loop
    // starts at c[N-2]; for M >= 1 Phase 0 handled c[N-1]..c[N-M],
    // so compensated loop starts at c[N-M-1].
    constexpr int __comp_start = (_Mp == 0) ? (_Np - 2) : (_Np - _Mp - 1);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __k = __comp_start; __k >= 0; --__k)
    {
      const _FpType __ckh = __c[__k].hi();

      // two_mult_fma: P + pi == xh * acc  (exact)
      _FpType __pval = __fpmp_mul_rn(__xh, __acc);
      _FpType __pi   = __fpmp_fma_rn(__xh, __acc, -__pval);

      // two_sum: S + sg == P + ckh  (exact, no magnitude assumption)
      _FpType __s  = __fpmp_add_rn(__pval, __ckh);
      _FpType __bb = __fpmp_sub_rn(__s, __pval);
      _FpType __t  = __fpmp_sub_rn(__s, __bb);
      _FpType __u  = __fpmp_sub_rn(__pval, __t);
      _FpType __v  = __fpmp_sub_rn(__ckh, __bb);
      _FpType __sg = __fpmp_add_rn(__u, __v);

      __err = __fpmp_fma_rn(__xh, __err, __fpmp_add_rn(__pi, __sg));
      __acc = __s;
    }
  }

  // === Phase 2a: contribution of c[k].lo (top M iterations skipped) ===
  _FpType __corr = static_cast<_FpType>(0);
  if constexpr (_Mp < _Np)
  {
    // For M == 0 we visit all N coefficients (k = N-1 .. 0);
    // for M >= 1 we skip the top M (their .lo() == 0 by contract).
    constexpr int __lo_start = (_Mp == 0) ? (_Np - 1) : (_Np - _Mp - 1);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __k = __lo_start; __k >= 0; --__k)
    {
      __corr = __fpmp_fma_rn(__xh, __corr, __c[__k].lo());
    }
  }

  // === Phase 2b: x.lo * p'(x.hi)  (full derivative, all N-1 terms) ===
  _FpType __dp = static_cast<_FpType>(0);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int __k = _Np - 1; __k >= 1; --__k)
  {
    __dp = __fpmp_fma_rn(__xh, __dp, __fpmp_mul_rn(static_cast<_FpType>(__k), __c[__k].hi()));
  }
  __corr = __fpmp_fma_rn(__xl, __dp, __corr);

  // === Phase 3: combine into normalized ff ===
  _FpType __lo  = __fpmp_add_rn(__err, __corr);
  _FpType __rhi = __fpmp_add_rn(__acc, __lo);
  _FpType __rlo = __fpmp_sub_rn(__lo, __fpmp_sub_rn(__rhi, __acc));
  return fpmp2<_FpType, _TypeAcc>(__rhi, __rlo);
} // poly_horner_comp

/*********************************************************************
 * Polynomial-evaluation strategy selector for `poly_eval`.
 *
 * Listed kernels currently route to a Horner backend; future
 * additions (e.g. factorized / Estrin / Knuth-Eve evaluation)
 * are expected to slot in as new enumerators here without
 * changing the dispatcher signature.
 *********************************************************************/
enum class __fpmp_poly_method
{
  horner_mixed = 0, // mixed-precision Horner (`poly_horner_mixed<M>`)
  horner_comp  = 1, // compensated  Horner    (`poly_horner_comp`)
};

/*********************************************************************
 * Polynomial-evaluation dispatcher
 * (internal building block -- namespace `fpmp`)
 *
 * Thin compile-time switch between the polynomial-evaluation kernels:
 *
 *     __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, M>(x, c)
 *         -> __fpmp_poly_horner_mixed<M>(x, c)
 *
 *     __fpmp_poly_eval<__fpmp_poly_method::horner_comp,  M>(x, c)
 *         -> __fpmp_poly_horner_comp <M>(x, c)
 *
 * Both backends share the same M-split semantics: the M HIGHEST-degree
 * coefficients are evaluated in plain FpType arithmetic (with the
 * convention that their .lo() == 0), and the remaining N-M coefficients
 * are evaluated in the precision-preserving regime of the selected
 * backend (ff-Horner for `horner_mixed`, error-tracking Horner for
 * `horner_comp`).  Switching between the two costs nothing more than
 * editing the strategy enumerator at the call site.
 *
 * Template params:
 *   strategy : poly_method::horner_mixed or poly_method::horner_comp
 *              (additional non-Horner methods may be added later).
 *   M        : split parameter forwarded to both backends.
 *              Defaults to 0 (= pure ff Horner for `horner_mixed`,
 *              = full compensated Horner for `horner_comp`).
 *   N        : number of coefficients (deduced).
 *   FpType   : float or double (deduced).
 *   met      : fpmp arithmetic accuracy level (deduced).
 *********************************************************************/
template <__fpmp_poly_method _Strategy, int _Mp = 0, int _Np, typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
__fpmp_poly_eval(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc> (&__c)[_Np]) noexcept
{
  if constexpr (_Strategy == __fpmp_poly_method::horner_mixed)
  {
    return __fpmp_poly_horner_mixed<_Mp>(__x, __c);
  }
  else /* poly_method::horner_comp */
  {
    return __fpmp_poly_horner_comp<_Mp>(__x, __c);
  }
} // poly_eval

/*
 * ============================================================================
 * Math function fallback bodies (fp32mp2)
 * ============================================================================
 * fp32mp2 implementations that delegate to the standard double-precision system
 * function with proper hi/lo splitting. Each defines the __internal_fpmp2_<name>
 * overload taking float; the fp64mp2 overload and the entry point sit next to
 * the invocation (see the dispatch macros below).
 * ============================================================================
 */

/* Defensive #undef for all fallback macros: protect against the (unlikely) case
 * that something earlier in the translation unit already defined them. They stay
 * defined for the per-family headers to invoke and are undefined again at the end
 * of the include block in fpmp_math.h. */
#  undef _CCCL_FPMP_MATH_FALLBACK_1A
#  undef _CCCL_FPMP_MATH_FALLBACK_2A
#  undef _CCCL_FPMP_MATH_FALLBACK_1A_RETLL
#  undef _CCCL_FPMP_MATH_FALLBACK_1A_RETL

// Uses explicit fpmp2 construction/conversion to avoid NVCC name resolution issues.
#  define _CCCL_FPMP_MATH_FALLBACK_1A(name)                                              \
    _CCCL_FPMP_CORE_API void __internal_fpmp2_##name(                                    \
      const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept \
    {                                                                                    \
      using __mp2_t = fpmp2<float>;                                                      \
      double __r    = ::name(static_cast<double>(__mp2_t(__x_hi, __x_lo)));              \
      __mp2_t __result(__r);                                                             \
      *__res_hi = __result.hi();                                                         \
      *__res_lo = __result.lo();                                                         \
    }

#  define _CCCL_FPMP_MATH_FALLBACK_2A(name)                                                                            \
    _CCCL_FPMP_CORE_API void __internal_fpmp2_##name(                                                                  \
      const float __x_hi,                                                                                              \
      const float __x_lo,                                                                                              \
      const float __y_hi,                                                                                              \
      const float __y_lo,                                                                                              \
      float* __res_hi,                                                                                                 \
      float* __res_lo) noexcept                                                                                        \
    {                                                                                                                  \
      using __mp2_t = fpmp2<float>;                                                                                    \
      double __r = ::name(static_cast<double>(__mp2_t(__x_hi, __x_lo)), static_cast<double>(__mp2_t(__y_hi, __y_lo))); \
      __mp2_t __result(__r);                                                                                           \
      *__res_hi = __result.hi();                                                                                       \
      *__res_lo = __result.lo();                                                                                       \
    }

#  define _CCCL_FPMP_MATH_FALLBACK_1A_RETLL(name)                                                              \
    _CCCL_FPMP_CORE_API long long int __internal_fpmp2_##name(const float __x_hi, const float __x_lo) noexcept \
    {                                                                                                          \
      using __mp2_t = fpmp2<float>;                                                                            \
      return ::name(static_cast<double>(__mp2_t(__x_hi, __x_lo)));                                             \
    }

#  define _CCCL_FPMP_MATH_FALLBACK_1A_RETL(name)                                                          \
    _CCCL_FPMP_CORE_API long int __internal_fpmp2_##name(const float __x_hi, const float __x_lo) noexcept \
    {                                                                                                     \
      using __mp2_t = fpmp2<float>;                                                                       \
      return ::name(static_cast<double>(__mp2_t(__x_hi, __x_lo)));                                        \
    }

/*
 * ============================================================================
 * Per-type dispatch for the math implementations
 * ============================================================================
 * A math function that needs a dedicated implementation per element type is
 * written as two non-template overloads named __internal_fpmp2_<name>, one
 * taking float (fp32mp2) and one taking double (fp64mp2), followed by the entry
 * point __fpmp2_<name> generated by one of the macros below. The two
 * implementations therefore sit side by side as peers, and every entry point
 * rejects an unsupported element type with the same message.
 *
 * Functions that are exact on the limb pair (floor, ldexp, copysign, ...) serve
 * both element types from a single template and do not use these macros.
 *
 * The suffix names the signature layout: 1A / 2A / 3A / 4A count the (hi, lo)
 * input pairs, RET* gives a return type other than a pair, INT_FP puts an int
 * argument first, 2OUT returns two pairs and QUO appends the remquo quotient.
 * ============================================================================
 */

#  undef _CCCL_FPMP_MATH_ONLY_FP32_FP64
#  undef _CCCL_FPMP_MATH_DISPATCH_1A
#  undef _CCCL_FPMP_MATH_DISPATCH_2A
#  undef _CCCL_FPMP_MATH_DISPATCH_2A_YX
#  undef _CCCL_FPMP_MATH_DISPATCH_3A
#  undef _CCCL_FPMP_MATH_DISPATCH_4A
#  undef _CCCL_FPMP_MATH_DISPATCH_1A_RETINT
#  undef _CCCL_FPMP_MATH_DISPATCH_1A_RETLL
#  undef _CCCL_FPMP_MATH_DISPATCH_1A_RETL
#  undef _CCCL_FPMP_MATH_DISPATCH_1A_2OUT
#  undef _CCCL_FPMP_MATH_DISPATCH_INT_FP
#  undef _CCCL_FPMP_MATH_DISPATCH_2A_QUO

#  define _CCCL_FPMP_MATH_ONLY_FP32_FP64(name) \
    "__fpmp2_" #name " is implemented for float (fp32mp2) and double (fp64mp2) only"

// The dispatch is written as if constexpr rather than a plain call guarded by a
// static_assert so that an unsupported element type produces this one message
// instead of trailing it with an overload-resolution failure.
#  define _CCCL_FPMP_MATH_DISPATCH_1A(name)                                                          \
    template <typename _FpType>                                                                      \
    _CCCL_FPMP_CORE_API void __fpmp2_##name(                                                         \
      const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept     \
    {                                                                                                \
      if constexpr (__fpmp2_is_supported_fp_v<_FpType>)                                              \
      {                                                                                              \
        __internal_fpmp2_##name(__x_hi, __x_lo, __res_hi, __res_lo);                                 \
      }                                                                                              \
      else                                                                                           \
      {                                                                                              \
        static_assert(::cuda::std::__always_false_v<_FpType>, _CCCL_FPMP_MATH_ONLY_FP32_FP64(name)); \
      }                                                                                              \
    }

#  define _CCCL_FPMP_MATH_DISPATCH_2A(name)                                                          \
    template <typename _FpType>                                                                      \
    _CCCL_FPMP_CORE_API void __fpmp2_##name(                                                         \
      const _FpType __x_hi,                                                                          \
      const _FpType __x_lo,                                                                          \
      const _FpType __y_hi,                                                                          \
      const _FpType __y_lo,                                                                          \
      _FpType* __res_hi,                                                                             \
      _FpType* __res_lo) noexcept                                                                    \
    {                                                                                                \
      if constexpr (__fpmp2_is_supported_fp_v<_FpType>)                                              \
      {                                                                                              \
        __internal_fpmp2_##name(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);                 \
      }                                                                                              \
      else                                                                                           \
      {                                                                                              \
        static_assert(::cuda::std::__always_false_v<_FpType>, _CCCL_FPMP_MATH_ONLY_FP32_FP64(name)); \
      }                                                                                              \
    }

// Same as 2A, for a function whose first pair is the y argument (atan2).
#  define _CCCL_FPMP_MATH_DISPATCH_2A_YX(name)                                                       \
    template <typename _FpType>                                                                      \
    _CCCL_FPMP_CORE_API void __fpmp2_##name(                                                         \
      const _FpType __y_hi,                                                                          \
      const _FpType __y_lo,                                                                          \
      const _FpType __x_hi,                                                                          \
      const _FpType __x_lo,                                                                          \
      _FpType* __res_hi,                                                                             \
      _FpType* __res_lo) noexcept                                                                    \
    {                                                                                                \
      if constexpr (__fpmp2_is_supported_fp_v<_FpType>)                                              \
      {                                                                                              \
        __internal_fpmp2_##name(__y_hi, __y_lo, __x_hi, __x_lo, __res_hi, __res_lo);                 \
      }                                                                                              \
      else                                                                                           \
      {                                                                                              \
        static_assert(::cuda::std::__always_false_v<_FpType>, _CCCL_FPMP_MATH_ONLY_FP32_FP64(name)); \
      }                                                                                              \
    }

#  define _CCCL_FPMP_MATH_DISPATCH_3A(name)                                                          \
    template <typename _FpType>                                                                      \
    _CCCL_FPMP_CORE_API void __fpmp2_##name(                                                         \
      const _FpType __a_hi,                                                                          \
      const _FpType __a_lo,                                                                          \
      const _FpType __b_hi,                                                                          \
      const _FpType __b_lo,                                                                          \
      const _FpType __c_hi,                                                                          \
      const _FpType __c_lo,                                                                          \
      _FpType* __res_hi,                                                                             \
      _FpType* __res_lo) noexcept                                                                    \
    {                                                                                                \
      if constexpr (__fpmp2_is_supported_fp_v<_FpType>)                                              \
      {                                                                                              \
        __internal_fpmp2_##name(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __res_hi, __res_lo); \
      }                                                                                              \
      else                                                                                           \
      {                                                                                              \
        static_assert(::cuda::std::__always_false_v<_FpType>, _CCCL_FPMP_MATH_ONLY_FP32_FP64(name)); \
      }                                                                                              \
    }

#  define _CCCL_FPMP_MATH_DISPATCH_4A(name)                                                                          \
    template <typename _FpType>                                                                                      \
    _CCCL_FPMP_CORE_API void __fpmp2_##name(                                                                         \
      const _FpType __a_hi,                                                                                          \
      const _FpType __a_lo,                                                                                          \
      const _FpType __b_hi,                                                                                          \
      const _FpType __b_lo,                                                                                          \
      const _FpType __c_hi,                                                                                          \
      const _FpType __c_lo,                                                                                          \
      const _FpType __d_hi,                                                                                          \
      const _FpType __d_lo,                                                                                          \
      _FpType* __res_hi,                                                                                             \
      _FpType* __res_lo) noexcept                                                                                    \
    {                                                                                                                \
      if constexpr (__fpmp2_is_supported_fp_v<_FpType>)                                                              \
      {                                                                                                              \
        __internal_fpmp2_##name(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __d_hi, __d_lo, __res_hi, __res_lo); \
      }                                                                                                              \
      else                                                                                                           \
      {                                                                                                              \
        static_assert(::cuda::std::__always_false_v<_FpType>, _CCCL_FPMP_MATH_ONLY_FP32_FP64(name));                 \
      }                                                                                                              \
    }

#  define _CCCL_FPMP_MATH_DISPATCH_1A_RETINT(name)                                                   \
    template <typename _FpType>                                                                      \
    _CCCL_FPMP_CORE_API int __fpmp2_##name(const _FpType __x_hi, const _FpType __x_lo) noexcept      \
    {                                                                                                \
      if constexpr (__fpmp2_is_supported_fp_v<_FpType>)                                              \
      {                                                                                              \
        return __internal_fpmp2_##name(__x_hi, __x_lo);                                              \
      }                                                                                              \
      else                                                                                           \
      {                                                                                              \
        static_assert(::cuda::std::__always_false_v<_FpType>, _CCCL_FPMP_MATH_ONLY_FP32_FP64(name)); \
        return 0;                                                                                    \
      }                                                                                              \
    }

#  define _CCCL_FPMP_MATH_DISPATCH_1A_RETLL(name)                                                         \
    template <typename _FpType>                                                                           \
    _CCCL_FPMP_CORE_API long long int __fpmp2_##name(const _FpType __x_hi, const _FpType __x_lo) noexcept \
    {                                                                                                     \
      if constexpr (__fpmp2_is_supported_fp_v<_FpType>)                                                   \
      {                                                                                                   \
        return __internal_fpmp2_##name(__x_hi, __x_lo);                                                   \
      }                                                                                                   \
      else                                                                                                \
      {                                                                                                   \
        static_assert(::cuda::std::__always_false_v<_FpType>, _CCCL_FPMP_MATH_ONLY_FP32_FP64(name));      \
        return 0;                                                                                         \
      }                                                                                                   \
    }

#  define _CCCL_FPMP_MATH_DISPATCH_1A_RETL(name)                                                     \
    template <typename _FpType>                                                                      \
    _CCCL_FPMP_CORE_API long int __fpmp2_##name(const _FpType __x_hi, const _FpType __x_lo) noexcept \
    {                                                                                                \
      if constexpr (__fpmp2_is_supported_fp_v<_FpType>)                                              \
      {                                                                                              \
        return __internal_fpmp2_##name(__x_hi, __x_lo);                                              \
      }                                                                                              \
      else                                                                                           \
      {                                                                                              \
        static_assert(::cuda::std::__always_false_v<_FpType>, _CCCL_FPMP_MATH_ONLY_FP32_FP64(name)); \
        return 0;                                                                                    \
      }                                                                                              \
    }

#  define _CCCL_FPMP_MATH_DISPATCH_1A_2OUT(name)                                                     \
    template <typename _FpType>                                                                      \
    _CCCL_FPMP_CORE_API void __fpmp2_##name(                                                         \
      const _FpType __x_hi,                                                                          \
      const _FpType __x_lo,                                                                          \
      _FpType* __sin_hi,                                                                             \
      _FpType* __sin_lo,                                                                             \
      _FpType* __cos_hi,                                                                             \
      _FpType* __cos_lo) noexcept                                                                    \
    {                                                                                                \
      if constexpr (__fpmp2_is_supported_fp_v<_FpType>)                                              \
      {                                                                                              \
        __internal_fpmp2_##name(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo);             \
      }                                                                                              \
      else                                                                                           \
      {                                                                                              \
        static_assert(::cuda::std::__always_false_v<_FpType>, _CCCL_FPMP_MATH_ONLY_FP32_FP64(name)); \
      }                                                                                              \
    }

#  define _CCCL_FPMP_MATH_DISPATCH_INT_FP(name)                                                                 \
    template <typename _FpType>                                                                                 \
    _CCCL_FPMP_CORE_API void __fpmp2_##name(                                                                    \
      const int __n, const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept \
    {                                                                                                           \
      if constexpr (__fpmp2_is_supported_fp_v<_FpType>)                                                         \
      {                                                                                                         \
        __internal_fpmp2_##name(__n, __x_hi, __x_lo, __res_hi, __res_lo);                                       \
      }                                                                                                         \
      else                                                                                                      \
      {                                                                                                         \
        static_assert(::cuda::std::__always_false_v<_FpType>, _CCCL_FPMP_MATH_ONLY_FP32_FP64(name));            \
      }                                                                                                         \
    }

#  define _CCCL_FPMP_MATH_DISPATCH_2A_QUO(name)                                                      \
    template <typename _FpType>                                                                      \
    _CCCL_FPMP_CORE_API void __fpmp2_##name(                                                         \
      const _FpType __x_hi,                                                                          \
      const _FpType __x_lo,                                                                          \
      const _FpType __y_hi,                                                                          \
      const _FpType __y_lo,                                                                          \
      _FpType* __res_hi,                                                                             \
      _FpType* __res_lo,                                                                             \
      int* __quo) noexcept                                                                           \
    {                                                                                                \
      if constexpr (__fpmp2_is_supported_fp_v<_FpType>)                                              \
      {                                                                                              \
        __internal_fpmp2_##name(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, __quo);          \
      }                                                                                              \
      else                                                                                           \
      {                                                                                              \
        static_assert(::cuda::std::__always_false_v<_FpType>, _CCCL_FPMP_MATH_ONLY_FP32_FP64(name)); \
      }                                                                                              \
    }

/*
 * ============================================================================
 * Double precision (fp64mp2) math backend
 * ============================================================================
 */

// Does the active binary128 backend provide erf/erfc?
//
// CUDA does not, as of 13.0: neither crt/device_fp128_functions.h nor libdevice
// declares __nv_fp128_erf or __nv_fp128_erfc (the declared set stops at exp, log,
// sin, pow, sqrt, fma and friends). libquadmath (erfq/erfcq) and 128-bit long double
// (erfl/erfcl) do have them, so this is a property of the backend and not of the
// function.
//
// The default is 0 on every backend, which keeps fp64mp2 erf and erfc at binary64
// accuracy everywhere and so keeps the host and the device in agreement -- valuable
// on its own, since a result that changes with the compilation pass is far worse than
// one that is uniformly less precise. Setting it to 1 moves every backend to true
// binary128 at once, which is the single change needed once CUDA gains the device
// intrinsics; the host halves already resolve to their quad entry points below.
//
// In library mode this has to match between the library build and its consumers, like
// the other fp128 knobs: it selects which implementation the fp64mp2 entry points get.
#  ifndef _CCCL_FPMP_FP128_QUAD_ERF
#    define _CCCL_FPMP_FP128_QUAD_ERF 0
#  endif

#  if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1)

#    if _CCCL_DEVICE_COMPILATION() && (defined(__aarch64__) || defined(_M_ARM64))     \
      && defined(_CCCL_FPMP_CUDA_FP128_INTRINSICS)                                    \
      && !(defined(__GNUC__) && !defined(__clang__) && !defined(__NVCOMPILER_MAJOR__) \
           && ((__GNUC__ > 13) || (__GNUC__ == 13 && __GNUC_MINOR__ >= 1)))           \
      && !defined(_CCCL_FLOAT128_CPP_SPELLING_ENABLED)
#      define _CCCL_FLOAT128_CPP_SPELLING_ENABLED
#    endif
} // namespace cuda::experimental
#    if _CCCL_DEVICE_COMPILATION()
  // CUDA device
#      include <crt/device_fp128_functions.h>
#    elif (_CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH == 1)
  // x86 host: libquadmath
#      include <quadmath.h>
#    elif (_CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128 == 1)
  // ARM64/s390x host: long double is 128-bit IEEE
#      include <cmath>
#    endif
namespace cuda::experimental
{
// ----------------------------------------------------------------------
// Branch 1 -- CUDA DEVICE with the *extended* NVVM fp128 intrinsics
//             (the primary GPU path).
// Target: device compile built with fp128 spelling that
// the crt header can declare the overloads under (__float128 on x86_64,
// or _Float128 on AArch64 with GCC >= 13.1). Serves both x86_64 and
// AArch64 devices: true binary128 via native __nv_fp128_* for every
// function that has one. Of the rest, cbrt is composed from the native
// pow/fabs/copysign and so stays in binary128, while atan2 -- and erf
// and erfc unless _CCCL_FPMP_FP128_QUAD_ERF -- widen through double.
// ----------------------------------------------------------------------
#    if _CCCL_DEVICE_COMPILATION() && defined(_CCCL_FPMP_CUDA_FP128_INTRINSICS) \
      && (defined(_CCCL_FLOAT128_CPP_SPELLING_ENABLED) || defined(__FLOAT128_C_SPELLING_ENABLED__))
#      define _CCCL_FPMP_EXPQ(x)          __nv_fp128_exp(x)
#      define _CCCL_FPMP_EXP2Q(x)         __nv_fp128_exp2(x)
#      define _CCCL_FPMP_EXP10Q(x)        __nv_fp128_exp10(x)
#      define _CCCL_FPMP_EXPM1Q(x)        __nv_fp128_expm1(x)
#      define _CCCL_FPMP_LOGQ(x)          __nv_fp128_log(x)
#      define _CCCL_FPMP_LOG2Q(x)         __nv_fp128_log2(x)
#      define _CCCL_FPMP_LOG10Q(x)        __nv_fp128_log10(x)
#      define _CCCL_FPMP_LOG1PQ(x)        __nv_fp128_log1p(x)
#      define _CCCL_FPMP_SINQ(x)          __nv_fp128_sin(x)
#      define _CCCL_FPMP_COSQ(x)          __nv_fp128_cos(x)
#      define _CCCL_FPMP_TANQ(x)          __nv_fp128_tan(x)
#      define _CCCL_FPMP_ASINQ(x)         __nv_fp128_asin(x)
#      define _CCCL_FPMP_ACOSQ(x)         __nv_fp128_acos(x)
#      define _CCCL_FPMP_ATANQ(x)         __nv_fp128_atan(x)
#      define _CCCL_FPMP_SINHQ(x)         __nv_fp128_sinh(x)
#      define _CCCL_FPMP_COSHQ(x)         __nv_fp128_cosh(x)
#      define _CCCL_FPMP_TANHQ(x)         __nv_fp128_tanh(x)
#      define _CCCL_FPMP_ASINHQ(x)        __nv_fp128_asinh(x)
#      define _CCCL_FPMP_ACOSHQ(x)        __nv_fp128_acosh(x)
#      define _CCCL_FPMP_ATANHQ(x)        __nv_fp128_atanh(x)
#      define _CCCL_FPMP_SQRTQ(x)         __nv_fp128_sqrt(x)
#      define _CCCL_FPMP_FABSQ(x)         __nv_fp128_fabs(x)
#      define _CCCL_FPMP_POWQ(x, y)       __nv_fp128_pow((x), (y))
#      define _CCCL_FPMP_FMODQ(x, y)      __nv_fp128_fmod((x), (y))
#      define _CCCL_FPMP_REMAINDERQ(x, y) __nv_fp128_remainder((x), (y))
#      define _CCCL_FPMP_FLOORQ(x)        __nv_fp128_floor(x)
#      define _CCCL_FPMP_CEILQ(x)         __nv_fp128_ceil(x)
#      define _CCCL_FPMP_TRUNCQ(x)        __nv_fp128_trunc(x)
#      define _CCCL_FPMP_ROUNDQ(x)        __nv_fp128_round(x)
#      define _CCCL_FPMP_RINTQ(x)         __nv_fp128_rint(x)
#      define _CCCL_FPMP_NEARBYINTQ(x)    __nv_fp128_rint(x)
#      define _CCCL_FPMP_CBRTQ(x) \
        __nv_fp128_copysign(__nv_fp128_pow(__nv_fp128_fabs(x), (__fpmp_fp128) 1 / (__fpmp_fp128) 3), (x))
#      define _CCCL_FPMP_ATAN2Q(y, x) ((__fpmp_fp128) atan2((double) (y), (double) (x)))
// erf/erfc have no __nv_fp128_* intrinsic to call; see _CCCL_FPMP_FP128_QUAD_ERF above.
#      if _CCCL_FPMP_FP128_QUAD_ERF
#        define _CCCL_FPMP_ERFQ(x)  __nv_fp128_erf(x)
#        define _CCCL_FPMP_ERFCQ(x) __nv_fp128_erfc(x)
#      else
#        define _CCCL_FPMP_ERFQ(x)  ((__fpmp_fp128) erf((double) (x)))
#        define _CCCL_FPMP_ERFCQ(x) ((__fpmp_fp128) erfc((double) (x)))
#      endif
// ----------------------------------------------------------------------
// Branch 2 -- HOST with libquadmath (the primary x86_64 host path).
// Target: host compilation pass where libquadmath is present (typically
// x86_64 GCC distributions). Reference math uses the true binary128
// libquadmath entry points (the `*q` suffix); __fpmp_fp128 is __float128
// here. The explicit _CCCL_HOST_COMPILATION() guard keeps the device pass on
// an x86_64 host (where _CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH is also 1) from
// matching this host-only branch.
// ----------------------------------------------------------------------
#    elif (_CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH == 1) && _CCCL_HOST_COMPILATION()
#      define _CCCL_FPMP_EXPQ(x)          expq(x)
#      define _CCCL_FPMP_EXPM1Q(x)        expm1q(x)
#      define _CCCL_FPMP_LOGQ(x)          logq(x)
#      define _CCCL_FPMP_LOG2Q(x)         log2q(x)
#      define _CCCL_FPMP_LOG10Q(x)        log10q(x)
#      define _CCCL_FPMP_LOG1PQ(x)        log1pq(x)
#      define _CCCL_FPMP_SINQ(x)          sinq(x)
#      define _CCCL_FPMP_COSQ(x)          cosq(x)
#      define _CCCL_FPMP_TANQ(x)          tanq(x)
#      define _CCCL_FPMP_ASINQ(x)         asinq(x)
#      define _CCCL_FPMP_ACOSQ(x)         acosq(x)
#      define _CCCL_FPMP_ATANQ(x)         atanq(x)
#      define _CCCL_FPMP_SINHQ(x)         sinhq(x)
#      define _CCCL_FPMP_COSHQ(x)         coshq(x)
#      define _CCCL_FPMP_TANHQ(x)         tanhq(x)
#      define _CCCL_FPMP_ASINHQ(x)        asinhq(x)
#      define _CCCL_FPMP_ACOSHQ(x)        acoshq(x)
#      define _CCCL_FPMP_ATANHQ(x)        atanhq(x)
#      define _CCCL_FPMP_SQRTQ(x)         sqrtq(x)
#      define _CCCL_FPMP_CBRTQ(x)         cbrtq(x)
#      define _CCCL_FPMP_FABSQ(x)         fabsq(x)
#      define _CCCL_FPMP_POWQ(x, y)       powq((x), (y))
#      define _CCCL_FPMP_ATAN2Q(y, x)     atan2q((y), (x))
#      define _CCCL_FPMP_FMODQ(x, y)      fmodq((x), (y))
#      define _CCCL_FPMP_REMAINDERQ(x, y) remainderq((x), (y))
#      define _CCCL_FPMP_ERFQ(x)          erfq(x)
#      define _CCCL_FPMP_ERFCQ(x)         erfcq(x)
#      define _CCCL_FPMP_FLOORQ(x)        floorq(x)
#      define _CCCL_FPMP_CEILQ(x)         ceilq(x)
#      define _CCCL_FPMP_TRUNCQ(x)        truncq(x)
#      define _CCCL_FPMP_ROUNDQ(x)        roundq(x)
#      define _CCCL_FPMP_RINTQ(x)         rintq(x)
#      define _CCCL_FPMP_NEARBYINTQ(x)    nearbyintq(x)
// libquadmath has no exp10q in any version and gained exp2q only in GCC 9. powq covers
// both without a version check, and stays in binary128.
#      define _CCCL_FPMP_EXP2Q(x)         powq((__float128) 2.0, (x))
#      define _CCCL_FPMP_EXP10Q(x)        powq((__float128) 10.0, (x))
// ----------------------------------------------------------------------
// Branch 3 -- HOST, no libquadmath, 128-bit `long double`
//             (the primary AArch64 / non-x86 host path).
// Target: host compilation pass on platforms whose C `long double` is a
// true 128-bit type (IEEE binary128 on AArch64 / PPC64LE, or 80-bit x87
// extended on x86 without libquadmath) AND where libquadmath is
// unavailable. Reference math uses the standard C `long double` libm entry
// points (the `*l` suffix).
// ----------------------------------------------------------------------
#    elif (_CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128 == 1) && _CCCL_HOST_COMPILATION() \
      && (_CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH == 0)
#      define _CCCL_FPMP_EXPQ(x)          expl(x)
#      define _CCCL_FPMP_EXP2Q(x)         exp2l(x)
#      define _CCCL_FPMP_EXPM1Q(x)        expm1l(x)
#      define _CCCL_FPMP_LOGQ(x)          logl(x)
#      define _CCCL_FPMP_LOG2Q(x)         log2l(x)
#      define _CCCL_FPMP_LOG10Q(x)        log10l(x)
#      define _CCCL_FPMP_LOG1PQ(x)        log1pl(x)
#      define _CCCL_FPMP_SINQ(x)          sinl(x)
#      define _CCCL_FPMP_COSQ(x)          cosl(x)
#      define _CCCL_FPMP_TANQ(x)          tanl(x)
#      define _CCCL_FPMP_ASINQ(x)         asinl(x)
#      define _CCCL_FPMP_ACOSQ(x)         acosl(x)
#      define _CCCL_FPMP_ATANQ(x)         atanl(x)
#      define _CCCL_FPMP_SINHQ(x)         sinhl(x)
#      define _CCCL_FPMP_COSHQ(x)         coshl(x)
#      define _CCCL_FPMP_TANHQ(x)         tanhl(x)
#      define _CCCL_FPMP_ASINHQ(x)        asinhl(x)
#      define _CCCL_FPMP_ACOSHQ(x)        acoshl(x)
#      define _CCCL_FPMP_ATANHQ(x)        atanhl(x)
#      define _CCCL_FPMP_SQRTQ(x)         sqrtl(x)
#      define _CCCL_FPMP_FABSQ(x)         fabsl(x)
#      define _CCCL_FPMP_POWQ(x, y)       powl((x), (y))
#      define _CCCL_FPMP_CBRTQ(x)         cbrtl(x)
#      define _CCCL_FPMP_ATAN2Q(y, x)     atan2l((y), (x))
#      define _CCCL_FPMP_FMODQ(x, y)      fmodl((x), (y))
#      define _CCCL_FPMP_REMAINDERQ(x, y) remainderl((x), (y))
#      define _CCCL_FPMP_ERFQ(x)          erfl(x)
#      define _CCCL_FPMP_ERFCQ(x)         erfcl(x)
#      define _CCCL_FPMP_FLOORQ(x)        floorl(x)
#      define _CCCL_FPMP_CEILQ(x)         ceill(x)
#      define _CCCL_FPMP_TRUNCQ(x)        truncl(x)
#      define _CCCL_FPMP_ROUNDQ(x)        roundl(x)
#      define _CCCL_FPMP_RINTQ(x)         rintl(x)
#      define _CCCL_FPMP_NEARBYINTQ(x)    nearbyintl(x)
#      define _CCCL_FPMP_EXP10Q(x)        ((long double) powl(10.0L, (x)))
// ----------------------------------------------------------------------
// Branch 4 -- CUDA DEVICE WITHOUT a usable native fp128 path
//             (the fp64 fallback).
// Target: any device build (x86_64 or AArch64) that did NOT enable the
// extended NVVM intrinsics (no _CCCL_FPMP_CUDA_FP128_INTRINSICS), or where the
// host toolchain cannot declare a usable __float128/_Float128 spelling
// for the crt overloads (e.g. AArch64 with GCC < 13.1).
//
// There is no "base subset" of __nv_fp128_* that links without the
// extended switch: on CUDA >= 12.8
// makes 128-bit FP unsupported in device code and rejects the ENTIRE
// __nv_fp128_* family at declaration (verified on 12.8 and 13.0). So when
// we land here, NO native fp128 is available and EVERY function degrades
// to a double-precision computation widened back to __fpmp_fp128 (i.e. the
// reference is effectively fp64-accurate on this target).
// ----------------------------------------------------------------------
#    elif _CCCL_DEVICE_COMPILATION()
#      define _CCCL_FPMP_EXPQ(x)          ((__fpmp_fp128) exp((double) (x)))
#      define _CCCL_FPMP_EXP2Q(x)         ((__fpmp_fp128) exp2((double) (x)))
#      define _CCCL_FPMP_EXP10Q(x)        ((__fpmp_fp128) exp10((double) (x)))
#      define _CCCL_FPMP_EXPM1Q(x)        ((__fpmp_fp128) expm1((double) (x)))
#      define _CCCL_FPMP_LOGQ(x)          ((__fpmp_fp128) log((double) (x)))
#      define _CCCL_FPMP_LOG2Q(x)         ((__fpmp_fp128) log2((double) (x)))
#      define _CCCL_FPMP_LOG10Q(x)        ((__fpmp_fp128) log10((double) (x)))
#      define _CCCL_FPMP_LOG1PQ(x)        ((__fpmp_fp128) log1p((double) (x)))
#      define _CCCL_FPMP_SINQ(x)          ((__fpmp_fp128) sin((double) (x)))
#      define _CCCL_FPMP_COSQ(x)          ((__fpmp_fp128) cos((double) (x)))
#      define _CCCL_FPMP_TANQ(x)          ((__fpmp_fp128) tan((double) (x)))
#      define _CCCL_FPMP_ASINQ(x)         ((__fpmp_fp128) asin((double) (x)))
#      define _CCCL_FPMP_ACOSQ(x)         ((__fpmp_fp128) acos((double) (x)))
#      define _CCCL_FPMP_ATANQ(x)         ((__fpmp_fp128) atan((double) (x)))
#      define _CCCL_FPMP_SINHQ(x)         ((__fpmp_fp128) sinh((double) (x)))
#      define _CCCL_FPMP_COSHQ(x)         ((__fpmp_fp128) cosh((double) (x)))
#      define _CCCL_FPMP_TANHQ(x)         ((__fpmp_fp128) tanh((double) (x)))
#      define _CCCL_FPMP_ASINHQ(x)        ((__fpmp_fp128) asinh((double) (x)))
#      define _CCCL_FPMP_ACOSHQ(x)        ((__fpmp_fp128) acosh((double) (x)))
#      define _CCCL_FPMP_ATANHQ(x)        ((__fpmp_fp128) atanh((double) (x)))
#      define _CCCL_FPMP_SQRTQ(x)         ((__fpmp_fp128) sqrt((double) (x)))
#      define _CCCL_FPMP_FABSQ(x)         ((__fpmp_fp128) fabs((double) (x)))
#      define _CCCL_FPMP_POWQ(x, y)       ((__fpmp_fp128) pow((double) (x), (double) (y)))
#      define _CCCL_FPMP_CBRTQ(x)         ((__fpmp_fp128) cbrt((double) (x)))
#      define _CCCL_FPMP_ATAN2Q(y, x)     ((__fpmp_fp128) atan2((double) (y), (double) (x)))
#      define _CCCL_FPMP_FMODQ(x, y)      ((__fpmp_fp128) fmod((double) (x), (double) (y)))
#      define _CCCL_FPMP_REMAINDERQ(x, y) ((__fpmp_fp128) remainder((double) (x), (double) (y)))
#      define _CCCL_FPMP_ERFQ(x)          ((__fpmp_fp128) erf((double) (x)))
#      define _CCCL_FPMP_ERFCQ(x)         ((__fpmp_fp128) erfc((double) (x)))
#      define _CCCL_FPMP_FLOORQ(x)        ((__fpmp_fp128) floor((double) (x)))
#      define _CCCL_FPMP_CEILQ(x)         ((__fpmp_fp128) ceil((double) (x)))
#      define _CCCL_FPMP_TRUNCQ(x)        ((__fpmp_fp128) trunc((double) (x)))
#      define _CCCL_FPMP_ROUNDQ(x)        ((__fpmp_fp128) round((double) (x)))
#      define _CCCL_FPMP_RINTQ(x)         ((__fpmp_fp128) rint((double) (x)))
#      define _CCCL_FPMP_NEARBYINTQ(x)    ((__fpmp_fp128) nearbyint((double) (x)))
#    endif // _CCCL_FPMP_FP128_MATH_FALLBACK == 1

// None of the four branches matched, so the bodies below would expand to undefined
// _CCCL_FPMP_*Q macros. The auto-detection above cannot produce this; forcing
// CCCL_FPMP_FP128_MATH_FALLBACK=1 on a pass with no binary128 math backend can.
#    if !defined(_CCCL_FPMP_EXPQ)
#      error \
        "CCCL_FPMP_FP128_MATH_FALLBACK is on but this compilation pass has no binary128 math backend (no <quadmath.h>, no 128-bit long double, no device fp128 intrinsics). Leave the knob unset to get the binary64 bodies, or provide a backend."
#    endif

/*
 * Simplified dispatch macro: uses __FPMP_*Q wrapper macros which already
 * handle CUDA/libquadmath/long double dispatching internally.
 */
#    define _CCCL_FPMP_CALL_FP64MP2_MATH(dfunc, qfunc, xhi, xlo, reshi, reslo) \
      __fpmp2_from_quad(qfunc(__fpmp2_to_quad(xhi, xlo)), reshi, reslo)
#    define _CCCL_FPMP_CALL_FP64MP2_MATH_2A(dfunc, qfunc, xhi, xlo, yhi, ylo, reshi, reslo) \
      __fpmp2_from_quad(qfunc(__fpmp2_to_quad(xhi, xlo), __fpmp2_to_quad(yhi, ylo)), reshi, reslo)
#  else
#    define _CCCL_FPMP_CALL_FP64MP2_MATH(dfunc, qfunc, xhi, xlo, reshi, reslo) \
      __fpmp2_from_double(::dfunc(__fpmp2_to_double(xhi, xlo)), reshi, reslo)
#    define _CCCL_FPMP_CALL_FP64MP2_MATH_2A(dfunc, qfunc, xhi, xlo, yhi, ylo, reshi, reslo) \
      __fpmp2_from_double(::dfunc(__fpmp2_to_double(xhi, xlo), __fpmp2_to_double(yhi, ylo)), reshi, reslo)
#  endif // _CCCL_FPMP_FP128_MATH_FALLBACK == 1

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_H
