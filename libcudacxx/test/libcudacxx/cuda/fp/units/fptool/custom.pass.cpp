// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: FP64 precision-emulation tool (fptool) with a reduced mantissa.
//
//  Instantiates fp_custom with a 23-bit (float-like) mantissa and exercises: basic
//  arithmetic, math functions (sqrt, fma), precision-sensitive operations
//  (small differences, catastrophic cancellation), accumulation error,
//  Newton-Raphson convergence, mantissa-truncation bit patterns, and
//  comparisons. Every check confirms the reduced-precision surface behaves as
//  expected relative to native double. A second section covers the smallest
//  mantissa the type accepts, zero bits, which quantizes to powers of two. The
//  same TEST_HOST_DEVICE_FUNC entry points run on the host and, under CUDA, on
//  the device. Further sections cover the type surface that keeps fp_custom a
//  drop-in for double (layout, trivial copyability, bit_cast, conversions,
//  volatile storage) and the standard math spellings, which must reduce rather
//  than fall back to a native-double computation.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fptool>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

// <cuda/stream> is only usable where the CUDA runtime is: under NVRTC cuda::stream_ref is left
// undefined while get_stream.h still returns it by value. The stream-based runtime-size tests
// below are host-side and carry the same guard.
#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)
#  include <cuda/stream>
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Reduced precision version (23 mantissa bits like float, full FP64 exponent range).
using fp_reduced = cudax::fp64_custom<11, 23>;

struct TestResults
{
  double add_n, add_r;
  double sub_n, sub_r;
  double mul_n, mul_r;
  double div_n, div_r;
  double neg_n, neg_r;

  double sqrt_n, sqrt_r;
  double fma_n, fma_r;

  double small_diff_n, small_diff_r;
  double cancel_n, cancel_r;
  double mul_prec_n, mul_prec_r;

  double accum_n, accum_r;
  double newton_n, newton_r;

  ::cuda::std::uint64_t bits_orig;
  ::cuda::std::uint64_t bits_native;
  ::cuda::std::uint64_t bits_reduced;

  double cmp_eq, cmp_lt, cmp_gt;
};

// Core computation, runs on both CPU and GPU. Each case computes the same expression twice,
// once in double and once in fp_reduced, and the braces on the fp_reduced declarations are
// the explicit constructor a format narrower than double asks for, not a style choice.
TEST_HOST_DEVICE_FUNC void run_precision_tests(TestResults* r)
{
  const double val_a = 1.12345678123456789;
  const double val_b = 2.12345678123456789;

  // Basic arithmetic.
  {
    double na = val_a, nb = val_b;
    fp_reduced ra{val_a}, rb{val_b};

    r->add_n = (double) (na + nb);
    r->add_r = (double) (ra + rb);
    r->sub_n = (double) (na - nb);
    r->sub_r = (double) (ra - rb);
    r->mul_n = (double) (na * nb);
    r->mul_r = (double) (ra * rb);
    r->div_n = (double) (na / nb);
    r->div_r = (double) (ra / rb);
    r->neg_n = (double) (-na);
    r->neg_r = (double) (-ra);
  }

  // Math functions.
  {
    double nx = 2.12345678123456789;
    fp_reduced rx{2.32145678123456789};

    r->sqrt_n = ::cuda::std::sqrt(nx);
    r->sqrt_r = (double) sqrt(rx);

    double na = val_a, nb = val_b, nc = 0.5;
    fp_reduced ra{val_a}, rb{val_b}, rc{0.5};

    r->fma_n = ::cuda::std::fma(na, nb, nc);
    r->fma_r = (double) fma(ra, rb, rc);
  }

  // Small difference: (1 + 1e-10) - 1.
  {
    double a  = 1.0 + 1e-10;
    double b  = 1.0;
    double na = a, nb = b;
    fp_reduced ra{a}, rb{b};
    r->small_diff_n = (double) (na - nb);
    r->small_diff_r = (double) (ra - rb);
  }

  // Catastrophic cancellation: (a + b) - a.
  {
    double a = 1.0, b = 1e-10;
    double na = a, nb = b;
    fp_reduced ra{a}, rb{b};
    r->cancel_n = (double) ((na + nb) - na);
    r->cancel_r = (double) ((ra + rb) - ra);
  }

  // Multiplication precision.
  {
    double a = 1.0000001, b = 1.0000002;
    double na = a, nb = b;
    fp_reduced ra{a}, rb{b};
    r->mul_prec_n = (double) (na * nb);
    r->mul_prec_r = (double) (ra * rb);
  }

  // Accumulation error (sum of 1/n, n=1..1000).
  {
    double native_sum = 0.0;
    fp_reduced reduced_sum{0.0};
    for (int n = 1; n <= 1000; n++)
    {
      double term = 1.0 / n;
      native_sum += double(term);
      reduced_sum += fp_reduced(term);
    }
    r->accum_n = (double) native_sum;
    r->accum_r = (double) reduced_sum;
  }

  // Newton-Raphson sqrt(2): x_{n+1} = 0.5 * (x_n + S/x_n).
  {
    double n_x = 1.0, n_S = 2.0, n_half = 0.5;
    fp_reduced r_x{1.0}, r_S{2.0}, r_half{0.5};
    for (int i = 0; i < 10; i++)
    {
      n_x = n_half * (n_x + n_S / n_x);
      r_x = r_half * (r_x + r_S / r_x);
    }
    r->newton_n = (double) n_x;
    r->newton_r = (double) r_x;
  }

  // Bit-pattern analysis (mantissa truncation).
  {
    double val   = 1.12345678123456789;
    double n_val = val;
    fp_reduced r_val{val};
    double n_result     = n_val + double(0.0);
    fp_reduced r_result = r_val + fp_reduced(0.0);
    double n_out        = (double) n_result;
    double r_out        = (double) r_result;
    ::cuda::std::memcpy(&r->bits_orig, &val, sizeof(::cuda::std::uint64_t));
    ::cuda::std::memcpy(&r->bits_native, &n_out, sizeof(::cuda::std::uint64_t));
    ::cuda::std::memcpy(&r->bits_reduced, &r_out, sizeof(::cuda::std::uint64_t));
  }

  // Comparison operators.
  {
    double na = val_a, nb = val_b;
    r->cmp_eq = (na == na) ? 1.0 : 0.0;
    r->cmp_lt = (na < nb) ? 1.0 : 0.0;
    r->cmp_gt = (na > nb) ? 1.0 : 0.0;
  }
}

// Verify the reduced-precision surface behaves as expected vs native double.
TEST_HOST_DEVICE_FUNC bool verify(const TestResults& r)
{
  bool ok = true;

  // Basic arithmetic: finite, and precision loss vs native where expected.
  ok = ok && ::cuda::std::isfinite(r.add_r) && ::cuda::std::isfinite(r.sub_r) && ::cuda::std::isfinite(r.mul_r)
    && ::cuda::std::isfinite(r.div_r) && ::cuda::std::isfinite(r.neg_r);
  ok = ok && (r.add_r != r.add_n) && (r.mul_r != r.mul_n) && (r.neg_r < 0.0);

  // Math functions finite.
  ok = ok && ::cuda::std::isfinite(r.sqrt_r) && ::cuda::std::isfinite(r.fma_r);

  // Precision-sensitive.
  ok = ok && (r.small_diff_n > 0.0) && (r.small_diff_r == 0.0);
  ok = ok && (r.cancel_n > 0.0) && (r.cancel_r == 0.0);
  ok = ok && (r.mul_prec_r != r.mul_prec_n);

  // Accumulation.
  ok = ok && (::cuda::std::fabs(r.accum_n - r.accum_r) > 0.0) && ::cuda::std::isfinite(r.accum_r);

  // Newton-Raphson.
  const double sqrt2        = ::cuda::std::sqrt(2.0);
  const double newton_err_n = ::cuda::std::fabs(r.newton_n - sqrt2);
  const double newton_err_r = ::cuda::std::fabs(r.newton_r - sqrt2);
  ok                        = ok && (newton_err_n < 1e-14) && (newton_err_r < 1e-5) && (newton_err_r > newton_err_n);

  // Bit patterns: reduced zeroes the low 29 bits, native keeps some.
  const ::cuda::std::uint64_t low_29_mask = (1ULL << 29) - 1;
  ok = ok && ((r.bits_native & low_29_mask) != 0) && ((r.bits_reduced & low_29_mask) == 0);

  // Comparisons.
  ok = ok && (r.cmp_eq == 1.0) && (r.cmp_lt == 1.0) && (r.cmp_gt == 0.0);

  return ok;
}

TEST_HOST_DEVICE_FUNC bool run_test()
{
  TestResults r{};
  run_precision_tests(&r);
  return verify(r);
}

// The smallest mantissa the type accepts keeps only the implicit leading 1, so every
// value collapses to the nearest power of two. Ties land on the even exponent, which is
// why 3 rounds down to 2 while 6 rounds up to 8.
TEST_HOST_DEVICE_FUNC bool test_power_of_two()
{
  using fp_po2 = cudax::fp64_custom<11, 0>;

  bool ok = true;

  // Reduction applies to the operands, so multiplying by one is enough to observe it.
  const fp_po2 one(1.0);
  ok = ok && static_cast<double>(fp_po2(1.4) * one) == 1.0;
  ok = ok && static_cast<double>(fp_po2(1.5) * one) == 2.0;
  ok = ok && static_cast<double>(fp_po2(3.0) * one) == 2.0;
  ok = ok && static_cast<double>(fp_po2(6.0) * one) == 8.0;
  ok = ok && static_cast<double>(fp_po2(0.3) * one) == 0.25;

  // Signs are kept, and so are the specials.
  ok = ok && static_cast<double>(fp_po2(-3.0) * one) == -2.0;
  ok = ok && static_cast<double>(fp_po2(-0.0) + fp_po2(-0.0)) == 0.0;
  ok = ok && ::cuda::std::signbit(static_cast<double>(fp_po2(-0.0) + fp_po2(-0.0)));
  ok = ok && ::cuda::std::isinf(static_cast<double>(fp_po2(::cuda::std::numeric_limits<double>::infinity()) + one));
  ok = ok && ::cuda::std::isnan(static_cast<double>(fp_po2(::cuda::std::numeric_limits<double>::quiet_NaN()) + one));

  return ok;
}

// The type stays a transparent stand-in for double: same layout, trivially copyable so
// it can be bit_cast and copied by memcpy, constructible from every arithmetic type
// double accepts, and usable as volatile storage.
static_assert(::cuda::std::is_trivially_copyable_v<fp_reduced>, "");
static_assert(::cuda::std::is_trivially_copy_constructible_v<fp_reduced>, "");
static_assert(::cuda::std::is_trivially_copy_assignable_v<fp_reduced>, "");
static_assert(sizeof(fp_reduced) == sizeof(double), "");
static_assert(alignof(fp_reduced) == alignof(double), "");

TEST_HOST_DEVICE_FUNC bool test_type_surface()
{
  bool ok = true;

  // bit_cast in both directions, which trivial copyability is what enables.
  const auto bits = ::cuda::std::bit_cast<::cuda::std::uint64_t>(fp_reduced(1.0));
  ok              = ok && bits == 0x3ff0000000000000ULL;
  ok              = ok && static_cast<double>(::cuda::std::bit_cast<fp_reduced>(bits)) == 1.0;

  // bool and character types convert, as they do for double.
  ok = ok && static_cast<double>(fp_reduced(true)) == 1.0;
  ok = ok && static_cast<double>(fp_reduced('A')) == 65.0;

  // Volatile storage: load, store and a volatile-to-volatile copy all preserve bits.
  volatile fp_reduced vsrc = fp_reduced(0.5);
  const fp_reduced loaded  = vsrc;
  volatile fp_reduced vdst = fp_reduced(0.0);
  vdst                     = loaded;
  volatile fp_reduced vcpy = fp_reduced(0.0);
  vcpy                     = vdst;
  ok                       = ok && static_cast<double>(loaded) == 0.5;
  ok                       = ok && static_cast<double>(fp_reduced(vcpy)) == 0.5;

  return ok;
}

// The way out of fp_custom follows the requested format. double is always implicit, the
// value being held in one. float is implicit exactly where binary32 represents the format,
// i.e. within 8 exponent and 23 mantissa bits, and takes a cast anywhere else.
//
// is_convertible cannot tell the two apart, because the implicit conversion to double
// reaches float through the standard double -> float conversion whatever the format is. What
// the specifier decides is overload resolution: a format that fits offers float and double
// on equal terms, which makes an overload set holding both ambiguous, while a wider format
// leaves double as the only way in. The cast is available either way.
static_assert(::cuda::std::is_convertible_v<fp_reduced, double>, "");
static_assert(::cuda::std::is_convertible_v<cudax::fp64_custom<>, double>, "");
static_assert(::cuda::std::is_constructible_v<float, fp_reduced>, "");
static_assert(::cuda::std::is_constructible_v<float, cudax::fp64_custom<8, 23>>, "");

TEST_HOST_DEVICE_FUNC int pick(float)
{
  return 1;
}
TEST_HOST_DEVICE_FUNC int pick(double)
{
  return 2;
}

template <class _Tp, class = void>
struct picks_ambiguously : ::cuda::std::true_type
{};
template <class _Tp>
struct picks_ambiguously<_Tp, decltype(void(pick(::cuda::std::declval<_Tp>())))> : ::cuda::std::false_type
{};

static_assert(picks_ambiguously<cudax::fp64_custom<8, 23>>::value, "");
static_assert(picks_ambiguously<cudax::fp64_custom<5, 10>>::value, "");
static_assert(!picks_ambiguously<fp_reduced>::value, ""); // 23 bits, but 11 of exponent
static_assert(!picks_ambiguously<cudax::fp64_custom<8, 24>>::value, "");
static_assert(!picks_ambiguously<cudax::fp64_custom<>>::value, "");
// A runtime size is unknown here, so it takes the explicit conversion, as it has to.
static_assert(!picks_ambiguously<cudax::fp64_custom<8, cudax::fp_custom_dynamic_size>>::value, "");
static_assert(!picks_ambiguously<cudax::fp64_custom<cudax::fp_custom_dynamic_size, 23>>::value, "");

// The way into fp_custom follows the same rank rule: implicit where the requested format is
// at least as wide as the source in both fields, explicit where either is narrower, with
// integers counting as double. Here is_convertible does decide it, there being no second
// path into fp_custom the way operator double() is one out of it.
static_assert(::cuda::std::is_convertible_v<double, cudax::fp64_custom<>>, "");
static_assert(::cuda::std::is_convertible_v<float, cudax::fp64_custom<>>, "");
static_assert(::cuda::std::is_convertible_v<int, cudax::fp64_custom<>>, "");
static_assert(::cuda::std::is_convertible_v<unsigned long long, cudax::fp64_custom<>>, "");
static_assert(::cuda::std::is_convertible_v<bool, cudax::fp64_custom<>>, "");
static_assert(::cuda::std::is_convertible_v<char, cudax::fp64_custom<>>, "");

// A format that holds binary32 but not binary64 takes a float implicitly, whatever the
// narrowing side is set to.
static_assert(::cuda::std::is_convertible_v<float, cudax::fp64_custom<8, 23>>, "");
static_assert(::cuda::std::is_convertible_v<float, cudax::fp64_custom<11, 23>>, "");
static_assert(::cuda::std::is_convertible_v<float, cudax::fp64_custom<8, 52>>, "");
static_assert(::cuda::std::is_convertible_v<float, fp_reduced>, "");

// A long double is not implicitly convertible at either setting, and for a different reason
// than the rank rule: it reaches the double constructor and the deleted 128-bit integer ones
// by a conversion apiece, so the call is ambiguous rather than explicit. What a long double
// source should do here is still open; asserting it holds either way pins today's behavior.
static_assert(!::cuda::std::is_convertible_v<long double, fp_reduced>, "");
static_assert(!::cuda::std::is_convertible_v<long double, cudax::fp64_custom<8, 23>>, "");

// The narrowing side is what CCCL_FP_CUSTOM_EXPLICIT_CASTS gates, so it is asserted at both
// settings: a cast by default, implicit where a codebase written against double is being
// moved onto the type. Nothing above this point changes with it.
#if CCCL_FP_CUSTOM_EXPLICIT_CASTS == 1

// A double, and an integer with it, reaches a format that holds binary32 only by cast.
static_assert(!::cuda::std::is_convertible_v<double, cudax::fp64_custom<8, 23>>, "");
static_assert(!::cuda::std::is_convertible_v<int, cudax::fp64_custom<8, 23>>, "");
static_assert(!::cuda::std::is_convertible_v<bool, cudax::fp64_custom<8, 23>>, "");
static_assert(!::cuda::std::is_convertible_v<char, cudax::fp64_custom<8, 23>>, "");

// The float constructor must not become a way in for the types that convert to float, which
// is what constrains it to a deduced float rather than naming one. fp_reduced is the case
// that shows the cost of getting this wrong: it holds binary32's mantissa and a wider
// exponent, so an implicit double would round twice and through the narrower range.
static_assert(!::cuda::std::is_convertible_v<double, fp_reduced>, "");

// Narrower than binary32 in either field, so neither source is implicit. fp64_custom<5, 52>
// is the unordered case: a wider mantissa than binary32 and a narrower exponent.
static_assert(!::cuda::std::is_convertible_v<float, cudax::fp64_custom<5, 10>>, "");
static_assert(!::cuda::std::is_convertible_v<float, cudax::fp64_custom<8, 22>>, "");
static_assert(!::cuda::std::is_convertible_v<float, cudax::fp64_custom<5, 52>>, "");

// A runtime size is unknown here, so every source takes the explicit constructor.
static_assert(!::cuda::std::is_convertible_v<double, cudax::fp64_custom<8, cudax::fp_custom_dynamic_size>>, "");
static_assert(!::cuda::std::is_convertible_v<float, cudax::fp64_custom<cudax::fp_custom_dynamic_size, 23>>, "");

#else // ^^^ CCCL_FP_CUSTOM_EXPLICIT_CASTS == 1 ^^^ / vvv == 0 vvv

// Every source named above is implicit here, which is the whole of the setting's effect:
// the sources are unchanged, and so are the formats.
static_assert(::cuda::std::is_convertible_v<double, cudax::fp64_custom<8, 23>>, "");
static_assert(::cuda::std::is_convertible_v<int, cudax::fp64_custom<8, 23>>, "");
static_assert(::cuda::std::is_convertible_v<bool, cudax::fp64_custom<8, 23>>, "");
static_assert(::cuda::std::is_convertible_v<char, cudax::fp64_custom<8, 23>>, "");
static_assert(::cuda::std::is_convertible_v<double, fp_reduced>, "");
static_assert(::cuda::std::is_convertible_v<float, cudax::fp64_custom<5, 10>>, "");
static_assert(::cuda::std::is_convertible_v<float, cudax::fp64_custom<8, 22>>, "");
static_assert(::cuda::std::is_convertible_v<float, cudax::fp64_custom<5, 52>>, "");
static_assert(::cuda::std::is_convertible_v<double, cudax::fp64_custom<8, cudax::fp_custom_dynamic_size>>, "");
static_assert(::cuda::std::is_convertible_v<float, cudax::fp64_custom<cudax::fp_custom_dynamic_size, 23>>, "");

#endif // CCCL_FP_CUSTOM_EXPLICIT_CASTS == 0

// Explicit, not absent: every source is still constructible into every format.
static_assert(::cuda::std::is_constructible_v<cudax::fp64_custom<8, 23>, double>, "");
static_assert(::cuda::std::is_constructible_v<cudax::fp64_custom<5, 10>, float>, "");
static_assert(::cuda::std::is_constructible_v<cudax::fp64_custom<5, 10>, int>, "");
static_assert(::cuda::std::is_constructible_v<cudax::fp64_custom<5, 10>, bool>, "");
static_assert(
  ::cuda::std::is_constructible_v<cudax::fp64_custom<cudax::fp_custom_dynamic_size, cudax::fp_custom_dynamic_size>,
                                  double>,
  "");

// What an explicit constructor leaves untouched: the operand of a mixed expression, which
// the hidden friends convert themselves, and the value, which arrives unreduced.
TEST_HOST_DEVICE_FUNC bool test_narrowing_construction()
{
  using fp_float = cudax::fp64_custom<8, 23>;

  bool ok = true;

  const fp_float from_double{1.0 / 3.0};
  const fp_float from_float = 1.0f / 3.0f;
  const fp_float from_int   = fp_float(3);

  // Stored in the base type, so the double arrives whole and the format applies at the
  // first operation, which here rounds it to what a float would hold.
  ok = ok && static_cast<double>(from_double) == 1.0 / 3.0;
  ok = ok && static_cast<double>(from_double + fp_float(0.0)) == static_cast<double>(1.0f / 3.0f);
  ok = ok && static_cast<double>(from_float) == static_cast<double>(1.0f / 3.0f);
  ok = ok && static_cast<double>(from_int) == 3.0;

  // A value outside the reduced exponent range survives construction and clamps on use.
  const fp_float huge{1e300};
  ok = ok && static_cast<double>(huge) == 1e300;
  ok = ok && ::cuda::std::isinf(static_cast<double>(huge + fp_float(0.0)));

  // Mixed arithmetic and comparison take a scalar operand whatever the format is.
  ok = ok && static_cast<double>(from_int + 1.0) == 4.0;
  ok = ok && static_cast<double>(1.0 + from_int) == 4.0;
  ok = ok && static_cast<double>(from_int * 2) == 6.0;
  ok = ok && from_int > 2.0 && from_int < 4;

  return ok;
}

// What the implicit conversion promises: the value arrives in the float unchanged, and the
// conversion does not draw mixed arithmetic away from fp_custom.
TEST_HOST_DEVICE_FUNC bool test_float_conversion()
{
  using fp_float = cudax::fp64_custom<8, 23>;
  using fp_half  = cudax::fp64_custom<5, 10>;

  bool ok = true;

  // A format wider than binary32 reaches a float sink only through double, which is what
  // the overload set below shows, and which is why float stays explicit there.
  ok = ok && pick(fp_reduced(1.0)) == 2;

  const fp_float third = fp_float(1.0) / fp_float(3.0);
  const float narrow   = third; // implicit, and exact
  ok                   = ok && static_cast<double>(narrow) == static_cast<double>(third);

  const fp_half tenth       = fp_half(1.0) / fp_half(10.0);
  const float half_as_float = tenth;
  ok                        = ok && static_cast<double>(half_as_float) == static_cast<double>(tenth);

  // Overflow of the reduced exponent range lands on an infinity, which float also has.
  const fp_float huge  = fp_float(1e30) * fp_float(1e30);
  const float huge_out = huge;
  ok                   = ok && ::cuda::std::isinf(huge_out);

  // Arithmetic still goes through fp_custom: a float operand is promoted rather than the
  // fp_custom value demoted.
  ok = ok && ::cuda::std::is_same_v<decltype(third + 1.0f), fp_float>;
  ok = ok && ::cuda::std::is_same_v<decltype(1.0f * third), fp_float>;

  return ok;
}

// A qualified cuda::std::sqrt or cuda::std::fma call suppresses ADL, and a plain double
// operand in an fma makes ::fma(double, double, double) viable. Both must still reduce
// rather than quietly compute at full FP64 precision.
TEST_HOST_DEVICE_FUNC bool test_standard_spellings()
{
  bool ok = true;

  const fp_reduced two(2.0);
  const double reduced_root = static_cast<double>(sqrt(two));
  ok                        = ok && static_cast<double>(::cuda::std::sqrt(two)) == reduced_root;
  ok                        = ok && reduced_root != ::cuda::std::sqrt(2.0);

  // A tiny third operand that only full precision could keep.
  const fp_reduced a(1.0 + 0x1p-30), b(1.0), zero(0.0);
  const double reduced_fma = static_cast<double>(fma(a, b, zero));
  ok                       = ok && static_cast<double>(::cuda::std::fma(a, b, zero)) == reduced_fma;
  ok                       = ok && static_cast<double>(fma(a, b, 0.0)) == reduced_fma;
  ok                       = ok && static_cast<double>(::cuda::std::fma(a, b, 0.0)) == reduced_fma;
  ok                       = ok && reduced_fma == 1.0 && ::cuda::std::fma(1.0 + 0x1p-30, 1.0, 0.0) != 1.0;

  return ok;
}

TEST_HOST_DEVICE_FUNC void test()
{
  assert(run_test());
  assert(test_power_of_two());
  assert(test_type_surface());
  assert(test_standard_spellings());
  assert(test_narrowing_construction());
  assert(test_float_conversion());
}

#if _CCCL_CUDA_COMPILATION()
// Runtime sizes, which fp_custom_dynamic_size selects, live in a device variable rather
// than in the type.
using fp_dynamic = cudax::fp64_custom<cudax::fp_custom_dynamic_size, cudax::fp_custom_dynamic_size>;

// Reports both what the arithmetic did with the current sizes and what the sizes are, so a
// stale device copy would be visible either way.
__global__ void dynamic_size_kernel(double* sum, int* mant_size)
{
  const fp_dynamic __one(1.0), __tiny(0x1p-30);
  *sum       = static_cast<double>(__one + __tiny);
  *mant_size = cudax::fp_custom_get_device_mantissa_size();
}

// The device-side setter, which is what a JIT-compiled program has instead of the host one.
// One block, so nothing else is reading the size while thread 0 writes it.
__global__ void device_set_size_kernel(int new_size, int* observed)
{
  cudax::fp_custom_set_device_mantissa_size(new_size);
  __syncthreads();
  *observed = cudax::fp_custom_get_device_mantissa_size();
}
#endif // _CCCL_CUDA_COMPILATION()

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)
// A device size is per-device state written through a stream: the write is ordered against
// the kernels that read it, so a kernel launched afterwards on the same stream must compute
// in the new format without any synchronization in between.
void test_runtime_sizes(cuda::stream_ref stream)
{
  double* sum           = nullptr;
  int* mant_size        = nullptr;
  const double sum_full = 1.0 + 0x1p-30;
  assert(cudaMallocManaged(&sum, sizeof(double)) == cudaSuccess);
  assert(cudaMallocManaged(&mant_size, sizeof(int)) == cudaSuccess);

  // Untouched, the sizes are the native ones, so the small term survives.
  assert(cudax::fp_custom_get_device_mantissa_size(stream) == 52);
  assert(cudax::fp_custom_get_device_exponent_size(stream) == 11);

  dynamic_size_kernel<<<1, 1, 0, stream.get()>>>(sum, mant_size);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  assert(*sum == sum_full);
  assert(*mant_size == 52);

  // 23 bits cannot hold a term 30 binades down, and the kernel needs no synchronization to
  // see the new size: the copy is ahead of it on the stream.
  cudax::fp_custom_set_device_mantissa_size(23, stream);
  dynamic_size_kernel<<<1, 1, 0, stream.get()>>>(sum, mant_size);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  assert(*sum == 1.0);
  assert(*mant_size == 23);
  assert(cudax::fp_custom_get_device_mantissa_size(stream) == 23);

  // The exponent is the other axis, and independent: 5 bits keep 1.0 but flush 2^-30.
  cudax::fp_custom_set_device_exponent_size(5, stream);
  assert(cudax::fp_custom_get_device_exponent_size(stream) == 5);
  assert(cudax::fp_custom_get_device_mantissa_size(stream) == 23);

  // The host copy of the sizes is separate state, which the device writes must not have
  // touched.
  assert(cudax::fp_custom_get_host_mantissa_size() == 52);
  assert(cudax::fp_custom_get_host_exponent_size() == 11);

  // A write from device code reaches the same variable the host accessors see.
  int* observed = nullptr;
  assert(cudaMallocManaged(&observed, sizeof(int)) == cudaSuccess);
  device_set_size_kernel<<<1, 32, 0, stream.get()>>>(40, observed);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  assert(*observed == 40);
  assert(cudax::fp_custom_get_device_mantissa_size(stream) == 40);

  // Leave the native format behind for anything that runs later.
  cudax::fp_custom_set_device_mantissa_size(52, stream);
  cudax::fp_custom_set_device_exponent_size(11, stream);
  assert(cudax::fp_custom_get_device_mantissa_size(stream) == 52);
  assert(cudax::fp_custom_get_device_exponent_size(stream) == 11);

  assert(cudaFree(sum) == cudaSuccess);
  assert(cudaFree(mant_size) == cudaSuccess);
  assert(cudaFree(observed) == cudaSuccess);
}
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  test();

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)
  // force_include.h runs this main on the host and then inside a kernel; only the host run
  // can create a stream and launch, so NV_IS_HOST selects the driver, not the code tested.
  NV_IF_TARGET(NV_IS_HOST,
               (cudaStream_t raw_stream = nullptr; //
                assert(cudaStreamCreate(&raw_stream) == cudaSuccess);
                test_runtime_sizes(cuda::stream_ref{raw_stream});
                assert(cudaStreamDestroy(raw_stream) == cudaSuccess);))
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

  return 0;
}
