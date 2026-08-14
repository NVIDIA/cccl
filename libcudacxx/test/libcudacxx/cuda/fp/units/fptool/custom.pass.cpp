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

// Core computation, runs on both CPU and GPU.
TEST_HOST_DEVICE_FUNC void run_precision_tests(TestResults* r)
{
  const double val_a = 1.12345678123456789;
  const double val_b = 2.12345678123456789;

  // Basic arithmetic.
  {
    double na = val_a, nb = val_b;
    fp_reduced ra = val_a, rb = val_b;

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
    double nx     = 2.12345678123456789;
    fp_reduced rx = 2.32145678123456789;

    r->sqrt_n = ::cuda::std::sqrt(nx);
    r->sqrt_r = (double) sqrt(rx);

    double na = val_a, nb = val_b, nc = 0.5;
    fp_reduced ra = val_a, rb = val_b, rc = 0.5;

    r->fma_n = ::cuda::std::fma(na, nb, nc);
    r->fma_r = (double) fma(ra, rb, rc);
  }

  // Small difference: (1 + 1e-10) - 1.
  {
    double a  = 1.0 + 1e-10;
    double b  = 1.0;
    double na = a, nb = b;
    fp_reduced ra = a, rb = b;
    r->small_diff_n = (double) (na - nb);
    r->small_diff_r = (double) (ra - rb);
  }

  // Catastrophic cancellation: (a + b) - a.
  {
    double a = 1.0, b = 1e-10;
    double na = a, nb = b;
    fp_reduced ra = a, rb = b;
    r->cancel_n = (double) ((na + nb) - na);
    r->cancel_r = (double) ((ra + rb) - ra);
  }

  // Multiplication precision.
  {
    double a = 1.0000001, b = 1.0000002;
    double na = a, nb = b;
    fp_reduced ra = a, rb = b;
    r->mul_prec_n = (double) (na * nb);
    r->mul_prec_r = (double) (ra * rb);
  }

  // Accumulation error (sum of 1/n, n=1..1000).
  {
    double native_sum      = 0.0;
    fp_reduced reduced_sum = 0.0;
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
    fp_reduced r_x = 1.0, r_S = 2.0, r_half = 0.5;
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
    double val          = 1.12345678123456789;
    double n_val        = val;
    fp_reduced r_val    = val;
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
}

int main(int, char**)
{
  test();

  return 0;
}
