// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: nvrtc
// note: the host half of this test resets and reads the device record through the CUDA
// runtime API, which is not available in NVRTC's device-only translation unit
// UNSUPPORTED: enable-tile
// error: the instrumentation updates its counters with atomics, which are unsupported in
// tile code

//===----------------------------------------------------------------------===//
//
//  Unit test: statistics-collecting fpmp2 wrapper (fpmp2_stat).
//
//  The wrapper must observe arithmetic without changing it, so the test computes
//  everything twice - once on fp32mp2 and once on fp32mp2_stat - and requires the
//  two to agree bit for bit: operators, compound assignments, increments, mixed
//  scalar and mixed wrapped-type forms, the untraced helpers (sqrt, rsqrt, fma,
//  mad, renormalize) and the math wrappers. It also covers the layout promise
//  (same size, alignment and trivial copyability as the wrapped type) and the
//  conversion surface.
//
//  Under CUDA a second part checks the record itself: that a reset arms the range
//  sentinels and zeroes the counters, and that a kernel with a known operation mix
//  produces exactly the expected counts. The parity part runs on the host and, under
//  CUDA, on the device.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fptool>
#include <cuda/std/cassert>
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

using base_t     = cudax::fp32mp2;
using stat_t     = cudax::fp32mp2_stat;
using base_low_t = cudax::fp32mp2_low;
using stat_low_t = cudax::fp32mp2_stat_low;

// The drop-in promise in memory.
static_assert(sizeof(stat_t) == sizeof(base_t));
static_assert(alignof(stat_t) == alignof(base_t));
static_assert(cuda::std::is_trivially_copyable_v<stat_t>);
static_assert(sizeof(cudax::fp64mp2_stat) == sizeof(cudax::fp64mp2));
static_assert(cuda::std::is_trivially_copyable_v<cudax::fp64mp2_stat>);

// The wrapper reports the same characteristics as the wrapped type.
static_assert(cuda::std::numeric_limits<stat_t>::digits == cuda::std::numeric_limits<base_t>::digits);
static_assert(cuda::std::numeric_limits<stat_t>::is_specialized);

// A pair of limbs, compared bitwise so that a sign of zero or a NaN payload cannot hide
// a difference.
struct pair_t
{
  float hi;
  float lo;
};

TEST_HOST_DEVICE_FUNC bool same(const pair_t& lhs, const pair_t& rhs)
{
  return cuda::std::memcmp(&lhs, &rhs, sizeof(pair_t)) == 0;
}

template <class _Tp>
TEST_HOST_DEVICE_FUNC pair_t limbs(const _Tp& value)
{
  return pair_t{value.hi(), value.lo()};
}

// Every operation below is computed on both types and the two results compared, so the
// harness needs no expected values: the wrapped type is the reference.
TEST_HOST_DEVICE_FUNC void test_parity()
{
  const float fa = 1.123456789f;
  const float fb = 2.987654321f;
  const float fc = 0.577215664f;

  const base_t ba(fa), bb(fb), bc(fc);
  const stat_t sa(fa), sb(fb), sc(fc);

  assert(same(limbs(ba + bb), limbs(sa + sb)));
  assert(same(limbs(ba - bb), limbs(sa - sb)));
  assert(same(limbs(ba * bb), limbs(sa * sb)));
  assert(same(limbs(ba / bb), limbs(sa / sb)));
  assert(same(limbs(-ba), limbs(-sa)));
  assert(same(limbs(renormalize(ba)), limbs(renormalize(sa))));

  { // compound assignment chain
    base_t b = ba;
    stat_t s = sa;
    b += bb;
    s += sb;
    b -= bc;
    s -= sc;
    b *= bb;
    s *= sb;
    b /= bc;
    s /= sc;
    b += fc;
    s += fc;
    b -= fc;
    s -= fc;
    assert(same(limbs(b), limbs(s)));
  }

  { // increment and decrement, prefix and postfix
    base_t b = ba;
    stat_t s = sa;
    ++b;
    ++s;
    --b;
    --s;
    b++;
    s++;
    b--;
    s--;
    assert(same(limbs(b), limbs(s)));
  }

  // mixed with a built-in scalar, both operand orders
  assert(same(limbs(fc * ba), limbs(fc * sa)));
  assert(same(limbs(ba * fc), limbs(sa * fc)));
  assert(same(limbs(ba + 2), limbs(sa + 2)));
  assert(same(limbs(3 - ba), limbs(3 - sa)));
  assert(same(limbs(ba / 2.0f), limbs(sa / 2.0f)));

  // mixed with the wrapped type, both operand orders: these must stay instrumented and
  // must not be ambiguous
  assert(same(limbs(ba + bb), limbs(sa + bb)));
  assert(same(limbs(ba + bb), limbs(ba + sb)));
  assert(same(limbs(ba - bb), limbs(sa - bb)));
  assert(same(limbs(ba * bb), limbs(ba * sb)));
  assert(same(limbs(ba / bb), limbs(sa / bb)));
  static_assert(cuda::std::is_same_v<decltype(sa + bb), stat_t>);
  static_assert(cuda::std::is_same_v<decltype(ba + sb), stat_t>);

  // untraced arithmetic helpers
  assert(same(limbs(sqrt(bb)), limbs(sqrt(sb))));
  assert(same(limbs(rsqrt(bb)), limbs(rsqrt(sb))));
  assert(same(limbs(fma(ba, bb, bc)), limbs(fma(sa, sb, sc))));
  assert(same(limbs(mad(ba, bb, bc)), limbs(mad(sa, sb, sc))));
  assert(same(limbs(fma(ba, bb, base_t(2.0f))), limbs(fma(sa, sb, 2.0f))));

  // math wrappers against the same functions on the wrapped type
  assert(same(limbs(exp(bc)), limbs(exp(sc))));
  assert(same(limbs(log(bb)), limbs(log(sb))));
  assert(same(limbs(pow(bb, bc)), limbs(pow(sb, sc))));
  assert(same(limbs(hypot(ba, bb)), limbs(hypot(sa, sb))));
  assert(same(limbs(fabs(-ba)), limbs(fabs(-sa))));
  assert(same(limbs(fmax(ba, bb)), limbs(fmax(sa, sb))));
  assert(same(limbs(ldexp(ba, 3)), limbs(ldexp(sa, 3))));
  assert(same(limbs(norm3d(ba, bb, bc)), limbs(norm3d(sa, sb, sc))));
  assert(ilogb(bb) == ilogb(sb));
  assert(lround(bb) == lround(sb));

  // the standard spellings must reach the emulated implementation, not narrow to double
  assert(same(limbs(cuda::std::exp(bc)), limbs(cuda::std::exp(sc))));
  assert(same(limbs(cuda::std::hypot(ba, bb)), limbs(cuda::std::hypot(sa, sb))));

  { // out-pointer functions
    base_t bs, bcos;
    stat_t ss, scos;
    sincos(bc, &bs, &bcos);
    sincos(sc, &ss, &scos);
    assert(same(limbs(bs), limbs(ss)));
    assert(same(limbs(bcos), limbs(scos)));

    base_t bi;
    stat_t si;
    const base_t bf = modf(bb, &bi);
    const stat_t sf = modf(sb, &si);
    assert(same(limbs(bi), limbs(si)));
    assert(same(limbs(bf), limbs(sf)));

    int bq = 0, sq = 0;
    assert(same(limbs(remquo(bb, bc, &bq)), limbs(remquo(sb, sc, &sq))));
    assert(bq == sq);
  }

  // comparisons, against the wrapper, the wrapped type and a scalar
  assert(sa == sa);
  assert(sa != sb);
  assert(sa < sb);
  assert(!(sa > sb));
  assert(sa <= sb);
  assert(sb >= sa);
  assert(sa == ba);
  assert(ba == sa);
  assert(sa < bb);
  assert(bb > sa);
  assert(sa != 0.0f);
  assert(0.0f < sa);

  // classification
  assert(fpmp_isfinite(sa) == fpmp_isfinite(ba));
  assert(fpmp_isnan(sa) == fpmp_isnan(ba));
  assert(isfinite(sa) != 0);
  assert(isnan(sa) == 0);

  // conversions: to and from the wrapped type, to built-in types, and across accuracy
  const base_t to_base   = sa; // implicit
  const stat_t from_base = ba; // implicit
  assert(same(limbs(to_base), limbs(sa)));
  assert(same(limbs(from_base), limbs(ba)));
  assert(static_cast<double>(sa) == static_cast<double>(ba));
  assert(static_cast<float>(sa) == static_cast<float>(ba));
  assert(static_cast<int>(sb) == static_cast<int>(bb));

  const base_low_t base_low       = static_cast<base_low_t>(ba);
  const stat_low_t stat_low       = static_cast<stat_low_t>(sa);
  const stat_low_t stat_from_base = static_cast<stat_low_t>(ba);
  assert(same(limbs(base_low), limbs(stat_low)));
  assert(same(limbs(base_low), limbs(stat_from_base)));

  { // volatile storage round-trip, the pattern used for shared-memory scalars
    volatile stat_t vs = sa;
    const stat_t loaded(vs);
    assert(same(limbs(sa), limbs(loaded)));
    assert(vs.hi() == sa.hi() && vs.lo() == sa.lo());
    vs = sb;
    const stat_t reloaded(vs);
    assert(same(limbs(sb), limbs(reloaded)));
  }

  { // trivially copyable in practice: a byte copy carries the value
    stat_t dst(0.0f);
    cuda::std::memcpy(&dst, &sa, sizeof(stat_t));
    assert(same(limbs(dst), limbs(sa)));
  }

  { // the double-double instantiation, so both limb types are exercised
    using dbase_t = cudax::fp64mp2;
    using dstat_t = cudax::fp64mp2_stat;
    const dbase_t db(1.25);
    const dstat_t ds(1.25);
    const dbase_t br = (db + db) * db / db - db;
    const dstat_t sr = (ds + ds) * ds / ds - ds;
    assert(br.hi() == sr.hi() && br.lo() == sr.lo());
    // sqrt is native to fpmp for both limb types; the transcendentals of a double-double
    // fall back to binary128 on the host, which would pull libquadmath into this test.
    assert(sqrt(db).hi() == sqrt(ds).hi());
    assert(fabs(-db).hi() == fabs(-ds).hi());
  }
}

#if _CCCL_CUDA_COMPILATION()

__global__ void parity_kernel()
{
  test_parity();
}

// A kernel by one thread with a hand-counted operation mix.
constexpr unsigned long long int expected_add = 3ull;
constexpr unsigned long long int expected_sub = 3ull;
constexpr unsigned long long int expected_mul = 2ull;
constexpr unsigned long long int expected_div = 2ull;
constexpr unsigned long long int expected_ops = expected_add + expected_sub + expected_mul + expected_div;

__global__ void counting_kernel(float* sink)
{
  stat_t a(1.5f), b(0.25f);

  stat_t s = a + b; // add
  s        = s - b; // sub
  s        = s * b; // mul
  s        = s / b; // div
  s += a; // add
  s -= b; // sub
  s *= a; // mul
  s /= a; // div
  ++s; // add
  --s; // sub

  *sink = static_cast<float>(s);
}

// The exact operands of the counting kernel produce a zero lo limb, so a gap is only
// sampled where the arithmetic actually needed the second limb: the range stays empty
// otherwise, which is the sentinel pair rather than an ordered range.
void check_slot_sampled(const cudax::fpmp2_stat_value& slot)
{
  assert(slot.min_exp <= slot.max_exp);
  assert(slot.min_hi_lo_gap <= slot.max_hi_lo_gap
         || (slot.min_hi_lo_gap == cuda::std::numeric_limits<int>::max()
             && slot.max_hi_lo_gap == cuda::std::numeric_limits<int>::min()));
  // The operands are small exact values, so nothing degenerate should appear.
  assert(slot.nan_count == 0ull);
  assert(slot.inf_count == 0ull);
  assert(slot.infnan_count == 0ull);
  assert(slot.zero_count == 0ull);
  assert(slot.denorm_count == 0ull);
  // def accuracy renormalizes, so no pair may have overlapping limbs, let alone inverted ones.
  assert(slot.overlap_count == 0ull);
  assert(slot.invert_count == 0ull);
}

// Atomic accumulation must reach the same total as the wrapped type, return the old
// value the same way, and be counted. One thread per lane adds its own value.
constexpr int atomic_threads = 32;

__global__ void atomic_kernel(base_t* base_total, stat_t* stat_total, float* base_olds, float* stat_olds)
{
  const int lane      = static_cast<int>(threadIdx.x);
  const float summand = 1.0f + static_cast<float>(lane) / 3.0f;

  base_olds[lane] = atomicAdd(base_total, base_t(summand)).hi();
  stat_olds[lane] = atomicAdd(stat_total, stat_t(summand)).hi();
}

// An inexact quotient needs both limbs, so its summary must carry a gap sample. In a
// normalized double-float |lo| <= ulp(hi)/2, so the reported gap - the exponent difference
// with the mantissa width taken out - must be at least zero.
__global__ void gap_kernel(float* sink)
{
  const stat_t third = stat_t(1.0f) / stat_t(3.0f);
  *sink              = third.lo();
}

// Subnormals must be recognized and must not distort the exponent or gap measurements,
// which is why a subnormal limb is measured by its leading bit rather than by its encoded
// exponent field. The product underflows into the subnormal range; the quotient keeps a
// normal hi whose tail cannot stay normal, since a normalized lo sits `digits` binades
// lower. Both are one operation each.
__global__ void denorm_kernel(float* sink)
{
  const stat_t product = stat_t(1e-30f) * stat_t(1e-12f); // about 1e-42, subnormal
  const stat_t tail    = stat_t(1e-35f) / stat_t(3.0f); // hi about 3.3e-36, lo subnormal
  *sink                = product.hi() + tail.lo();
}

// Overlapping limbs, which only the low accuracy level produces: it skips renormalization,
// so a chain of additions leaves pairs whose lo limb reaches up into hi's range. The gap
// range alone would say only how bad the worst pair was, which is why a count goes with it.
__global__ void overlap_kernel(float* sink)
{
  stat_low_t s = stat_low_t(1.0f);
  for (int i = 0; i < 8; ++i)
  {
    s = s + stat_low_t(1.0f / 3.0f);
  }
  *sink = s.hi();
}

// A pair whose hi cancels to exactly zero while lo survives, so the pair is not zero and its
// magnitude is lo's. Reachable through the two-limb constructor at any accuracy level, and
// through arithmetic at the low one. Only five leading bits are lost here, far short of the
// digits / 2 threshold, so measuring such a pair by its hi limb - whose exponent field is
// pinned for a zero - would both misplace the exponent and invent a deep cancellation.
__global__ void zero_hi_kernel(float* sink)
{
  const stat_low_t a = stat_low_t(1.0f, ldexpf(1.0f, -5));
  const stat_low_t b = stat_low_t(1.0f, 0.0f);
  const stat_low_t r = a - b;
  *sink              = r.hi() + r.lo();
}

// Inverted limbs, the worst shape a pair can take: the tail outweighs the head, so reading the
// value through hi answers with the wrong magnitude and the wrong sign. Both limbs are non-zero
// here, unlike in zero_hi_kernel, so the pair is also an overlap and must be counted as both.
// The subtrahend is ordinary, which keeps the count attributable to one operand slot.
__global__ void invert_kernel(float* sink)
{
  const stat_low_t a = stat_low_t(ldexpf(-1.0f, -20), 1.0f);
  const stat_low_t b = stat_low_t(1.0f, 0.0f);
  const stat_low_t r = a - b;
  *sink              = r.hi() + r.lo();
}

// One operation of each classified kind, plus the cases that must not be classified. The
// operand magnitudes are chosen so that float limbs reach the ends of their range.
__global__ void event_kernel(float* sink)
{
  float acc = 0.0f;

  // full cancellation: equal and opposite, so the difference is an exact zero
  acc += (stat_t(1.375f) + stat_t(-1.375f)).hi();
  acc += (stat_t(1.375f) - stat_t(1.375f)).hi();

  // partial cancellation: 24 of 46 significand bits go, and the result is not zero
  acc += (stat_t(1.0f) - stat_t(0.99999994f)).hi();

  // a routine 1-bit drop, which must not count as cancellation
  acc += (stat_t(1.0f) - stat_t(0.6f)).hi();
  // nor must an addition that keeps its magnitude
  acc += (stat_t(1.0f) + stat_t(1.0f)).hi();

  // complete underflow: too small for even a subnormal
  acc += (stat_t(1e-30f) * stat_t(1e-30f)).hi();
  acc += (stat_t(1e-30f) / stat_t(1e30f)).hi();

  // overflow, which an fpmp2 pair reports as a NaN rather than an infinity
  acc += static_cast<float>(fpmp_isnan(stat_t(1e30f) * stat_t(1e30f)));
  acc += static_cast<float>(fpmp_isnan(stat_t(3e38f) + stat_t(3e38f)));

  // division by zero is non-finite too, but one operand is zero, so it is not an overflow
  acc += static_cast<float>(fpmp_isfinite(stat_t(1.0f) / stat_t(0.0f)));
  // and a zero operand passing through is not an underflow
  acc += (stat_t(0.0f) * stat_t(3.0f)).hi();

  *sink = acc;
}

void test_event_counters()
{
  float* sink = nullptr;
  assert(cudaMalloc(&sink, sizeof(float)) == cudaSuccess);
  assert(cudax::fpmp2_stat_reset_device_data() == cudaSuccess);

  event_kernel<<<1, 1>>>(sink);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  cudax::fpmp2_stat_data d{};
  assert(cudax::fpmp2_stat_read_device_data(&d) == cudaSuccess);

  assert(d.full_cancel_count == 2ull);
  assert(d.partial_cancel_count == 1ull);
  assert(d.underflow_count == 2ull);
  assert(d.overflow_count == 2ull);

  // The classified operations are a subset of the counted ones.
  assert(d.full_cancel_count + d.partial_cancel_count + d.underflow_count + d.overflow_count < d.ops_count);

  assert(cudaFree(sink) == cudaSuccess);
}

void test_device_record()
{
  const int sentinel_max = cuda::std::numeric_limits<int>::max();
  const int sentinel_min = cuda::std::numeric_limits<int>::min();

  // The parity kernel runs first and also counts, so the record is reset afterwards.
  parity_kernel<<<1, 1>>>();
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  float* sink = nullptr;
  assert(cudaMalloc(&sink, sizeof(float)) == cudaSuccess);

  assert(cudax::fpmp2_stat_reset_device_data() == cudaSuccess);

  cudax::fpmp2_stat_data after_reset{};
  assert(cudax::fpmp2_stat_read_device_data(&after_reset) == cudaSuccess);

  assert(after_reset.ops_count == 0ull);
  assert(after_reset.add_count == 0ull);
  assert(after_reset.sub_count == 0ull);
  assert(after_reset.mul_count == 0ull);
  assert(after_reset.div_count == 0ull);
  assert(after_reset.full_cancel_count == 0ull);
  assert(after_reset.partial_cancel_count == 0ull);
  assert(after_reset.underflow_count == 0ull);
  assert(after_reset.overflow_count == 0ull);
  assert(after_reset.result.overlap_count == 0ull);
  // Armed ranges: empty, so the first sample replaces both ends.
  assert(after_reset.result.min_exp == sentinel_max);
  assert(after_reset.result.max_exp == sentinel_min);
  assert(after_reset.result.min_hi_lo_gap == sentinel_max);
  assert(after_reset.result.max_hi_lo_gap == sentinel_min);

  counting_kernel<<<1, 1>>>(sink);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  cudax::fpmp2_stat_data after_run{};
  assert(cudax::fpmp2_stat_read_device_data(&after_run) == cudaSuccess);

  assert(after_run.add_count == expected_add);
  assert(after_run.sub_count == expected_sub);
  assert(after_run.mul_count == expected_mul);
  assert(after_run.div_count == expected_div);
  assert(after_run.ops_count == expected_ops);
  assert(after_run.ops_count == after_run.add_count + after_run.sub_count + after_run.mul_count + after_run.div_count);
  // Ordinary arithmetic on small exact values is none of the classified events.
  assert(after_run.full_cancel_count == 0ull);
  assert(after_run.partial_cancel_count == 0ull);
  assert(after_run.underflow_count == 0ull);
  assert(after_run.overflow_count == 0ull);

  check_slot_sampled(after_run.arg[0]);
  check_slot_sampled(after_run.arg[1]);
  check_slot_sampled(after_run.result);

  // arg[2] is reserved for a future ternary operation and must stay untouched.
  assert(after_run.arg[2].min_exp == sentinel_max);
  assert(after_run.arg[2].max_exp == sentinel_min);

  // A second reset clears what the run recorded.
  assert(cudax::fpmp2_stat_reset_device_data() == cudaSuccess);
  cudax::fpmp2_stat_data after_second_reset{};
  assert(cudax::fpmp2_stat_read_device_data(&after_second_reset) == cudaSuccess);
  assert(after_second_reset.ops_count == 0ull);

  // An inexact result must be summarized with a gap that reflects a normalized pair.
  gap_kernel<<<1, 1>>>(sink);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  cudax::fpmp2_stat_data after_gap{};
  assert(cudax::fpmp2_stat_read_device_data(&after_gap) == cudaSuccess);

  assert(after_gap.div_count == 1ull);
  assert(after_gap.result.min_hi_lo_gap <= after_gap.result.max_hi_lo_gap);
  // The gap is measured against a tightly normalized pair, so def accuracy - which
  // renormalizes - must never report an overlap.
  assert(after_gap.result.min_hi_lo_gap >= 0);
  assert(after_gap.result.overlap_count == 0ull);
  // The operands 1 and 3 are exact, so their own lo limbs are zero.
  assert(after_gap.arg[0].zero_lo_count == 1ull);
  assert(after_gap.arg[1].zero_lo_count == 1ull);
  assert(after_gap.result.zero_lo_count == 0ull);
  // Nothing here comes near the bottom of the exponent range.
  assert(after_gap.result.denorm_count == 0ull);

  { // subnormals: recognized, and measured by their leading bit
    assert(cudax::fpmp2_stat_reset_device_data() == cudaSuccess);
    denorm_kernel<<<1, 1>>>(sink);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    cudax::fpmp2_stat_data after_denorm{};
    assert(cudax::fpmp2_stat_read_device_data(&after_denorm) == cudaSuccess);

    // The two results are subnormal; all four operands are ordinary values.
    assert(after_denorm.result.denorm_count == 2ull);
    assert(after_denorm.arg[0].denorm_count == 0ull);
    assert(after_denorm.arg[1].denorm_count == 0ull);
    // Underflow must reach the subnormal range rather than flush to zero.
    assert(after_denorm.result.zero_count == 0ull);
    // A subnormal lo would fake an overlap if it were measured by its encoded exponent
    // field, which is pinned at the format minimum: the quotient is normalized, so the gap
    // must come out non-negative all the same.
    assert(after_denorm.result.min_hi_lo_gap >= 0);
    assert(after_denorm.result.overlap_count == 0ull);
    // Subnormal exponents lie below the smallest normal one, which the pinned field could
    // never report.
    assert(after_denorm.result.min_exp < cuda::std::numeric_limits<float>::min_exponent - 1);
  }

  { // low accuracy: overlap must be both bounded and counted
    assert(cudax::fpmp2_stat_reset_device_data() == cudaSuccess);
    overlap_kernel<<<1, 1>>>(sink);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    cudax::fpmp2_stat_data after_overlap{};
    assert(cudax::fpmp2_stat_read_device_data(&after_overlap) == cudaSuccess);

    assert(after_overlap.add_count == 8ull);
    assert(after_overlap.result.min_hi_lo_gap < 0);
    // The point of the counter: without it a negative minimum could stand for a single value.
    assert(after_overlap.result.overlap_count > 0ull);
    assert(after_overlap.result.overlap_count <= after_overlap.ops_count);
  }

  { // a pair led by lo, its hi being zero: measured by lo, and not a deep cancellation
    assert(cudax::fpmp2_stat_reset_device_data() == cudaSuccess);
    zero_hi_kernel<<<1, 1>>>(sink);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    cudax::fpmp2_stat_data after_zero_hi{};
    assert(cudax::fpmp2_stat_read_device_data(&after_zero_hi) == cudaSuccess);

    assert(after_zero_hi.sub_count == 1ull);
    // The pair is 2^-5 with nothing in hi, so both ends of the range must report that and
    // not the pinned exponent field of the zero limb.
    assert(after_zero_hi.result.min_exp == -5);
    assert(after_zero_hi.result.max_exp == -5);
    assert(after_zero_hi.result.zero_count == 0ull);
    // Five bits short of the operands against a threshold of digits / 2.
    assert(after_zero_hi.partial_cancel_count == 0ull);
    assert(after_zero_hi.full_cancel_count == 0ull);
    // No gap to report with one limb zero, so the range stays armed and empty.
    assert(after_zero_hi.result.overlap_count == 0ull);
    assert(after_zero_hi.result.min_hi_lo_gap > after_zero_hi.result.max_hi_lo_gap);
    // The one counter that does name this shape: lo outweighs a hi of zero.
    assert(after_zero_hi.result.invert_count == 1ull);
  }

  { // inverted limbs: counted as both an inversion and an overlap
    assert(cudax::fpmp2_stat_reset_device_data() == cudaSuccess);
    invert_kernel<<<1, 1>>>(sink);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    cudax::fpmp2_stat_data after_invert{};
    assert(cudax::fpmp2_stat_read_device_data(&after_invert) == cudaSuccess);

    assert(after_invert.sub_count == 1ull);
    // The first operand is the inverted one, hi = -2^-20 against lo = 1.
    assert(after_invert.arg[0].invert_count == 1ull);
    // Inverting puts lo at or above hi, which is a gap of at most -digits, so an inverted
    // pair is always an overlap too.
    assert(after_invert.arg[0].overlap_count == 1ull);
    assert(after_invert.arg[0].min_hi_lo_gap <= -cuda::std::numeric_limits<float>::digits);
    // The ordinary subtrahend must not be counted, and the pair is measured by lo.
    assert(after_invert.arg[1].invert_count == 0ull);
    assert(after_invert.arg[0].max_exp == 0);
  }

  { // atomics: same total as the wrapped type, and counted
    base_t* base_total = nullptr;
    stat_t* stat_total = nullptr;
    float* base_olds   = nullptr;
    float* stat_olds   = nullptr;
    assert(cudaMallocManaged(&base_total, sizeof(base_t)) == cudaSuccess);
    assert(cudaMallocManaged(&stat_total, sizeof(stat_t)) == cudaSuccess);
    assert(cudaMallocManaged(&base_olds, atomic_threads * sizeof(float)) == cudaSuccess);
    assert(cudaMallocManaged(&stat_olds, atomic_threads * sizeof(float)) == cudaSuccess);
    *base_total = base_t(0.0f);
    *stat_total = stat_t(0.0f);

    assert(cudax::fpmp2_stat_reset_device_data() == cudaSuccess);
    atomic_kernel<<<1, atomic_threads>>>(base_total, stat_total, base_olds, stat_olds);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    assert(base_total->hi() == stat_total->hi());
    assert(base_total->lo() == stat_total->lo());

    cudax::fpmp2_stat_data after_atomics{};
    assert(cudax::fpmp2_stat_read_device_data(&after_atomics) == cudaSuccess);
    assert(after_atomics.add_count == static_cast<unsigned long long int>(atomic_threads));
    assert(after_atomics.ops_count == static_cast<unsigned long long int>(atomic_threads));

    assert(cudaFree(base_total) == cudaSuccess);
    assert(cudaFree(stat_total) == cudaSuccess);
    assert(cudaFree(base_olds) == cudaSuccess);
    assert(cudaFree(stat_olds) == cudaSuccess);
  }

  assert(cudaFree(sink) == cudaSuccess);
}

#endif // _CCCL_CUDA_COMPILATION()

int main(int, char**)
{
  test_parity();
#if _CCCL_CUDA_COMPILATION()
  test_device_record();
  test_event_counters();
#endif // _CCCL_CUDA_COMPILATION()
  return 0;
}
