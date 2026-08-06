// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// error: device math intrinsics are unsupported in tile code
// UNSUPPORTED: nvrtc
// note: the host half of this test launches the kernels through the CUDA runtime API,
// which is not available in NVRTC's device-only translation unit

//===----------------------------------------------------------------------===//
//
//  Unit test: fp32mp2 / fp64mp2 math functions.
//
//  Sanity test that calls every fpmp2 math entry point on a single input and
//  compares against the corresponding double-precision reference. It checks that
//  each function exists, is callable and lands in the right neighbourhood;
//  accuracy to the last bit is the accuracy suite's job, not this one's.
//
//  The functions divide by what their *reference* needs rather than by anything
//  fpmp2 lacks, since fpmp2 math is host and device alike apart from erfinv,
//  erfcinv and erfcx:
//
//    - Where the reference is in ISO <cmath>, the check lives in main(), which
//      force_include.h turns into a __host__ __device__ function that the
//      harness runs on the host and then in a kernel. Those get both.
//    - The rest compare against CUDA-only math (rcbrt, the pi-scaled trig,
//      normcdf, the inverse error functions, cyl_bessel_i*, the vector norms)
//      or against names that exist in glibc and CUDA but not in ISO C++, namely
//      exp10, sincos and the POSIX Bessel functions. Neither group has a
//      portable host spelling, so they stay in kernels and are device-only.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpmp_math>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Comparison helpers, shared by the portable checks and the device-only kernels.
TEST_FUNC bool approx_eq(double a, double b, double tol)
{
  if (a == b)
  {
    return true;
  }
  double diff = ::cuda::std::fabs(a - b);
  double mag  = ::cuda::std::fmax(::cuda::std::fabs(a), ::cuda::std::fabs(b));
  if (mag == 0.0)
  {
    return diff < tol;
  }
  return (diff / mag) < tol;
}

TEST_FUNC bool check(double fpmp_val, double ref_val, double tol)
{
  return approx_eq(fpmp_val, ref_val, tol);
}

TEST_FUNC bool check_int(long long fpmp_val, long long ref_val)
{
  return fpmp_val == ref_val;
}

// ---- checks whose reference is in ISO <cmath> ------------------------------
//
// Reached on the host directly and on the device through force_include.h, so
// these cover both the host fallbacks and the device paths.
template <typename MP2>
TEST_FUNC void test_host_device(double tol)
{
  const double x_val = 1.234567890123;
  const double y_val = 2.345678901234;
  const int n_val    = 3;

  // assert per check rather than an accumulated flag, so a failure names the
  // function that broke instead of just the test.
#define CHECK_1A(name, xv)     assert(check(static_cast<double>(name(MP2(xv))), ::name(xv), tol))
#define CHECK_2A(name, xv, yv) assert(check(static_cast<double>(name(MP2(xv), MP2(yv))), ::name(xv, yv), tol))

  // Exponential / logarithmic
  CHECK_1A(exp, x_val);
  CHECK_1A(exp2, x_val);
  CHECK_1A(expm1, x_val);
  CHECK_1A(log, x_val);
  CHECK_1A(log2, x_val);
  CHECK_1A(log10, x_val);
  CHECK_1A(log1p, x_val);
  CHECK_1A(logb, x_val);

  // Power / root
  CHECK_1A(cbrt, x_val);

  // Trigonometric
  CHECK_1A(sin, x_val);
  CHECK_1A(cos, x_val);
  CHECK_1A(tan, x_val);
  CHECK_1A(asin, 0.5);
  CHECK_1A(acos, 0.5);
  CHECK_1A(atan, x_val);

  // Hyperbolic
  CHECK_1A(sinh, x_val);
  CHECK_1A(cosh, x_val);
  CHECK_1A(tanh, x_val);
  CHECK_1A(acosh, x_val);
  CHECK_1A(asinh, x_val);
  CHECK_1A(atanh, 0.5);

  // Error and gamma
  CHECK_1A(erf, x_val);
  CHECK_1A(erfc, x_val);
  CHECK_1A(lgamma, x_val);
  CHECK_1A(tgamma, x_val);

  // Rounding and absolute value
  CHECK_1A(ceil, x_val);
  CHECK_1A(floor, x_val);
  CHECK_1A(trunc, x_val);
  CHECK_1A(round, x_val);
  CHECK_1A(rint, x_val);
  CHECK_1A(nearbyint, x_val);
  CHECK_1A(fabs, -x_val);

  // Two-argument
  CHECK_2A(pow, x_val, y_val);
  CHECK_2A(atan2, x_val, y_val);
  CHECK_2A(fmax, x_val, y_val);
  CHECK_2A(fmin, x_val, y_val);
  CHECK_2A(fmod, x_val, y_val);
  CHECK_2A(remainder, x_val, y_val);
  CHECK_2A(hypot, x_val, y_val);
  CHECK_2A(copysign, x_val, y_val);
  CHECK_2A(fdim, x_val, y_val);
  CHECK_2A(nextafter, x_val, y_val);

  // min / max are spelled without the f and have no ISO counterpart taking two
  // doubles, so the reference is the comparison itself.
  assert(check(static_cast<double>(max(MP2(x_val), MP2(y_val))), (x_val < y_val) ? y_val : x_val, tol));
  assert(check(static_cast<double>(min(MP2(x_val), MP2(y_val))), (y_val < x_val) ? y_val : x_val, tol));

  // The Boys function has no counterpart in the C library; its reference comes
  // from the definition, F_0(x) = 0.5 * sqrt(pi/x) * erf(sqrt(x)) for x > 0.
  assert(check(static_cast<double>(boys_f0(MP2(x_val))),
               0.5 * ::sqrt(3.14159265358979323846 / x_val) * ::erf(::sqrt(x_val)),
               tol));

  // Scaling by a power of two
  assert(check(static_cast<double>(ldexp(MP2(x_val), n_val)), ::ldexp(x_val, n_val), tol));
  assert(check(static_cast<double>(scalbn(MP2(x_val), n_val)), ::scalbn(x_val, n_val), tol));
  assert(check(static_cast<double>(scalbln(MP2(x_val), (long) n_val)), ::scalbln(x_val, (long) n_val), tol));

  // Integer-returning
  assert(check_int(ilogb(MP2(x_val)), ::ilogb(x_val)));
  assert(check_int(llrint(MP2(x_val)), ::llrint(x_val)));
  assert(check_int(llround(MP2(x_val)), ::llround(x_val)));
  assert(check_int(lrint(MP2(x_val)), ::lrint(x_val)));
  assert(check_int(lround(MP2(x_val)), ::lround(x_val)));

  // Classification
  assert(check_int(fpmp_isfinite(MP2(x_val)), ::cuda::std::isfinite(x_val) ? 1 : 0));
  assert(check_int(fpmp_isinf(MP2(x_val)), ::cuda::std::isinf(x_val) ? 1 : 0));
  assert(check_int(fpmp_isnan(MP2(x_val)), ::cuda::std::isnan(x_val) ? 1 : 0));
  assert(check_int(fpmp_signbit(MP2(x_val)), ::cuda::std::signbit(x_val) ? 1 : 0));

  // Functions returning a second result through a pointer
  {
    int e            = 0;
    const MP2 frac   = frexp(MP2(x_val), &e);
    int ref_e        = 0;
    const double ref = ::frexp(x_val, &ref_e);
    assert(check(static_cast<double>(frac), ref, tol));
    assert(check_int(e, ref_e));
  }
  {
    MP2 ipart;
    const MP2 frac     = modf(MP2(x_val), &ipart);
    double ref_ipart   = 0;
    const double ref_f = ::modf(x_val, &ref_ipart);
    assert(check(static_cast<double>(frac), ref_f, tol));
    assert(check(static_cast<double>(ipart), ref_ipart, tol));
  }
  {
    int quo          = 0;
    const MP2 res    = remquo(MP2(x_val), MP2(y_val), &quo);
    int ref_quo      = 0;
    const double ref = ::remquo(x_val, y_val, &ref_quo);
    assert(check(static_cast<double>(res), ref, tol));
    assert(check_int(quo, ref_quo));
  }

#undef CHECK_1A
#undef CHECK_2A
}

#if _CCCL_CUDA_COMPILATION()

#  define CUDA_CHECK(call)           \
    do                               \
    {                                \
      assert((call) == cudaSuccess); \
    } while (0)

// Result structure stored in managed memory.
struct Result
{
  double fpmp_val;
  double ref_val;
};

// One-argument kernels: f(x) -> fpmp2
#  define DEFINE_KERNEL_1A(name)                          \
    template <typename MP2>                               \
    __global__ void kernel_##name(double x_in, Result* r) \
    {                                                     \
      MP2 x       = MP2(x_in);                            \
      MP2 res     = name(x);                              \
      r->fpmp_val = static_cast<double>(res);             \
      r->ref_val  = ::name(x_in);                         \
    }

DEFINE_KERNEL_1A(exp10)
DEFINE_KERNEL_1A(j0)
DEFINE_KERNEL_1A(j1)
DEFINE_KERNEL_1A(y0)
DEFINE_KERNEL_1A(y1)
DEFINE_KERNEL_1A(cyl_bessel_i0)
DEFINE_KERNEL_1A(cyl_bessel_i1)
DEFINE_KERNEL_1A(sinpi)
DEFINE_KERNEL_1A(cospi)
DEFINE_KERNEL_1A(normcdf)
DEFINE_KERNEL_1A(rcbrt)
DEFINE_KERNEL_1A(erfcinv)
DEFINE_KERNEL_1A(erfinv)
DEFINE_KERNEL_1A(erfcx)

// Three-argument kernels: f(a,b,c) -> fpmp2
#  define DEFINE_KERNEL_3A(name)                                                    \
    template <typename MP2>                                                         \
    __global__ void kernel_##name(double a_in, double b_in, double c_in, Result* r) \
    {                                                                               \
      MP2 a = MP2(a_in), b = MP2(b_in), c = MP2(c_in);                              \
      MP2 res     = name(a, b, c);                                                  \
      r->fpmp_val = static_cast<double>(res);                                       \
      r->ref_val  = ::name(a_in, b_in, c_in);                                       \
    }

DEFINE_KERNEL_3A(norm3d)
DEFINE_KERNEL_3A(rnorm3d)

// Four-argument kernels: f(a,b,c,d) -> fpmp2
#  define DEFINE_KERNEL_4A(name)                                                                 \
    template <typename MP2>                                                                      \
    __global__ void kernel_##name(double a_in, double b_in, double c_in, double d_in, Result* r) \
    {                                                                                            \
      MP2 a = MP2(a_in), b = MP2(b_in), c = MP2(c_in), d = MP2(d_in);                            \
      MP2 res     = name(a, b, c, d);                                                            \
      r->fpmp_val = static_cast<double>(res);                                                    \
      r->ref_val  = ::name(a_in, b_in, c_in, d_in);                                              \
    }

DEFINE_KERNEL_4A(norm4d)
DEFINE_KERNEL_4A(rnorm4d)

// Two-argument kernels: f(x,y) -> fpmp2
#  define DEFINE_KERNEL_2A(name)                                       \
    template <typename MP2>                                            \
    __global__ void kernel_##name(double x_in, double y_in, Result* r) \
    {                                                                  \
      MP2 x       = MP2(x_in);                                         \
      MP2 y       = MP2(y_in);                                         \
      MP2 res     = name(x, y);                                        \
      r->fpmp_val = static_cast<double>(res);                          \
      r->ref_val  = ::name(x_in, y_in);                                \
    }

DEFINE_KERNEL_2A(rhypot)

// sincos / sincospi (use sin+cos / sinpi+cospi to avoid overload clash).
template <typename MP2>
__global__ void kernel_sincos(double x_in, Result* r_sin, Result* r_cos)
{
  MP2 x           = MP2(x_in);
  r_sin->fpmp_val = static_cast<double>(sin(x));
  r_cos->fpmp_val = static_cast<double>(cos(x));
  double sd, cd;
  ::sincos(x_in, &sd, &cd);
  r_sin->ref_val = sd;
  r_cos->ref_val = cd;
}

template <typename MP2>
__global__ void kernel_sincospi(double x_in, Result* r_sin, Result* r_cos)
{
  MP2 x           = MP2(x_in);
  r_sin->fpmp_val = static_cast<double>(sinpi(x));
  r_cos->fpmp_val = static_cast<double>(cospi(x));
  double sd, cd;
  ::sincospi(x_in, &sd, &cd);
  r_sin->ref_val = sd;
  r_cos->ref_val = cd;
}

// normcdfinv (input must be in (0,1)).
template <typename MP2>
__global__ void kernel_normcdfinv(double x_in, Result* r)
{
  MP2 x       = MP2(x_in);
  MP2 res     = normcdfinv(x);
  r->fpmp_val = static_cast<double>(res);
  r->ref_val  = ::normcdfinv(x_in);
}

// Mixed-signature kernels.
template <typename MP2>
__global__ void kernel_jn(int n, double x_in, Result* r)
{
  MP2 x       = MP2(x_in);
  MP2 res     = jn(n, x);
  r->fpmp_val = static_cast<double>(res);
  r->ref_val  = ::jn(n, x_in);
}

template <typename MP2>
__global__ void kernel_yn(int n, double x_in, Result* r)
{
  MP2 x       = MP2(x_in);
  MP2 res     = yn(n, x);
  r->fpmp_val = static_cast<double>(res);
  r->ref_val  = ::yn(n, x_in);
}

// ---- checks whose reference exists only in CUDA ----------------------------
//
// Launched from the host half, so these run on the device only.
template <typename MP2>
static bool test_device(double tol)
{
  const double x_val = 1.234567890123;
  const double y_val = 2.345678901234;
  const double p_val = 0.3;
  const int n_val    = 3;

  Result *r1, *r2;

  CUDA_CHECK(cudaMallocManaged(&r1, sizeof(Result)));
  CUDA_CHECK(cudaMallocManaged(&r2, sizeof(Result)));

  bool ok = true;

#  define RUN_1A(name, xv)                \
    kernel_##name<MP2><<<1, 1>>>(xv, r1); \
    CUDA_CHECK(cudaDeviceSynchronize());  \
    ok = check(r1->fpmp_val, r1->ref_val, tol) && ok;

#  define RUN_2A(name, xv, yv)                \
    kernel_##name<MP2><<<1, 1>>>(xv, yv, r1); \
    CUDA_CHECK(cudaDeviceSynchronize());      \
    ok = check(r1->fpmp_val, r1->ref_val, tol) && ok;

#  define RUN_3A(name, av, bv, cv)                \
    kernel_##name<MP2><<<1, 1>>>(av, bv, cv, r1); \
    CUDA_CHECK(cudaDeviceSynchronize());          \
    ok = check(r1->fpmp_val, r1->ref_val, tol) && ok;

#  define RUN_4A(name, av, bv, cv, dv)                \
    kernel_##name<MP2><<<1, 1>>>(av, bv, cv, dv, r1); \
    CUDA_CHECK(cudaDeviceSynchronize());              \
    ok = check(r1->fpmp_val, r1->ref_val, tol) && ok;

  // Base-10 exponential: exp10 is in glibc and CUDA but not in ISO C++.
  RUN_1A(exp10, x_val)

  // Reciprocal cube root
  RUN_1A(rcbrt, x_val)

  // Error / Probability
  RUN_1A(erfcinv, p_val)
  RUN_1A(erfinv, p_val)
  RUN_1A(erfcx, x_val)
  RUN_1A(normcdf, x_val)

  // Bessel: j0/j1/y0/y1 are POSIX rather than ISO, and MSVC spells them _j0.
  RUN_1A(j0, x_val)
  RUN_1A(j1, x_val)
  RUN_1A(y0, x_val)
  RUN_1A(y1, x_val)
  RUN_1A(cyl_bessel_i0, x_val)
  RUN_1A(cyl_bessel_i1, x_val)

  // CUDA trigonometric (pi-scaled)
  RUN_1A(sinpi, x_val)
  RUN_1A(cospi, x_val)

  // Inverse CDF
  kernel_normcdfinv<MP2><<<1, 1>>>(p_val, r1);
  CUDA_CHECK(cudaDeviceSynchronize());
  ok = check(r1->fpmp_val, r1->ref_val, tol) && ok;

  // Reciprocal hypotenuse
  RUN_2A(rhypot, x_val, y_val)

  // Vector norm (3/4 args)
  RUN_3A(norm3d, x_val, y_val, p_val)
  RUN_3A(rnorm3d, x_val, y_val, p_val)
  RUN_4A(norm4d, x_val, y_val, p_val, 0.7)
  RUN_4A(rnorm4d, x_val, y_val, p_val, 0.7)

  // sincos / sincospi
  kernel_sincos<MP2><<<1, 1>>>(x_val, r1, r2);
  CUDA_CHECK(cudaDeviceSynchronize());
  ok = check(r1->fpmp_val, r1->ref_val, tol) && ok;
  ok = check(r2->fpmp_val, r2->ref_val, tol) && ok;

  kernel_sincospi<MP2><<<1, 1>>>(x_val, r1, r2);
  CUDA_CHECK(cudaDeviceSynchronize());
  ok = check(r1->fpmp_val, r1->ref_val, tol) && ok;
  ok = check(r2->fpmp_val, r2->ref_val, tol) && ok;

  // Bessel functions taking an order
  kernel_jn<MP2><<<1, 1>>>(n_val, x_val, r1);
  CUDA_CHECK(cudaDeviceSynchronize());
  ok = check(r1->fpmp_val, r1->ref_val, tol) && ok;
  kernel_yn<MP2><<<1, 1>>>(n_val, x_val, r1);
  CUDA_CHECK(cudaDeviceSynchronize());
  ok = check(r1->fpmp_val, r1->ref_val, tol) && ok;

#  undef RUN_1A
#  undef RUN_2A
#  undef RUN_3A
#  undef RUN_4A

  cudaFree(r1);
  cudaFree(r2);
  return ok;
}

// The launches must live outside the NV_IF_TARGET(NV_IS_HOST) block in main(): nvcc's device
// pass discards that block, so the kernel templates would never be instantiated for the device
// and every launch would fail with cudaErrorInvalidDeviceFunction.
bool test_device_all()
{
  bool ok = test_device<cudax::fp32mp2>(1e-5);
  ok      = test_device<cudax::fp64mp2>(1e-12) && ok;
  return ok;
}
#endif // _CCCL_CUDA_COMPILATION()

int main(int, char**)
{
  // force_include.h makes this main __host__ __device__ and runs it twice, on the
  // host and then in a kernel, so these cover both without any launch of our own.
  test_host_device<cudax::fp32mp2>(1e-5);
  test_host_device<cudax::fp64mp2>(1e-12);

#if _CCCL_CUDA_COMPILATION()
  // The remaining functions have no portable host reference and so are checked
  // through kernels, which only the host half can launch.
  NV_IF_TARGET(NV_IS_HOST, (assert(test_device_all());))
#endif // _CCCL_CUDA_COMPILATION()
  return 0;
}
