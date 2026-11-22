// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

// fp32mp2 sqrt, rsqrt, fdim, hypot, rhypot, norm3d, rnorm3d, norm4d and rnorm4d:
// within a few u^2 (u = 2^-24) of an MPFR reference over the whole range of each,
// subnormal operands and results and operands whose squares overflow or underflow included;
// the results that must be exact; and the special values (IEEE 754 but for sqrt and rsqrt,
// which take no branch for them), with those of fabs, copysign, fmax and fmin.
//
// sqrt and rsqrt have two kernels, neither of which takes a branch (a positive, finite operand
// with a normal high limb): the default and low accuracy levels get a short one (within 2.5 u^2
// and 8 u^2 for fp32mp2, 13 u^2 for rsqrt on the device), and fpmp2_accuracy::high one that adds
// the second-order term of its Newton step (1.3 u^2, and 4.2 u^2 for rsqrt). Every check runs
// on the default, low and high types; sqrt and rsqrt are
// held to the bound of the kernel the type gets, and are not asked for positive operands below
// their range. The norms and fdim take their roots from the short kernel at every level, except
// the fp32mp2 norm3d, norm4d, rnorm3d and rnorm4d at fpmp2_accuracy::high, which are computed
// in double and held to 2 u^2.

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

enum sqrt_fn
{
  fn_sqrt,
  fn_rsqrt,
  fn_fdim,
  fn_hypot,
  fn_rhypot,
  fn_norm3d,
  fn_rnorm3d,
  fn_norm4d,
  fn_rnorm4d,
  fn_fabs,
  fn_copysign,
  fn_fmax,
  fn_fmin,
  fn_count
};

template <class T>
TEST_HOST_DEVICE_FUNC T call(sqrt_fn fn, T a, T b, T c, T d)
{
  switch (fn)
  {
    case fn_sqrt:
      return cuda::sqrt(a);
    case fn_rsqrt:
      return cuda::rsqrt(a);
    case fn_fdim:
      return cuda::fdim(a, b);
    case fn_hypot:
      return cuda::hypot(a, b);
    case fn_rhypot:
      return cuda::rhypot(a, b);
    case fn_norm3d:
      return cuda::norm3d(a, b, c);
    case fn_rnorm3d:
      return cuda::rnorm3d(a, b, c);
    case fn_norm4d:
      return cuda::norm4d(a, b, c, d);
    case fn_rnorm4d:
      return cuda::rnorm4d(a, b, c, d);
    case fn_fabs:
      return cuda::fabs(a);
    case fn_copysign:
      return cuda::copysign(a, b);
    case fn_fmax:
      return cuda::fmax(a, b);
    default:
      return cuda::fmin(a, b);
  }
}

// Operands as (hi, lo) pairs, unused ones zero; the reference as three doubles that sum to it.
template <class F>
struct acc_case
{
  sqrt_fn fn;
  F x[8];
  double ref[3];
};

// The smallest high limb the sqrt and rsqrt kernels are asked for: below it the residual of the
// Newton step is subnormal.
template <class F>
TEST_HOST_DEVICE_FUNC F short_kernel_min()
{
  return sizeof(F) == 4 ? F(0x1p-78) : F(0x1p-916);
}

// The largest error MPFR sweeps of 1M operand sets per range found on the host and on the
// device, in u^2 of the result. sqrt and rsqrt at the high level: 1.3 and 4.2; at the default
// and low levels: 2.5 and 8 on the host, 13 on the device, whose seed is the approximate
// reciprocal square root. fdim: 2.5. The norms take the roots of the short kernels at every
// level: 4 for hypot, norm3d and norm4d, and the error of the short rsqrt for rhypot, rnorm3d
// and rnorm4d. At the high level norm3d, norm4d, rnorm3d and rnorm4d are computed in double:
// 0.55. The bounds for a double limb type are those of the fp64mp2 kernels, which this test
// does not instantiate.
template <class T, class F, int N>
TEST_HOST_DEVICE_FUNC void check_accuracy(const acc_case<F> (&cases)[N], bool high)
{
  // The short fp64mp2 square root reaches 5.8 u^2 on the host, where its seed is 1 / sqrt(x).
  const double sqrt_default = sizeof(F) == 4 ? 3.5 : 7.0;
  const double norm_high    = sizeof(F) == 4 ? 2.0 : 7.5;
  const double rnorm_high   = sizeof(F) == 4 ? 2.0 : 16.0;
  const double tolerance_high[fn_rnorm4d + 1] = {2.5, 5.0, 3.5, 7.5, 16.0, norm_high, rnorm_high, norm_high, rnorm_high};
  const double tolerance_default[fn_rnorm4d + 1] = {sqrt_default, 16.0, 3.5, 7.5, 16.0, 7.5, 16.0, 7.5, 16.0};
  const double* tolerance                        = high ? tolerance_high : tolerance_default;
  const double u                                 = static_cast<double>(cuda::std::numeric_limits<F>::epsilon()) / 2;
  const double denorm                            = static_cast<double>(cuda::std::numeric_limits<F>::denorm_min());
  for (const acc_case<F>& c : cases)
  {
    if ((c.fn == fn_sqrt || c.fn == fn_rsqrt) && c.x[0] < short_kernel_min<F>())
    {
      continue;
    }
    const T r = call(c.fn, T(c.x[0], c.x[1]), T(c.x[2], c.x[3]), T(c.x[4], c.x[5]), T(c.x[6], c.x[7]));
    // Both differences are exact or nearly so: each limb is within a few ulps of its reference.
    const double d = ((static_cast<double>(r.hi()) - c.ref[0]) + (static_cast<double>(r.lo()) - c.ref[1])) - c.ref[2];
    // A low limb in the subnormal range is rounded to a multiple of the smallest subnormal.
    assert(cuda::std::fabs(d) <= tolerance[c.fn] * u * u * cuda::std::fabs(c.ref[0]) + 1.5 * denorm);
  }
}

// MPFR 4.2.2: the function, its operands, and its value. The ends of each range, operands with a
// low limb, subnormal operands and results, operands whose squares overflow or underflow, and the
// worst operands of the sweeps.
template <class T>
TEST_HOST_DEVICE_FUNC void test_accuracy_fp32mp2(bool high)
{
  const acc_case<float> cases[] = {
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_sqrt, {0x1p-149f, 0x0p+0f}, {0x1.6a09e667f3bcdp-75, -0x1.bdd3413b26456p-129, 0x1.57d3e3adec175p-183}},
    {fn_sqrt, {0x1.8p-148f, 0x0p+0f}, {0x1.3988e1409212ep-74, 0x1.f40c86450c869p-128, 0x1.56473db022875p-182}},
    {fn_sqrt, {0x1.fffffcp-127f, 0x0p+0f}, {0x1.fffffdffffffp-64, -0x1.000001400001cp-136, -0x1.5000021000036p-207}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_sqrt, {0x1p-126f, 0x0p+0f}, {0x1p-63, 0x0p+0, 0x0p+0}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_sqrt, {0x1p-126f, 0x1p-149f}, {0x1.000000ffffff8p-63, 0x1.fffffd8000038p-137, -0x1.4ffffdf000036p-207}},
    {fn_sqrt, {0x1p-125f, -0x1p-149f}, {0x1.6a09e5b2eec96p-63, 0x1.f365ecbbfdd7dp-117, 0x1.c524c1d7584d4p-173}},
    {fn_sqrt, {0x1p-103f, 0x1p-128f}, {0x1.6a09e6c276365p-52, 0x1.ed54bf3a8263cp-106, -0x1.35546047b03d5p-166}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_sqrt, {0x1p+1f, 0x0p+0f}, {0x1.6a09e667f3bcdp+0, -0x1.bdd3413b26456p-54, 0x1.57d3e3adec175p-108}},
    {fn_sqrt, {0x1p+1f, 0x1p-24f}, {0x1.6a09e6c276365p+0, 0x1.ed54bf3a8263cp-54, -0x1.35546047b03d5p-114}},
    {fn_sqrt, {0x1.8p+1f, -0x1p-23f}, {0x1.bb67adf1b7906p+0, 0x1.0bac1667072fap-55, 0x1.ebf0922839aep-113}},
    {fn_sqrt, {0x1p-1f, 0x0p+0f}, {0x1.6a09e667f3bcdp-1, -0x1.bdd3413b26456p-55, 0x1.57d3e3adec175p-109}},
    {fn_sqrt, {0x1p+0f, 0x1p-25f}, {0x1.0000004p+0, -0x1.ffffff8000002p-54, -0x1.fffffc8000015p-108}},
    {fn_sqrt, {0x1p+0f, -0x1p-26f}, {0x1.ffffffcp-1, -0x1.0000002p-55, -0x1.4000003800001p-109}},
    {fn_sqrt, {0x1.fffffep+125f, 0x0p+0f}, {0x1.fffffeffffffcp+62, -0x1.000000a000007p-13, -0x1.500001080000dp-87}},
    {fn_sqrt, {0x1.fffffep+127f, 0x0p+0f}, {0x1.fffffeffffffcp+63, -0x1.000000a000007p-12, -0x1.500001080000dp-86}},
    {fn_sqrt,
     {0x1.fffffep+127f, 0x1.fffffep+102f},
     {0x1.ffffff7fffff7p+63, -0x1.200000ea00009p-12, 0x1.1fffe44bffeb2p-66}},
    {fn_sqrt,
     {0x1.fffffep+127f, -0x1.fffffep+102f},
     {0x1.fffffe7ffffffp+63, -0x1.800001280000fp-14, 0x1.7fffd0afffd8fp-68}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rsqrt, {0x1p-149f, 0x0p+0f}, {0x1.6a09e667f3bcdp+74, -0x1.bdd3413b26456p+20, 0x1.57d3e3adec175p-34}},
    {fn_rsqrt, {0x1.8p-148f, 0x0p+0f}, {0x1.a20bd700c2c3ep+73, -0x1.fde99f28943c7p+14, -0x1.25d6dfe8fb206p-42}},
    {fn_rsqrt, {0x1.fffffcp-127f, 0x0p+0f}, {0x1.0000010000018p+63, 0x1.400002300003fp-8, 0x1.ce00035a00065p-78}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rsqrt, {0x1p-126f, 0x0p+0f}, {0x1p+63, 0x0p+0, 0x0p+0}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rsqrt, {0x1p-126f, 0x1p-149f}, {0x1.fffffe000003p+62, -0x1.3ffffdd00003fp-8, 0x1.cdfffca600065p-78}},
    {fn_rsqrt, {0x1p-125f, -0x1p-149f}, {0x1.6a09e71cf8b08p+62, 0x1.31920ded8fe5cp+8, -0x1.55eafda73a632p-49}},
    {fn_rsqrt, {0x1p-103f, 0x1p-128f}, {0x1.6a09e60d71435p+51, 0x1.f962a8d761c99p-6, 0x1.da388300f7af3p-61}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rsqrt, {0x1p+1f, 0x0p+0f}, {0x1.6a09e667f3bcdp-1, -0x1.bdd3413b26456p-55, 0x1.57d3e3adec175p-109}},
    {fn_rsqrt, {0x1p+1f, 0x1p-24f}, {0x1.6a09e60d71435p-1, 0x1.f962a8d761c99p-58, 0x1.da388300f7af3p-113}},
    {fn_rsqrt, {0x1.8p+1f, -0x1p-23f}, {0x1.279a74bb8c037p-1, -0x1.ba1b67046534dp-55, 0x1.48186c4712af7p-110}},
    {fn_rsqrt, {0x1p-1f, 0x0p+0f}, {0x1.6a09e667f3bcdp+0, -0x1.bdd3413b26456p-54, 0x1.57d3e3adec175p-108}},
    {fn_rsqrt, {0x1p+0f, 0x1p-25f}, {0x1.ffffff8000003p-1, -0x1.3fffff7400004p-77, 0x1.00001cdffff2ap-133}},
    {fn_rsqrt, {0x1p+0f, -0x1p-26f}, {0x1.0000002p+0, 0x1.8000005000001p-54, 0x1.800003f00000ep-110}},
    {fn_rsqrt, {0x1.fffffep+125f, 0x0p+0f}, {0x1.0000008000006p-63, 0x1.400001180001p-137, -0x1.ffff8c7fff94cp-192}},
    {fn_rsqrt, {0x1.fffffep+127f, 0x0p+0f}, {0x1.0000008000006p-64, 0x1.400001180001p-138, -0x1.ffff8c7fff94cp-193}},
    {fn_rsqrt,
     {0x1.fffffep+127f, 0x1.fffffep+102f},
     {0x1.0000004000006p-64, -0x1.fffff17ffff16p-118, -0x1.fffce87ffd101p-172}},
    {fn_rsqrt,
     {0x1.fffffep+127f, -0x1.fffffep+102f},
     {0x1.000000c00000ap-64, -0x1.ffffe07fffe4ep-118, -0x1.fffa017ffa949p-172}},
    {fn_hypot,
     {0x1p+0f, 0x0p+0f, 0x1p+0f, 0x0p+0f},
     {0x1.6a09e667f3bcdp+0, -0x1.bdd3413b26456p-54, 0x1.57d3e3adec175p-108}},
    {fn_hypot,
     {0x1p+0f, 0x1p-25f, 0x1p-23f, 0x0p+0f},
     {0x1.000000800002p+0, -0x1.0000017ffffd4p-72, 0x1.1ffff87000033p-144}},
    {fn_hypot,
     {0x1.8p+1f, 0x1p-24f, -0x1p+2f, 0x1p-23f},
     {0x1.3fffffc000002p+2, -0x1.99999851eb852p-52, 0x1.47adfa43fe54bp-108}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_hypot,
     {0x1p-149f, 0x0p+0f, 0x1p-149f, 0x0p+0f},
     {0x1.6a09e667f3bcdp-149, -0x1.bdd3413b26456p-203, 0x1.57d3e3adec175p-257}},
    {fn_hypot,
     {0x1p-126f, 0x0p+0f, -0x1p-126f, 0x0p+0f},
     {0x1.6a09e667f3bcdp-126, -0x1.bdd3413b26456p-180, 0x1.57d3e3adec175p-234}},
    {fn_hypot,
     {0x1.fffffcp-127f, 0x0p+0f, 0x1.8p-148f, 0x0p+0f},
     {0x1.fffffc000024p-127, 0x1.1ffffd2fffe62p-193, -0x1.55ffdf3bff742p-261}},
    {fn_hypot,
     {0x1p-80f, 0x0p+0f, 0x1p-80f, 0x0p+0f},
     {0x1.6a09e667f3bcdp-80, -0x1.bdd3413b26456p-134, 0x1.57d3e3adec175p-188}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_hypot,
     {0x1.8p+80f, 0x0p+0f, -0x1.8p+80f, 0x0p+0f},
     {0x1.0f876ccdf6cd9p+81, 0x1.b1a18f13a34cp+27, -0x1.fc422a7b1dddp-28}},
    {fn_hypot,
     {0x1.fffffep+126f, 0x0p+0f, 0x1.fffffep+126f, 0x0p+0f},
     {0x1.6a09e4fde9d66p+127, 0x1.14f6320ab3ef8p+71, 0x1.0b29d2b0c19d2p+16}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_hypot,
     {0x1.fffffep+125f, 0x1.fffffep+99f, 0x1p-149f, 0x0p+0f},
     {0x1.fffffe7fffff8p+125, 0x1.000000c00000dp-425, 0x1.9800019a0001ap-498}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_hypot, {0x1p+0f, 0x0p+0f, 0x1p-80f, 0x0p+0f}, {0x1p+0, 0x1p-161, -0x1p-323}},
    {fn_hypot, {0x1.8p+80f, 0x0p+0f, 0x1p+0f, 0x0p+0f}, {0x1.8p+80, 0x1.5555555555555p-82, 0x1.5555555555555p-136}},
    {fn_hypot,
     {0x1.fffffep+127f, 0x0p+0f, 0x1p-126f, 0x0p+0f},
     {0x1.fffffep+127, 0x1.000001000001p-381, 0x1.000001000001p-453}},
    {fn_hypot,
     {0x1.2a05f2p+33f, 0x0p+0f, 0x1.b7cdfep-34f, 0x0p+0f},
     {0x1.2a05f2p+33, 0x1.4484c0801e864p-101, 0x1.ba504d94403afp-156}},
    {fn_rhypot,
     {0x1p+0f, 0x0p+0f, 0x1p+0f, 0x0p+0f},
     {0x1.6a09e667f3bcdp-1, -0x1.bdd3413b26456p-55, 0x1.57d3e3adec175p-109}},
    {fn_rhypot,
     {0x1p+0f, 0x1p-25f, 0x1p-23f, 0x0p+0f},
     {0x1.fffffefffffc8p-1, 0x1.70000187fff9cp-71, -0x1.ffffd6fff4bc8p-126}},
    {fn_rhypot,
     {0x1.8p+1f, 0x1p-24f, -0x1p+2f, 0x1p-23f},
     {0x1.999999eb851eap-3, 0x1.fbe76872b01ffp-57, 0x1.7247735c18513p-112}},
    {fn_rhypot,
     {0x1p-126f, 0x0p+0f, -0x1p-126f, 0x0p+0f},
     {0x1.6a09e667f3bcdp+125, -0x1.bdd3413b26456p+71, 0x1.57d3e3adec175p+17}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rhypot,
     {0x1.fffffcp-127f, 0x0p+0f, 0x1.8p-148f, 0x0p+0f},
     {0x1.000001fffff2p+126, -0x1.8ffffee7ffca1p+60, 0x1.429ff63bbfaf4p-5}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rhypot,
     {0x1p-80f, 0x0p+0f, 0x1p-80f, 0x0p+0f},
     {0x1.6a09e667f3bcdp+79, -0x1.bdd3413b26456p+25, 0x1.57d3e3adec175p-29}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rhypot,
     {0x1.8p+80f, 0x0p+0f, -0x1.8p+80f, 0x0p+0f},
     {0x1.e2b7dddfefa66p-82, 0x1.60eea419de8e2p-138, -0x1.ac812e1629b1fp-193}},
    {fn_rhypot,
     {0x1.fffffep+126f, 0x0p+0f, 0x1.fffffep+126f, 0x0p+0f},
     {0x1.6a09e7d1fda4ap-128, -0x1.3e6a1a79905fdp-182, -0x1.0c44115e57fa1p-236}},
    {fn_rhypot,
     {0x1.fffffep+125f, 0x1.fffffep+99f, 0x1p-149f, 0x0p+0f},
     {0x1.000000c00000dp-126, 0x1.9800019a0001ap-199, -0x1.9fff9997ff99ap-253}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rhypot, {0x1p+0f, 0x0p+0f, 0x1p-80f, 0x0p+0f}, {0x1p+0, -0x1p-161, 0x1.8p-322}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rhypot,
     {0x1.8p+80f, 0x0p+0f, 0x1p+0f, 0x0p+0f},
     {0x1.5555555555555p-81, 0x1.5555555555555p-135, 0x1.5555555555555p-189}},
    {fn_rhypot,
     {0x1.fffffep+127f, 0x0p+0f, 0x1p-126f, 0x0p+0f},
     {0x1.000001000001p-128, 0x1.000001000001p-200, 0x1.000001000001p-272}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rhypot,
     {0x1.2a05f2p+33f, 0x0p+0f, 0x1.b7cdfep-34f, 0x0p+0f},
     {0x1.b7cdfd9d7bdbbp-34, -0x1.20a5465df8d2cp-88, 0x1.09a3658026f64p-142}},
    {fn_norm3d,
     {0x1p+0f, 0x0p+0f, 0x1p+0f, 0x0p+0f, 0x1p+0f, 0x0p+0f},
     {0x1.bb67ae8584caap+0, 0x1.cec95d0b5c1e3p-54, -0x1.f11db689f2ccfp-110}},
    {fn_norm3d,
     {0x1p+0f, 0x1p-25f, -0x1p+1f, 0x0p+0f, 0x1.8p+1f, -0x1p-23f},
     {0x1.deeea05a5d10ep+1, -0x1.57e8c781ac2d9p-53, 0x1.b4db12b1763cap-107}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_norm3d,
     {0x1p-149f, 0x0p+0f, 0x1p-149f, 0x0p+0f, 0x1p-149f, 0x0p+0f},
     {0x1.bb67ae8584caap-149, 0x1.cec95d0b5c1e3p-203, -0x1.f11db689f2ccfp-259}},
    {fn_norm3d,
     {0x1p-126f, 0x0p+0f, 0x1p-126f, 0x1p-149f, -0x1p-126f, 0x0p+0f},
     {0x1.bb67afad1f3fcp-126, 0x1.53fe66e824e1ap-180, 0x1.88e9a66bfda3p-238}},
    {fn_norm3d,
     {0x1p-80f, 0x0p+0f, 0x1p-80f, 0x0p+0f, -0x1p-80f, 0x0p+0f},
     {0x1.bb67ae8584caap-80, 0x1.cec95d0b5c1e3p-134, -0x1.f11db689f2ccfp-190}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_norm3d,
     {0x1.8p+80f, 0x0p+0f, 0x1.8p+80f, 0x0p+0f, 0x1.8p+80f, 0x0p+0f},
     {0x1.4c8dc2e42398p+81, -0x1.49d1f46ef5d2cp+26, 0x1.4594db8c44f32p-28}},
    {fn_norm3d,
     {0x1.fffffep+126f, 0x0p+0f, 0x1.fffffep+125f, 0x0p+0f, 0x1.fffffep+126f, 0x0p+0f},
     {0x1.7ffffe8p+127, 0x0p+0, 0x0p+0}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_norm3d,
     {0x1.fffffep+126f, 0x0p+0f, 0x1p+0f, 0x0p+0f, 0x1p-149f, 0x0p+0f},
     {0x1.fffffep+126, 0x1.000001000001p-128, 0x1.000001000001p-200}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rnorm3d,
     {0x1p+0f, 0x0p+0f, 0x1p+0f, 0x0p+0f, 0x1p+0f, 0x0p+0f},
     {0x1.279a74590331cp-1, 0x1.34863e0792bedp-55, -0x1.a82f9e6c53222p-109}},
    {fn_rnorm3d,
     {0x1p+0f, 0x1p-25f, -0x1p+1f, 0x0p+0f, 0x1.8p+1f, -0x1p-23f},
     {0x1.11aceec18632ap-2, -0x1.556cacb71d45fp-57, 0x1.9b79be860d3b9p-111}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rnorm3d,
     {0x1p-126f, 0x0p+0f, 0x1p-126f, 0x1p-149f, -0x1p-126f, 0x0p+0f},
     {0x1.279a7393f18eep+125, -0x1.30078696fcc6ap+69, -0x1.269fc8130debbp+11}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rnorm3d,
     {0x1p-80f, 0x0p+0f, 0x1p-80f, 0x0p+0f, -0x1p-80f, 0x0p+0f},
     {0x1.279a74590331cp+79, 0x1.34863e0792bedp+25, -0x1.a82f9e6c53222p-29}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rnorm3d,
     {0x1.8p+80f, 0x0p+0f, 0x1.8p+80f, 0x0p+0f, 0x1.8p+80f, 0x0p+0f},
     {0x1.8a2345cc04426p-82, -0x1.0f4d02a091ac5p-136, 0x1.ca6b2cc4e67d2p-190}},
    {fn_rnorm3d,
     {0x1.fffffep+126f, 0x0p+0f, 0x1.fffffep+125f, 0x0p+0f, 0x1.fffffep+126f, 0x0p+0f},
     {0x1.555556aaaaacp-128, 0x1.555556aaaaacp-200, 0x1.555556aaaaacp-272}},
    {fn_rnorm3d,
     {0x1.fffffep+126f, 0x0p+0f, 0x1p+0f, 0x0p+0f, 0x1p-149f, 0x0p+0f},
     {0x1.000001000001p-127, 0x1.000001000001p-199, 0x1.000001000001p-271}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_norm4d,
     {0x1p+0f, 0x0p+0f, 0x1p+1f, 0x0p+0f, 0x1.8p+1f, 0x0p+0f, 0x1.4p+2f, 0x0p+0f},
     {0x1.8fae0c15ad38ap+2, -0x1.db7adb6817f6dp-54, 0x1.792d5c133a726p-108}},
    {fn_norm4d,
     {0x1p+0f, 0x1p-25f, -0x1p+1f, 0x0p+0f, 0x1.8p+1f, -0x1p-23f, 0x1p-1f, 0x1p-26f},
     {0x1.e3307c0f2b837p+1, 0x1.86820e34e6f19p-53, -0x1.77142565d7df7p-110}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_norm4d,
     {0x1p-149f, 0x0p+0f, 0x1p-149f, 0x0p+0f, 0x1p-149f, 0x0p+0f, 0x1p-148f, 0x0p+0f},
     {0x1.52a7fa9d2f8eap-148, -0x1.21c62b033c079p-202, -0x1.77caad6200612p-259}},
    {fn_norm4d,
     {0x1p-126f, 0x0p+0f, 0x1p-126f, 0x1p-149f, -0x1p-126f, 0x0p+0f, 0x1p-149f, 0x0p+0f},
     {0x1.bb67afad1f40fp-126, -0x1.8ac96836eefeap-181, 0x1.1616a400288c7p-235}},
    {fn_norm4d,
     {0x1p-80f, 0x0p+0f, 0x1p-80f, 0x0p+0f, -0x1p-80f, 0x0p+0f, 0x1p-80f, 0x0p+0f},
     {0x1p-79, 0x0p+0, 0x0p+0}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_norm4d,
     {0x1.8p+80f, 0x0p+0f, 0x1.8p+80f, 0x0p+0f, 0x1.8p+80f, 0x0p+0f, -0x1.8p+80f, 0x0p+0f},
     {0x1.8p+81, 0x0p+0, 0x0p+0}},
    {fn_norm4d,
     {0x1.fffffep+125f, 0x0p+0f, 0x1.fffffep+125f, 0x0p+0f, 0x1.fffffep+125f, 0x0p+0f, 0x1.fffffep+125f, 0x0p+0f},
     {0x1.fffffep+126, 0x0p+0, 0x0p+0}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_norm4d,
     {0x1.fffffep+126f, 0x0p+0f, 0x1p+0f, 0x0p+0f, 0x1p-149f, 0x0p+0f, -0x1.fffffep+126f, 0x0p+0f},
     {0x1.6a09e4fde9d66p+127, 0x1.14f6320ab3ef8p+71, 0x1.0b29d2b0c19d2p+16}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rnorm4d,
     {0x1p+0f, 0x0p+0f, 0x1p+1f, 0x0p+0f, 0x1.8p+1f, 0x0p+0f, 0x1.4p+2f, 0x0p+0f},
     {0x1.47f144fe17f9fp-3, 0x1.590ac8b1276ap-59, 0x1.357a8d3098fb6p-113}},
    {fn_rnorm4d,
     {0x1p+0f, 0x1p-25f, -0x1p+1f, 0x0p+0f, 0x1.8p+1f, -0x1p-23f, 0x1p-1f, 0x1p-26f},
     {0x1.0f43a4c330827p-2, -0x1.df553e4f4ac41p-56, -0x1.3b8629c269355p-111}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rnorm4d,
     {0x1p-126f, 0x0p+0f, 0x1p-126f, 0x1p-149f, -0x1p-126f, 0x0p+0f, 0x1p-149f, 0x0p+0f},
     {0x1.279a7393f18e2p+125, -0x1.906a389d761a9p+71, 0x1.203623792c5c8p+17}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rnorm4d,
     {0x1p-80f, 0x0p+0f, 0x1p-80f, 0x0p+0f, -0x1p-80f, 0x0p+0f, 0x1p-80f, 0x0p+0f},
     {0x1p+79, 0x0p+0, 0x0p+0}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rnorm4d,
     {0x1.8p+80f, 0x0p+0f, 0x1.8p+80f, 0x0p+0f, 0x1.8p+80f, 0x0p+0f, -0x1.8p+80f, 0x0p+0f},
     {0x1.5555555555555p-82, 0x1.5555555555555p-136, 0x1.5555555555555p-190}},
    {fn_rnorm4d,
     {0x1.fffffep+125f, 0x0p+0f, 0x1.fffffep+125f, 0x0p+0f, 0x1.fffffep+125f, 0x0p+0f, 0x1.fffffep+125f, 0x0p+0f},
     {0x1.000001000001p-127, 0x1.000001000001p-199, 0x1.000001000001p-271}},
    {fn_rnorm4d,
     {0x1.fffffep+126f, 0x0p+0f, 0x1p+0f, 0x0p+0f, 0x1p-149f, 0x0p+0f, -0x1.fffffep+126f, 0x0p+0f},
     {0x1.6a09e7d1fda4ap-128, -0x1.3e6a1a79905fdp-182, -0x1.0c44115e57fa1p-236}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_fdim, {0x1.8p+1f, 0x1p-24f, 0x1.8p+1f, -0x1p-24f}, {0x1p-23, 0x0p+0, 0x0p+0}},
    {fn_fdim, {0x1p+0f, 0x1p-25f, 0x1p+0f, -0x1p-25f}, {0x1p-24, 0x0p+0, 0x0p+0}},
    {fn_fdim, {0x1p+0f, 0x0p+0f, 0x1p-23f, 0x0p+0f}, {0x1.fffffcp-1, 0x0p+0, 0x0p+0}},
    {fn_fdim, {0x1p+1f, 0x0p+0f, -0x1.8p+1f, 0x1p-23f}, {0x1.3fffff8p+2, 0x0p+0, 0x0p+0}},
    {fn_fdim, {0x1.fffffep+127f, 0x0p+0f, 0x1.fffffep+126f, 0x0p+0f}, {0x1.fffffep+126, 0x0p+0, 0x0p+0}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_fdim, {0x1p-126f, 0x1p-149f, 0x1p-126f, 0x0p+0f}, {0x1p-149, 0x0p+0, 0x0p+0}},
    {fn_fdim, {0x1p-125f, 0x0p+0f, 0x1p-126f, 0x1p-149f}, {0x1.fffffcp-127, 0x0p+0, 0x0p+0}},
    {fn_fdim, {0x1p-149f, 0x0p+0f, -0x1p-149f, 0x0p+0f}, {0x1p-148, 0x0p+0, 0x0p+0}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_fdim, {0x1.2a05f2p+33f, 0x0p+0f, 0x1p+0f, 0x1p-25f}, {0x1.2a05f1ff8p+33, -0x1p-25, 0x0p+0}},
    {fn_fdim, {0x1.fffffep+126f, 0x0p+0f, -0x1.fffffep+125f, 0x0p+0f}, {0x1.7ffffe8p+127, 0x0p+0, 0x0p+0}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_sqrt, {0x1.002e4p-130f, 0x0p+0f}, {0x1.00171ef4b6248p-65, 0x1.a94f3e9d1fe91p-123, -0x1.89ce7b90bf812p-183}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_sqrt, {0x1.000016p-122f, 0x0p+0f}, {0x1.00000affffc38p-61, 0x1.4cbfee20b1134p-124, -0x1.106f566d56e11p-178}},
    {fn_sqrt,
     {0x1.002aacp-46f, -0x1.6f7e2p-71f},
     {0x1.001554c09ea38p-23, 0x1.1acbfdd11453ep-83, -0x1.954186a0f708cp-137}},
    {fn_sqrt, {0x1.00b1d4p-2f, 0x0p+0f}, {0x1.0058da947fa38p-1, 0x1.1af2d0da8992bp-62, 0x1.2f45056f4e4efp-117}},
    {fn_sqrt,
     {0x1.008416p+20f, 0x1.3acc88p-5f},
     {0x1.004202cbfa878p+10, -0x1.3d46781491626p-49, -0x1.9c2626dce6db9p-104}},
    {fn_sqrt, {0x1.000016p+126f, 0x0p+0f}, {0x1.00000affffc38p+63, 0x1.4cbfee20b1134p+0, -0x1.106f566d56e11p-54}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rsqrt, {0x1.c3989p-127f, 0x0p+0f}, {0x1.10956f3d72b49p+63, 0x1.240a15b2a3e4p+2, 0x1.94f0fa6d7f059p-56}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rsqrt,
     {0x1.eefe6ep-81f, 0x1.cc7946p-108f},
     {0x1.045c47532a309p+40, 0x1.9981cc07cc85cp-18, 0x1.f5100c5026639p-73}},
    {fn_rsqrt,
     {0x1.03249ep-32f, 0x1.f5bc24p-57f},
     {0x1.fce2b68b3cbf5p+15, 0x1.ad3c09520aa2fp-39, -0x1.df9ef00198c03p-93}},
    {fn_rsqrt,
     {0x1.8c4942p+0f, 0x1.dc32b4p-25f},
     {0x1.9b83cb52010b3p-1, -0x1.cdc12a3e139e5p-55, 0x1.d471a939d8739p-109}},
    {fn_rsqrt,
     {0x1.95ef96p+39f, 0x1.d35fap+12f},
     {0x1.1f816b4b176efp-20, 0x1.23cc8752a0335p-75, 0x1.3a641c1cdd29bp-130}},
    {fn_rsqrt,
     {0x1.e33274p+109f, 0x1.14948cp+84f},
     {0x1.0784ff3f80709p-55, 0x1.8864dbdbea8d1p-109, 0x1.568dde8b4f74cp-163}},
    {fn_fdim,
     {0x1.f98a72p-1f, -0x1.2c0872p-26f, 0x1.f2f6bep-2f, 0x1.e0f396p-27f},
     {0x1.000f11f1bee18p-1, 0x0p+0, 0x0p+0}},
    {fn_fdim,
     {0x1.0a9de6p+0f, -0x1.c87482p-25f, 0x1.e3aecap-2f, -0x1.fcc5bcp-28f},
     {0x1.2364657724358p-1, 0x0p+0, 0x0p+0}},
    {fn_fdim,
     {0x1.ad45f4p-1f, -0x1.627536p-26f, 0x1.588002p-2f, 0x1.a6406ap-27f},
     {0x1.0105f1e5354a8p-1, 0x0p+0, 0x0p+0}},
    {fn_fdim,
     {0x1.03bbb2p+124f, 0x1.9e2ddp+99f, 0x1.fde566p+122f, 0x1.888a16p+95f},
     {0x1.0884b285a52eap+123, 0x0p+0, 0x0p+0}},
    {fn_fdim,
     {0x1.004d3p+126f, 0x1.a3cb54p+101f, 0x1.fa9306p+124f, 0x1.d29352p+96f},
     {0x1.0350de9536b97p+125, 0x0p+0, 0x0p+0}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_fdim,
     {0x1.007efcp-95f, -0x1.75f9cep-120f, 0x1.ffe9c2p-97f, 0x1.48b49p-125f},
     {0x1.0109157fc08d8p-96, 0x0p+0, 0x0p+0}},
    {fn_fdim, {0x1.112372p-105f, 0x0p+0f, 0x1.ccb254p-107f, 0x0p+0f}, {0x1.3bedbap-106, 0x0p+0, 0x0p+0}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_fdim,
     {0x1.0702ccp-38f, -0x1.4a73e2p-63f, 0x1.ca0a7ap-40f, 0x1.f37ac4p-66f},
     {0x1.290059771cc58p-39, 0x0p+0, 0x0p+0}},
    {fn_hypot,
     {0x1.250d2p-2f, -0x1.354802p-29f, -0x1.28464cp-1f, 0x1.fa8778p-26f},
     {0x1.4a87066071bf9p-1, -0x1.deda0a793d8a7p-56, -0x1.13681f221833ep-110}},
    {fn_hypot,
     {0x1.3d773ap-1f, -0x1.e4711cp-26f, 0x1.0b1ddep-13f, -0x1.2f96bp-41f},
     {0x1.3d77397e27ce5p-1, -0x1.ffc0ff19fde73p-55, -0x1.bf625bb893f05p-110}},
    {fn_hypot,
     {-0x1.69bef8p-6f, -0x1.237ac8p-34f, 0x1.37485ap-1f, -0x1.d7b292p-26f},
     {0x1.377ce121af4bfp-1, -0x1.51486b34136f7p-57, -0x1.eac5d898de5ap-111}},
    {fn_hypot,
     {0x1.c42eeep+71f, 0x0p+0f, 0x1.3357dp+74f, 0x1.b3a04ep+49f},
     {0x1.387f50b69180cp+74, -0x1.842039d47ff43p+20, 0x1.db45b5d147bccp-34}},
    {fn_hypot,
     {-0x1.678c5p+123f, 0x1.0d9e26p+98f, 0x1.020938p+126f, -0x1.297d44p+101f},
     {0x1.05ebb88b47b8ep+126, -0x1.84d8e34423d69p+72, -0x1.b11437f53c0e8p+18}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_hypot,
     {0x1.03bf5ep-94f, 0x0p+0f, -0x1.00128ep-91f, 0x1.8eead4p-117f},
     {0x1.021f671b9bec5p-91, 0x1.a8282455841e3p-147, -0x1.0ecd4422702cap-202}},
    {fn_hypot,
     {-0x1.b33496p-103f, -0x0p+0f, 0x1.14fa12p-103f, 0x0p+0f},
     {0x1.01ef1ea943066p-102, -0x1.c55cf9cf1f3d5p-156, 0x1.f17f6b69d4397p-215}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_hypot,
     {0x1.1d5e86p+75f, 0x1.e016fap+50f, -0x1.06a966p+87f, -0x1.e25f22p+62f},
     {0x1.06a9678c34ab3p+87, -0x1.b586cb0a39ca4p+32, -0x1.36fca7dffe123p-22}},
    {fn_rhypot,
     {-0x1.5e70b4p-2f, -0x1.e66292p-27f, 0x1.7df972p-3f, 0x0p+0f},
     {0x1.486a4770b6a8dp+1, -0x1.3ae54d838a1d6p-54, -0x1.715b0687043cbp-108}},
    {fn_rhypot,
     {0x1.33056ep-16f, -0x1.65da04p-44f, -0x1.0224c4p+1f, -0x1.a7bd54p-24f},
     {0x1.fbbf9384f75d1p-2, -0x1.d3763983c285ap-56, 0x1.1ec30abc8486ap-111}},
    {fn_rhypot,
     {-0x1.5797f2p-26f, 0x1.f8c4cp-52f, -0x1.0b0424p-2f, 0x1.fe0b1cp-27f},
     {0x1.eae06cb40201ep+1, -0x1.65b5c9ea1a23fp-54, -0x1.243be00be14eep-112}},
    {fn_rhypot,
     {0x1.6be1bp+65f, 0x1.c41cfep+39f, -0x1.166f08p+67f, 0x1.32a3fp+42f},
     {0x1.bf785014bf06dp-68, -0x1.33faf5442b36ap-122, -0x1.70843bf1e4905p-178}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rhypot,
     {0x1.4db81ap+125f, -0x1.0678dp+98f, -0x1.1a288ap+126f, 0x1.1eb314p+100f},
     {0x1.8fd945000017ep-127, 0x1.d2a8ef35991a7p-181, 0x1.7446834434326p-238}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rhypot,
     {0x1.0e9712p-68f, -0x1.e428a8p-93f, 0x1.7d1186p-69f, 0x0p+0f},
     {0x1.8c0ed12b3d61p+67, 0x1.08757cd55433bp+13, -0x1.ea8893826155ap-43}},
    {fn_rhypot,
     {0x1.8d718ep-124f, 0x0p+0f, -0x1.87d03ep-124f, -0x0p+0f},
     {0x1.d5b48b7a131fdp+122, -0x1.8cec451745674p+68, 0x1.e981f50cd4ed9p+13}},
    {fn_rhypot,
     {-0x1.13bc6p+20f, -0x1.5444f2p-5f, 0x1.8c09ecp+14f, -0x1.d5cca4p-14f},
     {0x1.db3bdc90e1bedp-21, -0x1.7f6ba7cc92aadp-75, -0x1.8ffedba51b21p-129}},
    {fn_norm3d,
     {-0x1.7b984ep+0f, -0x1.661f54p-26f, 0x1.ae7602p+0f, 0x1.ffa354p-25f, 0x1.99e054p-1f, 0x1.bccceap-26f},
     {0x1.30b554075070ap+1, -0x1.c34975f9a3aep-54, -0x1.5cf699eda395bp-108}},
    {fn_norm3d,
     {0x1.553194p-13f, -0x1.8c4208p-38f, -0x1.18aa6ap-1f, 0x1.ce53a4p-26f, -0x1.119b4p-16f, -0x1.21a2c4p-44f},
     {0x1.18aa69ea4ed4dp-1, -0x1.efdbfe4d903fap-56, 0x1.2c65c75888d7p-110}},
    {fn_norm3d,
     {0x1.b961cap-6f, -0x1.e5d9aap-32f, -0x1.06dc02p+0f, 0x1.b8adf6p-25f, -0x1.09f09ep-6f, 0x1.9943f2p-33f},
     {0x1.06fb90e41c7fp+0, 0x1.4aac97164fd7cp-54, -0x1.518a9e4420dc2p-111}},
    {fn_norm3d,
     {0x1.6e2276p+97f, -0x1.278edap+69f, -0x1.e100acp+98f, 0x1.6210ecp+73f, 0x1.69cc52p+96f, -0x1.47b596p+68f},
     {0x1.05465f23110ap+99, 0x1.1431c999f3114p+42, -0x1.acd86a7e82531p-12}},
    {fn_norm3d,
     {0x1.61f1cap+125f, 0x1.61c854p+97f, 0x1.329p+126f, 0x1.ce4eacp+101f, 0x1.a72c6p+124f, -0x1.e4e0b4p+99f},
     {0x1.7172bdd6a1b34p+126, -0x1.529a95f8b7bd5p+69, -0x1.a9869a06ce71p+12}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_norm3d,
     {0x1.04b9bp-83f, 0x1.1db1cp-108f, 0x1.5cb392p-86f, -0x1.138568p-113f, -0x1.d75c24p-86f, -0x1.92d4acp-113f},
     {0x1.0ed49b9f67688p-83, 0x1.128026b38db9cp-137, -0x1.6118accd10cbp-191}},
    {fn_norm3d,
     {0x1.3b05c6p-104f, 0x0p+0f, -0x1.ef90d2p-103f, -0x0p+0f, 0x1.a47b48p-105f, 0x0p+0f},
     {0x1.0941d5eef31e9p-102, 0x1.25a5d17184b32p-157, 0x1.a7be1f7d5f7b2p-213}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_norm3d,
     {-0x1.04fc3ap+55f, -0x1.c80812p+30f, 0x1.fb3f6ap+48f, 0x1.8bffd6p+21f, 0x1.bf5c24p+35f, -0x1.c0cc54p+8f},
     {0x1.0503ee89a6856p+55, 0x1.8ec884529b31fp+0, 0x1.a09c05494932bp-54}},
    {fn_rnorm3d,
     {-0x1.994148p+0f, -0x1.f07f24p-25f, -0x1.bce7bp-1f, -0x1.0979f4p-27f, -0x1.7de8cep-1f, 0x1.be2118p-28f},
     {0x1.045c10882003cp-1, -0x1.f710a06cf8b75p-55, 0x1.f039c379fe574p-109}},
    {fn_rnorm3d,
     {0x1.51ae5ap-13f, -0x1.470e9ap-39f, -0x1.3817b8p-2f, -0x1.fe417cp-27f, -0x1.36366ep-28f, 0x1.b20354p-56f},
     {0x1.a3fa4f0b31dafp+1, -0x1.e47109150fd9ep-53, 0x1.7d0db96946c1bp-107}},
    {fn_rnorm3d,
     {0x1.5749dcp-17f, -0x1.cf73c8p-45f, -0x1.0a2798p-38f, 0x1.d7a894p-65f, -0x1.052514p+1f, -0x1.df488cp-24f},
     {0x1.f5e9bb8225ddfp-2, -0x1.bd9ca3a57061bp-56, 0x1.8fea08b438287p-110}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rnorm3d,
     {0x1.1d157ap+85f, -0x1.4dce1ep+59f, 0x1.0b7756p+86f, -0x1.f958eap+61f, 0x1.b8468ep+84f, 0x0p+0f},
     {0x1.967e222e42bf4p-87, -0x1.267d5900cb7e4p-141, 0x1.c4e253f119ed3p-200}},
    {fn_rnorm3d,
     {0x1.88981p+125f, -0x1.cd7a42p+100f, -0x1.939c2ap+123f, -0x0p+0f, 0x1.ccbd96p+126f, 0x1.244974p+99f},
     {0x1.04666d0000383p-127, -0x1.2781cbfe68893p-182, 0x1.d050b7a855447p-237}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rnorm3d,
     {0x1.d47de8p-97f, -0x1.87a142p-122f, 0x1.7ed16ap-97f, -0x1.73cbccp-125f, 0x1.7b8b3ap-99f, -0x1.2e7eb8p-126f},
     {0x1.ac0ed4094bdd4p+95, 0x1.ec62bb599ffbcp+39, -0x1.807a3af9aa0a1p-16}},
    {fn_rnorm3d,
     {0x1.c86c48p-118f, 0x0p+0f, 0x1.249b3cp-115f, 0x0p+0f, 0x1.23321ap-117f, 0x0p+0f},
     {0x1.ab1dcbd9e4fb1p+114, 0x1.20398ecad90b8p+60, -0x1.fc9ab1718365ep+5}},
    {fn_rnorm3d,
     {0x1.156e1ap+48f, 0x1.61d422p+23f, -0x1.047e4p+58f, -0x1.537a6ep+33f, -0x1.b26116p+55f, -0x1.6ddd4cp+29f},
     {0x1.ec94d8cc5cd5bp-59, -0x1.85d4720a51b17p-113, 0x1.7ef6f4df6264cp-169}},
    {fn_norm4d,
     {-0x1.229acp-2f,
      -0x1.fc353p-27f,
      -0x1.c67b92p-3f,
      -0x0p+0f,
      0x1.0bed22p-3f,
      0x1.bbc13ep-29f,
      -0x1.906914p-3f,
      -0x0p+0f},
     {0x1.b8968d805534bp-2, 0x1.76c7ae9906c1ap-57, -0x1.052daace2ded9p-111}},
    {fn_norm4d,
     {0x1.0f510ap-4f,
      -0x1.7aa022p-29f,
      0x1.77fcap-1f,
      -0x1.dd7336p-27f,
      -0x1.5f98aep-17f,
      0x1.a3d5ccp-42f,
      0x1.b60a64p-13f,
      -0x1.46284cp-39f},
     {0x1.7983677fb4d83p-1, 0x1.f1e4d02eb1777p-56, 0x1.2d583bcef492p-111}},
    {fn_norm4d,
     {0x1.84cb0ap-24f,
      -0x1.0eba88p-49f,
      -0x1.1e052p-9f,
      0x1.cb18aep-36f,
      0x1.713b94p-1f,
      -0x1.ad260ep-26f,
      0x1.d4ae8ep-18f,
      -0x1.6358bep-43f},
     {0x1.713c01f16af2bp-1, 0x1.0c438b39256e1p-55, 0x1.0316e9af55107p-112}},
    {fn_norm4d,
     {0x1.35f3c8p+95f,
      0x1.2182fcp+70f,
      0x1.7401bap+98f,
      0x1.cbc196p+73f,
      0x1.9fc3ap+96f,
      -0x1.9ef2cp+70f,
      0x1.461fe8p+96f,
      -0x1.4e003ep+70f},
     {0x1.8ca99e6e31effp+98, -0x1.860d4616ccd74p+42, -0x1.c0ea6bfc19f73p-15}},
    {fn_norm4d,
     {-0x1.8ed43p+126f,
      -0x1.f82b2p+100f,
      0x1.370ccep+124f,
      -0x1.b951d2p+97f,
      0x1.7f1704p+124f,
      0x1.b8c9ap+99f,
      -0x1.5a4168p+124f,
      -0x1.a5681p+99f},
     {0x1.aa5a7260540cfp+126, 0x1.75edba660ccabp+72, -0x1.af79a03a47c4ep+18}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_norm4d,
     {-0x1.f045d8p-94f,
      -0x1.2f61d8p-121f,
      -0x1.0ab7d6p-91f,
      -0x1.e7ebep-116f,
      -0x1.0f93fep-93f,
      -0x0p+0f,
      0x1.36ea78p-94f,
      -0x1.ff4176p-121f},
     {0x1.1ccaf0b528421p-91, -0x1.e4395c4a34e63p-145, -0x1.83d2c05f99de9p-199}},
    {fn_norm4d,
     {0x1.530f96p-103f, 0x0p+0f, 0x1.c78b02p-103f, 0x0p+0f, -0x1.c1c186p-106f, -0x0p+0f, 0x1.e71ec4p-104f, 0x0p+0f},
     {0x1.363a2edb5b8acp-102, 0x1.9340ff833fac6p-156, -0x1.5bb89a7dbc63bp-211}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_norm4d,
     {-0x1.0ad55cp+24f,
      0x1.bdc254p-1f,
      -0x1.fed53p+10f,
      -0x0p+0f,
      -0x1.72220ap+12f,
      -0x1.eea49p-16f,
      -0x1.1d2938p+15f,
      0x1.2a4a1ep-13f},
     {0x1.0ad582584cb67p+24, -0x1.4d4edc6afaaacp-30, -0x1.2f35b98aaebc1p-85}},
    {fn_rnorm4d,
     {-0x1.295002p+0f,
      -0x1.c7e838p-27f,
      -0x1.9aebcp+0f,
      0x1.8df444p-25f,
      0x1.e49346p+1f,
      0x1.ed6be8p-24f,
      0x1.067e9ep+0f,
      -0x1.8cf2c6p-26f},
     {0x1.d212ec227a752p-3, -0x1.c0313d74ca95p-58, -0x1.1be9863fb539dp-112}},
    {fn_rnorm4d,
     {-0x1.7062a6p-9f,
      0x1.e52d0ep-36f,
      0x1.484ad4p+1f,
      -0x1.f1e61ap-24f,
      0x1.a4cd0ap-27f,
      0x1.348332p-52f,
      0x1.0a9166p-8f,
      0x0p+0f},
     {0x1.8f40d42380fc1p-2, 0x1.389d8eefec2a9p-56, -0x1.b697dbf0d9cedp-110}},
    {fn_rnorm4d,
     {0x1.77955cp+0f,
      -0x1.586bb6p-25f,
      0x1.bf4d3cp-12f,
      0x1.692b28p-38f,
      0x1.886574p-20f,
      0x1.0e5938p-48f,
      0x1.214edep-12f,
      0x0p+0f},
     {0x1.5cfb7c534c2a9p-1, 0x1.7e2f0a9a0d874p-56, 0x1.9d0ca0d07cedfp-113}},
    {fn_rnorm4d,
     {0x1.c353aep+71f,
      -0x1.58228ap+43f,
      0x1.08a8c2p+73f,
      -0x1.f6778p+48f,
      0x1.0f9bbap+71f,
      0x1.f1fe12p+46f,
      -0x1.339752p+70f,
      -0x1.ab2266p+42f},
     {0x1.b7b06e995c647p-74, 0x1.5e751d817a892p-128, 0x1.3cfdb693c0765p-183}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rnorm4d,
     {0x1.1566bcp+123f,
      0x1.a9006ap+96f,
      0x1.1c54d8p+126f,
      0x1.644798p+100f,
      -0x1.2ad41cp+126f,
      -0x0p+0f,
      -0x1.9f7ee4p+123f,
      0x1.724572p+98f},
     {0x1.3a2ebaffff0ecp-127, 0x1.e59cfe96bf91fp-181, 0x1.edcfcafb8bfb5p-235}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rnorm4d,
     {-0x1.121144p-95f,
      -0x1.64a262p-121f,
      -0x1.6d65cap-92f,
      0x1.d51e0ep-117f,
      -0x1.9ae48p-95f,
      -0x1.34d0f8p-121f,
      -0x1.56f528p-94f,
      0x1.3cfbd4p-120f},
     {0x1.58982afaaae84p+91, -0x1.00e8f38ecc09dp+34, -0x1.fff8202eac68dp-20}},
    {fn_rnorm4d,
     {-0x1.429f7ep-117f, -0x0p+0f, -0x1.bc1e5ep-117f, -0x0p+0f, 0x1.4b830ap-120f, 0x0p+0f, 0x1.730ad6p-118f, 0x0p+0f},
     {0x1.c342b03434794p+115, 0x1.f23cddd8c287ap+59, -0x1.b59096368e6c6p+3}},
    {fn_rnorm4d,
     {-0x1.2d44a2p+48f,
      -0x1.554038p+22f,
      -0x1.5c6cd6p+63f,
      -0x0p+0f,
      -0x1.c858b2p+50f,
      -0x1.ba7338p+23f,
      0x1.02a0ccp+72f,
      0x1.6987d6p+47f},
     {0x1.facba01380591p-73, -0x1.cdb9994ae7409p-127, -0x1.142cb2f395599p-182}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rsqrt, {0x1.b879p-133f, 0x0p+0f}, {0x1.14010a0430955p+66, 0x1.afacb132828cbp+8, 0x1.0fab218d82efbp-46}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rsqrt,
     {0x1.67f1c2p-99f, -0x1.d9fea6p-125f},
     {0x1.315247f151308p+49, -0x1.2f116707179fap-5, 0x1.d6ff878051348p-59}},
    {fn_rsqrt,
     {0x1.b8d01ap-11f, 0x1.891002p-36f},
     {0x1.13e5c3c243a0dp+5, -0x1.721f385210a15p-54, -0x1.0645def7ba6bap-109}},
    {fn_rsqrt,
     {0x1.fc22dap+1f, 0x1.67f9d4p-25f},
     {0x1.00f8b1fc4dac3p-1, 0x1.69c098090330ep-56, -0x1.7027e7d00a37bp-111}},
    {fn_rsqrt,
     {0x1.c92916p+47f, 0x1.e24658p+22f},
     {0x1.0eeb71c1868fap-24, 0x1.4b2aea62aa0abp-78, 0x1.dbbaac0b21f46p-132}},
    {fn_rsqrt,
     {0x1.c937dap+99f, 0x1.cf387ep+74f},
     {0x1.0ee711d521b29p-50, 0x1.98b41e62dea91p-104, -0x1.d14fc21cae90fp-158}},
    {fn_rhypot,
     {0x1.2536ccp+0f, 0x1.eaee56p-28f, 0x1.7df1bp+0f, -0x1.71101ep-26f},
     {0x1.10354341f8567p-1, 0x1.32d42fa4df9bdp-56, 0x1.a6fc8be70426dp-111}},
    {fn_rhypot,
     {-0x1.490c7ep-30f, 0x1.865c1cp-55f, 0x1.7533a6p-1f, 0x1.da93bap-26f},
     {0x1.5f35a0cfb77f4p+0, -0x1.73c57f1155758p-55, -0x1.c05133979dfbcp-111}},
    {fn_rhypot,
     {0x1.722186p-1f, 0x1.b1710cp-26f, 0x1.226ad6p-50f, 0x1.1538b2p-76f},
     {0x1.621f90e190befp+0, 0x1.4dbab9b2373d2p-55, -0x1.372f0cb8481e8p-109}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rhypot,
     {-0x1.48adfcp+93f, -0x1.51f976p+66f, 0x1.45c4b6p+94f, -0x1.112a26p+69f},
     {0x1.6739d6f0acb65p-95, 0x1.e8e5bc229eb99p-149, 0x1.7c2b0411bab3fp-205}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rhypot,
     {-0x1.660184p-115f, -0x0p+0f, -0x1.199512p-117f, -0x0p+0f},
     {0x1.673ce503665a9p+114, -0x1.0b6a3d0c26694p+58, 0x1.03270c4bb87fcp+4}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rhypot,
     {-0x1.058fep+82f, 0x1.fe3658p+57f, -0x1.202adp+62f, 0x1.6a2526p+35f},
     {0x1.f51cd05037823p-83, 0x1.a062f8f2fcb28p-138, -0x1.1a13a4a46e2c3p-193}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rnorm3d,
     {-0x1.760b96p+0f, -0x1.88f5d8p-27f, 0x1.97192p+0f, 0x1.b527dcp-28f, -0x1.dc6d92p+0f, -0x1.3956aap-28f},
     {0x1.6731e759db432p-2, -0x1.720e55980322bp-56, -0x1.b814ada44eb5p-110}},
    {fn_rnorm3d,
     {0x1.3d512cp-15f, 0x1.fe4d48p-43f, 0x1.6f9e8cp-1f, -0x1.db60e8p-26f, -0x1.3dadcep-9f, -0x0p+0f},
     {0x1.648a6c94c1747p+0, 0x1.200938ea57952p-54, -0x1.4b0599f44a962p-109}},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rnorm3d,
     {-0x1.dd0caap+82f, -0x0p+0f, 0x1.63b8ecp+82f, -0x1.b2a4eap+57f, 0x1.a659e8p+82f, 0x1.1c1d5cp+54f},
     {0x1.673ce4ddb3641p-84, -0x1.34f7d150b0f1bp-138, 0x1.a4b8309c7bfbcp-195}},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rnorm3d,
     {0x1.a1eb36p-73f, 0x1.22ba16p-101f, -0x1.9fec2cp-71f, -0x1.363a82p-96f, -0x1.3e2deep-74f, -0x1.8fabcep-102f},
     {0x1.30557b3c077e6p+70, -0x1.372221214425ep+15, -0x1.202cf4ab9b16fp-41}},
    {fn_rnorm3d,
     {-0x1.f59adep-111f, -0x0p+0f, -0x1.625c0cp-111f, -0x0p+0f, -0x1.80163ap-111f, -0x0p+0f},
     {0x1.69e56cd8b7fabp+109, -0x1.78d2fc9889b92p+54, 0x1.b9241189c17ecp+0}},
    {fn_rnorm3d,
     {-0x1.2f3ae8p-5f, -0x1.81681ap-31f, -0x1.156c02p-28f, 0x1.cbb876p-56f, -0x1.64e78ap-3f, -0x1.47bd7cp-29f},
     {0x1.673b957e9c5eap+2, -0x1.a48aaac4d0b9fp-52, -0x1.10c740d988021p-107}},
    {fn_rnorm4d,
     {0x1.1ff2eep-2f,
      -0x1.1caa38p-30f,
      0x1.1c43f4p-2f,
      -0x1.261f42p-30f,
      0x1.f40504p-2f,
      -0x1.fdbaaap-27f,
      -0x1.58c1fp-2f,
      -0x1.2e58b2p-27f},
     {0x1.6733f10c6b3ep+0, -0x1.e48e384a907fdp-55, -0x1.357fe63b8fa29p-110}},
    {fn_rnorm4d,
     {0x1.531cep-10f,
      0x1.621fbp-38f,
      -0x1.ac3a9ep-23f,
      0x1.76bdd2p-51f,
      0x1.88d1a8p-21f,
      -0x1.1eabacp-49f,
      -0x1.1d60aap+1f,
      0x1.e6af38p-24f},
     {0x1.cb4b0e4af4d5ep-2, -0x1.e953c8f02990ap-58, -0x1.34b351fafd5a6p-113}},
    {fn_rnorm4d,
     {0x1.0157dep-55f,
      -0x1.97279p-80f,
      0x1.185bb6p+1f,
      0x1.dab95ep-24f,
      0x1.9e71fp-11f,
      -0x1.98c19ap-38f,
      0x1.87a192p-12f,
      0x0p+0f},
     {0x1.d3841c9d5ba79p-2, -0x1.66f6a244bf3afp-56, -0x1.4f83cbfac8be7p-111}},
    {fn_rnorm4d,
     {0x1.f6e7a2p-114f, 0x0p+0f, -0x1.c94ef2p-114f, -0x0p+0f, 0x1.04e576p-117f, 0x0p+0f, -0x1.47505ep-115f, -0x0p+0f},
     {0x1.76886f89bd57ap+112, 0x1.6d190163211acp+58, -0x1.b61a2900d7212p+3}},
    {fn_rnorm4d,
     {0x1.0f2af6p+12f,
      0x1.fed1e8p-13f,
      -0x1.e8ecp-5f,
      0x1.c864d4p-33f,
      0x1.f53758p-4f,
      -0x1.627a9ep-32f,
      -0x1.d93944p-11f,
      -0x1.6bf84ep-36f},
     {0x1.e35c758ff421ap-13, 0x1.1392a7ddb2423p-67, 0x1.f73184cdf2a8ep-122}},
  };
  check_accuracy<T>(cases, high);
}

template <class F>
struct exact_case
{
  sqrt_fn fn;
  F x[4];
  F r; // expected high limb, compared bit for bit (NaN matches any NaN)
};

template <class F>
TEST_HOST_DEVICE_FUNC bool same(F x, F y)
{
  if (cuda::std::isnan(y))
  {
    return cuda::std::isnan(x);
  }
  return x == y && cuda::std::signbit(x) == cuda::std::signbit(y);
}

// A NaN operand is NaN in both limbs.
template <class T, class F>
TEST_HOST_DEVICE_FUNC T operand(F x)
{
  return cuda::std::isnan(x) ? T(x, x) : T(x);
}

// The high limb is the expected one, and the low limb is zero, or NaN if the result is NaN.
// sqrt and rsqrt are not asked for positive operands below their range.
template <class T, class F, int N>
TEST_HOST_DEVICE_FUNC void check_exact(const exact_case<F> (&cases)[N])
{
  for (const exact_case<F>& c : cases)
  {
    if ((c.fn == fn_sqrt || c.fn == fn_rsqrt) && c.x[0] > F(0) && c.x[0] < short_kernel_min<F>())
    {
      continue;
    }
    const T r = call(c.fn, operand<T, F>(c.x[0]), operand<T, F>(c.x[1]), operand<T, F>(c.x[2]), operand<T, F>(c.x[3]));
    assert(same(r.hi(), c.r));
    if (cuda::std::isnan(c.r))
    {
      assert(cuda::std::isnan(r.lo()));
    }
    else
    {
      assert(r.lo() == F(0));
    }
  }
}

// Results that are exact, with a zero low limb: roots of squares, their reciprocals where those
// are powers of two, Pythagorean operands, and operands scaled to both ends of the range.
template <class T, class F>
TEST_HOST_DEVICE_FUNC void test_exact()
{
  const F maxn                = (cuda::std::numeric_limits<F>::max)();
  const F minn                = (cuda::std::numeric_limits<F>::min)();
  const F mind                = cuda::std::numeric_limits<F>::denorm_min();
  const F big                 = sizeof(F) == 4 ? F(0x1p123) : F(0x1p1019);
  const F tiny                = sizeof(F) == 4 ? 2 * mind : 4 * mind; // an even power of two
  const exact_case<F> cases[] = {
    {fn_sqrt, {F(4)}, F(2)},
    {fn_sqrt, {F(1)}, F(1)},
    {fn_sqrt, {F(0.25)}, F(0.5)},
    {fn_sqrt, {F(1764)}, F(42)},
    {fn_sqrt, {minn}, sizeof(F) == 4 ? F(0x1p-63) : F(0x1p-511)},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_sqrt, {tiny}, sizeof(F) == 4 ? F(0x1p-74) : F(0x1p-536)},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_rsqrt, {F(4)}, F(0.5)},
    {fn_rsqrt, {F(1)}, F(1)},
    {fn_rsqrt, {F(0.0625)}, F(4)},
    {fn_rsqrt, {minn}, sizeof(F) == 4 ? F(0x1p63) : F(0x1p511)},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rsqrt, {tiny}, sizeof(F) == 4 ? F(0x1p74) : F(0x1p536)},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_hypot, {F(3), F(4)}, F(5)},
    {fn_hypot, {F(-5), F(12)}, F(13)},
    {fn_hypot, {F(0), F(-7)}, F(7)},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_hypot, {F(-0.0), -mind}, mind},
    {fn_hypot, {3 * mind, 4 * mind}, 5 * mind},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_hypot, {3 * minn, -4 * minn}, 5 * minn},
    {fn_hypot, {3 * big, 4 * big}, 5 * big},
    {fn_rhypot, {F(0), F(4)}, F(0.25)},
    {fn_rhypot, {F(-2), F(0)}, F(0.5)},
    {fn_norm3d, {F(1), F(2), F(2)}, F(3)},
    {fn_norm3d, {F(2), F(-3), F(6)}, F(7)},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_norm3d, {mind, 2 * mind, -2 * mind}, 3 * mind},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_norm3d, {big, 2 * big, 2 * big}, 3 * big},
    {fn_rnorm3d, {F(0), F(0), F(-8)}, F(0.125)},
    {fn_norm4d, {F(1), F(1), F(1), F(1)}, F(2)},
    {fn_norm4d, {F(2), F(4), F(5), F(6)}, F(9)},
    {fn_norm4d, {minn, minn, -minn, minn}, 2 * minn},
    {fn_norm4d, {big, big, big, -big}, 2 * big},
    {fn_rnorm4d, {F(1), F(1), F(1), F(1)}, F(0.5)},
    {fn_fdim, {F(5), F(3)}, F(2)},
    {fn_fdim, {F(3), F(5)}, F(0)},
    {fn_fdim, {F(-3), F(-5)}, F(2)},
    {fn_fdim, {F(3), F(3)}, F(0)},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_fdim, {mind, -mind}, 2 * mind},
#endif // !TEST_HAS_CUDA_FAST_MATH
  };
  check_exact<T>(cases);
  // The largest operand is its own norm. The short fp64mp2 square root is not exact on it.
  if (sizeof(F) == 4)
  {
    const exact_case<F> max_cases[] = {
      {fn_hypot, {maxn, F(0)}, maxn},
      {fn_norm3d, {F(0), F(0), maxn}, maxn},
    };
    check_exact<T>(max_cases);
  }
}

// IEEE 754 / C Annex F special values, and what sqrt and rsqrt return in their place.
template <class T, class F>
TEST_HOST_DEVICE_FUNC void test_special_values()
{
  const F inf  = cuda::std::numeric_limits<F>::infinity();
  const F nan  = cuda::std::numeric_limits<F>::quiet_NaN();
  const F maxn = (cuda::std::numeric_limits<F>::max)();
  const F minn = (cuda::std::numeric_limits<F>::min)();
  const F mind = cuda::std::numeric_limits<F>::denorm_min();
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operands and result: flushed to zero on the device
  // the smallest operands: sqrt(2) times the smallest subnormal rounds to it, not to zero
  {
    const T h = cuda::hypot(T(mind), T(mind));
    assert(h.hi() == mind && h.lo() == F(0));
  }
#endif // !TEST_HAS_CUDA_FAST_MATH
  const exact_case<F> cases[] = {
    // sqrt and rsqrt take no branch: sqrt(0) is 0, and every other operand that is not
    // positive and finite gives NaN, the zero and +inf of rsqrt included (a subnormal operand
    // is outside their range on the device and is not asked for)
    {fn_sqrt, {F(0)}, F(0)},
    {fn_sqrt, {inf}, nan},
    {fn_sqrt, {-inf}, nan},
    {fn_sqrt, {F(-1)}, nan},
    {fn_sqrt, {-maxn}, nan},
    {fn_sqrt, {nan}, nan},
    {fn_sqrt, {-nan}, nan},
    {fn_rsqrt, {F(0)}, nan},
    {fn_rsqrt, {F(-0.0)}, nan},
    {fn_rsqrt, {inf}, nan},
    {fn_rsqrt, {-inf}, nan},
    {fn_rsqrt, {F(-1)}, nan},
    {fn_rsqrt, {nan}, nan},
    // hypot: an infinity wins over a NaN; overflow
    {fn_hypot, {F(0), F(0)}, F(0)},
    {fn_hypot, {F(-0.0), F(-0.0)}, F(0)},
    {fn_hypot, {inf, F(1)}, inf},
    {fn_hypot, {F(1), -inf}, inf},
    {fn_hypot, {inf, nan}, inf},
    {fn_hypot, {nan, -inf}, inf},
    {fn_hypot, {nan, F(1)}, nan},
    {fn_hypot, {F(0), nan}, nan},
    {fn_hypot, {maxn, nan}, nan},
    {fn_hypot, {nan, nan}, nan},
    {fn_hypot, {maxn, maxn}, inf},
    {fn_hypot, {-maxn, maxn / 2}, inf},
    // rhypot: +0 for an infinity, +inf for two zeros; overflow
    {fn_rhypot, {F(0), F(-0.0)}, inf},
    {fn_rhypot, {inf, F(1)}, F(0)},
    {fn_rhypot, {nan, -inf}, F(0)},
    {fn_rhypot, {nan, F(1)}, nan},
    {fn_rhypot, {F(0), nan}, nan},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rhypot, {mind, F(0)}, inf},
    {fn_rhypot, {mind, -mind}, inf},
#endif // !TEST_HAS_CUDA_FAST_MATH
    // norm3d, norm4d and their reciprocals
    {fn_norm3d, {F(0), F(-0.0), F(0)}, F(0)},
    {fn_norm3d, {nan, inf, F(1)}, inf},
    {fn_norm3d, {F(1), nan, -inf}, inf},
    {fn_norm3d, {F(1), nan, F(0)}, nan},
    {fn_norm3d, {F(0), F(0), nan}, nan},
    {fn_norm3d, {maxn, maxn, maxn}, inf},
    {fn_rnorm3d, {F(0), F(0), F(0)}, inf},
    {fn_rnorm3d, {inf, nan, F(0)}, F(0)},
    {fn_rnorm3d, {F(1), nan, F(0)}, nan},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_rnorm3d, {mind, mind, mind}, inf},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_norm4d, {F(0), F(0), F(0), F(-0.0)}, F(0)},
    {fn_norm4d, {nan, F(1), -inf, F(0)}, inf},
    {fn_norm4d, {nan, nan, nan, inf}, inf},
    {fn_norm4d, {F(1), F(2), F(3), nan}, nan},
    {fn_norm4d, {maxn, F(0), F(0), maxn}, inf},
    {fn_rnorm4d, {F(0), F(0), F(0), F(0)}, inf},
    {fn_rnorm4d, {F(1), -inf, nan, nan}, F(0)},
    {fn_rnorm4d, {F(1), F(2), nan, F(4)}, nan},
    // fdim: +0 unless x > y, NaN for a NaN operand
    {fn_fdim, {inf, inf}, F(0)},
    {fn_fdim, {-inf, -inf}, F(0)},
    {fn_fdim, {inf, -inf}, inf},
    {fn_fdim, {-inf, inf}, F(0)},
    {fn_fdim, {inf, F(1)}, inf},
    {fn_fdim, {F(1), -inf}, inf},
    {fn_fdim, {F(-0.0), F(0)}, F(0)},
    {fn_fdim, {F(0), F(-0.0)}, F(0)},
    {fn_fdim, {maxn, -maxn}, inf},
    {fn_fdim, {nan, F(1)}, nan},
    {fn_fdim, {F(1), nan}, nan},
    {fn_fdim, {nan, inf}, nan},
    {fn_fdim, {-inf, nan}, nan},
    // fabs and copysign act on the sign alone
    {fn_fabs, {F(-0.0)}, F(0)},
    {fn_fabs, {-inf}, inf},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_fabs, {-mind}, mind},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_fabs, {-maxn}, maxn},
    {fn_fabs, {nan}, nan},
    {fn_copysign, {F(1), F(-0.0)}, F(-1)},
    {fn_copysign, {F(-1), F(0)}, F(1)},
    {fn_copysign, {F(0), F(-2)}, F(-0.0)},
    {fn_copysign, {inf, -minn}, -inf},
    {fn_copysign, {-inf, inf}, inf},
    {fn_copysign, {F(3), -inf}, F(-3)},
    {fn_copysign, {nan, F(-1)}, nan},
    // fmax and fmin: a NaN operand is missing data, and +0 is above -0
    {fn_fmax, {F(0), F(-0.0)}, F(0)},
    {fn_fmax, {F(-0.0), F(0)}, F(0)},
    {fn_fmax, {F(-0.0), F(-0.0)}, F(-0.0)},
    {fn_fmax, {nan, F(-1)}, F(-1)},
    {fn_fmax, {F(-1), nan}, F(-1)},
    {fn_fmax, {nan, -inf}, -inf},
    {fn_fmax, {nan, nan}, nan},
    {fn_fmax, {inf, maxn}, inf},
    {fn_fmax, {-inf, -maxn}, -maxn},
    {fn_fmin, {F(0), F(-0.0)}, F(-0.0)},
    {fn_fmin, {F(-0.0), F(0)}, F(-0.0)},
    {fn_fmin, {F(0), F(0)}, F(0)},
    {fn_fmin, {nan, F(1)}, F(1)},
    {fn_fmin, {F(1), nan}, F(1)},
    {fn_fmin, {inf, nan}, inf},
    {fn_fmin, {nan, nan}, nan},
    {fn_fmin, {-inf, -maxn}, -inf},
    {fn_fmin, {inf, maxn}, maxn},
  };
  check_exact<T>(cases);
}

int main(int, char**)
{
  test_accuracy_fp32mp2<cuda::fp32mp2>(false);
  test_accuracy_fp32mp2<cuda::fp32mp2_low>(false);
  test_accuracy_fp32mp2<cuda::fp32mp2_high>(true);
  test_exact<cuda::fp32mp2, float>();
  test_exact<cuda::fp32mp2_low, float>();
  test_exact<cuda::fp32mp2_high, float>();
  test_special_values<cuda::fp32mp2, float>();
  test_special_values<cuda::fp32mp2_low, float>();
  test_special_values<cuda::fp32mp2_high, float>();

  return 0;
}
