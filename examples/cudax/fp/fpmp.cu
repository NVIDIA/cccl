//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/*
    fpmp.cpp - Multi-precision arithmetic on pairs of floating-point values
    ======================================================================

    An fpmp2 value represents a number as the unevaluated sum of two IEEE floats, which
    roughly doubles the available mantissa:

        fp32mp2   float-float     ~46 effective mantissa bits
        fp64mp2   double-double  ~104 effective mantissa bits

    Each width comes at a selectable accuracy level, which decides how much work goes into
    keeping the trailing limb correct:

        fp32mp2_low    fast arithmetic with no renormalization, so the two limbs are
                       allowed to drift into overlap
        fp32mp2_mid    Dekker-based splitting and error accumulation
                       preserving normalization of the result
        fp32mp2_high   Higher accuracy for key operations via Thall-based splitting
        fp32mp2        the default selector, equal to fp32mp2_mid

    and the same four names for fp64mp2. The level can also be chosen per key operations
    rather than per type, through add<>, sub<>, mul<>, div<> and fma<>.

    Both types offer the same interface and are used the same way, so this example runs one
    kernel per type over the same set of operations:

      - construction from literals of several source types
      - arithmetic: +, -, *, /
      - mixed-type arithmetic, an fpmp2 combined directly with a built-in scalar
      - sqrt, rsqrt and fma
      - math functions, here exp and sin, from <cuda/fpmp_math>
      - comparison operators
      - the hi/lo components the value is stored as
      - the accuracy levels: the same sum on the default and the fpmp2_accuracy::high
        type, then the same high-accuracy addition applied to low-accuracy operands with
        add<>, which selects the accuracy per operation instead of per type
      - renormalize(), which repairs a value whose limbs have come to overlap after a run
        of fpmp2_accuracy::low operations
      - the same source running on host and device
*/
#include <cstdio>

// Multi-precision types and operations, plus the transcendental math functions.
// The core type alone is available as <cuda/fpmp>, which does not pay the math
// compile-time cost.
// Note: The math functions are in <cuda/fpmp_math>
//       and it already includes <cuda/fpmp>.
#include <cuda/fpmp>
#include <cuda/fpmp_math>

// The CCCL FP component lives in cuda::experimental (later cuda::),
// abbreviated here rather than pulled in with a using-directive.
//
// Two spellings appear below, and the split is deliberate. Type names and the
// component's own functions - renormalize, the accuracy-selecting add<> -
// carry the cudax:: prefix, since they have no counterpart for double and so
// only occur in code written against the component. The standard-named math
// functions are left unqualified and found by argument-dependent lookup
// instead, which is what lets an existing body of double code keep its call
// sites unchanged when the type underneath is swapped.
namespace cudax = cuda::experimental;

// A value reported as the two limbs it is stored as, rather than rounded to one
// double. The accuracy levels differ below the width of a single double, so
// they are only distinguishable this way.
struct limb_pair
{
  double hi, lo;
};

// One kernel result, as doubles so the host can print it without knowing the
// type. Everything except the limb pairs is the operation result rounded to a
// single double for display.
struct fpmp_results
{
  double a, b, c, d; // the four inputs
  double sum, diff, prod, quot; // arithmetic operations
  double sqrt_d, rsqrt_d, fma_abc; // sqrt, rsqrt and fma operations
  double exp_a, sin_a; // math functions
  bool is_gt, is_lt, is_eq; // comparison operations
  double mixed_add; // fpmp2 + a double literal
  double mixed_mul; // fpmp2 * an int
  double mixed_lhs; // an int * fpmp2, scalar on the left
  double mixed_acc; // += a scalar, the optimized accumulate path
  double hi, lo; // the two components the value is stored as
  limb_pair p, q; // The same sum p + q at each accuracy level.
  limb_pair add_def; // the default type
  limb_pair add_high; // the fpmp2_accuracy::high type
  limb_pair add_low_as_high; // low-accuracy operands, added at high accuracy
  // a - big + big evaluated in the low-accuracy type, before and after
  // renormalize().
  limb_pair renorm_a, renorm_big, renorm_raw, renorm_fixed;
};

template <typename T>
__host__ __device__ limb_pair limbs_of(const T& value)
{
  return limb_pair{static_cast<double>(value.hi()), static_cast<double>(value.lo())};
}

// The float-float operations.
__host__ __device__ void float_float_operations(fpmp_results* out)
{
  // Construction. A double literal carries more precision than the fp32mp2 pair
  // holds, so the cast is explicit by default (see CCCL_FPMP_EXPLICIT_CASTS);
  // float and integer sources convert implicitly.
  const cudax::fp32mp2 a{1.234567890123456789};
  const cudax::fp32mp2 b{9.876543210987654321};
  const cudax::fp32mp2 c{2.71828182f};
  const cudax::fp32mp2 d{5u};

  // Arithmetic. An operation on two fp32mp2 values yields an fp32mp2, so
  // results stay in the pair representation and can feed further operations at
  // full precision.
  const cudax::fp32mp2 sum  = a + b;
  const cudax::fp32mp2 diff = a - b;
  const cudax::fp32mp2 prod = a * b;
  const cudax::fp32mp2 quot = a / b;

  // Unqualified on purpose: ADL finds the fpmp2 overloads, so a call site
  // written for double needs no edit when the type underneath it changes.
  const cudax::fp32mp2 sqrt_d  = sqrt(d);
  const cudax::fp32mp2 rsqrt_d = rsqrt(d);
  const cudax::fp32mp2 fma_abc = fma(a, b, c);

  const cudax::fp32mp2 exp_a = exp(a);
  const cudax::fp32mp2 sin_a = sin(a);

  // Comparisons return plain bool.
  const bool is_gt = a > b;
  const bool is_lt = a < b;
  const bool is_eq = a == b;

  // Mixed-type operations. An fpmp2 combines directly with the built-in
  // arithmetic types: the scalar is converted to the fpmp2 type and the
  // arithmetic is carried out at full fpmp2 precision, so no temporary has to
  // be built by hand. This holds even for a double literal, which is wider than
  // an fp32mp2 pair and so would need an explicit cast to be stored on its own.
  // The scalar may sit on either side, and the result is an fp32mp2 rather than
  // a built-in type.
  const cudax::fp32mp2 mixed_add = a + 0.9876543210987654;
  const cudax::fp32mp2 mixed_mul = a * 3;
  const cudax::fp32mp2 mixed_lhs = 2 * a;

  // Accumulating a single float has a dedicated path, cheaper than widening it
  // and running a full pair-with-pair addition.
  cudax::fp32mp2 acc = a;
  acc += 0.5f;

  // The two components a value is built from: a + b is exact in the pair, and
  // lo holds what a single float would have dropped.
  const cudax::fp32mp2 s = a + b;

  // Accuracy levels. p and q are given directly as limb pairs so that both
  // limbs carry information, and their leading limbs almost cancel. What is
  // left of the sum then comes from the trailing limbs, which is where the
  // accuracy levels part company.
  const cudax::fp32mp2 p{1.0f, 4.4e-8f};
  const cudax::fp32mp2 q{-(1.0f - 1.0e-7f), 3.1e-8f};
  const cudax::fp32mp2 add_def = p + q;

  // fpmp2_accuracy::high spends more operations to recover the trailing limb
  // that the default level rounds away.
  const cudax::fp32mp2_high high_p{1.0f, 4.4e-8f};
  const cudax::fp32mp2_high high_q{-(1.0f - 1.0e-7f), 3.1e-8f};
  const cudax::fp32mp2_high add_high = high_p + high_q;

  // The accuracy can also be picked per operation rather than per type: add<>
  // takes low-accuracy operands and applies the high-accuracy algorithm to
  // them, so a value stored as fp32mp2_low can still be summed carefully where
  // it matters.
  const cudax::fp32mp2_low low_p{1.0f, 4.4e-8f};
  const cudax::fp32mp2_low low_q{-(1.0f - 1.0e-7f), 3.1e-8f};
  const cudax::fp32mp2_low add_low_as_high = cudax::renormalize(cudax::add<cudax::fpmp2_accuracy::high>(low_p, low_q));

  // Renormalization. fpmp2_accuracy::low is cheap because it omits the step
  // that keeps the two limbs separated. Detouring through a much larger value
  // and coming back returns exactly a, but the trailing limb was sized against
  // big along the way and is never resized, so it ends up far too large for the
  // small result it is attached to. hi is then no longer the correctly rounded
  // leading part, though hi + lo still is the value. renormalize()
  // redistributes the two.
  const cudax::fp32mp2_low low_a{1.2345679f, 2.9e-8f};
  const cudax::fp32mp2_low low_big{1234.5678f, 5.0e-5f};
  const cudax::fp32mp2_low drifted  = low_a - low_big + low_big;
  const cudax::fp32mp2_low renormed = cudax::renormalize(drifted);

  // Everything above stayed in the pair representation. Converting for display
  // happens here, in one place: to a single double where that is enough, and to
  // both limbs where the interesting part lies below what one double can show.
  out->a = static_cast<double>(a);
  out->b = static_cast<double>(b);
  out->c = static_cast<double>(c);
  out->d = static_cast<double>(d);

  out->sum  = static_cast<double>(sum);
  out->diff = static_cast<double>(diff);
  out->prod = static_cast<double>(prod);
  out->quot = static_cast<double>(quot);

  out->sqrt_d  = static_cast<double>(sqrt_d);
  out->rsqrt_d = static_cast<double>(rsqrt_d);
  out->fma_abc = static_cast<double>(fma_abc);

  out->exp_a = static_cast<double>(exp_a);
  out->sin_a = static_cast<double>(sin_a);

  out->is_gt = is_gt;
  out->is_lt = is_lt;
  out->is_eq = is_eq;

  out->mixed_add = static_cast<double>(mixed_add);
  out->mixed_mul = static_cast<double>(mixed_mul);
  out->mixed_lhs = static_cast<double>(mixed_lhs);
  out->mixed_acc = static_cast<double>(acc);

  out->hi = static_cast<double>(s.hi());
  out->lo = static_cast<double>(s.lo());

  out->p               = limbs_of(p);
  out->q               = limbs_of(q);
  out->add_def         = limbs_of(add_def);
  out->add_high        = limbs_of(add_high);
  out->add_low_as_high = limbs_of(add_low_as_high);

  out->renorm_a     = limbs_of(low_a);
  out->renorm_big   = limbs_of(low_big);
  out->renorm_raw   = limbs_of(drifted);
  out->renorm_fixed = limbs_of(renormed);
} // float_float_operations

// The double-double operations.
__host__ __device__ void double_double_operations(fpmp_results* out)
{
  // The same sequence as above on the wider type; only the type names and the
  // width-dependent constants differ.

  // Construction. Here double is the component type itself, so the literal
  // lands whole in hi and leaves lo at zero. Nothing is narrowed and no
  // explicit cast is involved, unlike the fp32mp2 case above where the same
  // literal has to be split across both limbs. Float and integer sources
  // convert as before.
  const cudax::fp64mp2 a{1.234567890123456789};
  const cudax::fp64mp2 b{9.876543210987654321};
  const cudax::fp64mp2 c{2.71828182f};
  const cudax::fp64mp2 d{5u};

  // Arithmetic. As on the narrower type, results stay in the pair
  // representation.
  const cudax::fp64mp2 sum  = a + b;
  const cudax::fp64mp2 diff = a - b;
  const cudax::fp64mp2 prod = a * b;
  const cudax::fp64mp2 quot = a / b;

  const cudax::fp64mp2 sqrt_d  = sqrt(d);
  const cudax::fp64mp2 rsqrt_d = rsqrt(d);
  const cudax::fp64mp2 fma_abc = fma(a, b, c);

  const cudax::fp64mp2 exp_a = exp(a);
  const cudax::fp64mp2 sin_a = sin(a);

  const bool is_gt = a > b;
  const bool is_lt = a < b;
  const bool is_eq = a == b;

  // Mixed-type operations, as above. The scalar is converted to fp64mp2 and the
  // arithmetic runs at full pair precision; at this width the double literal is
  // already the component type, so that conversion costs nothing and loses
  // nothing.
  const cudax::fp64mp2 mixed_add = a + 0.9876543210987654;
  const cudax::fp64mp2 mixed_mul = a * 3;
  const cudax::fp64mp2 mixed_lhs = 2 * a;

  // Accumulating a single double takes the same dedicated path.
  cudax::fp64mp2 acc = a;
  acc += 0.5;

  // The two components a value is built from: a + b is exact in the pair, and
  // lo holds what a single double would have dropped.
  const cudax::fp64mp2 s = a + b;

  // Accuracy levels, exactly as above but on the wider type: the limb values
  // shrink to match the wider mantissa, nothing else changes.
  const cudax::fp64mp2 p{1.0, 7.3e-17};
  const cudax::fp64mp2 q{-(1.0 - 1.0e-16), 5.1e-17};
  const cudax::fp64mp2 add_def = p + q;

  // fpmp2_accuracy::high recovers the trailing limb the default level rounds
  // away. At this width that limb sits far below anything a single double could
  // show, which is why these results are reported as limb pairs rather than as
  // one number.
  const cudax::fp64mp2_high high_p{1.0, 7.3e-17};
  const cudax::fp64mp2_high high_q{-(1.0 - 1.0e-16), 5.1e-17};
  const cudax::fp64mp2_high add_high = high_p + high_q;

  // The same per-operation choice: low-accuracy operands summed at high
  // accuracy.
  const cudax::fp64mp2_low low_p{1.0, 7.3e-17};
  const cudax::fp64mp2_low low_q{-(1.0 - 1.0e-16), 5.1e-17};
  const cudax::fp64mp2_low add_low_as_high = cudax::renormalize(cudax::add<cudax::fpmp2_accuracy::high>(low_p, low_q));

  // Renormalization, as above: the detour through big leaves the trailing limb
  // sized for big rather than for the small result, and renormalize()
  // redistributes the two. big is around 1e7 here so that the drift lands where
  // a double can still show it.
  const cudax::fp64mp2_low low_a{1.2345678901234567, 5.5e-17};
  const cudax::fp64mp2_low low_big{12345678.901234567, 8.0e-10};
  const cudax::fp64mp2_low drifted  = low_a - low_big + low_big;
  const cudax::fp64mp2_low renormed = cudax::renormalize(drifted);

  // The single conversion point, as in the fp32mp2 kernel.
  out->a = static_cast<double>(a);
  out->b = static_cast<double>(b);
  out->c = static_cast<double>(c);
  out->d = static_cast<double>(d);

  out->sum  = static_cast<double>(sum);
  out->diff = static_cast<double>(diff);
  out->prod = static_cast<double>(prod);
  out->quot = static_cast<double>(quot);

  out->sqrt_d  = static_cast<double>(sqrt_d);
  out->rsqrt_d = static_cast<double>(rsqrt_d);
  out->fma_abc = static_cast<double>(fma_abc);

  out->exp_a = static_cast<double>(exp_a);
  out->sin_a = static_cast<double>(sin_a);

  out->is_gt = is_gt;
  out->is_lt = is_lt;
  out->is_eq = is_eq;

  out->mixed_add = static_cast<double>(mixed_add);
  out->mixed_mul = static_cast<double>(mixed_mul);
  out->mixed_lhs = static_cast<double>(mixed_lhs);
  out->mixed_acc = static_cast<double>(acc);

  out->hi = static_cast<double>(s.hi());
  out->lo = static_cast<double>(s.lo());

  out->p               = limbs_of(p);
  out->q               = limbs_of(q);
  out->add_def         = limbs_of(add_def);
  out->add_high        = limbs_of(add_high);
  out->add_low_as_high = limbs_of(add_low_as_high);

  out->renorm_a     = limbs_of(low_a);
  out->renorm_big   = limbs_of(low_big);
  out->renorm_raw   = limbs_of(drifted);
  out->renorm_fixed = limbs_of(renormed);
} // double_double_operations

__global__ void float_float_kernel(fpmp_results* out)
{
  float_float_operations(out);
}

__global__ void double_double_kernel(fpmp_results* out)
{
  double_double_operations(out);
}

// Print the results of the fpmp operations. `where` names the side that produced
// them, since a single run reports both.
static void print_results(const char* where, const char* type_name, const char* description, const fpmp_results& r)
{
  printf("\n");
  printf("====================================================================="
         "===========\n");
  printf("  %s on the %s - %s\n", type_name, where, description);
  printf("====================================================================="
         "===========\n");

  printf("\ninputs\n");
  printf("  a = %.17g   (from a double literal)\n", r.a);
  printf("  b = %.17g   (from a double literal)\n", r.b);
  printf("  c = %.17g   (from a float literal)\n", r.c);
  printf("  d = %.17g   (from an unsigned literal)\n", r.d);

  printf("\narithmetic\n");
  printf("  a + b            = %.17g\n", r.sum);
  printf("  a - b            = %.17g\n", r.diff);
  printf("  a * b            = %.17g\n", r.prod);
  printf("  a / b            = %.17g\n", r.quot);

  printf("\nsqrt, rsqrt, fma\n");
  printf("  sqrt(d)          = %.17g\n", r.sqrt_d);
  printf("  rsqrt(d)         = %.17g\n", r.rsqrt_d);
  printf("  fma(a, b, c)     = %.17g\n", r.fma_abc);

  printf("\nmath functions\n");
  printf("  exp(a)           = %.17g\n", r.exp_a);
  printf("  sin(a)           = %.17g\n", r.sin_a);

  printf("\ncomparisons\n");
  printf("  a > b            = %s\n", r.is_gt ? "true" : "false");
  printf("  a < b            = %s\n", r.is_lt ? "true" : "false");
  printf("  a == b           = %s\n", r.is_eq ? "true" : "false");

  printf("\nmixed-type operations, an fpmp2 combined with a built-in scalar\n");
  printf("  a + 0.9876543210987654  = %.17g\n", r.mixed_add);
  printf("  a * 3                   = %.17g\n", r.mixed_mul);
  printf("  2 * a                   = %.17g\n", r.mixed_lhs);
  printf("  acc = a; acc += 0.5     = %.17g\n", r.mixed_acc);

  printf("\nstored components of a + b\n");
  printf("  hi               = %.17g\n", r.hi);
  printf("  lo               = %.17g\n", r.lo);

  // Shown as hi + lo rather than as one double: the leading limbs of the sum
  // agree at every accuracy level, and the whole difference lies in the
  // trailing limb.
  printf("\naccuracy levels, on a sum whose leading limbs almost cancel\n");
  printf("                                       %-24s %s\n", "hi", "lo");
  printf("  p                                    %-24.17g %.17g\n", r.p.hi, r.p.lo);
  printf("  q                                    %-24.17g %.17g\n", r.q.hi, r.q.lo);
  printf("  p + q, default type                  %-24.17g %.17g\n", r.add_def.hi, r.add_def.lo);
  printf("  p + q, accuracy::high type           %-24.17g %.17g\n", r.add_high.hi, r.add_high.lo);
  printf("  add<accuracy::high>(low p, low q)    %-24.17g %.17g\n", r.add_low_as_high.hi, r.add_low_as_high.lo);

  printf("\nrenormalization, on a low-accuracy result whose limbs overlap\n");
  printf("                                       %-24s %s\n", "hi", "lo");
  printf("  a,   in the accuracy::low type       %-24.17g %.17g\n", r.renorm_a.hi, r.renorm_a.lo);
  printf("  big, in the accuracy::low type       %-24.17g %.17g\n", r.renorm_big.hi, r.renorm_big.lo);
  printf("  a - big + big, unnormalized result   %-24.17g %.17g\n", r.renorm_raw.hi, r.renorm_raw.lo);
  printf("  renormalize(a - big + big)           %-24.17g %.17g\n", r.renorm_fixed.hi, r.renorm_fixed.lo);
}

int main()
{
  // The operations above are __host__ __device__, so the build carries a host and
  // a device copy of each and one run can report both. The two agree wherever the
  // arithmetic is defined by the format; single-precision sqrt and rsqrt are the
  // exception, as they bottom out in a hardware primitive that the two sides are
  // not required to round identically.
  fpmp_results host_float_float{};
  fpmp_results host_double_double{};

  float_float_operations(&host_float_float);
  double_double_operations(&host_double_double);

  print_results("host", "fp32mp2", "float-float, ~46 effective mantissa bits", host_float_float);
  print_results("host", "fp64mp2", "double-double, ~104 effective mantissa bits", host_double_double);

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
  {
    cudaGetLastError(); // discard the sticky error so it cannot mask a later one
    printf("\nno CUDA device available, so only the host results are shown\n\n");
    return 0;
  }

  // Name the GPU the device results came from. Its architecture is worth having next to
  // the numbers, since that is what decides which instructions the arithmetic is built on.
  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, device_id);
  printf("\nthe device results below come from %s, sm_%d%d\n", props.name, props.major, props.minor);

  fpmp_results* float_float;
  fpmp_results* double_double;

  cudaMallocManaged(&float_float, sizeof(fpmp_results));
  cudaMallocManaged(&double_double, sizeof(fpmp_results));

  // The same two functions, this time from a kernel.
  float_float_kernel<<<1, 1>>>(float_float);
  double_double_kernel<<<1, 1>>>(double_double);

  cudaDeviceSynchronize();

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    cudaFree(float_float);
    cudaFree(double_double);
    return 1;
  }

  print_results("device", "fp32mp2", "float-float, ~46 effective mantissa bits", *float_float);
  print_results("device", "fp64mp2", "double-double, ~104 effective mantissa bits", *double_double);

  cudaFree(float_float);
  cudaFree(double_double);

  printf("\n");

  return 0;
} // main
