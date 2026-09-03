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
    fpemu.cpp - IEEE-754 compliant double-precision arithmetic software implementation
    ==================================================================================

    fpemu provides an IEEE-754 double built from 32-bit integer and float operations
    rather than native FP64 instructions, for GPUs where FP64 throughput is limited or
    absent. The accuracy level is chosen per type:

        fp64emu_high   correctly rounded, full IEEE-754 range (INF, NaN, subnormals)
        fp64emu_mid    up to 1-2 least significant mantissa bits of error, and the
                       normal range only, without subnormals or special values
        fp64emu_low    up to half the mantissa bits may be lost, same normal range
        fp64emu        the default selector, currently the same as fp64emu_high

    The value comes in two representations, and this example runs one kernel per
    representation over the same set of operations:

      fp64emu           packed: the 64-bit IEEE bit pattern, as a double would be stored.
                        Constructs implicitly from the built-in arithmetic types, and at
                        the high accuracy level reproduces double bit for bit.

      fp64emu_unpacked  unpacked: separate sign, exponent and mantissa fields, with 9
                        extra bits extending the 53-bit significand. Arithmetic therefore
                        runs on 62 significand bits and rounds to the storage format
                        once, when the value is packed, instead of after every operation.

    Both offer the same operations, so most of the two kernels below read identically. The
    differences worth watching for are that the unpacked type needs a written-out cast
    wherever the packed one converts implicitly, and that deferring the rounding pays off
    over a long run of operations, which the last section measures.

    Demonstrated:

      - construction from literals of several source types
      - arithmetic: +, -, *, /
      - mixed-type arithmetic, an fpemu value combined directly with a built-in scalar
      - sqrt and fma
      - comparison operators
      - the accuracy levels, on the two operations where they visibly disagree
      - accumulating terms that fall below one ulp of the running total, where the
        unpacked form's deferred rounding separates it from double and from packed
      - converting between the two representations
      - the same source running on host and device
*/
#include <cuda/fpemu>

#include <cstdio>

// The CCCL FP component lives in cuda::experimental (later cuda::),
// abbreviated here rather than pulled in with a using-directive. Type names
// carry the cudax:: prefix; the standard-named math functions are left
// unqualified and found by argument-dependent lookup, so a body of double
// code keeps its call sites when the type underneath is swapped.
namespace cudax = cuda::experimental;

// The accumulation demo below adds this many copies of this term onto 1.0. One
// ulp of 1.0 is about 2.22e-16, so a single term is some twenty times too small
// to move a double at all: the interesting part is what a long run of such
// additions does. Named because both kernels and the printer need to agree on
// them.
static constexpr int fpemu_drift_terms   = 1000;
static constexpr double fpemu_drift_term = 1e-17;

// One kernel result, as doubles so the host can print it without knowing which
// representation produced it.
struct fpemu_results
{
  double a, b, c, d; // the four inputs
  double sum, diff, prod, quot; // arithmetic operations
  double sqrt_d, fma_abc; // sqrt and fma operations
  bool is_gt, is_lt, is_eq; // comparison operations
  double mixed_add; // fpemu + a double literal
  double mixed_mul; // fpemu * an int
  double mixed_lhs; // an int * fpemu, scalar on the left
  double mixed_acc; // += a scalar
  // The same two operations at each accuracy level. Subtraction and
  // multiplication are used because division and sqrt agree across the levels
  // for these operands.
  double diff_high, diff_mid, diff_low; // subtraction at each accuracy level
  double prod_high, prod_mid, prod_low; // multiplication at each accuracy level
  double drift_sum; // the accumulation described above
  double converted; // round trip through the other representation
};

// The packed form of the fpemu type, which holds the same bit pattern as a double.
__host__ __device__ void fpemu_packed_operations(fpemu_results* out)
{
  // Construction. The packed form holds the same bit pattern a double would, so
  // every built-in arithmetic type converts implicitly, exactly as it would to
  // double.
  const cudax::fp64emu a = 1.234567890123456789;
  const cudax::fp64emu b = 9.876543210987654321;
  const cudax::fp64emu c = 2.71828182f;
  const cudax::fp64emu d = 5u;

  // Arithmetic. An operation on two fp64emu values yields an fp64emu, so
  // results stay in the emulated type and can feed further operations without a
  // conversion in between.
  const cudax::fp64emu sum  = a + b;
  const cudax::fp64emu diff = a - b;
  const cudax::fp64emu prod = a * b;
  const cudax::fp64emu quot = a / b;

  // Unqualified on purpose: ADL finds the fpemu overloads, so a call site
  // written for double needs no edit when the type underneath it changes.
  const cudax::fp64emu sqrt_d  = sqrt(d);
  const cudax::fp64emu fma_abc = fma(a, b, c);

  // Comparisons return plain bool.
  const bool is_gt = a > b;
  const bool is_lt = a < b;
  const bool is_eq = a == b;

  // Mixed-type operations. An fpemu value combines directly with the built-in
  // arithmetic types, on either side, without building a temporary by hand. The
  // result is still an fp64emu, not a double.
  const cudax::fp64emu mixed_add = a + 0.9876543210987654;
  const cudax::fp64emu mixed_mul = a * 3;
  const cudax::fp64emu mixed_lhs = 2 * a;

  // Compound assignment takes a bare scalar here, since the conversion is
  // implicit.
  cudax::fp64emu acc = a;
  acc += 0.5;

  // Accuracy levels. Only the type differs; the expressions are the same. high
  // is correctly rounded, mid gives up a low bit or two, and low trades away
  // enough of the mantissa to be visible in the seventh decimal place.
  const cudax::fp64emu_high a_high = 1.234567890123456789;
  const cudax::fp64emu_high b_high = 9.876543210987654321;
  const cudax::fp64emu_mid a_mid   = 1.234567890123456789;
  const cudax::fp64emu_mid b_mid   = 9.876543210987654321;
  const cudax::fp64emu_low a_low   = 1.234567890123456789;
  const cudax::fp64emu_low b_low   = 9.876543210987654321;

  const cudax::fp64emu_high diff_high = a_high - b_high;
  const cudax::fp64emu_mid diff_mid   = a_mid - b_mid;
  const cudax::fp64emu_low diff_low   = a_low - b_low;
  const cudax::fp64emu_high prod_high = a_high * b_high;
  const cudax::fp64emu_mid prod_mid   = a_mid * b_mid;
  const cudax::fp64emu_low prod_low   = a_low * b_low;

  // Accumulation. The packed form rounds to the storage format after every
  // operation, so each term vanishes on its own and the total never leaves 1.0
  // — the same outcome a plain double gives, which is what being bit-compatible
  // with double means here.
  cudax::fp64emu drift = 1.0;
  for (int i = 0; i < fpemu_drift_terms; ++i)
  {
    drift += fpemu_drift_term;
  }

  // Converting to the other representation and back. Both directions are
  // explicit.
  const cudax::fp64emu_unpacked as_unpacked{a};
  const cudax::fp64emu round_trip{as_unpacked};

  // Everything above stayed in the emulated type. Converting to double happens
  // here, in one place, only because the host has to be handed something it can
  // print.
  out->a = static_cast<double>(a);
  out->b = static_cast<double>(b);
  out->c = static_cast<double>(c);
  out->d = static_cast<double>(d);

  out->sum  = static_cast<double>(sum);
  out->diff = static_cast<double>(diff);
  out->prod = static_cast<double>(prod);
  out->quot = static_cast<double>(quot);

  out->sqrt_d  = static_cast<double>(sqrt_d);
  out->fma_abc = static_cast<double>(fma_abc);

  out->is_gt = is_gt;
  out->is_lt = is_lt;
  out->is_eq = is_eq;

  out->mixed_add = static_cast<double>(mixed_add);
  out->mixed_mul = static_cast<double>(mixed_mul);
  out->mixed_lhs = static_cast<double>(mixed_lhs);
  out->mixed_acc = static_cast<double>(acc);

  out->diff_high = static_cast<double>(diff_high);
  out->diff_mid  = static_cast<double>(diff_mid);
  out->diff_low  = static_cast<double>(diff_low);
  out->prod_high = static_cast<double>(prod_high);
  out->prod_mid  = static_cast<double>(prod_mid);
  out->prod_low  = static_cast<double>(prod_low);

  out->drift_sum = static_cast<double>(drift);
  out->converted = static_cast<double>(round_trip);
} // fpemu_packed_operations

// The unpacked form of the fpemu type, which holds
// the sign, exponent and mantissa separately,
// with 9 extra bits extending the 53-bit significand.
__host__ __device__ void fpemu_unpacked_operations(fpemu_results* out)
{
  // Construction. The unpacked form keeps sign, exponent and mantissa apart
  // rather than in a double's layout, so it does not pretend to be a built-in
  // number: every conversion in is written out. Braces read best, and a cast
  // does the same job.
  const cudax::fp64emu_unpacked a{1.234567890123456789};
  const cudax::fp64emu_unpacked b{9.876543210987654321};
  const cudax::fp64emu_unpacked c{2.71828182f};
  const cudax::fp64emu_unpacked d{5u};

  // Arithmetic, as on the packed form: the result of every operation is another
  // fp64emu_unpacked. Keeping it in this type is the whole point here, since
  // that is what holds the guard bits back from being rounded away.
  const cudax::fp64emu_unpacked sum  = a + b;
  const cudax::fp64emu_unpacked diff = a - b;
  const cudax::fp64emu_unpacked prod = a * b;
  const cudax::fp64emu_unpacked quot = a / b;

  const cudax::fp64emu_unpacked sqrt_d  = sqrt(d);
  const cudax::fp64emu_unpacked fma_abc = fma(a, b, c);

  const bool is_gt = a > b;
  const bool is_lt = a < b;
  const bool is_eq = a == b;

  // Mixed-type operations work as they do for the packed form: the binary
  // operators take a built-in scalar on either side, with no cast needed at the
  // call site.
  const cudax::fp64emu_unpacked mixed_add = a + 0.9876543210987654;
  const cudax::fp64emu_unpacked mixed_mul = a * 3;
  const cudax::fp64emu_unpacked mixed_lhs = 2 * a;

  // Compound assignment is the exception, and follows from construction being
  // explicit: there is no scalar overload, and the fp64emu_unpacked one cannot
  // be reached from a double implicitly, so the scalar is cast.
  cudax::fp64emu_unpacked acc = a;
  acc += cudax::fp64emu_unpacked{0.5};

  // Accuracy levels, as above on the other representation. high and mid can
  // agree here where the packed ones did not: the guard bits absorb the
  // difference between the two algorithms before it reaches the stored value.
  const cudax::fp64emu_unpacked_high a_high{1.234567890123456789};
  const cudax::fp64emu_unpacked_high b_high{9.876543210987654321};
  const cudax::fp64emu_unpacked_mid a_mid{1.234567890123456789};
  const cudax::fp64emu_unpacked_mid b_mid{9.876543210987654321};
  const cudax::fp64emu_unpacked_low a_low{1.234567890123456789};
  const cudax::fp64emu_unpacked_low b_low{9.876543210987654321};

  const cudax::fp64emu_unpacked_high diff_high = a_high - b_high;
  const cudax::fp64emu_unpacked_mid diff_mid   = a_mid - b_mid;
  const cudax::fp64emu_unpacked_low diff_low   = a_low - b_low;
  const cudax::fp64emu_unpacked_high prod_high = a_high * b_high;
  const cudax::fp64emu_unpacked_mid prod_mid   = a_mid * b_mid;
  const cudax::fp64emu_unpacked_low prod_low   = a_low * b_low;

  // Accumulation, and the reason this representation exists. The 9 guard bits
  // sit exactly where each term would otherwise be rounded off, so the terms
  // keep contributing while the value stays unpacked; the single rounding
  // happens on the way out. A double and a packed fpemu both round after every
  // operation and so never leave 1.0.
  cudax::fp64emu_unpacked drift{1.0};
  for (int i = 0; i < fpemu_drift_terms; ++i)
  {
    drift += cudax::fp64emu_unpacked{fpemu_drift_term};
  }

  // Converting to the other representation and back.
  const cudax::fp64emu as_packed{a};
  const cudax::fp64emu_unpacked round_trip{as_packed};

  // The single conversion point, as in the packed kernel. For this
  // representation it is also where the one rounding to the 53-bit significand
  // happens.
  out->a = static_cast<double>(a);
  out->b = static_cast<double>(b);
  out->c = static_cast<double>(c);
  out->d = static_cast<double>(d);

  out->sum  = static_cast<double>(sum);
  out->diff = static_cast<double>(diff);
  out->prod = static_cast<double>(prod);
  out->quot = static_cast<double>(quot);

  out->sqrt_d  = static_cast<double>(sqrt_d);
  out->fma_abc = static_cast<double>(fma_abc);

  out->is_gt = is_gt;
  out->is_lt = is_lt;
  out->is_eq = is_eq;

  out->mixed_add = static_cast<double>(mixed_add);
  out->mixed_mul = static_cast<double>(mixed_mul);
  out->mixed_lhs = static_cast<double>(mixed_lhs);
  out->mixed_acc = static_cast<double>(acc);

  out->diff_high = static_cast<double>(diff_high);
  out->diff_mid  = static_cast<double>(diff_mid);
  out->diff_low  = static_cast<double>(diff_low);
  out->prod_high = static_cast<double>(prod_high);
  out->prod_mid  = static_cast<double>(prod_mid);
  out->prod_low  = static_cast<double>(prod_low);

  out->drift_sum = static_cast<double>(drift);
  out->converted = static_cast<double>(round_trip);
} // fpemu_unpacked_operations

__global__ void fpemu_packed_kernel(fpemu_results* out)
{
  fpemu_packed_operations(out);
}

__global__ void fpemu_unpacked_kernel(fpemu_results* out)
{
  fpemu_unpacked_operations(out);
}

// Print the results of the fpemu operations. `where` names the side that
// produced them, since a single run reports both.
static void print_results(
  const char* where, const char* type_name, const char* description, const fpemu_results& r, double drift_double)
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

  printf("\nsqrt, fma\n");
  printf("  sqrt(d)          = %.17g\n", r.sqrt_d);
  printf("  fma(a, b, c)     = %.17g\n", r.fma_abc);

  printf("\ncomparisons\n");
  printf("  a > b            = %s\n", r.is_gt ? "true" : "false");
  printf("  a < b            = %s\n", r.is_lt ? "true" : "false");
  printf("  a == b           = %s\n", r.is_eq ? "true" : "false");

  printf("\nmixed-type operations, an fpemu value combined with a built-in "
         "scalar\n");
  printf("  a + 0.9876543210987654  = %.17g\n", r.mixed_add);
  printf("  a * 3                   = %.17g\n", r.mixed_mul);
  printf("  2 * a                   = %.17g\n", r.mixed_lhs);
  printf("  acc = a; acc += 0.5     = %.17g\n", r.mixed_acc);

  printf("\naccuracy levels\n");
  printf("                     %-24s %s\n", "a - b", "a * b");
  printf("  high             %-24.17g %.17g\n", r.diff_high, r.prod_high);
  printf("  mid              %-24.17g %.17g\n", r.diff_mid, r.prod_mid);
  printf("  low              %-24.17g %.17g\n", r.diff_low, r.prod_low);

  // The exact result needs no accumulation, so a single double operation
  // reaches it: 1e-14 is well within what a double next to 1.0 can represent.
  // Only adding the terms one at a time is lossy.
  const double exact = 1.0 + fpemu_drift_terms * fpemu_drift_term;
  printf("\n%d additions of %g onto 1.0, each term ~22x below one ulp of 1.0\n", fpemu_drift_terms, fpemu_drift_term);
  printf("  exact, as one operation  = %.17g\n", exact);
  printf("  accumulated in double    = %.17g\n", drift_double);
  printf("  accumulated in %-9s = %.17g\n", type_name, r.drift_sum);

  printf("\nround trip through the other representation\n");
  printf("  a                = %.17g\n", r.converted);
}

int main()
{
  // The plain-double baseline for the accumulation section, so each report can
  // show what the hardware type does with the same sequence.
  double drift_double = 1.0;
  for (int i = 0; i < fpemu_drift_terms; ++i)
  {
    drift_double += fpemu_drift_term;
  }

  // The operations above are __host__ __device__, so the build carries a host and
  // a device copy of each and one run can report both. The emulation is integer
  // and FP64 arithmetic either way, so the two sides agree throughout.
  fpemu_results host_packed{};
  fpemu_results host_unpacked{};

  fpemu_packed_operations(&host_packed);
  fpemu_unpacked_operations(&host_unpacked);

  print_results("host", "fp64emu", "packed, the 64-bit IEEE layout", host_packed, drift_double);
  print_results("host", "unpacked", "sign/exponent/mantissa, 9 guard bits", host_unpacked, drift_double);

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

  fpemu_results* packed;
  fpemu_results* unpacked;

  cudaMallocManaged(&packed, sizeof(fpemu_results));
  cudaMallocManaged(&unpacked, sizeof(fpemu_results));

  // The fpemu template instantiations are stack-heavy.
  cudaDeviceSetLimit(cudaLimitStackSize, 16384);

  // The same two functions, this time from a kernel.
  fpemu_packed_kernel<<<1, 1>>>(packed);
  fpemu_unpacked_kernel<<<1, 1>>>(unpacked);

  cudaDeviceSynchronize();

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    cudaFree(packed);
    cudaFree(unpacked);
    return 1;
  }

  print_results("device", "fp64emu", "packed, the 64-bit IEEE layout", *packed, drift_double);
  print_results("device", "unpacked", "sign/exponent/mantissa, 9 guard bits", *unpacked, drift_double);

  cudaFree(packed);
  cudaFree(unpacked);

  printf("\n");

  return 0;
} // main
