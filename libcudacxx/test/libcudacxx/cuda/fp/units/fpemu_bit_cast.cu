// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: bit_cast on emulated double values (packed fpemu and unpacked).
//
//  Verifies that:
//    - the packed fpemu<double> round-trips through its 64-bit IEEE-754
//      representation via ::cuda::std::bit_cast and is bit-identical to the
//      native double (bits is private; bit_cast is the supported reinterpret),
//    - the unpacked emulated double is layout-compatible with and trivially
//      copyable to its raw {sign, exponent, mantissa} representation, so an
//      equal-size ::cuda::std::bit_cast round-trips values exactly (there is no
//      size-changing bit_cast overload), produces the expected result of a simple
//      arithmetic expression, and yields an identical raw representation for a
//      plain conversion across all accuracy levels.
//  The same _CCCL_HOST_DEVICE run_test() runs on the host and, under CUDA, on the
//  device.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cstdio>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpemu>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Same-size (16-byte) mirror of the unpacked representation {sign, exponent, mantissa}.
// fpemu_unpacked keeps its storage private and intentionally offers no size-changing
// bit_cast overload, so raw access goes through an equal-size ::cuda::std::bit_cast.
struct fpemu_unpacked_bits
{
  uint32_t sign;
  uint32_t exponent;
  uint64_t mantissa;
};

_CCCL_HOST_DEVICE bool run_test()
{
  bool ok = true;

  // Packed fpemu<double>: reinterpret to/from the 64-bit IEEE-754 representation
  // via the standard bit_cast (fpemu::bits is private, so this is the only way).
  const double packed_vals[6] = {1.5, -2.0, 0.0, -0.0, 42.0, 3.14159265358979323846};
  for (int i = 0; i < 6; i++)
  {
    const fpemu<double> p(packed_vals[i]);
    const uint64_t pbits = ::cuda::std::bit_cast<uint64_t>(p);
    // fpemu<double> is a faithful double, so its bits match the native double's.
    ok = ok && (pbits == ::cuda::std::bit_cast<uint64_t>(packed_vals[i]));
    // uint64_t -> fpemu<double> -> double round-trips the value exactly.
    ok = ok && (static_cast<double>(::cuda::std::bit_cast<fpemu<double>>(pbits)) == packed_vals[i]);
  }

  // Unpacked fpemu is layout-compatible with, and trivially copyable to, its raw
  // {sign, exponent, mantissa} representation, so an equal-size bit_cast is the
  // supported way to reach the storage (there is no size-changing overload).
  static_assert(sizeof(fpemu_unpacked<double, fpemu_accuracy::def>) == sizeof(fpemu_unpacked_bits),
                "unpacked fpemu must be bit-compatible with its representation");
  static_assert(::cuda::std::is_trivially_copyable_v<fpemu_unpacked<double, fpemu_accuracy::def>>,
                "unpacked fpemu must be trivially copyable for bit_cast");

  // Round-trip: double -> unpacked -> (equal-size) bits -> unpacked -> value.
  const double test_vals[5] = {1.5, -2.0, 0.0, 42.0, 3.14159265358979323846};
  for (int i = 0; i < 5; i++)
  {
    fpemu_unpacked<double, fpemu_accuracy::def> x(test_vals[i]);
    const auto rep = ::cuda::std::bit_cast<fpemu_unpacked_bits>(x);
    const auto y   = ::cuda::std::bit_cast<fpemu_unpacked<double, fpemu_accuracy::def>>(rep);
    ok             = ok && (static_cast<double>(y) == test_vals[i]);
  }

  // Arithmetic result via value conversion: 2 * 3 + 1 == 7.
  fpemu_unpacked<double, fpemu_accuracy::def> a(2.0), b(3.0), c(1.0);
  ok = ok && (::cuda::std::fabs(static_cast<double>(a * b + c) - 7.0) <= 1e-10);

  // A plain conversion produces an identical raw representation across accuracy levels.
  const double pi     = 3.14159265358979323846;
  const auto rep_def  = ::cuda::std::bit_cast<fpemu_unpacked_bits>(fpemu_unpacked<double, fpemu_accuracy::def>(pi));
  const auto rep_high = ::cuda::std::bit_cast<fpemu_unpacked_bits>(fpemu_unpacked<double, fpemu_accuracy::high>(pi));
  const auto rep_low  = ::cuda::std::bit_cast<fpemu_unpacked_bits>(fpemu_unpacked<double, fpemu_accuracy::low>(pi));
  ok                  = ok && (rep_def.sign == rep_high.sign) && (rep_def.exponent == rep_high.exponent)
    && (rep_def.mantissa == rep_high.mantissa);
  ok = ok && (rep_def.sign == rep_low.sign) && (rep_def.exponent == rep_low.exponent)
    && (rep_def.mantissa == rep_low.mantissa);

  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu bit_cast (packed and unpacked)", "[fpemu]")
{
  fp_ran_on_host();
  REQUIRE(run_test());

#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
  bool* d_ok = nullptr;
  REQUIRE_CUDART(cudaMallocManaged(&d_ok, sizeof(bool)));
  *d_ok = false;
  run_test_kernel<<<1, 1>>>(d_ok);
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(*d_ok);
  REQUIRE_CUDART(cudaFree(d_ok));
#endif // _CCCL_CUDA_COMPILATION()
}
