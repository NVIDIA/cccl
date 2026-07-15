// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: converting constructors between fpemu representations / accuracies.
//
//  The accuracy conversion (same representation, different accuracy) and the
//  packed <-> unpacked conversions are exposed as EXPLICIT converting
//  constructors (they used to be conversion operators). This test verifies that:
//    - each conversion is explicit (not implicitly convertible) yet explicitly
//      constructible,
//    - all four directions round-trip a value exactly:
//        packed(accA)   -> packed(accB),
//        packed         -> unpacked,
//        unpacked       -> packed,
//        unpacked(accA) -> unpacked(accB),
//    - both class templates remain trivially copyable.
//  The same _CCCL_HOST_DEVICE run_test() runs on the host and, under CUDA, on the
//  device.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include <cstdio>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpemu>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

using P_hi = fpemu<double, fpemu_accuracy::high>;
using P_lo = fpemu<double, fpemu_accuracy::low>;
using U_hi = fpemu_unpacked<double, fpemu_accuracy::high>;
using U_lo = fpemu_unpacked<double, fpemu_accuracy::low>;

// All conversions are explicit: not implicitly convertible ...
static_assert(!::cuda::std::is_convertible_v<P_hi, P_lo>, "accuracy conversion must be explicit");
static_assert(!::cuda::std::is_convertible_v<U_hi, U_lo>, "accuracy conversion must be explicit");
static_assert(!::cuda::std::is_convertible_v<P_hi, U_hi>, "packed -> unpacked must be explicit");
static_assert(!::cuda::std::is_convertible_v<U_hi, P_hi>, "unpacked -> packed must be explicit");
// ... but explicitly constructible.
static_assert(::cuda::std::is_constructible_v<P_lo, P_hi>, "");
static_assert(::cuda::std::is_constructible_v<U_lo, U_hi>, "");
static_assert(::cuda::std::is_constructible_v<U_hi, P_hi>, "");
static_assert(::cuda::std::is_constructible_v<P_hi, U_hi>, "");
// The added converting ctors must not break trivial copyability.
static_assert(::cuda::std::is_trivially_copyable_v<P_hi>, "");
static_assert(::cuda::std::is_trivially_copyable_v<U_hi>, "");

_CCCL_HOST_DEVICE bool run_test()
{
  bool ok               = true;
  constexpr double kTol = 1e-10;

  const double vals[6] = {0.0, 1.5, -2.0, 42.0, 1234.5678, -9.999e12};
  for (int i = 0; i < 6; i++)
  {
    const double d = vals[i];

    const P_hi p(d);
    const P_lo p2(p); // packed accuracy converting ctor
    const U_hi u(p); // packed -> unpacked ctor
    const P_hi back(u); // unpacked -> packed ctor
    const U_lo u2(u); // unpacked accuracy converting ctor

    ok = ok && (::cuda::std::fabs(static_cast<double>(p2) - d) <= kTol);
    ok = ok && (::cuda::std::fabs(static_cast<double>(u) - d) <= kTol);
    ok = ok && (::cuda::std::fabs(static_cast<double>(back) - d) <= kTol);
    ok = ok && (::cuda::std::fabs(static_cast<double>(u2) - d) <= kTol);
  }

  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu converting constructors", "[fpemu]")
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
