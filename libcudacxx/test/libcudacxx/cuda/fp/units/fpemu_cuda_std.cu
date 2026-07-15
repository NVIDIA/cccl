// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: cuda::std math overloads for the emulated double types.
//
//  The emulated operations live in cuda::experimental, but a qualified
//  cuda::std::fma / cuda::std::sqrt call suppresses ADL. Without dedicated
//  overloads in namespace cuda::std, such a call would silently narrow the
//  fpemu/fpemu_unpacked argument to double (via the implicit conversion) and
//  compute a native-double result. This test verifies that:
//    - cuda::std::fma and cuda::std::sqrt select the emulated implementation for
//      both the packed fpemu<double> and the unpacked fpemu_unpacked<double>,
//    - the RETURN TYPE is the emulated type (not double) -- the compile-time
//      guard that proves the fallback overload was not chosen,
//    - mixed fpemu + built-in arithmetic operands are also handled,
//    - the numeric results match the expected values.
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

_CCCL_HOST_DEVICE bool run_test()
{
  bool ok               = true;
  constexpr double kTol = 1e-10;

  // ---- packed fpemu<double> --------------------------------------------------
  {
    const fpemu<double> a(2.0), b(3.0), c(1.0);

    // Return type must be the emulated type, proving cuda::std::fma did not fall
    // back to the double overload via the implicit fpemu -> double conversion.
    static_assert(::cuda::std::is_same_v<decltype(::cuda::std::fma(a, b, c)), fpemu<double>>);
    static_assert(::cuda::std::is_same_v<decltype(::cuda::std::sqrt(a)), fpemu<double>>);

    const fpemu<double> r = ::cuda::std::fma(a, b, c); // 2*3 + 1
    ok                    = ok && (::cuda::std::fabs(static_cast<double>(r) - 7.0) <= kTol);

    const fpemu<double> s = ::cuda::std::sqrt(fpemu<double>(4.0));
    ok                    = ok && (::cuda::std::fabs(static_cast<double>(s) - 2.0) <= kTol);

    // Mixed fpemu + built-in arithmetic operands.
    static_assert(::cuda::std::is_same_v<decltype(::cuda::std::fma(a, 3.0, c)), fpemu<double>>);
    const fpemu<double> rm = ::cuda::std::fma(a, 3.0, c); // 2*3 + 1
    ok                     = ok && (::cuda::std::fabs(static_cast<double>(rm) - 7.0) <= kTol);
  }

  // ---- unpacked fpemu_unpacked<double> --------------------------------------
  {
    const fpemu_unpacked<double> a(2.0), b(3.0), c(1.0);

    static_assert(::cuda::std::is_same_v<decltype(::cuda::std::fma(a, b, c)), fpemu_unpacked<double>>);
    static_assert(::cuda::std::is_same_v<decltype(::cuda::std::sqrt(a)), fpemu_unpacked<double>>);

    const fpemu_unpacked<double> r = ::cuda::std::fma(a, b, c); // 2*3 + 1
    ok                             = ok && (::cuda::std::fabs(static_cast<double>(r) - 7.0) <= kTol);

    const fpemu_unpacked<double> s = ::cuda::std::sqrt(fpemu_unpacked<double>(9.0));
    ok                             = ok && (::cuda::std::fabs(static_cast<double>(s) - 3.0) <= kTol);

    static_assert(::cuda::std::is_same_v<decltype(::cuda::std::fma(1.0, b, 2.0)), fpemu_unpacked<double>>);
    const fpemu_unpacked<double> rm = ::cuda::std::fma(1.0, b, 2.0); // 1*3 + 2
    ok                              = ok && (::cuda::std::fabs(static_cast<double>(rm) - 5.0) <= kTol);
  }

  // ---- non-def accuracy levels still route through the emulated overloads ----
  {
    const fpemu<double, fpemu_accuracy::high> a(2.0), b(3.0), c(1.0);
    static_assert(::cuda::std::is_same_v<decltype(::cuda::std::fma(a, b, c)), fpemu<double, fpemu_accuracy::high>>);
    const fpemu<double, fpemu_accuracy::high> r = ::cuda::std::fma(a, b, c);
    ok                                          = ok && (::cuda::std::fabs(static_cast<double>(r) - 7.0) <= kTol);
  }

  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu cuda::std math overloads", "[fpemu]")
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
