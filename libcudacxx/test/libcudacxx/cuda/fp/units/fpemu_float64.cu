// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fpemu<_Float64> behaves like fpemu<double>.
//
//  C++23's _Float64 (the type behind std::float64_t) is a *distinct* type from
//  double even though it is bit-identical (is_same_v<double, _Float64> is false on
//  GCC/clang in C++23 mode). fpemu accepts it as a bit-identical alias so that
//  fpemu<_Float64> instantiates and behaves exactly like fpemu<double>. The whole
//  test is guarded by the standard feature-test macro __STDCPP_FLOAT64_T__, which
//  is defined precisely when the _Float64 type is available and distinct; on older
//  language modes the type is either absent or an alias for double, so there is
//  nothing extra to check.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cstdio>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpemu>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if defined(__STDCPP_FLOAT64_T__) && (__STDCPP_FLOAT64_T__ == 1)

// _Float64 is a distinct type here, yet fpemu<_Float64> must still be a valid,
// trivially copyable emulated double that constructs from / converts to double.
static_assert(!::cuda::std::is_same_v<double, _Float64>, "expected _Float64 to be a distinct type in this mode");
static_assert(::cuda::std::is_trivially_copyable_v<fpemu<_Float64>>, "");
static_assert(::cuda::std::is_trivially_copyable_v<fpemu_unpacked<_Float64>>, "");
static_assert(sizeof(fpemu<_Float64>) == sizeof(fpemu<double>), "");
static_assert(::cuda::std::is_constructible_v<fpemu<_Float64>, double>, "");
static_assert(::cuda::std::is_constructible_v<fpemu<_Float64>, int>, "");

_CCCL_HOST_DEVICE bool run_test()
{
  bool ok             = true;
  const double vals[] = {0.0, 1.5, -3.25, 1234.5678, -9.999e12};
  for (double d : vals)
  {
    fpemu<_Float64> a(d);
    fpemu<double> b(d);
    // Same value in, same 64-bit result out as the double instantiation.
    ok = ok && (::cuda::std::bit_cast<uint64_t>((double) a) == ::cuda::std::bit_cast<uint64_t>((double) b))
      && (::cuda::std::bit_cast<uint64_t>((double) a) == ::cuda::std::bit_cast<uint64_t>(d));
  }
  return ok;
}

#  if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#  endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu _Float64 behaves like double", "[fpemu]")
{
  fp_ran_on_host();
  REQUIRE(run_test());

#  if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
  bool* d_ok = nullptr;
  REQUIRE_CUDART(cudaMallocManaged(&d_ok, sizeof(bool)));
  *d_ok = false;
  run_test_kernel<<<1, 1>>>(d_ok);
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(*d_ok);
  REQUIRE_CUDART(cudaFree(d_ok));
#  endif // _CCCL_CUDA_COMPILATION()
}

#else // ^^^ __STDCPP_FLOAT64_T__ ^^^ / vvv no distinct _Float64 vvv

C2H_TEST("fpemu _Float64 behaves like double", "[fpemu]")
{
  // _Float64 is unavailable or an alias for double in this language mode; the
  // fpemu<double> path already covers it. Nothing distinct to exercise.
  fp_ran_on_host();
  SUCCEED("no distinct _Float64 in this language mode");
}

#endif // __STDCPP_FLOAT64_T__
