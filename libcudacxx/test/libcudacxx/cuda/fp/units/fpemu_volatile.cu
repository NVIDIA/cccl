// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu volatile constructors / assignment + trivial copyability.
//
//  Verifies that the packed (fp64emu*) and unpacked (fp64emu_unpacked*) types are
//  trivially copyable (required for cooperative_groups, __shfl, etc.) and that
//  they correctly support construction from volatile, assignment to volatile, and
//  assignment from volatile, preserving values through volatile round-trips. The
//  same _CCCL_HOST_DEVICE run_test() runs on the host and, under CUDA, on device.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <type_traits>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpemu>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Compile-time: every accuracy variant, packed and unpacked, is trivially copyable.
static_assert(std::is_trivially_copyable<fp64emu>::value, "fp64emu must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64emu_low>::value, "fp64emu_low must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64emu_high>::value, "fp64emu_high must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64emu_unpacked>::value, "fp64emu_unpacked must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64emu_unpacked_low>::value, "fp64emu_unpacked_low must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64emu_unpacked_high>::value,
              "fp64emu_unpacked_high must be trivially copyable");

// Exercise the four volatile paths for one emulated type; values are exact double
// bit patterns so the round-trips must be exactly preserved.
template <typename emu_type>
_CCCL_HOST_DEVICE bool vol_ok()
{
  const double v1 = 3.141592653589793;
  const double v2 = 2.718281828459045;
  bool ok         = true;

  // Construct from volatile.
  {
    volatile emu_type vol;
    const emu_type tmp(v1);
    vol = tmp;
    emu_type non_vol(vol);
    ok = ok && ((double) non_vol == v1);
  }

  // Assign to volatile, read back via construct-from-volatile.
  {
    emu_type src(v1);
    volatile emu_type vol;
    vol = src;
    emu_type readback(vol);
    ok = ok && ((double) readback == v1);
  }

  // Assign from volatile.
  {
    volatile emu_type vol;
    const emu_type tmp(v2);
    vol = tmp;
    emu_type dst;
    dst = vol;
    ok  = ok && ((double) dst == v2);
  }

  // Volatile round-trip preserves the value.
  {
    emu_type src(v1);
    volatile emu_type vol;
    vol = src;
    emu_type dst(vol);
    ok = ok && ((double) src == (double) dst);
  }

  return ok;
}

_CCCL_HOST_DEVICE bool run_test()
{
  return vol_ok<fp64emu>() && vol_ok<fp64emu_low>() && vol_ok<fp64emu_high>() && vol_ok<fp64emu_unpacked>()
      && vol_ok<fp64emu_unpacked_low>() && vol_ok<fp64emu_unpacked_high>();
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu volatile constructors + assignment", "[fpemu]")
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
