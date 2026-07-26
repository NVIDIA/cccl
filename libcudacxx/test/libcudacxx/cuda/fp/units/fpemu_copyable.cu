// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu / fp64emu_unpacked trivial copyability + volatile round-trip.
//
//  Compile-time static_asserts check that both the packed (fp64emu) and unpacked
//  (fp64emu_unpacked) types are trivially copyable; run_test() confirms a value
//  survives a round-trip through a volatile object for each. The same
//  _CCCL_HOST_DEVICE run_test() runs on the host and, under CUDA, on the device.
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

static_assert(std::is_trivially_copyable<fp64emu>::value, "fp64emu must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64emu_unpacked>::value, "fp64emu_unpacked must be trivially copyable");

// Round-trip both the packed and unpacked types through a volatile object.
_CCCL_HOST_DEVICE bool run_test()
{
  bool ok = true;

  // Packed type (fp64emu).
  {
    volatile fp64emu vx[1];
    fp64emu x[1] = {fp64emu(1.0e+20)};
    vx[0]        = x[0];
    fp64emu readback(vx[0]); // template volatile copy constructor
    ok = ok && !(readback != x[0]);
  }

  // Unpacked type (fp64emu_unpacked).
  {
    volatile fp64emu_unpacked vx[1];
    fp64emu_unpacked x[1] = {fp64emu_unpacked(1.0e+20)};
    vx[0]                 = x[0];
    fp64emu_unpacked readback(vx[0]);
    ok = ok && !(readback != x[0]);
  }

  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu trivially-copyable + volatile round-trip", "[fpemu]")
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
