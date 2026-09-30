// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fpmp2 volatile constructors / assignment + trivial copyability.
//
//  Verifies that the fp32mp2 / fp64mp2 accuracy variants are trivially copyable
//  (required for cooperative_groups, __shfl, etc.) and that they correctly support
//  construction from volatile, assignment to volatile, assignment from volatile,
//  assignment between two volatile objects and reading hi/lo off a volatile object,
//  preserving hi/lo through volatile round-trips.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Compile-time: every accuracy variant is trivially copyable.
static_assert(::cuda::std::is_trivially_copyable<cudax::fp32mp2>::value, "fp32mp2 must be trivially copyable");
static_assert(::cuda::std::is_trivially_copyable<cudax::fp32mp2_low>::value, "fp32mp2_low must be trivially copyable");
static_assert(::cuda::std::is_trivially_copyable<cudax::fp32mp2_high>::value,
              "fp32mp2_high must be trivially copyable");
static_assert(::cuda::std::is_trivially_copyable<cudax::fp64mp2>::value, "fp64mp2 must be trivially copyable");
static_assert(::cuda::std::is_trivially_copyable<cudax::fp64mp2_low>::value, "fp64mp2_low must be trivially copyable");
static_assert(::cuda::std::is_trivially_copyable<cudax::fp64mp2_high>::value,
              "fp64mp2_high must be trivially copyable");

// Exercise the volatile paths for one fpmp2 type. Value checks use a
// tolerance (the double-word truncates the source double); the bit-preserving
// round-trip is checked exactly on hi/lo.
template <typename mp_type>
TEST_HOST_DEVICE_FUNC void vol_ok()
{
  const double v1  = 3.141592653589793;
  const double v2  = 2.718281828459045;
  const double tol = 1e-6;

  // Construct from volatile.
  {
    volatile mp_type vol;
    const mp_type tmp(v1);
    vol = tmp;
    mp_type non_vol(vol);
    assert(::cuda::std::fabs((double) non_vol - v1) < tol);
  }

  // Assign to volatile, read back via construct-from-volatile.
  {
    mp_type src(v1);
    volatile mp_type vol;
    vol = src;
    mp_type readback(vol);
    assert(::cuda::std::fabs((double) readback - v1) < tol);
  }

  // Assign from volatile.
  {
    volatile mp_type vol;
    const mp_type tmp(v2);
    vol = tmp;
    mp_type dst;
    dst = vol;
    assert(::cuda::std::fabs((double) dst - v2) < tol);
  }

  // Volatile round-trip preserves hi/lo exactly.
  {
    mp_type src(v1);
    volatile mp_type vol;
    vol = src;
    mp_type dst(vol);
    assert((src.hi() == dst.hi()) && (src.lo() == dst.lo()));
  }

  // Assign one volatile object to another, e.g. a shared-memory to shared-memory copy.
  {
    const mp_type src(v1);
    volatile mp_type src_vol;
    volatile mp_type dst_vol;
    src_vol = src;
    dst_vol = src_vol;
    mp_type dst(dst_vol);
    assert((src.hi() == dst.hi()) && (src.lo() == dst.lo()));
  }

  // Read the limbs straight off a volatile object, without copying it out first.
  {
    const mp_type src(v2);
    volatile mp_type vol;
    vol = src;
    assert((vol.hi() == src.hi()) && (vol.lo() == src.lo()));
  }
}

TEST_HOST_DEVICE_FUNC void run_test()
{
  vol_ok<cudax::fp32mp2>();
  vol_ok<cudax::fp32mp2_low>();
  vol_ok<cudax::fp32mp2_high>();
  vol_ok<cudax::fp64mp2>();
  vol_ok<cudax::fp64mp2_low>();
  vol_ok<cudax::fp64mp2_high>();
}

TEST_HOST_DEVICE_FUNC void test()
{
  run_test();
}

int main(int, char**)
{
  test();

  return 0;
}
