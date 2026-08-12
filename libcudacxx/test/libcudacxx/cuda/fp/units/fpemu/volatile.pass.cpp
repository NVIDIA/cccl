// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu volatile constructors / assignment + trivial copyability.
//
//  Verifies that the packed (fp64emu*) and unpacked (fp64emu_unpacked*) types are
//  trivially copyable (required for cooperative_groups, __shfl, etc.) and that
//  they correctly support construction from volatile, assignment to volatile, and
//  assignment from volatile, preserving values through volatile round-trips.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Compile-time: every accuracy variant, packed and unpacked, is trivially copyable.
static_assert(cuda::std::is_trivially_copyable_v<cudax::fp64emu>);
static_assert(cuda::std::is_trivially_copyable_v<cudax::fp64emu_low>);
static_assert(cuda::std::is_trivially_copyable_v<cudax::fp64emu_high>);
static_assert(cuda::std::is_trivially_copyable_v<cudax::fp64emu_unpacked>);
static_assert(cuda::std::is_trivially_copyable_v<cudax::fp64emu_unpacked_low>);
static_assert(cuda::std::is_trivially_copyable_v<cudax::fp64emu_unpacked_high>);

// Exercise the four volatile paths for one emulated type; values are exact double
// bit patterns so the round-trips must be exactly preserved.
template <class emu_type>
TEST_HOST_DEVICE_FUNC void test()
{
  const double v1 = 3.141592653589793;
  const double v2 = 2.718281828459045;

  // Construct from volatile.
  {
    volatile emu_type vol;
    const emu_type tmp(v1);
    vol = tmp;
    emu_type non_vol(vol);
    assert((double) non_vol == v1);
  }

  // Assign to volatile, read back via construct-from-volatile.
  {
    emu_type src(v1);
    volatile emu_type vol;
    vol = src;
    emu_type readback(vol);
    assert((double) readback == v1);
  }

  // Assign from volatile.
  {
    volatile emu_type vol;
    const emu_type tmp(v2);
    vol = tmp;
    emu_type dst;
    dst = vol;
    assert((double) dst == v2);
  }

  // Volatile round-trip preserves the value.
  {
    emu_type src(v1);
    volatile emu_type vol;
    vol = src;
    emu_type dst(vol);
    assert((double) src == (double) dst);
  }
}

TEST_HOST_DEVICE_FUNC void test()
{
  test<cudax::fp64emu>();
  test<cudax::fp64emu_low>();
  test<cudax::fp64emu_high>();
  test<cudax::fp64emu_unpacked>();
  test<cudax::fp64emu_unpacked_low>();
  test<cudax::fp64emu_unpacked_high>();
}

int main(int, char**)
{
  test();

  return 0;
}
