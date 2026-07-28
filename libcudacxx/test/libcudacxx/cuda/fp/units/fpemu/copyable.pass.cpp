// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu / fp64emu_unpacked trivial copyability + volatile round-trip.
//
//  Compile-time static_asserts check that both the packed (fp64emu) and unpacked
//  (fp64emu_unpacked) types are trivially copyable; run_test() confirms a value
//  survives a round-trip through a volatile object for each.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

static_assert(cuda::std::is_trivially_copyable_v<fp64emu>, "fp64emu must be trivially copyable");
static_assert(cuda::std::is_trivially_copyable_v<fp64emu_unpacked>, "fp64emu_unpacked must be trivially copyable");

// Round-trip both the packed and unpacked types through a volatile object.
TEST_FUNC void test()
{
  // Packed type (fp64emu).
  {
    volatile fp64emu vx[1];
    fp64emu x[1] = {fp64emu(1.0e+20)};
    vx[0]        = x[0];
    fp64emu readback(vx[0]); // template volatile copy constructor
    assert(!(readback != x[0]));
  }

  // Unpacked type (fp64emu_unpacked).
  {
    volatile fp64emu_unpacked vx[1];
    fp64emu_unpacked x[1] = {fp64emu_unpacked(1.0e+20)};
    vx[0]                 = x[0];
    fp64emu_unpacked readback(vx[0]);
    assert(!(readback != x[0]));
  }
}

int main(int, char**)
{
  test();

  return 0;
}
