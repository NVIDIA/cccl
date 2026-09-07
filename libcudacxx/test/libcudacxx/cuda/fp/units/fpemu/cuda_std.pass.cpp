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
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

TEST_HOST_DEVICE_FUNC void test()
{
  constexpr double kTol = 1e-10;

  // ---- packed fpemu<double> --------------------------------------------------
  {
    const cudax::fpemu<double> a(2.0), b(3.0), c(1.0);

    // Return type must be the emulated type, proving cuda::std::fma did not fall
    // back to the double overload via the implicit fpemu -> double conversion.
    static_assert(cuda::std::is_same_v<decltype(cuda::std::fma(a, b, c)), cudax::fpemu<double>>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::sqrt(a)), cudax::fpemu<double>>);

    const cudax::fpemu<double> r = cuda::std::fma(a, b, c); // 2*3 + 1
    assert(cuda::std::fabs(static_cast<double>(r) - 7.0) <= kTol);

    const cudax::fpemu<double> s = cuda::std::sqrt(cudax::fpemu<double>(4.0));
    assert(cuda::std::fabs(static_cast<double>(s) - 2.0) <= kTol);

    // Mixed fpemu + built-in arithmetic operands.
    static_assert(cuda::std::is_same_v<decltype(cuda::std::fma(a, 3.0, c)), cudax::fpemu<double>>);
    const cudax::fpemu<double> rm = cuda::std::fma(a, 3.0, c); // 2*3 + 1
    assert(cuda::std::fabs(static_cast<double>(rm) - 7.0) <= kTol);
  }

  // ---- unpacked fpemu_unpacked<double> --------------------------------------
  {
    const cudax::fpemu_unpacked<double> a(2.0), b(3.0), c(1.0);

    static_assert(cuda::std::is_same_v<decltype(cuda::std::fma(a, b, c)), cudax::fpemu_unpacked<double>>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::sqrt(a)), cudax::fpemu_unpacked<double>>);

    const cudax::fpemu_unpacked<double> r = cuda::std::fma(a, b, c); // 2*3 + 1
    assert(cuda::std::fabs(static_cast<double>(r) - 7.0) <= kTol);

    const cudax::fpemu_unpacked<double> s = cuda::std::sqrt(cudax::fpemu_unpacked<double>(9.0));
    assert(cuda::std::fabs(static_cast<double>(s) - 3.0) <= kTol);

    static_assert(cuda::std::is_same_v<decltype(cuda::std::fma(1.0, b, 2.0)), cudax::fpemu_unpacked<double>>);
    const cudax::fpemu_unpacked<double> rm = cuda::std::fma(1.0, b, 2.0); // 1*3 + 2
    assert(cuda::std::fabs(static_cast<double>(rm) - 5.0) <= kTol);
  }

  // ---- non-def accuracy levels still route through the emulated overloads ----
  {
    const cudax::fpemu<double, cudax::fpemu_accuracy::high> a(2.0), b(3.0), c(1.0);
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::fma(a, b, c)), cudax::fpemu<double, cudax::fpemu_accuracy::high>>);
    const cudax::fpemu<double, cudax::fpemu_accuracy::high> r = cuda::std::fma(a, b, c);
    assert(cuda::std::fabs(static_cast<double>(r) - 7.0) <= kTol);
  }
}

int main(int, char**)
{
  test();

  return 0;
}
