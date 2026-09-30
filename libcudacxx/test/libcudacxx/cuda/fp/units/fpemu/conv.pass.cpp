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

using P_hi = cudax::fpemu<double, cudax::fpemu_accuracy::high>;
using P_lo = cudax::fpemu<double, cudax::fpemu_accuracy::low>;
using U_hi = cudax::fpemu_unpacked<double, cudax::fpemu_accuracy::high>;
using U_lo = cudax::fpemu_unpacked<double, cudax::fpemu_accuracy::low>;

// All conversions are explicit: not implicitly convertible ...
static_assert(!cuda::std::is_convertible_v<P_hi, P_lo>, "accuracy conversion must be explicit");
static_assert(!cuda::std::is_convertible_v<U_hi, U_lo>, "accuracy conversion must be explicit");
static_assert(!cuda::std::is_convertible_v<P_hi, U_hi>, "packed -> unpacked must be explicit");
static_assert(!cuda::std::is_convertible_v<U_hi, P_hi>, "unpacked -> packed must be explicit");
// ... but explicitly constructible.
static_assert(cuda::std::is_constructible_v<P_lo, P_hi>);
static_assert(cuda::std::is_constructible_v<U_lo, U_hi>);
static_assert(cuda::std::is_constructible_v<U_hi, P_hi>);
static_assert(cuda::std::is_constructible_v<P_hi, U_hi>);
// The added converting ctors must not break trivial copyability.
static_assert(cuda::std::is_trivially_copyable_v<P_hi>);
static_assert(cuda::std::is_trivially_copyable_v<U_hi>);

TEST_HOST_DEVICE_FUNC void test()
{
  constexpr double kTol = 1e-10;

  const double vals[6] = {0.0, 1.5, -2.0, 42.0, 1234.5678, -9.999e12};
  for (const double d : vals)
  {
    const P_hi p(d);
    const P_lo p2(p); // packed accuracy converting ctor
    const U_hi u(p); // packed -> unpacked ctor
    const P_hi back(u); // unpacked -> packed ctor
    const U_lo u2(u); // unpacked accuracy converting ctor

    assert(cuda::std::fabs(static_cast<double>(p2) - d) <= kTol);
    assert(cuda::std::fabs(static_cast<double>(u) - d) <= kTol);
    assert(cuda::std::fabs(static_cast<double>(back) - d) <= kTol);
    assert(cuda::std::fabs(static_cast<double>(u2) - d) <= kTol);
  }
}

int main(int, char**)
{
  test();

  return 0;
}
