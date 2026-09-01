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

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

// nvcc doesn't currently support _Float64 in device code.
// UNSUPPORTED: nvcc

#include <cuda/fpemu>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

TEST_HOST_DEVICE_FUNC void test()
{
#if __STDCPP_FLOAT64_T__ == 1
  // _Float64 is a distinct type here, yet fpemu<_Float64> must still be a valid,
  // trivially copyable emulated double that constructs from / converts to double.
  static_assert(!cuda::std::is_same_v<double, _Float64>, "expected _Float64 to be a distinct type in this mode");
  static_assert(cuda::std::is_trivially_copyable_v<cudax::fpemu<_Float64>>);
  static_assert(cuda::std::is_trivially_copyable_v<cudax::fpemu_unpacked<_Float64>>);
  static_assert(sizeof(cudax::fpemu<_Float64>) == sizeof(cudax::fpemu<double>));
  static_assert(cuda::std::is_constructible_v<cudax::fpemu<_Float64>, double>);
  static_assert(cuda::std::is_constructible_v<cudax::fpemu<_Float64>, int>);

  const double vals[] = {0.0, 1.5, -3.25, 1234.5678, -9.999e12};
  for (double d : vals)
  {
    cudax::fpemu<_Float64> a(d);
    cudax::fpemu<double> b(d);
    // Same value in, same 64-bit result out as the double instantiation.
    assert(cuda::std::bit_cast<uint64_t>((double) a) == cuda::std::bit_cast<uint64_t>((double) b));
    assert(cuda::std::bit_cast<uint64_t>((double) a) == cuda::std::bit_cast<uint64_t>(d));
  }
#endif // __STDCPP_FLOAT64_T__ == 1
}

int main(int, char**)
{
  test();

  return 0;
}
