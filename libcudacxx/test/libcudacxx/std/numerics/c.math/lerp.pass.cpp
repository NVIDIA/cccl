//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cmath>

// constexpr float lerp(float a, float b, float t) noexcept;
// constexpr double lerp(double a, double b, double t) noexcept;
// constexpr long double lerp(long double a, long double b, long double t) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "fp_compare.h"
#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr bool constexpr_test()
{
  return cuda::std::lerp(T(0.0), T(12), T(0.0)) == T(0.0) && cuda::std::lerp(T(12), T(0.0), T(0.5)) == T(6)
      && cuda::std::lerp(T(0.0), T(12), T(2)) == T(24);
}

template <typename T>
__host__ __device__ void test()
{
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::lerp(T(), T(), T()))>);
  static_assert(noexcept(cuda::std::lerp(T(), T(), T())), "");

  const T maxV = cuda::std::numeric_limits<T>::max();
  const T inf  = cuda::std::numeric_limits<T>::infinity();

  //  Things that can be compared exactly
  assert((cuda::std::lerp(T(0.0), T(12), T(0.0)) == T(0.0)));
  assert((cuda::std::lerp(T(0.0), T(12), T(1)) == T(12)));
  assert((cuda::std::lerp(T(12), T(0.0), T(0.0)) == T(12)));
  assert((cuda::std::lerp(T(12), T(0.0), T(1)) == T(0.0)));

  assert((cuda::std::lerp(T(0.0), T(12), T(0.5)) == T(6)));
  assert((cuda::std::lerp(T(12), T(0.0), T(0.5)) == T(6)));
  assert((cuda::std::lerp(T(0.0), T(12), T(2)) == T(24)));
  assert((cuda::std::lerp(T(12), T(0.0), T(2)) == T(-12)));

  assert((cuda::std::lerp(maxV, maxV / T(10), T(0.0)) == maxV));
  assert((cuda::std::lerp(maxV / T(10), maxV, T(1)) == maxV));

  assert((cuda::std::lerp(T(2.3), T(2.3), inf) == T(2.3)));

  assert(cuda::std::lerp(T(0.0), T(0.0), T(23)) == T(0.0));

  // __half and __nvbfloat have precision issues here
  if constexpr (!cuda::std::__is_extended_floating_point_v<T>)
  {
    assert(cuda::std::isnan(cuda::std::lerp(T(0.0), T(0.0), T(inf))));
  }
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  static_assert(constexpr_test<float>(), "");
  static_assert(constexpr_test<double>(), "");

  return 0;
}
