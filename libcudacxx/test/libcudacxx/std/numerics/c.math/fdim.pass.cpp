//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cmath>

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "comparison.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::fdim(val, T())), ret>);

  assert(eq(cuda::std::fdim(val, T()), val));
  assert(eq(cuda::std::fdim(T(5), T(2)), T(3)));

  // negative values arer clamped to 0
  assert(eq(cuda::std::fdim(T(), val), T()));

  if constexpr (cuda::std::is_signed_v<T>)
  {
    assert(eq(cuda::std::fdim(-T(2), -T(5)), T(3)));
    assert(eq(cuda::std::fdim(-T(5), -T(2)), T(0)));
  }

  if constexpr (!cuda::std::is_integral_v<T>)
  {
    // If a range error due to overflow occurs, +HUGE_VAL, +HUGE_VALF, or +HUGE_VALL is returned.
    assert(eq(cuda::std::fdim(cuda::std::numeric_limits<T>::max(), cuda::std::numeric_limits<T>::min()),
              cuda::std::numeric_limits<T>::max()));

    // If a range error due to underflow occurs, the correct value (after rounding) is returned.
    assert(
      eq(cuda::std::fdim(cuda::std::numeric_limits<T>::denorm_min(), T()), cuda::std::numeric_limits<T>::denorm_min()));
  }

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::fdimf(val, T()), val));
    assert(eq(cuda::std::fdimf(T(5), T(2)), T(3)));

    // negative values arer clamped to 0
    assert(eq(cuda::std::fdimf(T(), val), T()));

    assert(eq(cuda::std::fdimf(-T(2), -T(5)), T(3)));
    assert(eq(cuda::std::fdimf(-T(5), -T(2)), T(0)));

    // If a range error due to overflow occurs, +HUGE_VAL, +HUGE_VALF, or +HUGE_VALL is returned.
    assert(eq(cuda::std::fdimf(cuda::std::numeric_limits<T>::max(), cuda::std::numeric_limits<T>::min()),
              cuda::std::numeric_limits<T>::max()));

    // If a range error due to underflow occurs, the correct value (after rounding) is returned.
    assert(eq(cuda::std::fdimf(cuda::std::numeric_limits<T>::denorm_min(), T()),
              cuda::std::numeric_limits<T>::denorm_min()));
  }

#if _CCCL_HAS_LONG_DOUBLE()
  if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::fdiml(val, T()), val));
    assert(eq(cuda::std::fdiml(T(5), T(2)), T(3)));

    // negative values arer clamped to 0
    assert(eq(cuda::std::fdiml(T(), val), T()));

    assert(eq(cuda::std::fdiml(-T(2), -T(5)), T(3)));
    assert(eq(cuda::std::fdiml(-T(5), -T(2)), T(0)));

    // If a range error due to overflow occurs, +HUGE_VAL, +HUGE_VALF, or +HUGE_VALL is returned.
    assert(eq(cuda::std::fdiml(cuda::std::numeric_limits<T>::max(), cuda::std::numeric_limits<T>::min()),
              cuda::std::numeric_limits<T>::max()));

    // If a range error due to underflow occurs, the correct value (after rounding) is returned.
    assert(eq(cuda::std::fdiml(cuda::std::numeric_limits<T>::denorm_min(), T()),
              cuda::std::numeric_limits<T>::denorm_min()));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

__host__ __device__ bool test(float val)
{
  test<float>(val);
  test<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>(val);
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>(::__float2half(val));
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__nv_bfloat16>(::__float2bfloat16(val));
#endif // _LIBCUDACXX_HAS_NVFP16()

  test<signed char>(static_cast<signed char>(val));
  test<unsigned char>(static_cast<unsigned char>(val));
  test<signed short>(static_cast<signed short>(val));
  test<unsigned short>(static_cast<unsigned short>(val));
  test<signed int>(static_cast<signed int>(val));
  test<unsigned int>(static_cast<unsigned int>(val));
  test<signed long>(static_cast<signed long>(val));
  test<unsigned long>(static_cast<unsigned long>(val));
  test<signed long long>(static_cast<signed long long>(val));
  test<unsigned long long>(static_cast<unsigned long long>(val));
  // clang-cuda fails to convert from `float` to `__int128` on device
#if _CCCL_HAS_INT128() && !TEST_CUDA_COMPILER(CLANG)
  test<__int128_t>(static_cast<__int128_t>(static_cast<int>(val)));
  test<__uint128_t>(static_cast<__uint128_t>(static_cast<int>(val)));
#endif // _CCCL_HAS_INT128() && !TEST_CUDA_COMPILER(CLANG)

  return true;
}

__global__ void test_global_kernel(float* val)
{
  test(*val);
}

int main(int, char**)
{
  volatile float val = 1.0f;
  test(val);
  return 0;
}
