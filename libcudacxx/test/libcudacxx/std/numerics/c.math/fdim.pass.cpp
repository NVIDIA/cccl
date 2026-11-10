//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error calling a __host__ __device__ function from a __host__ __device__ __tile__ function is not allowed

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

// <cmath>

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#if _CCCL_HOST_COMPILATION()
#  include <cfenv>
#endif // _CCCL_HOST_COMPILATION()

#include <nv/target>

#include "comparison.h"
#include "test_macros.h"

template <class T>
TEST_HOST_DEVICE_FUNC void test(T val)
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

    // Fast device math flushes float subnormals, including bfloat16 values promoted to float, to zero.
    T expected = cuda::std::numeric_limits<T>::denorm_min();
    if constexpr (TEST_HAS_CUDA_FAST_MATH
                  && (cuda::std::is_same_v<T, float>
#if _LIBCUDACXX_HAS_NVBF16()
                      || cuda::std::is_same_v<T, __nv_bfloat16>
#endif // _LIBCUDACXX_HAS_NVBF16()
                      ) )
    {
      NV_IF_TARGET(NV_IS_DEVICE, (expected = T(0);))
    }
    assert(eq(cuda::std::fdim(cuda::std::numeric_limits<T>::denorm_min(), T()), expected));
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

    T expected = cuda::std::numeric_limits<T>::denorm_min();
    if constexpr (TEST_HAS_CUDA_FAST_MATH)
    {
      NV_IF_TARGET(NV_IS_DEVICE, (expected = T(0);))
    }
    assert(eq(cuda::std::fdimf(cuda::std::numeric_limits<T>::denorm_min(), T()), expected));
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

#if _CCCL_HAS_FLOAT128()
TEST_HOST_DEVICE_FUNC void test_float128_edge_cases()
{
  using T = __float128;
  using L = cuda::std::numeric_limits<T>;

  static_assert(cuda::std::is_same_v<decltype(cuda::std::fdim(T(), T())), T>);

  const T zero = T(0);
  const T nan  = L::quiet_NaN();
  const T inf  = L::infinity();

  // positive, negative, equal and unequal inputs
  assert(eq(cuda::std::fdim(T(5), T(2)), T(3)));
  assert(eq(cuda::std::fdim(T(2), T(5)), zero));
  assert(eq(cuda::std::fdim(T(-2), T(-5)), T(3)));
  assert(eq(cuda::std::fdim(T(-5), T(-2)), zero));
  assert(eq(cuda::std::fdim(T(1), T(-1)), T(2)));
  assert(eq(cuda::std::fdim(T(-1), T(1)), zero));
  assert(eq(cuda::std::fdim(T(3), T(3)), zero));
  assert(!cuda::std::signbit(cuda::std::fdim(T(3), T(3))));
  assert(!cuda::std::signbit(cuda::std::fdim(T(-3), T(-3))));
  assert(!cuda::std::signbit(cuda::std::fdim(T(2), T(5))));

  // the difference carries the full binary128 precision
  assert(eq(cuda::std::fdim(T(1) + L::epsilon(), T(1)), L::epsilon()));
  assert(eq(cuda::std::fdim(T(1), T(1) - L::epsilon() / T(2)), L::epsilon() / T(2)));
  assert(eq(cuda::std::fdim(T(1ull << 62) + T(1), T(1ull << 62)), T(1)));

  // signed zeros: the result is always +0
  const T neg_zero = -zero;
  assert(eq(cuda::std::fdim(zero, neg_zero), zero));
  assert(eq(cuda::std::fdim(neg_zero, zero), zero));
  assert(eq(cuda::std::fdim(neg_zero, neg_zero), zero));
  assert(eq(cuda::std::fdim(zero, zero), zero));
  assert(!cuda::std::signbit(cuda::std::fdim(zero, neg_zero)));
  assert(!cuda::std::signbit(cuda::std::fdim(neg_zero, zero)));
  assert(!cuda::std::signbit(cuda::std::fdim(neg_zero, neg_zero)));
  assert(!cuda::std::signbit(cuda::std::fdim(zero, zero)));
  assert(eq(cuda::std::fdim(T(1), neg_zero), T(1)));
  assert(eq(cuda::std::fdim(neg_zero, T(1)), zero));
  assert(eq(cuda::std::fdim(neg_zero, T(-1)), T(1)));

  // infinities
  assert(eq(cuda::std::fdim(inf, T(1)), inf));
  assert(eq(cuda::std::fdim(T(1), inf), zero));
  assert(eq(cuda::std::fdim(inf, -inf), inf));
  assert(eq(cuda::std::fdim(-inf, inf), zero));
  assert(eq(cuda::std::fdim(T(1), -inf), inf));
  assert(eq(cuda::std::fdim(-inf, T(1)), zero));
  assert(eq(cuda::std::fdim(inf, inf), zero));
  assert(eq(cuda::std::fdim(-inf, -inf), zero));
  assert(!cuda::std::signbit(cuda::std::fdim(inf, inf)));
  assert(!cuda::std::signbit(cuda::std::fdim(-inf, -inf)));
  assert(eq(cuda::std::fdim(L::max(), -inf), inf));
  assert(eq(cuda::std::fdim(inf, L::max()), inf));

  // NaNs propagate from either argument
  assert(cuda::std::isnan(cuda::std::fdim(nan, T(1))));
  assert(cuda::std::isnan(cuda::std::fdim(T(1), nan)));
  assert(cuda::std::isnan(cuda::std::fdim(nan, nan)));
  assert(cuda::std::isnan(cuda::std::fdim(nan, inf)));
  assert(cuda::std::isnan(cuda::std::fdim(inf, nan)));
  assert(cuda::std::isnan(cuda::std::fdim(nan, -inf)));
  assert(cuda::std::isnan(cuda::std::fdim(-inf, nan)));
  assert(cuda::std::isnan(cuda::std::fdim(nan, zero)));
  assert(cuda::std::isnan(cuda::std::fdim(zero, nan)));

  // subnormals
  const T dmin = L::denorm_min();
  assert(eq(cuda::std::fdim(dmin, zero), dmin));
  assert(eq(cuda::std::fdim(zero, dmin), zero));
  assert(eq(cuda::std::fdim(T(2) * dmin, dmin), dmin));
  assert(eq(cuda::std::fdim(dmin, T(2) * dmin), zero));
  assert(eq(cuda::std::fdim(dmin, dmin), zero));
  assert(eq(cuda::std::fdim(-dmin, -T(2) * dmin), dmin));
  assert(eq(cuda::std::fdim(dmin, -dmin), T(2) * dmin));
  // min() - denorm_min() is the largest subnormal and exactly representable
  const T largest_subnormal = cuda::std::fdim(L::min(), dmin);
  assert(largest_subnormal > zero);
  assert(largest_subnormal < L::min());
  assert(eq(largest_subnormal + dmin, L::min()));
  // the difference of two normals can be subnormal
  assert(eq(cuda::std::fdim(L::min() + dmin, L::min()), dmin));

  // large values and overflow
  assert(eq(cuda::std::fdim(L::max(), zero), L::max()));
  assert(eq(cuda::std::fdim(L::max(), L::min()), L::max()));
  assert(eq(cuda::std::fdim(L::max(), T(1)), L::max()));
  assert(eq(cuda::std::fdim(L::max(), -T(1)), L::max()));
  assert(eq(cuda::std::fdim(zero, L::max()), zero));
  assert(eq(cuda::std::fdim(L::max(), L::max()), zero));
  assert(eq(cuda::std::fdim(-L::max(), L::max()), zero));
  assert(eq(cuda::std::fdim(L::lowest(), L::max()), zero));
  // 2 * max overflows to +inf when rounding to nearest
  assert(eq(cuda::std::fdim(L::max(), -L::max()), inf));
}

#  if _CCCL_HOST_COMPILATION()
// The device has no dynamic rounding modes or exception flags, so these only run on the host
__float128 fdim_float128_in_mode(__float128 x, __float128 y, int mode, int* raised = nullptr)
{
  const int old_mode = std::fegetround();
  std::fesetround(mode);
  std::feclearexcept(FE_ALL_EXCEPT);
  // volatile prevents constant folding under the wrong mode
  volatile __float128 vx = x;
  volatile __float128 vy = y;
  volatile __float128 r  = cuda::std::fdim(vx, vy);
  if (raised != nullptr)
  {
    *raised = std::fetestexcept(FE_ALL_EXCEPT);
  }
  const __float128 res = r;
  std::fesetround(old_mode);
  return res;
}

void test_float128_rounding_and_exceptions()
{
  using T = __float128;
  using L = cuda::std::numeric_limits<T>;

  // max - (-max) overflows: +inf when rounding to nearest or up, max when rounding down or towards zero
  assert(eq(fdim_float128_in_mode(L::max(), -L::max(), FE_TONEAREST), L::infinity()));
  assert(eq(fdim_float128_in_mode(L::max(), -L::max(), FE_UPWARD), L::infinity()));
  assert(eq(fdim_float128_in_mode(L::max(), -L::max(), FE_DOWNWARD), L::max()));
  assert(eq(fdim_float128_in_mode(L::max(), -L::max(), FE_TOWARDZERO), L::max()));

  // 1 +- 2^-200 is not representable and rounds depending on the mode
  T tiny = T(1);
  for (int i = 0; i < 200; ++i)
  {
    tiny /= T(2);
  }
  const T below_one = T(1) - L::epsilon() / T(2); // the predecessor of 1
  const T above_one = T(1) + L::epsilon(); // the successor of 1
  assert(eq(fdim_float128_in_mode(T(1), tiny, FE_TONEAREST), T(1)));
  assert(eq(fdim_float128_in_mode(T(1), tiny, FE_UPWARD), T(1)));
  assert(eq(fdim_float128_in_mode(T(1), tiny, FE_DOWNWARD), below_one));
  assert(eq(fdim_float128_in_mode(T(1), tiny, FE_TOWARDZERO), below_one));
  assert(eq(fdim_float128_in_mode(T(1), -tiny, FE_TONEAREST), T(1)));
  assert(eq(fdim_float128_in_mode(T(1), -tiny, FE_UPWARD), above_one));
  assert(eq(fdim_float128_in_mode(T(1), -tiny, FE_DOWNWARD), T(1)));
  assert(eq(fdim_float128_in_mode(T(1), -tiny, FE_TOWARDZERO), T(1)));

  // exact results do not depend on the rounding mode and raise no exceptions
  const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_TOWARDZERO, FE_UPWARD};
  for (const int mode : modes)
  {
    int raised = 0;
    assert(eq(fdim_float128_in_mode(T(5), T(2), mode, &raised), T(3)));
    assert(raised == 0);
    assert(eq(fdim_float128_in_mode(T(2), T(5), mode, &raised), T(0)));
    assert(raised == 0);
    assert(eq(fdim_float128_in_mode(L::infinity(), T(1), mode, &raised), L::infinity()));
    assert(raised == 0);
    assert(cuda::std::isnan(fdim_float128_in_mode(L::quiet_NaN(), T(1), mode, &raised)));
    assert((raised & FE_INVALID) == 0);
  }

  int raised = 0;
  fdim_float128_in_mode(L::max(), -L::max(), FE_TONEAREST, &raised);
  assert((raised & FE_OVERFLOW) != 0);
  fdim_float128_in_mode(T(1), tiny, FE_TONEAREST, &raised);
  assert((raised & FE_INEXACT) != 0);
}
#  endif // _CCCL_HOST_COMPILATION()

TEST_HOST_DEVICE_FUNC void test_float128()
{
  test_float128_edge_cases();
  NV_IF_TARGET(NV_IS_HOST, (test_float128_rounding_and_exceptions();))
}
#endif // _CCCL_HAS_FLOAT128()

TEST_HOST_DEVICE_FUNC bool test(float val)
{
  test<float>(val);
  test<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>(val);
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _CCCL_HAS_FLOAT128()
  test_float128();
#endif // _CCCL_HAS_FLOAT128()

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
