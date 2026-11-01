//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cmath>

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#if _CCCL_HOST_COMPILATION()
#  include <cfenv>
#endif // _CCCL_HOST_COMPILATION()

#include <nv/target>

#include "comparison.h"
#include "fp_compare.h"
#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4244) // conversion from 'const double' to 'int', possible loss of data
TEST_DIAG_SUPPRESS_MSVC(4146) // unary minus operator applied to unsigned type, result still unsigned

template <typename T>
TEST_FUNC void test_sqrt(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::sqrt(T{})), ret>);

  assert(eq(cuda::std::sqrt(val), T(8.0)));
  assert(eq(cuda::std::sqrt(T(0.0)), T(0.0)));
  assert(eq(cuda::std::sqrt(T(cuda::std::numeric_limits<T>::infinity())), cuda::std::numeric_limits<T>::infinity()));
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::sqrtf(val), T(8.0)));
    assert(eq(cuda::std::sqrtf(T(0.0)), T(0.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::sqrtl(val), T(8)));
    assert(eq(cuda::std::sqrtl(T(0.0)), T(0.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
TEST_FUNC void test_cbrt(T val)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::cbrt(T{})), ret>);

  assert(eq(cuda::std::cbrt(val), T(2)));
  assert(eq(cuda::std::cbrt(T(0.0)), T(0.0)));
  assert(eq(cuda::std::cbrt(-T(0.0)), -T(0.0)));
  assert(eq(cuda::std::cbrt(T(cuda::std::numeric_limits<T>::infinity())), cuda::std::numeric_limits<T>::infinity()));
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    assert(eq(cuda::std::cbrtf(val), T(2)));
    assert(eq(cuda::std::cbrtf(T(0.0)), T(0.0)));
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    assert(eq(cuda::std::cbrtl(val), T(2)));
    assert(eq(cuda::std::cbrtl(T(0.0)), T(0.0)));
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <typename T>
TEST_FUNC void test(const T val)
{
  test_sqrt<T>(val);
  test_cbrt<T>(val / T(8));
}

#if _CCCL_HAS_FLOAT128()
TEST_FUNC void test_sqrt_float128_edge_cases()
{
  using T = __float128;

  // squares of integers with more significant bits than double can hold are exact in binary128
  const T a = T(1ull << 55) + T(1);
  assert(eq(cuda::std::sqrt(a * a), a));
  const T b = T((1ull << 56) - 1);
  assert(eq(cuda::std::sqrt(b * b), b));

  // the result carries the full binary128 precision
  const T sqrt2 = cuda::std::sqrt(T(2));
  assert(cuda::std::fabs(sqrt2 * sqrt2 - T(2)) <= T(4) * cuda::std::numeric_limits<T>::epsilon());

  // exponents beyond the range of double
  T big      = T(1);
  T big_root = T(1);
  for (int i = 0; i < 10; ++i)
  {
    big *= T(1ull << 60) * T(1ull << 60);
    big_root *= T(1ull << 60);
  }
  assert(eq(cuda::std::sqrt(big), big_root)); // sqrt(2^1200) == 2^600
  assert(eq(cuda::std::sqrt(T(1) / big), T(1) / big_root));
  assert(cuda::std::isfinite(cuda::std::sqrt(cuda::std::numeric_limits<T>::max())));

  // sqrt(2^-16494) == 2^-8247
  T denorm_root = T(1);
  for (int i = 0; i < 137; ++i)
  {
    denorm_root /= T(1ull << 60);
  }
  denorm_root /= T(1ull << 27);
  assert(eq(cuda::std::sqrt(cuda::std::numeric_limits<T>::denorm_min()), denorm_root));

  // sqrt(-0) is -0
  assert(eq(cuda::std::sqrt(-T(0.0)), -T(0.0)));
  assert(cuda::std::signbit(cuda::std::sqrt(-T(0.0))));

  // negative arguments and NaN produce NaN
  assert(cuda::std::isnan(cuda::std::sqrt(T(-1))));
  assert(cuda::std::isnan(cuda::std::sqrt(-cuda::std::numeric_limits<T>::infinity())));
  assert(cuda::std::isnan(cuda::std::sqrt(cuda::std::numeric_limits<T>::quiet_NaN())));
}

#  if _CCCL_HOST_COMPILATION()
// The device has no dynamic rounding modes, so the rounding mode tests only run on the host
__float128 sqrt_float128_in_mode(__float128 x, int mode)
{
  const int old_mode = std::fegetround();
  std::fesetround(mode);
  // volatile forces sqrt to execute between the fesetround calls and prevents constant folding under the wrong mode
  volatile __float128 vx = x;
  volatile __float128 r  = cuda::std::sqrt(vx);
  const __float128 res   = r;
  std::fesetround(old_mode);
  return res;
}

// sqrt(x) lies strictly between the representable values with storage lower and lower + 1
void test_sqrt_float128_rounding_mode(__float128 x, __uint128_t lower, bool nearest_rounds_up)
{
  assert(cuda::std::bit_cast<__uint128_t>(sqrt_float128_in_mode(x, FE_DOWNWARD)) == lower);
  assert(cuda::std::bit_cast<__uint128_t>(sqrt_float128_in_mode(x, FE_TOWARDZERO)) == lower);
  assert(cuda::std::bit_cast<__uint128_t>(sqrt_float128_in_mode(x, FE_UPWARD)) == lower + 1);
  assert(cuda::std::bit_cast<__uint128_t>(sqrt_float128_in_mode(x, FE_TONEAREST))
         == lower + (nearest_rounds_up ? 1 : 0));
}

void test_sqrt_float128_rounding_modes()
{
  // sqrt(2) rounds down and sqrt(5) rounds up to nearest
  test_sqrt_float128_rounding_mode(2, (__uint128_t{0x3fff6a09e667f3bc} << 64) | 0xc908b2fb1366ea95, false);
  test_sqrt_float128_rounding_mode(5, (__uint128_t{0x40001e3779b97f4a} << 64) | 0x7c15f39cc0605ced, true);

  // exact roots do not depend on the rounding mode
  const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_TOWARDZERO, FE_UPWARD};
  for (const int mode : modes)
  {
    assert(eq(sqrt_float128_in_mode(4, mode), __float128(2)));
  }
}
#  endif // _CCCL_HOST_COMPILATION()

// Only sqrt supports __float128 so far, so the generic driver can not be used
TEST_FUNC void test_float128(const float val)
{
  test_sqrt<__float128>(val);
  test_sqrt_float128_edge_cases();
  NV_IF_TARGET(NV_IS_HOST, (test_sqrt_float128_rounding_modes();))
}
#endif // _CCCL_HAS_FLOAT128()

TEST_FUNC void test(const float val)
{
  test<float>(val);
  test<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>(val);
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>(val);
#endif // _LIBCUDACXX_HAS_NVBF16()
#if _CCCL_HAS_FLOAT128()
  test_float128(val);
#endif // _CCCL_HAS_FLOAT128()

  test<unsigned short>(static_cast<unsigned short>(val));
  test<int>(static_cast<int>(val));
  test<unsigned int>(static_cast<unsigned int>(val));
  test<long>(static_cast<long>(val));
  test<unsigned long>(static_cast<unsigned long>(val));
  test<long long>(static_cast<long long>(val));
  test<unsigned long long>(static_cast<unsigned long long>(val));
}

__global__ void test_global_kernel(float* val)
{
  test(*val);
}

int main(int, char**)
{
  volatile float val = 64.f;
  test(val);
  return 0;
}
