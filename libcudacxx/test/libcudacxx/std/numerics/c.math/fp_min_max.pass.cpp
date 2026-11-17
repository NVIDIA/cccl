//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cmath>

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <nv/target>

#include "test_macros.h"

template <class T, class U>
TEST_FUNC constexpr bool eq(T lhs, U rhs)
{
  return !cuda::std::isnan(lhs) && !cuda::std::isnan(rhs) && !cuda::std::islessgreater(lhs, static_cast<T>(rhs));
}

template <class T, class U>
TEST_FUNC constexpr auto invoke_fmax(T x, U y)
{
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    DoNotOptimize(x);
    DoNotOptimize(y);
  }
  return cuda::std::fmax(x, y);
}

template <class T, class U>
TEST_FUNC constexpr auto invoke_fmin(T x, U y)
{
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    DoNotOptimize(x);
    DoNotOptimize(y);
  }
  return cuda::std::fmin(x, y);
}

template <class T>
TEST_FUNC constexpr void test_fmax(T val)
{
  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(invoke_fmax(val, T()), val));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::fmax(val, T())), double>);
  }
  else
  {
    using Limits = cuda::std::numeric_limits<T>;

    assert(eq(invoke_fmax(val, Limits::lowest()), val));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::fmax(val, Limits::lowest())), T>);

    // Mixed arguments require a type supported by arithmetic promotion.
    if constexpr (!cuda::std::__is_ext_nv_fp_v<T>
                  || cuda::std::__fp_format_of_v<T> == cuda::std::__fp_format::__binary16
                  || cuda::std::__fp_format_of_v<T> == cuda::std::__fp_format::__bfloat16)
    {
      using promote_float = cuda::std::__promote_t<float, T>;
      assert(invoke_fmax(val, 0.0f) == promote_float(1));
      assert(invoke_fmax(0.0f, val) == promote_float(1));

      using promote_double = cuda::std::__promote_t<double, T>;
      assert(invoke_fmax(val, 0.0) == promote_double(1));
      assert(invoke_fmax(0.0, val) == promote_double(1));
    }

    // We always return the other value when passing a NaN
    if constexpr (Limits::has_quiet_NaN)
    {
      assert(eq(invoke_fmax(val, Limits::quiet_NaN()), val));
      assert(eq(invoke_fmax(Limits::quiet_NaN(), val), val));
    }
    if constexpr (Limits::has_signaling_NaN)
    {
      assert(eq(invoke_fmax(val, Limits::signaling_NaN()), val));
      assert(eq(invoke_fmax(Limits::signaling_NaN(), val), val));
    }

    if constexpr (Limits::has_infinity)
    {
      // check infinities
      assert(eq(invoke_fmax(val, Limits::infinity()), Limits::infinity()));
      assert(eq(invoke_fmax(Limits::infinity(), val), Limits::infinity()));
      if constexpr (Limits::is_signed)
      {
        T negative_inf = Limits::infinity();
        if constexpr (cuda::std::__is_ext_nv_fp_v<T>)
        {
          negative_inf = cuda::std::copysign(negative_inf, Limits::lowest());
        }
        else
        {
          negative_inf = -negative_inf;
        }
        assert(eq(invoke_fmax(val, negative_inf), val));
        assert(eq(invoke_fmax(negative_inf, val), val));
      }
    }
  }
}

template <class T>
TEST_FUNC constexpr void test_fmin(T val)
{
  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(cuda::std::fmin(val, T()) == T());
    static_assert(cuda::std::is_same_v<decltype(cuda::std::fmin(val, T())), double>);
  }
  else
  {
    using Limits = cuda::std::numeric_limits<T>;

    assert(eq(invoke_fmin(val, Limits::lowest()), Limits::lowest()));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::fmin(val, Limits::lowest())), T>);

    // Mixed arguments require a type supported by arithmetic promotion.
    if constexpr (!cuda::std::__is_ext_nv_fp_v<T>
                  || cuda::std::__fp_format_of_v<T> == cuda::std::__fp_format::__binary16
                  || cuda::std::__fp_format_of_v<T> == cuda::std::__fp_format::__bfloat16)
    {
      using promote_float = cuda::std::__promote_t<float, T>;
      assert(invoke_fmin(val, 2.0f) == promote_float(1));
      assert(invoke_fmin(2.0f, val) == promote_float(1));

      using promote_double = cuda::std::__promote_t<double, T>;
      assert(invoke_fmin(val, 2.0) == promote_double(1));
      assert(invoke_fmin(2.0, val) == promote_double(1));
    }

    // We always return the other value when passing a NaN
    if constexpr (cuda::std::__fp_has_nan_v<cuda::std::__fp_format_of_v<T>>)
    {
      assert(eq(invoke_fmin(val, Limits::quiet_NaN()), val));
      assert(eq(invoke_fmin(Limits::quiet_NaN(), val), val));
    }
    if constexpr (Limits::has_signaling_NaN)
    {
      assert(eq(invoke_fmin(val, Limits::signaling_NaN()), val));
      assert(eq(invoke_fmin(Limits::signaling_NaN(), val), val));
    }

    if constexpr (Limits::has_infinity)
    {
      // check infinities
      assert(eq(invoke_fmin(val, Limits::infinity()), val));
      assert(eq(invoke_fmin(Limits::infinity(), val), val));

      if constexpr (cuda::std::__fp_is_signed_v<cuda::std::__fp_format_of_v<T>>)
      {
        T negative_inf = Limits::infinity();
        if constexpr (cuda::std::__is_ext_nv_fp_v<T>)
        {
          negative_inf = cuda::std::copysign(negative_inf, Limits::lowest());
        }
        else
        {
          negative_inf = -negative_inf;
        }
        assert(eq(invoke_fmin(val, negative_inf), negative_inf));
        assert(eq(invoke_fmin(negative_inf, val), negative_inf));
      }
    }
  }
}

TEST_FUNC void test_fp64_min_max(double x, double y, double max, double min)
{
  assert(eq(invoke_fmax(x, y), max));
  assert(eq(invoke_fmin(x, y), min));
}

// double has an integer implementation on slow fp64 targets
// check the cases where the integer mapping can go wrong
TEST_FUNC void test_fp64_edge_cases(float val)
{
  // values are derived from val at runtime
  double one   = val;
  double two   = one + one;
  double zero  = one - one;
  double nzero = -zero;
  double inf   = cuda::std::numeric_limits<double>::infinity() * one;
  double max   = cuda::std::numeric_limits<double>::max() * one;
  double dmin  = cuda::std::numeric_limits<double>::denorm_min() * one;
  double nan   = cuda::std::numeric_limits<double>::quiet_NaN();
  double nnan  = cuda::std::copysign(nan, -one);
  // nvcc quiets signaling NaN constants in device code, so the bits are built at runtime
  double snan =
    cuda::std::bit_cast<double>(cuda::std::uint64_t{0x7ff4'0000'0000'0000} * static_cast<cuda::std::uint64_t>(val));

  //                x      y      fmax   fmin
  // signed zeros compare equal
  // only the value is checked
  test_fp64_min_max(zero, nzero, zero, zero);
  test_fp64_min_max(nzero, zero, zero, zero);
  assert(cuda::std::signbit(cuda::std::fmax(nzero, nzero)));
  assert(cuda::std::signbit(cuda::std::fmin(nzero, nzero)));

  // negative values are ordered in reverse to their bit patterns
  test_fp64_min_max(-one, -two, -one, -two);
  test_fp64_min_max(-two, -one, -one, -two);
  test_fp64_min_max(-one, one, one, -one);
  test_fp64_min_max(-one, -one, -one, -one);

  // infinities and limits
  test_fp64_min_max(-inf, -max, -max, -inf);
  test_fp64_min_max(max, inf, inf, max);

  // denormals
  test_fp64_min_max(dmin, zero, dmin, zero);
  test_fp64_min_max(-dmin, nzero, nzero, -dmin);
  test_fp64_min_max(-dmin, dmin, dmin, -dmin);

  // negative NaNs map below -inf, but the other value must still be returned
  test_fp64_min_max(nnan, -one, -one, -one);
  test_fp64_min_max(-one, nnan, -one, -one);
  test_fp64_min_max(nnan, -inf, -inf, -inf);
  test_fp64_min_max(snan, one, one, one);
  test_fp64_min_max(one, snan, one, one);

  // two NaNs return a NaN, the payload and whether a signaling NaN is quieted are unspecified
  assert(cuda::std::isnan(invoke_fmax(nan, nan)));
  assert(cuda::std::isnan(invoke_fmin(nan, nan)));
  assert(cuda::std::isnan(invoke_fmax(nnan, nan)));
  assert(cuda::std::isnan(invoke_fmin(nnan, nan)));
  assert(cuda::std::isnan(invoke_fmax(snan, snan)));
  assert(cuda::std::isnan(invoke_fmin(snan, snan)));
  assert(cuda::std::isnan(invoke_fmax(nan, snan)));
  assert(cuda::std::isnan(invoke_fmin(snan, nan)));

  // mixed arguments are promoted to double
  assert(eq(invoke_fmax(-val, -0.5 * one), -0.5));
  assert(eq(invoke_fmin(-val, -0.5 * one), -1.0));

  // on device -0 < +0 for fmax and fmin
  // libcu++ does not use the fmax/fmin builtins under NVRTC, the fallback on fast fp64 targets may return either zero
#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (assert(!cuda::std::signbit(invoke_fmax(zero, nzero))); assert(!cuda::std::signbit(invoke_fmax(nzero, zero)));
     assert(cuda::std::signbit(invoke_fmin(zero, nzero)));
     assert(cuda::std::signbit(invoke_fmin(nzero, zero)));))
#endif // !TEST_COMPILER(NVRTC)
}

template <class T>
TEST_FUNC constexpr bool test(T val)
{
  test_fmax<T>(val);
  test_fmin<T>(val);

  return true;
}

TEST_FUNC bool test(float val)
{
  test<float>(val);
  test<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>(val);
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test<__float128>(val);
#endif // _CCCL_HAS_FLOAT128()

#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>(::__float2half(val));
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16>(::__float2bfloat16(val));
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3>(__nv_fp8_e4m3(val));
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2>(__nv_fp8_e5m2(val));
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_UE5M3()
  test<__nv_fp8_ue5m3>(__nv_fp8_ue5m3(val));
#endif // _CCCL_HAS_NVFP8_UE5M3()
#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0>(__nv_fp8_e8m0(val));
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3>(__nv_fp6_e2m3(val));
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2>(__nv_fp6_e3m2(val));
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1>(__nv_fp4_e2m1(val));
#endif // _CCCL_HAS_NVFP4_E2M1()

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
#if _CCCL_HAS_INT128()
#  if TEST_CUDA_COMPILER(CLANG)
  // Clang's NVPTX backend cannot lower runtime conversions from 128-bit integers to double.
  NV_IF_TARGET(NV_IS_HOST,
               (test<__int128_t>(static_cast<__int128_t>(static_cast<int>(val)));
                test<__uint128_t>(static_cast<__uint128_t>(static_cast<int>(val)));))
#  else // TEST_CUDA_COMPILER(CLANG)
  test<__int128_t>(static_cast<__int128_t>(static_cast<int>(val)));
  test<__uint128_t>(static_cast<__uint128_t>(static_cast<int>(val)));
#  endif // !TEST_CUDA_COMPILER(CLANG)
#endif // _CCCL_HAS_INT128()

  test_fp64_edge_cases(val);

  return true;
}

template <class T>
TEST_FUNC constexpr bool test()
{
  if constexpr (cuda::std::is_integral_v<T>)
  {
    test_fmax<T>(T{1});
    test_fmin<T>(T{1});
  }
  else
  {
    test_fmax<T>(cuda::std::__fp_one<T>());
    test_fmin<T>(cuda::std::__fp_one<T>());
  }

  return true;
}

TEST_FUNC constexpr bool test_constexpr()
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test<__float128>();
#endif // _CCCL_HAS_FLOAT128()

#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_UE5M3()
  test<__nv_fp8_ue5m3>();
#endif // _CCCL_HAS_NVFP8_UE5M3()
#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()

  test<signed char>();
  test<unsigned char>();
  test<signed short>();
  test<unsigned short>();
  test<signed int>();
  test<unsigned int>();
  test<signed long>();
  test<unsigned long>();
  test<signed long long>();
  test<unsigned long long>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()

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
  static_assert(test_constexpr());
  return 0;
}
