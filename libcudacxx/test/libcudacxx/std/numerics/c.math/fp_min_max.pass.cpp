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
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "comparison.h"
#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_fmax(T val)
{
  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(eq(cuda::std::fmax(val, T()), val));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::fmax(val, T())), double>);
  }
  else
  {
    assert(eq(cuda::std::fmax(val, cuda::std::__fp_zero<T>()), val));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::fmax(val, cuda::std::__fp_zero<T>())), T>);

    // Ensure that we can have different arguments
    using promote_float = cuda::std::__promote_t<float, T>;
    assert(cuda::std::fmax(val, 0.0f) == promote_float(1));
    assert(cuda::std::fmax(0.0f, val) == promote_float(1));

    using promote_double = cuda::std::__promote_t<double, T>;
    assert(cuda::std::fmax(val, 0.0) == promote_double(1));
    assert(cuda::std::fmax(0.0, val) == promote_double(1));

    // We always return the other value when passing a NaN
    if constexpr (cuda::std::__fp_has_nan_v<cuda::std::__fp_format_of_v<T>>)
    {
      assert(eq(cuda::std::fmax(val, cuda::std::numeric_limits<T>::quiet_NaN()), val));
      assert(eq(cuda::std::fmax(cuda::std::numeric_limits<T>::quiet_NaN(), val), val));
      assert(eq(cuda::std::fmax(val, cuda::std::numeric_limits<T>::signaling_NaN()), val));
      assert(eq(cuda::std::fmax(cuda::std::numeric_limits<T>::signaling_NaN(), val), val));
    }

    // check infinities
    assert(
      eq(cuda::std::fmax(val, cuda::std::numeric_limits<T>::infinity()), cuda::std::numeric_limits<T>::infinity()));
    assert(
      eq(cuda::std::fmax(cuda::std::numeric_limits<T>::infinity(), val), cuda::std::numeric_limits<T>::infinity()));
    if constexpr (cuda::std::__fp_is_signed_v<cuda::std::__fp_format_of_v<T>>)
    {
      assert(eq(cuda::std::fmax(val, -cuda::std::numeric_limits<T>::infinity()), val));
      assert(eq(cuda::std::fmax(-cuda::std::numeric_limits<T>::infinity(), val), val));
    }
  }
}

template <class T>
__host__ __device__ constexpr void test_fmin(T val)
{
  if constexpr (cuda::std::is_integral_v<T>)
  {
    assert(cuda::std::fmin(val, T()) == T());
    static_assert(cuda::std::is_same_v<decltype(cuda::std::fmin(val, T())), double>);
  }
  else
  {
    assert(eq(cuda::std::fmin(val, cuda::std::__fp_zero<T>()), cuda::std::__fp_zero<T>()));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::fmin(val, cuda::std::__fp_zero<T>())), T>);

    // Ensure that we can have different arguments
    using promote_float = cuda::std::__promote_t<float, T>;
    assert(cuda::std::fmin(val, 2.0f) == promote_float(1));
    assert(cuda::std::fmin(2.0f, val) == promote_float(1));

    using promote_double = cuda::std::__promote_t<double, T>;
    assert(cuda::std::fmin(val, 2.0) == promote_double(1));
    assert(cuda::std::fmin(2.0, val) == promote_double(1));

    // We always return the other value when passing a NaN
    if constexpr (cuda::std::__fp_has_nan_v<cuda::std::__fp_format_of_v<T>>)
    {
      assert(eq(cuda::std::fmin(val, cuda::std::numeric_limits<T>::quiet_NaN()), val));
      assert(eq(cuda::std::fmin(cuda::std::numeric_limits<T>::quiet_NaN(), val), val));
      assert(eq(cuda::std::fmin(val, cuda::std::numeric_limits<T>::signaling_NaN()), val));
      assert(eq(cuda::std::fmin(cuda::std::numeric_limits<T>::signaling_NaN(), val), val));
    }

    // check infinities
    assert(eq(cuda::std::fmin(val, cuda::std::numeric_limits<T>::infinity()), val));
    assert(eq(cuda::std::fmin(cuda::std::numeric_limits<T>::infinity(), val), val));

    if constexpr (cuda::std::__fp_is_signed_v<cuda::std::__fp_format_of_v<T>>)
    {
      assert(
        eq(cuda::std::fmin(val, -cuda::std::numeric_limits<T>::infinity()), -cuda::std::numeric_limits<T>::infinity()));
      assert(
        eq(cuda::std::fmin(-cuda::std::numeric_limits<T>::infinity(), val), -cuda::std::numeric_limits<T>::infinity()));
    }
  }
}

template <class T>
__host__ __device__ constexpr bool test(T val)
{
  test_fmax<T>(val);
  test_fmin<T>(val);

  return true;
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
#if _CCCL_HAS_INT128()
  test<__int128_t>(static_cast<__int128_t>(static_cast<int>(val)));
  test<__uint128_t>(static_cast<__uint128_t>(static_cast<int>(val)));
#endif // _CCCL_HAS_INT128()

  return true;
}

template <class T>
__host__ __device__ constexpr bool test()
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

__host__ __device__ constexpr bool test_constexpr()
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

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
