//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cmath>

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_fmax(T val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmax((T) 0, (T) 0)), T>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmax((float) 0, (T) 0)), cuda::std::__promote_t<float, T>>),
                "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmax((double) 0, (T) 0)), cuda::std::__promote_t<double, T>>),
                "");
  assert(cuda::std::fmax(val, (T) 0) == val);
}

__host__ __device__ constexpr void test_fmax(float val)
{
  test_fmax<float>(val);
  test_fmax<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test_fmax<long double>(val);
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    test_fmax<__half>(__float2half(val));
  }
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    test_fmax<__nv_bfloat16>(__float2bfloat16(val));
  }
#endif // _LIBCUDACXX_HAS_NVBF16()

  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmax((int) 0, (int) 0)), double>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmax((int) 0, (long long) 0)), double>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmax((int) 0, (unsigned long long) 0)), double>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmax((float) 0, (unsigned int) 0)), double>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmax((double) 0, (long) 0)), double>), "");

  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmax((bool) 0, (float) 0)), double>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmax((unsigned short) 0, (double) 0)), double>), "");

  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmaxf(0, 0)), float>), "");
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmax((long double) 0, (unsigned long) 0)), long double>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmaxl(0, 0)), long double>), "");
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <class T>
__host__ __device__ constexpr void test_fmin(T val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmin((T) 0, (T) 0)), T>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmin((float) 0, (T) 0)), cuda::std::__promote_t<float, T>>),
                "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmin((double) 0, (T) 0)), cuda::std::__promote_t<double, T>>),
                "");
  assert(cuda::std::fmin(val, (T) 0) == T(0));
}

__host__ __device__ constexpr void test_fmin(float val)
{
  test_fmin<float>(val);
  test_fmin<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test_fmin<long double>(val);
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    test_fmin<__half>(__float2half(val));
  }
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    test_fmin<__nv_bfloat16>(__float2bfloat16(val));
  }
#endif // _LIBCUDACXX_HAS_NVBF16()

  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmin((int) 0, (int) 0)), double>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmin((int) 0, (long long) 0)), double>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmin((int) 0, (unsigned long long) 0)), double>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmin((float) 0, (unsigned int) 0)), double>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmin((double) 0, (long) 0)), double>), "");

  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmin((bool) 0, (float) 0)), double>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmin((unsigned short) 0, (double) 0)), double>), "");

  static_assert((cuda::std::is_same_v<decltype(cuda::std::fminf(0, 0)), float>), "");
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fmin((long double) 0, (unsigned long) 0)), long double>), "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::fminl(0, 0)), long double>), "");
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <class _Tp>
__host__ __device__ constexpr bool test(float val)
{
  test_fmax(val);
  test_fmin(val);

  return true;
}

__host__ __device__ constexpr bool test(float val)
{
  test<float>(val);
  test<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>(val);
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test<__half>(val);
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16>(val);
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3>(val);
#endif // _CCCL_HAS_NVFP8_E4M3
#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2>(val);
#endif // _CCCL_HAS_NVFP8_E5M2
#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0>(val);
#endif // _CCCL_HAS_NVFP8_E8M0
#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3>(val);
#endif // _CCCL_HAS_NVFP6_E2M3
#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2>(val);
#endif // _CCCL_HAS_NVFP6_E3M2
#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1>(val);
#endif // _CCCL_HAS_NVFP4_E2M1

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
  test<__int128_t>(static_cast<__int128_t>(val));
  test<__uint128_t>(static_cast<__uint128_t>(val));
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
  static_assert(test(1.0f));
  return 0;
}
