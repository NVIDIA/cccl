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
__host__ __device__ void test_log(T value)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::log(value)), ret>, "");
  assert(cuda::std::log(value) == ret{0});
}

template <class T>
__host__ __device__ void test_log10(T value)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::log10(value)), ret>, "");
  assert(cuda::std::log10(value) == ret{0});
}

template <class T>
__host__ __device__ _CCCL_CONSTEXPR_BIT_CAST void test_ilogb(T value)
{
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ilogb(value)), int>, "");
  assert(cuda::std::ilogb(value) == 0);
}

template <class T>
__host__ __device__ void test_log1p(T value)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::log1p(value)), ret>, "");
  assert(cuda::std::log1p(value - value) == ret{0});
}

template <class T>
__host__ __device__ void test_log2(T value)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::log2(value)), ret>, "");
  assert(cuda::std::log2(value) == ret{0});
}

template <class T>
__host__ __device__ _CCCL_CONSTEXPR_BIT_CAST void test_logb(T value)
{
  using ret = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::logb(value)), ret>, "");
  assert(cuda::std::logb(value) == ret{0});
}

template <class T>
__host__ __device__ void test(T value)
{
  test_log<T>(value);
  test_log10<T>(value);
  test_ilogb<T>(value);
  test_log1p<T>(value);
  test_log2<T>(value);
  test_logb<T>(value);
}

__host__ __device__ void test(float value)
{
  test<float>(value);
  test<double>(value);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>(value);
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>(__float2half(value));
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>(__float2bfloat16(value));
#endif // _LIBCUDACXX_HAS_NVBF16()

  test<unsigned short>(static_cast<unsigned short>(value));
  test<int>(static_cast<int>(value));
  test<unsigned int>(static_cast<unsigned int>(value));
  test<long>(static_cast<long>(value));
  test<unsigned long>(static_cast<unsigned long>(value));
  test<long long>(static_cast<long long>(value));
  test<unsigned long long>(static_cast<unsigned long long>(value));
}

__global__ void test_global_kernel(float* value)
{
  test(*value);
}

template <class T>
__host__ __device__ _CCCL_CONSTEXPR_BIT_CAST void test_constexpr()
{
  if constexpr (cuda::std::is_integral_v<T>)
  {
    test_ilogb<T>(static_cast<T>(1));
    test_logb<T>(static_cast<T>(1));
  }
  else
  {
    test_ilogb<T>(cuda::std::__fp_one<T>());
    test_logb<T>(cuda::std::__fp_one<T>());
  }
}

__host__ __device__ _CCCL_CONSTEXPR_BIT_CAST bool test_constexpr()
{
  test_constexpr<float>();
  test_constexpr<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_constexpr<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

  test_constexpr<unsigned short>();
  test_constexpr<int>();
  test_constexpr<unsigned int>();
  test_constexpr<long>();
  test_constexpr<unsigned long>();
  test_constexpr<long long>();
  test_constexpr<unsigned long long>();

  return true;
}

int main(int, char**)
{
  volatile float value = 1.0f;
  test(value);

#if _CCCL_HAS_CONSTEXPR_BIT_CAST()
  static_assert(test_constexpr());
#endif // _CCCL_HAS_CONSTEXPR_BIT_CAST()
  return 0;
}
