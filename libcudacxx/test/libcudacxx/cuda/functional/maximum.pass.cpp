//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/cassert>
#include <cuda/type_traits>

#include "test_macros.h"

template <typename OpT, typename T, typename U, typename Result>
__host__ __device__ constexpr void test_op(const T lhs, const U rhs, const Result expected)
{
  if constexpr (!cuda::std::__is_extended_floating_point_v<T> && !cuda::std::__is_extended_floating_point_v<U>)
  {
    static_assert(noexcept(OpT{}(cuda::std::declval<T>(), cuda::std::declval<U>())), "OpT is not noexcept");
  }
  static_assert(cuda::std::is_same_v<decltype(OpT{}(lhs, rhs)), Result>, "OpT is not the expected type");
  assert((OpT{}(lhs, rhs) == expected) && (OpT{}(lhs, rhs) == OpT{}(rhs, lhs)));
}

template <typename T, typename U, typename Result>
__host__ __device__ constexpr void test(const T lhs, const U rhs, const Result expected)
{
  if constexpr (cuda::std::is_same_v<T, U> && cuda::std::is_same_v<Result, T>)
  {
    test_op<cuda::maximum<T>>(lhs, rhs, expected);
  }
  else
  {
    test_op<cuda::maximum<>>(lhs, rhs, expected);
    test_op<cuda::maximum<void>>(lhs, rhs, expected);
  }
}

__host__ __device__ constexpr bool test()
{
  test<int, int, int>(0, 1, 1);
  test<int, int, int>(1, 0, 1);
  test<int, int, int>(0, 0, 0);
  test<int, int, int>(-1, 1, 1);
  test<char, char, char>('a', 'b', 'b');
  test<float, float, float>(1.0f, 2.0f, 2.0f);
  test<double, double, double>(1.0, 2.0, 2.0);

  test<float, double, double>(1.0f, 2.0, 2.0);
#if _CCCL_HAS_FLOAT128()
  test<__float128, __float128, __float128>(__float128(1.0f), __float128(2.0f), __float128(2.0f));
  test<float, __float128, __float128>(1.0f, __float128(2.0f), __float128(2.0f));
  test<double, __float128, __float128>(2.0, __float128(2.0f), __float128(2.0f));
#endif // _CCCL_HAS_FLOAT128()
  return true;
}

__host__ __device__ bool runtime_test()
{
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half, __half, __half>(__half(1.0f), __half(2.0f), __half(2.0f));
  test<__half, float, float>(__half(1.0f), 2.0f, 2.0f);
// MSVC with C++17 doesn't allow operator== when mixing __half and __nv_bfloat16 with double
#  if _LIBCUDACXX_HAS_NVFP16() && !(_CCCL_COMPILER(MSVC) && _CCCL_STD_VER == 2017)
  test<__half, double, double>(__half(1.0f), 2.0, 2.0);
#  endif
#endif
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(__nv_bfloat16(1.0f), __nv_bfloat16(2.0f), __nv_bfloat16(2.0f));
  test<__nv_bfloat16, float, float>(__nv_bfloat16(1.0f), 2.0f, 2.0f);
#  if _LIBCUDACXX_HAS_NVFP16() && !(_CCCL_COMPILER(MSVC) && _CCCL_STD_VER == 2017)
  test<__nv_bfloat16, double, double>(__nv_bfloat16(1.0f), 2.0, 2.0);
#  endif
#endif
  return true;
}

int main(int, char**)
{
  assert(test());
  assert(runtime_test());
  static_assert(test());
  return 0;
}
