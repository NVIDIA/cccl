//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

using F  = float;
using D  = double;
using LD = long double;

__host__ __device__ F fabs_overload(F value)
{
  ASSERT_SAME_TYPE(F, decltype(cuda::std::fabsf(F{})));
  // static_assert(noexcept(cuda::std::fabsf(F{})), "");
  return cuda::std::fabsf(value);
}

__host__ __device__ D fabs_overload(D value)
{
  ASSERT_SAME_TYPE(D, decltype(cuda::std::fabs(D{})));
  // static_assert(noexcept(cuda::std::fabs(D{})), "");
  return cuda::std::fabs(value);
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
__host__ __device__ LD fabs_overload(LD value)
{
  ASSERT_SAME_TYPE(LD, decltype(cuda::std::fabsl(LD{})));
  // static_assert(noexcept(cuda::std::fabsl(LD{})), "");
  return cuda::std::fabsl(value);
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

template <class T, class = void>
struct OverloadTester;

template <class T>
struct OverloadTester<T, cuda::std::enable_if_t<!cuda::std::__is_extended_floating_point<T>::value>>
{
  __host__ __device__ static bool test_fabs(T in, T ref, T zero_value)
  {
    assert(fabs_overload(zero_value + in) == ref);
    assert(cuda::std::abs(zero_value + in) == ref);
    return true;
  }

  __host__ __device__ static bool test_fabs_nan(T zero_value)
  {
    assert(cuda::std::isnan(fabs_overload(zero_value + cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::std::abs(zero_value + cuda::std::numeric_limits<T>::quiet_NaN())));
    return true;
  }
};

template <class T>
struct OverloadTester<T, cuda::std::enable_if_t<cuda::std::__is_extended_floating_point<T>::value>>
{
  __host__ __device__ static bool test_fabs(T in, T ref, T zero_value)
  {
    assert(cuda::std::abs(zero_value + in) == ref);
    return true;
  }

  __host__ __device__ static bool test_fabs_nan(T zero_value)
  {
    assert(cuda::std::isnan(cuda::std::abs(zero_value + cuda::std::numeric_limits<T>::quiet_NaN())));
    return true;
  }
};

template <class T>
__host__ __device__ bool test_fabs(T in, T ref, T zero_value)
{
  return OverloadTester<T>::test_fabs(in, ref, zero_value);
}

template <class T>
__host__ __device__ bool test_fabs_nan(T zero_value)
{
  return OverloadTester<T>::test_fabs_nan(zero_value);
}

template <class T>
__host__ __device__ bool test_fabs(T zero_value)
{
  test_fabs(T{0.0}, T{0.0}, zero_value);
  test_fabs(T{-0.0}, T{0.0}, zero_value);

  test_fabs(T{1.0}, T{1.0}, zero_value);
  test_fabs(T{-1.0}, T{1.0}, zero_value);

  test_fabs(T{257.0}, T{257.0}, zero_value);
  test_fabs(T{-257.0}, T{257.0}, zero_value);

  test_fabs(cuda::std::numeric_limits<T>::max(), cuda::std::numeric_limits<T>::max(), zero_value);
  test_fabs(-cuda::std::numeric_limits<T>::max(), cuda::std::numeric_limits<T>::max(), zero_value);

  test_fabs(cuda::std::numeric_limits<T>::min(), cuda::std::numeric_limits<T>::min(), zero_value);
  test_fabs(-cuda::std::numeric_limits<T>::min(), cuda::std::numeric_limits<T>::min(), zero_value);

  test_fabs(cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity(), zero_value);
  test_fabs(-cuda::std::numeric_limits<T>::infinity(), cuda::std::numeric_limits<T>::infinity(), zero_value);

  // __half and __nvbfloat have precision issues here
  if (!cuda::std::__is_extended_floating_point<T>::value)
  {
    test_fabs_nan(zero_value);
  }

  return true;
}

__host__ __device__ bool test(F zero_value)
{
  test_fabs(zero_value);
  test_fabs(static_cast<D>(zero_value));
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_fabs(static_cast<LD>(zero_value));
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_fabs(__float2half(zero_value));
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_fabs(__float2bfloat16(zero_value));
#endif // _LIBCUDACXX_HAS_NVBF16

  return true;
}

__global__ void test_global_kernel(F* zero_value)
{
  test(*zero_value);
  // #if TEST_STD_VER >= 2014
  //   static_assert(test(0.f), "");
  // #endif // TEST_STD_VER >= 2014
}

int main(int, char**)
{
  volatile F zero_value = 0.f;
  assert(test(zero_value));
  // #if TEST_STD_VER >= 2014
  //   static_assert(test(0.f), "");
  // #endif // TEST_STD_VER >= 2014
  return 0;
}
