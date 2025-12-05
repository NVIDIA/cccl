//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/cmath>

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

__host__ __device__ bool is_about(float x, float y)
{
  return (cuda::std::abs((x - y) / (x + y)) < 1.e-6);
}

__host__ __device__ bool is_about(double x, double y)
{
  return (cuda::std::abs((x - y) / (x + y)) < 1.e-14);
}

#if _CCCL_HAS_LONG_DOUBLE()
__host__ __device__ bool is_about(long double x, long double y)
{
  return (cuda::std::abs((x - y) / (x + y)) < 1.e-14);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
__host__ __device__ bool is_about(__half x, __half y)
{
  return (cuda::std::fabs((x - y) / (x + y)) <= __half(1e-3));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
__host__ __device__ bool is_about(__nv_bfloat16 x, __nv_bfloat16 y)
{
  return (cuda::std::fabs((x - y) / (x + y)) <= __nv_bfloat16(5e-3));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class T>
__host__ __device__ void test_type(float zero)
{
  using Result = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;

  // 1. Test signature.
  static_assert(cuda::std::is_same_v<cuda::sincos_result<Result>, decltype(cuda::sincos(T{}))>);
  static_assert(noexcept(cuda::sincos(cuda::std::declval<T>())));

  // 2. Test sincos(0).
  {
    auto result = cuda::sincos(static_cast<T>(zero));
    static_assert(cuda::std::is_same_v<Result, decltype(result.sin)>);
    static_assert(cuda::std::is_same_v<Result, decltype(result.cos)>);
    assert(result.sin == Result{0});
    assert(result.cos == Result{1});
  }

  // 3. Test sincos(value) to result of separate sin/cos calls.
  {
    const auto value = static_cast<T>(4);
    auto result      = cuda::sincos(value);
    assert(is_about(result.sin, cuda::std::sin(value)));
    assert(is_about(result.cos, cuda::std::cos(value)));
  }

  // 4. Test sincos(+-inf)
  if constexpr (cuda::std::numeric_limits<T>::has_infinity && cuda::std::numeric_limits<T>::has_quiet_NaN)
  {
    auto pos_result = cuda::sincos(cuda::std::numeric_limits<T>::infinity());
    assert(cuda::std::isnan(pos_result.sin));
    assert(cuda::std::isnan(pos_result.cos));

    auto neg_result = cuda::sincos(-cuda::std::numeric_limits<T>::infinity());
    assert(cuda::std::isnan(neg_result.sin));
    assert(cuda::std::isnan(neg_result.cos));
  }

  // 5. Test sincos(+-nan)
  if constexpr (cuda::std::numeric_limits<T>::has_quiet_NaN)
  {
    auto pos_result = cuda::sincos(cuda::std::numeric_limits<T>::quiet_NaN());
    assert(cuda::std::isnan(pos_result.sin));
    assert(cuda::std::isnan(pos_result.cos));

    auto neg_result = cuda::sincos(-cuda::std::numeric_limits<T>::quiet_NaN());
    assert(cuda::std::isnan(neg_result.sin));
    assert(cuda::std::isnan(neg_result.cos));
  }
}

__host__ __device__ void test(float zero)
{
  test_type<float>(zero);
  test_type<double>(zero);
#if _CCCL_HAS_LONG_DOUBLE()
  test_type<long double>(zero);
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test_type<__half>(zero);
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_type<__nv_bfloat16>(zero);
#endif // _LIBCUDACXX_HAS_NVBF16()

  // todo: add tests for f128 once supported

  test_type<signed char>(zero);
  test_type<signed short>(zero);
  test_type<signed int>(zero);
  test_type<signed long>(zero);
  test_type<signed long long>(zero);
#if _CCCL_HAS_INT128() && !_CCCL_CUDA_COMPILER(CLANG) // clang-cuda crashes with int128
  test_type<__int128_t>(static_cast<int>(zero));
#endif // _CCCL_HAS_INT128() && !_CCCL_CUDA_COMPILER(CLANG)

  test_type<unsigned char>(zero);
  test_type<unsigned short>(zero);
  test_type<unsigned int>(zero);
  test_type<unsigned long>(zero);
  test_type<unsigned long long>(zero);
#if _CCCL_HAS_INT128() && !_CCCL_CUDA_COMPILER(CLANG) // clang-cuda crashes with int128
  test_type<__uint128_t>(static_cast<int>(zero));
#endif // _CCCL_HAS_INT128() && !_CCCL_CUDA_COMPILER(CLANG)
}

__global__ void kernel()
{
  test(0.f);
}

int main(int, char**)
{
  volatile float zero = 0.0f;
  test(zero);
  return 0;
}
