//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

// <cuda/complex>

#include <cuda/__complex_>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#if !_CCCL_COMPILER(NVRTC)
#  include <complex>
#endif // !_CCCL_COMPILER(NVRTC)

template <class From, class To>
__host__ __device__ constexpr bool is_implicit_conversion()
{
  if constexpr (cuda::std::__is_fp_v<From> && cuda::std::__is_fp_v<To>)
  {
    return cuda::std::__fp_is_implicit_conversion_v<From, To>;
  }
  else
  {
    return true;
  }
}

template <class T, class C>
__host__ __device__ constexpr void test_constructor_from_complex(const C& other)
{
  using U = typename C::value_type;

  // 1. Test that cuda::complex<T> constructible from C
  static_assert(cuda::std::is_constructible_v<cuda::complex<T>, const C&>);

  // 2. Test that the constructor is noexcept
  static_assert(noexcept(cuda::complex<T>{cuda::std::declval<const C&>()}));

  // 3. Test that the constructor takes into account floating-point conversion rank order
  static_assert(cuda::std::is_convertible_v<const C&, cuda::complex<T>> == is_implicit_conversion<U, T>());

  const cuda::complex<T> v2{other};
  assert(v2.real() == T(1));
  assert(v2.imag() == T(2));
}

template <class T, class U>
__host__ __device__ constexpr bool test_cccl_types()
{
  if constexpr (!cuda::std::is_same_v<T, U>)
  {
    test_constructor_from_complex<T>(cuda::complex<U>{U(1), U(2)});
  }
  test_constructor_from_complex<T>(cuda::std::complex<U>{U(1), U(2)});

  return true;
}

template <class T, class U>
__host__ __device__ void test_types()
{
  test_cccl_types<T, U>();
  static_assert(test_cccl_types<T, U>());

#if !_CCCL_COMPILER(NVRTC)
  // std::complex is not required to support other than standard floating-point types
  if constexpr (cuda::std::__is_std_fp_v<T>)
  {
    NV_IF_TARGET(NV_IS_HOST, (test_constructor_from_complex<T>(std::complex<U>{U(1), U(2)});))
  }
#endif // !_CCCL_COMPILER(NVRTC)
}

template <class T>
__host__ __device__ void test()
{
  test_types<T, float>();
  test_types<T, double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_types<T, long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_types<T, __float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_types<T, signed char>();
  test_types<T, signed short>();
  test_types<T, signed int>();
  test_types<T, signed long>();
  test_types<T, signed long long>();
#if _CCCL_HAS_INT128()
  test_types<T, __int128_t>();
#endif // _CCCL_HAS_INT128()

  test_types<T, unsigned char>();
  test_types<T, unsigned short>();
  test_types<T, unsigned int>();
  test_types<T, unsigned long>();
  test_types<T, unsigned long long>();
#if _CCCL_HAS_INT128()
  test_types<T, __uint128_t>();
#endif // _CCCL_HAS_INT128()
}

__host__ __device__ void test()
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test<signed char>();
  test<signed short>();
  test<signed int>();
  test<signed long>();
  test<signed long long>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test<unsigned char>();
  test<unsigned short>();
  test<unsigned int>();
  test<unsigned long>();
  test<unsigned long long>();
#if _CCCL_HAS_INT128()
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()
}

int main(int, char**)
{
  test();
  return 0;
}
