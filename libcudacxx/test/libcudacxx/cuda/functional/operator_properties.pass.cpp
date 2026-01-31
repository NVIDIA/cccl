//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/cassert>

#include "test_macros.h"

/***********************************************************************************************************************
 * Associativity and Commutativity
 **********************************************************************************************************************/

template <template <class...> class Op, class T>
__host__ __device__ constexpr bool is_commutative()
{
  return cuda::is_commutative_v<Op<T>, T> && cuda::is_commutative_v<Op<>, T> && //
         cuda::is_commutative_v<Op<T>, const T> && cuda::is_commutative_v<Op<T>, const T> && //
         cuda::is_commutative_v<Op<T>, volatile T> && cuda::is_commutative_v<Op<T>, volatile T> && //
         cuda::is_commutative_v<Op<T>, const volatile T> && cuda::is_commutative_v<Op<T>, const volatile T>;
}

template <template <class...> class Op, class T>
__host__ __device__ constexpr bool is_associative()
{
  return cuda::is_associative_v<Op<T>, T> && cuda::is_associative_v<Op<>, T> && //
         cuda::is_associative_v<Op<T>, const T> && cuda::is_associative_v<Op<T>, const T> && //
         cuda::is_associative_v<Op<T>, volatile T> && cuda::is_associative_v<Op<T>, volatile T> && //
         cuda::is_associative_v<Op<T>, const volatile T> && cuda::is_associative_v<Op<T>, const volatile T>;
}

template <template <class...> class Op, class T>
__host__ __device__ constexpr bool is_commutative_and_associative()
{
  return is_commutative<Op, T>() && is_associative<Op, T>();
}

template <class T>
__host__ __device__ constexpr void test_associative_commutative_integral()
{
  static_assert(is_commutative_and_associative<cuda::std::plus, T>());
  static_assert(is_commutative_and_associative<cuda::std::multiplies, T>());
  static_assert(!is_commutative<cuda::std::minus, T>() && !is_associative<cuda::std::minus, T>());
  static_assert(!is_commutative<cuda::std::divides, T>() && !is_associative<cuda::std::divides, T>());
  static_assert(!is_commutative<cuda::std::modulus, T>() && !is_associative<cuda::std::modulus, T>());
  static_assert(is_commutative_and_associative<cuda::std::bit_and, T>());
  static_assert(is_commutative_and_associative<cuda::std::bit_or, T>());
  static_assert(is_commutative_and_associative<cuda::std::bit_xor, T>());
  static_assert(is_commutative_and_associative<cuda::minimum, T>());
  static_assert(is_commutative_and_associative<cuda::maximum, T>());
}

__host__ __device__ constexpr void test_associative_commutative_integral()
{
  static_assert(is_commutative_and_associative<cuda::std::logical_and, bool>());
  static_assert(is_commutative_and_associative<cuda::std::logical_or, bool>());
  test_associative_commutative_integral<signed char>();
  test_associative_commutative_integral<unsigned char>();
  test_associative_commutative_integral<short>();
  test_associative_commutative_integral<unsigned short>();
  test_associative_commutative_integral<int>();
  test_associative_commutative_integral<unsigned int>();
  test_associative_commutative_integral<long>();
  test_associative_commutative_integral<unsigned long>();
  test_associative_commutative_integral<long long>();
  test_associative_commutative_integral<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_associative_commutative_integral<__int128_t>();
  test_associative_commutative_integral<__uint128_t>();
#endif // _CCCL_HAS_INT128()
}

//----------------------------------------------------------------------------------------------------------------------
// floating-point

template <class T>
__host__ __device__ constexpr void test_associative_commutative_floating_point()
{
  static_assert(is_commutative<cuda::std::plus, T>() && !is_associative<cuda::std::plus, T>());
  static_assert(is_commutative<cuda::std::multiplies, T>() && !is_associative<cuda::std::multiplies, T>());
  static_assert(!is_commutative<cuda::std::minus, T>() && !is_associative<cuda::std::minus, T>());
  static_assert(!is_commutative<cuda::std::divides, T>() && !is_associative<cuda::std::divides, T>());
  static_assert(is_commutative_and_associative<cuda::minimum, T>());
  static_assert(is_commutative_and_associative<cuda::maximum, T>());
}

__host__ __device__ constexpr void test_associative_commutative_floating_point()
{
  test_associative_commutative_floating_point<float>();
  test_associative_commutative_floating_point<double>();
#if _CCCL_HAS_NVFP16()
  test_associative_commutative_floating_point<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_associative_commutative_floating_point<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_FLOAT128()
  test_associative_commutative_floating_point<__float128>();
#endif // _CCCL_HAS_FLOAT128()
}

/***********************************************************************************************************************
 * Test dispatch
 **********************************************************************************************************************/

__host__ __device__ constexpr bool test()
{
  test_associative_commutative_integral();
  test_associative_commutative_floating_point();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
