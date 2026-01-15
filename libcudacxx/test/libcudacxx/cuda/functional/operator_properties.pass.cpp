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
#include <cuda/std/__cccl/dialect.h>
#include <cuda/std/cassert>
#include <cuda/std/limits>

#include "test_macros.h"

/***********************************************************************************************************************
 * Associativity and Commutativity
 **********************************************************************************************************************/

template <template <class...> class Op, class T>
__host__ __device__ constexpr bool is_commutative()
{
  return cuda::is_commutative_v<Op<T>, T> && cuda::is_commutative_v<Op<>, T>;
}

template <template <class...> class Op, class T>
__host__ __device__ constexpr bool is_associative()
{
  return cuda::is_associative_v<Op<T>, T> && cuda::is_associative_v<Op<>, T>;
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
 * Identity
 **********************************************************************************************************************/
template <class T>
__host__ __device__ constexpr T get_value()
{
  if constexpr (cuda::std::is_same_v<T, bool>)
  {
    return true;
  }
  else
  {
    if constexpr (::cuda::std::__is_extended_floating_point_v<T>)
    {
      return cuda::std::__fp_max<T>();
    }
    else
    {
      return T{42};
    }
  }
}

template <class Op, class T>
__host__ __device__ constexpr void test_identity_impl2(T identity)
{
  assert((identity == cuda::identity_element_v<Op, T>) );
  Op op{};
  T value      = get_value<T>();
  T identity1  = cuda::identity_element_v<Op, T>;
  T result_lhs = op(value, identity1);
  T result_rhs = op(identity1, value);
  assert(result_lhs == value);
  assert(result_rhs == value);
}

template <class Op, class T>
__host__ __device__ constexpr void test_identity_impl(bool has_identity, [[maybe_unused]] T identity)
{
  assert((has_identity == cuda::has_identity_element_v<Op, T>) );
  if constexpr (cuda::has_identity_element_v<Op, T>)
  {
    // handle extended floating-point types separately
    if constexpr (!::cuda::std::__is_extended_floating_point_v<T>)
    {
      test_identity_impl2<Op, T>(identity);
    }
    else
    {
      _CCCL_IF_NOT_CONSTEVAL
      {
        test_identity_impl2<Op, T>(identity);
      }
    }
  }
}

template <template <class...> class Op, class T>
__host__ __device__ constexpr void test_identity(bool has_identity, T identity)
{
  test_identity_impl<Op<T>, T>(has_identity, identity);
  test_identity_impl<Op<>, T>(has_identity, identity);
}

template <class T>
__host__ __device__ constexpr void test_identity_integral()
{
  test_identity<cuda::std::plus, T>(true, T{});
  test_identity<cuda::std::multiplies, T>(true, T{1});
  test_identity<cuda::std::bit_and, T>(true, static_cast<T>(~T{}));
  test_identity<cuda::std::bit_or, T>(true, T{});
  test_identity<cuda::std::bit_xor, T>(true, T{});
  test_identity<cuda::minimum, T>(true, cuda::std::numeric_limits<T>::max());
  test_identity<cuda::maximum, T>(true, cuda::std::numeric_limits<T>::lowest());
}

__host__ __device__ constexpr void test_identity_integral()
{
  test_identity<cuda::std::logical_and, bool>(true, true);
  test_identity<cuda::std::logical_or, bool>(true, false);
  test_identity_integral<signed char>();
  test_identity_integral<unsigned char>();
  test_identity_integral<short>();
  test_identity_integral<unsigned short>();
  test_identity_integral<int>();
  test_identity_integral<unsigned int>();
  test_identity_integral<long>();
  test_identity_integral<unsigned long>();
  test_identity_integral<long long>();
  test_identity_integral<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_identity_integral<__int128_t>();
  test_identity_integral<__uint128_t>();
#endif // _CCCL_HAS_INT128()
}

//----------------------------------------------------------------------------------------------------------------------
// floating-point

template <class T>
__host__ __device__ constexpr void test_identity_floating_point()
{
  test_identity<cuda::std::plus, T>(true, cuda::std::__fp_neg(T{}));
  test_identity<cuda::std::multiplies, T>(true, cuda::std::__fp_one<T>());
  test_identity<cuda::minimum, T>(true, cuda::std::numeric_limits<T>::infinity());
  test_identity<cuda::maximum, T>(true, cuda::std::__fp_neg(::cuda::std::__fp_inf<T>()));
}

__host__ __device__ constexpr void test_identity_floating_point()
{
  test_identity_floating_point<float>();
  test_identity_floating_point<double>();
#if _CCCL_HAS_NVFP16()
  test_identity_floating_point<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_identity_floating_point<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_FLOAT128()
  test_identity_floating_point<__float128>();
#endif // _CCCL_HAS_FLOAT128()
}

/***********************************************************************************************************************
 * Absorbing element
 **********************************************************************************************************************/

template <class Op, class T>
__host__ __device__ constexpr void test_absorbing_impl2([[maybe_unused]] T absorbing)
{
  assert((absorbing == cuda::absorbing_element_v<Op, T>) );
  Op op{};
  T value      = get_value<T>();
  T absorbing1 = cuda::absorbing_element_v<Op, T>;
  T result_lhs = op(value, absorbing1);
  T result_rhs = op(absorbing1, value);
  assert(result_lhs == absorbing1);
  assert(result_rhs == absorbing1);
}

template <class Op, class T>
__host__ __device__ constexpr void test_absorbing_impl(bool has_absorbing, [[maybe_unused]] T absorbing)
{
  assert((has_absorbing == cuda::has_absorbing_element_v<Op, T>) );
  if constexpr (cuda::has_absorbing_element_v<Op, T>)
  {
    // handle extended floating-point types separately
    if constexpr (!::cuda::std::__is_extended_floating_point_v<T>)
    {
      test_absorbing_impl2<Op, T>(absorbing);
    }
    else
    {
      _CCCL_IF_NOT_CONSTEVAL
      {
        test_absorbing_impl2<Op, T>(absorbing);
      }
    }
  }
}

template <template <class...> class Op, class T>
__host__ __device__ constexpr void test_absorbing(bool has_absorbing, T absorbing)
{
  test_absorbing_impl<Op<T>, T>(has_absorbing, absorbing);
  test_absorbing_impl<Op<>, T>(has_absorbing, absorbing);
}

template <class T>
__host__ __device__ constexpr void test_absorbing_integral()
{
  test_absorbing<cuda::std::multiplies, T>(true, T{});
  test_absorbing<cuda::std::bit_and, T>(true, T{});
  test_absorbing<cuda::std::bit_or, T>(true, static_cast<T>(~T{}));
  test_absorbing<cuda::minimum, T>(true, cuda::std::numeric_limits<T>::lowest());
  test_absorbing<cuda::maximum, T>(true, cuda::std::numeric_limits<T>::max());
}

__host__ __device__ constexpr void test_absorbing_integral()
{
  test_absorbing<cuda::std::logical_and, bool>(true, false);
  test_absorbing<cuda::std::logical_or, bool>(true, true);
  test_absorbing_integral<signed char>();
  test_absorbing_integral<unsigned char>();
  test_absorbing_integral<short>();
  test_absorbing_integral<unsigned short>();
  test_absorbing_integral<int>();
  test_absorbing_integral<unsigned int>();
  test_absorbing_integral<long>();
  test_absorbing_integral<unsigned long>();
  test_absorbing_integral<long long>();
  test_absorbing_integral<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_absorbing_integral<__int128_t>();
  test_absorbing_integral<__uint128_t>();
#endif // _CCCL_HAS_INT128()
}

//----------------------------------------------------------------------------------------------------------------------
// floating-point

template <class T>
__host__ __device__ constexpr void test_absorbing_floating_point()
{
  test_absorbing<cuda::minimum, T>(true, cuda::std::__fp_neg(cuda::std::numeric_limits<T>::infinity()));
  test_absorbing<cuda::maximum, T>(true, cuda::std::numeric_limits<T>::infinity());
}

__host__ __device__ constexpr void test_absorbing_floating_point()
{
  test_absorbing_floating_point<float>();
  test_absorbing_floating_point<double>();
#if _CCCL_HAS_NVFP16()
  test_absorbing_floating_point<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_absorbing_floating_point<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_FLOAT128()
  test_absorbing_floating_point<__float128>();
#endif // _CCCL_HAS_FLOAT128()
}

/***********************************************************************************************************************
 * Test dispatch
 **********************************************************************************************************************/

__host__ __device__ constexpr bool test()
{
  test_associative_commutative_integral();
  test_associative_commutative_floating_point();

  test_identity_integral();
  test_identity_floating_point();
  test_absorbing_integral();
  test_absorbing_floating_point();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
