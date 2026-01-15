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
#include <cuda/std/limits>

#include "test_macros.h"

template <class T, class Op, class Identity, class Absorbing>
__host__ __device__ constexpr void test_operator_properties(
  bool is_commutative,
  bool is_associative,
  bool has_identity_element,
  bool has_absorbing_element,
  Identity identity,
  Absorbing absorbing)
{
  using props = cuda::operator_properties<Op, T>;
  assert(props::is_commutative == is_commutative);
  assert(props::is_associative == is_associative);
  assert(props::has_identity_element == has_identity_element);
  assert(props::has_absorbing_element == has_absorbing_element);
  if constexpr (props::has_identity_element)
  {
    assert(props::identity_element() == identity);
  }
  if constexpr (props::has_absorbing_element)
  {
    assert(props::absorbing_element() == absorbing);
  }
}

template <class T, template <class...> class Op, class Identity, class Absorbing>
__host__ __device__ constexpr void test_operator_properties(
  bool is_commutative,
  bool is_associative,
  bool has_identity_element,
  bool has_absorbing_element,
  Identity identity,
  Absorbing absorbing)
{
  test_operator_properties<T, Op<T>, Identity, Absorbing>(
    is_commutative, is_associative, has_identity_element, has_absorbing_element, identity, absorbing);
  test_operator_properties<T, Op<>, Identity, Absorbing>(
    is_commutative, is_associative, has_identity_element, has_absorbing_element, identity, absorbing);
}

/***********************************************************************************************************************
 * Test all integral types
 **********************************************************************************************************************/

template <class T>
__host__ __device__ constexpr void test_integral_types()
{
  // is_commutative, is_associative, has_identity_element, has_absorbing_element
  // identity, absorbing
  test_operator_properties<T, cuda::std::plus<T>, T, T>(true, true, true, false, T{}, T{});
  test_operator_properties<T, cuda::std::multiplies<T>, T, T>(true, true, true, true, T{1}, T{});
  test_operator_properties<T, cuda::std::minus<T>, T, T>(false, false, false, false, T{}, T{});
  test_operator_properties<T, cuda::std::divides<T>, T, T>(false, false, false, false, T{}, T{});
  test_operator_properties<T, cuda::std::modulus<T>, T, T>(false, false, false, false, T{}, T{});
  if constexpr (cuda::std::__cccl_is_unsigned_integer_v<T>)
  {
    test_operator_properties<T, cuda::std::bit_and<T>, T, T>(true, true, true, true, static_cast<T>(~T{}), T{});
    test_operator_properties<T, cuda::std::bit_or<T>, T, T>(true, true, true, true, T{}, static_cast<T>(~T{}));
    test_operator_properties<T, cuda::std::bit_xor<T>, T, T>(true, true, true, false, T{}, T{});
  }
  test_operator_properties<T, cuda::minimum<T>, T, T>(
    true, true, true, true, cuda::std::numeric_limits<T>::max(), cuda::std::numeric_limits<T>::lowest());
  test_operator_properties<T, cuda::maximum<T>, T, T>(
    true, true, true, true, cuda::std::numeric_limits<T>::lowest(), cuda::std::numeric_limits<T>::max());
}

__host__ __device__ constexpr void test_integral_types()
{
  // is_commutative, is_associative, has_identity_element, has_absorbing_element
  // identity, absorbing
  test_operator_properties<bool, cuda::std::logical_and<bool>, bool, bool>(true, true, true, true, true, false);
  test_operator_properties<bool, cuda::std::logical_or<bool>, bool, bool>(true, true, true, true, false, true);

  test_integral_types<char>();
  test_integral_types<signed char>();
  test_integral_types<unsigned char>();
  test_integral_types<short>();
  test_integral_types<unsigned short>();
  test_integral_types<int>();
  test_integral_types<unsigned int>();
  test_integral_types<long>();
  test_integral_types<unsigned long>();
  test_integral_types<long long>();
  test_integral_types<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_integral_types<__int128_t>();
  test_integral_types<__uint128_t>();
#endif // _CCCL_HAS_INT128()
}

/***********************************************************************************************************************
 * Test floating-point types (associativity should be false for plus/multiplies)
 **********************************************************************************************************************/

template <class T>
__host__ __device__ constexpr void test_floating_point_types()
{
  // is_commutative, is_associative, has_identity_element, has_absorbing_element
  // identity, absorbing
  // Note: plus and multiplies are NOT associative for floating-point due to rounding
  test_operator_properties<T, cuda::std::plus<T>, T, T>(true, false, true, false, T{}, T{});
  test_operator_properties<T, cuda::std::multiplies<T>, T, T>(true, false, true, true, T{1}, T{});
  test_operator_properties<T, cuda::std::minus<T>, T, T>(false, false, false, false, T{}, T{});
  test_operator_properties<T, cuda::std::divides<T>, T, T>(false, false, false, false, T{}, T{});
  test_operator_properties<T, cuda::minimum<T>, T, T>(
    true, true, true, true, cuda::std::numeric_limits<T>::max(), cuda::std::numeric_limits<T>::lowest());
  test_operator_properties<T, cuda::maximum<T>, T, T>(
    true, true, true, true, cuda::std::numeric_limits<T>::lowest(), cuda::std::numeric_limits<T>::max());
}

__host__ __device__ constexpr void test_floating_point_types()
{
  test_floating_point_types<float>();
  test_floating_point_types<double>();
  // Verify associativity is false for floating-point arithmetic
  static_assert(!cuda::operator_properties<cuda::std::plus<float>, float>::is_associative);
  static_assert(!cuda::operator_properties<cuda::std::plus<double>, double>::is_associative);
  static_assert(!cuda::operator_properties<cuda::std::multiplies<float>, float>::is_associative);
  static_assert(!cuda::operator_properties<cuda::std::multiplies<double>, double>::is_associative);
}

/***********************************************************************************************************************
 * Test that identity and absorbing elements work correctly with operators
 **********************************************************************************************************************/

template <class T, class Op>
__host__ __device__ constexpr void test_identity_property()
{
  // Test that op(x, identity) == x for operators with identity
  using props = cuda::operator_properties<Op, T>;
  if constexpr (props::has_identity_element)
  {
    Op op{};
    T value      = T{42};
    T identity   = props::identity_element();
    T result_lhs = op(value, identity);
    T result_rhs = op(identity, value);
    assert(result_lhs == value);
    assert(result_rhs == value);
  }
}

template <class T, class Op>
__host__ __device__ constexpr void test_absorbing_property()
{
  // Test that op(x, absorbing) == absorbing for operators with absorbing element
  using props = cuda::operator_properties<Op, T>;
  if constexpr (props::has_absorbing_element)
  {
    Op op{};
    T value      = T{42};
    T absorbing  = props::absorbing_element();
    T result_lhs = op(value, absorbing);
    T result_rhs = op(absorbing, value);
    assert(result_lhs == absorbing);
    assert(result_rhs == absorbing);
  }
}

template <class T, class Op>
__host__ __device__ constexpr void test_algebraic_properties()
{
  test_identity_property<T, Op>();
  test_absorbing_property<T, Op>();
}

template <class T>
__host__ __device__ constexpr void test_algebraic_properties()
{
  test_algebraic_properties<T, cuda::std::plus<T>>();
  test_algebraic_properties<T, cuda::std::multiplies<T>>();
  if constexpr (cuda::std::__cccl_is_unsigned_integer_v<T>)
  {
    test_algebraic_properties<T, cuda::std::bit_and<T>>();
    test_algebraic_properties<T, cuda::std::bit_or<T>>();
    test_algebraic_properties<T, cuda::std::bit_xor<T>>();
  }
  test_algebraic_properties<T, cuda::minimum<T>>();
  test_algebraic_properties<T, cuda::maximum<T>>();
}

__host__ __device__ constexpr bool test_algebraic_properties()
{
  test_algebraic_properties<int>();
  test_algebraic_properties<unsigned int>();
  test_algebraic_properties<float>();
  test_algebraic_properties<double>();
  return true;
}

__host__ __device__ constexpr bool test()
{
  test_integral_types();
  test_floating_point_types();
  test_algebraic_properties();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
