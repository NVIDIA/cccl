//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__simd_>

// [simd.mask.conv], basic_mask conversions
//
// explicit operator basic_vec<U, A>() const noexcept;   // sizeof(U) != Bytes
// operator basic_vec<U, A>() const noexcept;            // sizeof(U) == Bytes (implicit)
// constexpr bitset<size()> to_bitset() const noexcept;
// constexpr unsigned long long to_ullong() const;

#include <cuda/std/__simd_>
#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// implicit conversion to basic_vec (sizeof(U) == Bytes)

template <typename T, int N>
TEST_FUNC constexpr void test_implicit_conv()
{
  using Mask = simd::basic_mask<sizeof(T), simd::fixed_size<N>>;
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  Mask mask(is_even{});

  static_assert(cuda::std::is_convertible_v<const Mask&, Vec>);
  static_assert(noexcept(static_cast<Vec>(mask)));

  Vec vec = mask;
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == static_cast<T>(i % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// explicit conversion to basic_vec (sizeof(U) != Bytes)

template <int Bytes, typename U, int N>
TEST_FUNC constexpr void test_explicit_conv()
{
  static_assert(sizeof(U) != Bytes);
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  using Vec  = simd::basic_vec<U, simd::fixed_size<N>>;
  Mask mask(is_even{});

  static_assert(!cuda::std::is_convertible_v<const Mask&, Vec>);
  static_assert(cuda::std::is_same_v<decltype(static_cast<Vec>(mask)), Vec>);
  static_assert(noexcept(static_cast<Vec>(mask)));

  Vec vec = static_cast<Vec>(mask);
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == static_cast<U>(i % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// to_bitset

template <int Bytes, int N>
TEST_FUNC constexpr void test_to_bitset()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Mask mask(true);

  static_assert(cuda::std::is_same_v<decltype(mask.to_bitset()), cuda::std::bitset<N>>);
  static_assert(noexcept(mask.to_bitset()));
  static_assert(is_const_member_function_v<decltype(&Mask::to_bitset)>);
  unused(mask);

  Mask all_false(false);
  auto bitset_false = all_false.to_bitset();
  assert(bitset_false.none());

  Mask all_true(true);
  auto bitset_true = all_true.to_bitset();
  assert(bitset_true.all());

  Mask mixed(is_even{});
  auto bitset = mixed.to_bitset();
  for (int i = 0; i < N; ++i)
  {
    assert(bitset[i] == (i % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// to_ullong

// N is guaranteed to be in [1, 64]
template <int Bytes, int N>
TEST_FUNC constexpr void test_to_ullong()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Mask mask(true);

  static_assert(cuda::std::is_same_v<decltype(mask.to_ullong()), unsigned long long>);
  static_assert(!noexcept(mask.to_ullong()));
  static_assert(is_const_member_function_v<decltype(&Mask::to_ullong)>);
  unused(mask);

  Mask all_false(false);
  assert(all_false.to_ullong() == 0ULL);

  Mask all_true(true);
  constexpr unsigned long long expected = (N == 64) ? ~0ULL : (~0ULL >> (64 - N));
  assert(all_true.to_ullong() == expected);

  Mask mixed(is_even{});
  unsigned long long expected_mixed = 0ULL;
  for (int i = 0; i < N; ++i)
  {
    if (i % 2 == 0)
    {
      expected_mixed |= (1ULL << i);
    }
  }
  assert(mixed.to_ullong() == expected_mixed);
}

//----------------------------------------------------------------------------------------------------------------------
// SFINAE constraints

template <int Bytes, int N>
TEST_FUNC constexpr void test_sfinae_negative()
{
  using Mask    = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  using Integer = integer_from_t<Bytes>;

  // mismatched element count: conversion is fully rejected
  using WrongSizeVec = simd::basic_vec<Integer, simd::fixed_size<N + 1>>;
  static_assert(!cuda::std::is_constructible_v<WrongSizeVec, const Mask&>);
  static_assert(!cuda::std::is_convertible_v<const Mask&, WrongSizeVec>);
}

//----------------------------------------------------------------------------------------------------------------------

template <int Bytes, int N>
TEST_FUNC constexpr void test_size()
{
  test_to_bitset<Bytes, N>();
  test_to_ullong<Bytes, N>();
  test_sfinae_negative<Bytes, N>();
}

template <int Bytes>
TEST_FUNC constexpr void test_bytes()
{
  test_size<Bytes, 1>();
  test_size<Bytes, 4>();
}

TEST_FUNC constexpr bool test()
{
  test_bytes<1>();
  test_bytes<2>();
  test_bytes<4>();
  test_bytes<8>();
#if _CCCL_HAS_INT128()
  test_bytes<16>();
#endif

  test_implicit_conv<int, 1>();
  test_implicit_conv<int, 8>();
  test_implicit_conv<short, 4>();
  test_implicit_conv<long long, 4>();

  test_explicit_conv<4, short, 4>();
  test_explicit_conv<2, int, 4>();
  test_explicit_conv<8, int, 4>();
  test_explicit_conv<4, long long, 4>();
  test_explicit_conv<1, short, 4>();
  test_explicit_conv<2, long long, 8>();

  test_to_ullong<1, 64>();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
