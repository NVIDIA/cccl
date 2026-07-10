//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/numeric>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class Res, class L, class R, class InL, class InR, class Ref>
TEST_FUNC constexpr void test_mul_overflow(InL lhs_in, InR rhs_in, Ref expected, bool overflow)
{
  const auto lhs = cuda::overflow_cast<L>(lhs_in);
  const auto rhs = cuda::overflow_cast<R>(rhs_in);

  if (lhs.overflow || rhs.overflow)
  {
    return;
  }

  {
    const auto result = cuda::mul_overflow<Res>(lhs.value, rhs.value);
    assert(result.value == static_cast<Res>(expected));
    assert(result.overflow == overflow);
  }
  {
    Res result{};
    assert(cuda::mul_overflow(result, lhs.value, rhs.value) == overflow);
    assert(result == static_cast<Res>(expected));
  }
}

template <class Res, class L, class R, class InL, class InR, class Ref>
TEST_FUNC constexpr void test_mul_overflow(InL lhs_in, InR rhs_in, Ref expected)
{
  test_mul_overflow<Res, L, R, InL, InR, Ref>(lhs_in, rhs_in, expected, !cuda::std::in_range<Res>(expected));
}

template <class Res, class L, class R>
TEST_FUNC constexpr bool test_type()
{
  static_assert(
    cuda::std::is_same_v<decltype(cuda::mul_overflow(L{}, R{})), cuda::overflow_result<cuda::std::common_type_t<L, R>>>);
  static_assert(cuda::std::is_same_v<decltype(cuda::mul_overflow<Res>(L{}, R{})), cuda::overflow_result<Res>>);
  static_assert(cuda::std::is_same_v<decltype(cuda::mul_overflow(cuda::std::declval<Res&>(), L{}, R{})), bool>);

  static_assert(noexcept(cuda::mul_overflow(L{}, R{})));
  static_assert(noexcept(cuda::mul_overflow<Res>(L{}, R{})));
  static_assert(noexcept(cuda::mul_overflow(cuda::std::declval<Res&>(), L{}, R{})));

  // 1. Multiplying zeros should always result in zero
  test_mul_overflow<Res, L, R>(0, 0, 0);

  // 2. Test correctness of multiplication
  test_mul_overflow<Res, L, R>(1, 1, 1);
  test_mul_overflow<Res, L, R>(-1, 1, -1);
  test_mul_overflow<Res, L, R>(1, -1, -1);
  test_mul_overflow<Res, L, R>(-1, -1, 1);

  // 3. Test T(-1) * T_MIN case
  if constexpr (cuda::std::is_signed_v<L> && cuda::std::is_signed_v<R>)
  {
    // nvcc 12.0 seems to have some issues during constant evaluation of this test. Works fine with later versions, so
    // let's just disable this test.
#if !_CCCL_CUDA_COMPILER(NVCC, ==, 12, 0)
    test_mul_overflow<Res, L, R>(
      cuda::std::numeric_limits<L>::min(), -1, cuda::uabs(cuda::std::numeric_limits<L>::min()));
    test_mul_overflow<Res, L, R>(
      -1, cuda::std::numeric_limits<R>::min(), cuda::uabs(cuda::std::numeric_limits<R>::min()));
#endif // !_CCCL_CUDA_COMPILER(NVCC, ==, 12, 0)
  }

  // 4. Test other numbers
  test_mul_overflow<Res, L, R>(17, 14, 238);
  test_mul_overflow<Res, L, R>(-254, 127, -32258);
  test_mul_overflow<Res, L, R>(1657, -13748, -22780436);
  test_mul_overflow<Res, L, R>(-2147483647, 4294967295, -9223372030412324865);
  if constexpr (cuda::std::is_unsigned_v<L> && cuda::std::is_unsigned_v<Res>)
  {
    test_mul_overflow<Res, L, R>(
      cuda::std::numeric_limits<L>::max(),
      4,
      static_cast<Res>(cuda::std::numeric_limits<L>::max()) << 2,
      sizeof(L) >= sizeof(Res));
  }
  return true;
}

TEST_FUNC constexpr void test_corner_cases()
{
  constexpr auto int_max  = cuda::std::numeric_limits<int>::max();
  constexpr auto int_min  = cuda::std::numeric_limits<int>::min();
  constexpr auto uint_max = cuda::std::numeric_limits<unsigned>::max();

  // 1. Boundary edge-cases
  test_mul_overflow<int, int, int>(int_min, 0, 0, false);
  test_mul_overflow<int, int, int>(0, int_max, 0, false);
  test_mul_overflow<int, int, int>(-1, int_min, int_min, true);
  test_mul_overflow<int, int, int>(int_min, -1, int_min, true);
  test_mul_overflow<int, int, int>(int_min, int_min, 0, true);
  test_mul_overflow<unsigned int, int, int>(2, int_max, static_cast<uint32_t>(int_max) * 2u, false);
  test_mul_overflow<unsigned int, unsigned int, unsigned int>(uint_max, 2ull, uint_max - 1ull, true);

  // 2. Explicit wider Result type
  test_mul_overflow<long long, int, int>(int_min, -1, int64_t{int_min} * (-1ll), false);
  test_mul_overflow<int64_t, int32_t, int32_t>(
    int_min, int_min, static_cast<int64_t>(int_min) * static_cast<int64_t>(int_min), false);
  test_mul_overflow<int64_t, int32_t, int32_t>(
    static_cast<int32_t>(uint_max), static_cast<int32_t>(uint_max), static_cast<int64_t>(uint_max * uint_max), false);
  test_mul_overflow<uint64_t, uint32_t, uint32_t>(
    uint_max, uint_max, static_cast<uint64_t>(uint_max) * static_cast<uint64_t>(uint_max), false);
  test_mul_overflow<int64_t, uint32_t, uint32_t>(uint_max, uint_max, -8589934591, true);

  // 3. Both operands negative, large magnitude (non-overflow and overflow)
  test_mul_overflow<int, int, int>(-40000, -50000, 2000000000, false);
  test_mul_overflow<int, int, int>(-50000, -50000, -1794967296, true);

  // 4. Overflow from downcasting
  test_mul_overflow<int8_t, int32_t, int32_t>(1000, 1000, static_cast<int8_t>(64), true);
  test_mul_overflow<int8_t, signed char, unsigned long>(
    static_cast<signed char>(17), static_cast<unsigned long>(14), static_cast<int8_t>(-18), true);

#if _CCCL_HAS_INT128()
  // 5. __uint128_t
  constexpr auto uint128_max = cuda::std::numeric_limits<__uint128_t>::max();

  test_mul_overflow<__uint128_t, __uint128_t, __uint128_t>(3, 4, 12, false);
  test_mul_overflow<__uint128_t, __uint128_t, __uint128_t>(
    static_cast<__uint128_t>(~0ull), // 2^64 - 1
    static_cast<__uint128_t>(1ull) << 63, // 2^63
    (static_cast<__uint128_t>(0x7fffffffffffffffULL) << 64) | static_cast<__uint128_t>(0x8000000000000000ULL),
    false);
  test_mul_overflow<__uint128_t, __uint128_t, __uint128_t>(
    static_cast<__uint128_t>(1) << 100, static_cast<__uint128_t>(1) << 100, static_cast<__uint128_t>(0), true);
  test_mul_overflow<__uint128_t, __uint128_t, __uint128_t>(
    uint128_max, static_cast<__uint128_t>(2), uint128_max - 1, true);
  test_mul_overflow<__uint128_t, __uint128_t, __uint128_t>(
    uint128_max, static_cast<__uint128_t>(0), static_cast<__uint128_t>(0), false);
  test_mul_overflow<__uint128_t, unsigned long long, __uint128_t>(
    ~0ull,
    static_cast<__uint128_t>(5) << 100,
    (static_cast<__uint128_t>(0xffffffb000000000ULL) << 64) | static_cast<__uint128_t>(0),
    true);
#endif // _CCCL_HAS_INT128()
}

using TypeList = cuda::std::tuple<
  signed char,
  unsigned char,
  short,
  unsigned short,
  int,
  unsigned int,
  long,
  unsigned long,
  long long,
  unsigned long long
#if _CCCL_HAS_INT128()
  ,
  __int128_t,
  __uint128_t
#endif // _CCCL_HAS_INT128()
  >;

using TypeListIndexSeq = cuda::std::make_index_sequence<cuda::std::tuple_size_v<TypeList>>;

template <class... Ts, cuda::std::size_t... Is>
TEST_FUNC constexpr void test_exhaustive(cuda::std::index_sequence<Is...>)
{
  if constexpr (sizeof...(Ts) < 3)
  {
    (test_exhaustive<Ts..., cuda::std::tuple_element_t<Is, TypeList>>(TypeListIndexSeq{}), ...);
  }
  else
  {
    test_type<Ts...>();
    static_assert(test_type<Ts...>());
  }
}

int main(int arg, char** argv)
{
  test_exhaustive(TypeListIndexSeq{});
  test_corner_cases();
  return 0;
}
