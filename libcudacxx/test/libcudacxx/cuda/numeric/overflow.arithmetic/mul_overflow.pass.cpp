//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/numeric>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

template <class Res, class L, class R, class InL, class InR, class Ref>
__host__ __device__ constexpr void test_mul_overflow(InL lhs_in, InR rhs_in, Ref expected)
{
  const auto lhs      = cuda::overflow_cast<L>(lhs_in);
  const auto rhs      = cuda::overflow_cast<R>(rhs_in);
  const bool overflow = !cuda::std::in_range<Res>(expected);

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

template <class Res, class L, class R>
__host__ __device__ constexpr void test_type()
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
    test_mul_overflow<Res, L, R>(
      cuda::std::numeric_limits<L>::min(), -1, cuda::uabs(cuda::std::numeric_limits<L>::min()));
    test_mul_overflow<Res, L, R>(
      -1, cuda::std::numeric_limits<R>::min(), cuda::uabs(cuda::std::numeric_limits<R>::min()));
  }

  // 4. Test other numbers
  test_mul_overflow<Res, L, R>(17, 14, 238);
  test_mul_overflow<Res, L, R>(-254, 127, -32258);
  test_mul_overflow<Res, L, R>(1657, -13748, -22780436);
  test_mul_overflow<Res, L, R>(-2147483647, 4294967295, -9223372030412324865);
}

template <class Res, class L>
__host__ __device__ constexpr void test_type()
{
  test_type<Res, L, signed char>();
  test_type<Res, L, unsigned char>();
  test_type<Res, L, short>();
  test_type<Res, L, unsigned short>();
  test_type<Res, L, int>();
  test_type<Res, L, unsigned int>();
  test_type<Res, L, long>();
  test_type<Res, L, unsigned long>();
  test_type<Res, L, long long>();
  test_type<Res, L, unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<Res, L, __int128_t>();
  test_type<Res, L, __uint128_t>();
#endif // _CCCL_HAS_INT128()
}

template <class Res>
__host__ __device__ constexpr void test_type()
{
  test_type<Res, signed char>();
  test_type<Res, unsigned char>();
  test_type<Res, short>();
  test_type<Res, unsigned short>();
  test_type<Res, int>();
  test_type<Res, unsigned int>();
  test_type<Res, long>();
  test_type<Res, unsigned long>();
  test_type<Res, long long>();
  test_type<Res, unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<Res, __int128_t>();
  test_type<Res, __uint128_t>();
#endif // _CCCL_HAS_INT128()
}

__host__ __device__ constexpr bool test()
{
  test_type<signed char>();
  test_type<unsigned char>();
  test_type<short>();
  test_type<unsigned short>();
  test_type<int>();
  test_type<unsigned int>();
  test_type<long>();
  test_type<unsigned long>();
  test_type<long long>();
  test_type<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
  test_type<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int arg, char** argv)
{
  test();
  // static_assert(test());
  return 0;
}
