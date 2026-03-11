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

template <class Res, class L, class R, class InL, class InR, class Ref>
__host__ __device__ constexpr void test_mul_overflow(InL lhs_in, InR rhs_in, Ref expected, bool overflow)
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
__host__ __device__ constexpr void test_mul_overflow(InL lhs_in, InR rhs_in, Ref expected)
{
  test_mul_overflow<Res, L, R, InL, InR, Ref>(lhs_in, rhs_in, expected, !cuda::std::in_range<Res>(expected));
}

template <class Res, class L, class R>
__host__ __device__ constexpr bool test_type()
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

using TypeList = cuda::std::tuple<
  signed char,
  unsigned char,
  short,
  unsigned short,
  int,
  unsigned long,
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
__host__ __device__ constexpr void test(cuda::std::index_sequence<Is...>)
{
  if constexpr (sizeof...(Ts) < 3)
  {
    (test<Ts..., cuda::std::tuple_element_t<Is, TypeList>>(TypeListIndexSeq{}), ...);
  }
  else
  {
    test_type<Ts...>();
    static_assert(test_type<Ts...>());
  }
}

int main(int arg, char** argv)
{
  test(TypeListIndexSeq{});
  return 0;
}
