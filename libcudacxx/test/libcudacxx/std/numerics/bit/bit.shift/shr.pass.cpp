//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class T, class S>
// constexpr T shr(T x, S s) noexcept;

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/climits>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/numeric>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T, class S>
TEST_FUNC constexpr T invoke_shr(T v, S shift)
{
  if (cuda::std::__cccl_default_is_constant_evaluated())
  {
    return cuda::std::shr(v, shift);
  }
  else
  {
    DoNotOptimize(v);
    DoNotOptimize(shift);
    return cuda::std::shr(v, shift);
  }
}

template <class T, class S>
TEST_FUNC constexpr void test_pos(T v, S shift)
{
  assert(cuda::std::cmp_greater_equal(shift, 0));

  auto result = invoke_shr(v, shift);
  if (cuda::std::cmp_less(shift, (sizeof(T) * CHAR_BIT)))
  {
    assert(result == static_cast<T>(v >> shift));
  }
  else
  {
    assert(result == T(cuda::std::cmp_less(v, 0) ? -1 : 0));
  }
}

template <class T, class S>
TEST_FUNC constexpr void test_neg(T v, S shift)
{
  assert(shift < 0);

  // There is a bug on sm100+ when cuda::std::shr returns invalid result when the shift is int64_t min. Re-enable
  // once nvbug 6471375 is resolved.
  NV_IF_TARGET(NV_PROVIDES_SM_100, ({
                 if constexpr (sizeof(T) == sizeof(cuda::std::int64_t) && sizeof(S) == sizeof(cuda::std::int64_t))
                 {
                   if (shift == cuda::std::numeric_limits<S>::min())
                   {
                     return;
                   }
                 }
               }))

  auto result = invoke_shr(v, shift);
  if (cuda::std::cmp_less(cuda::uabs(shift), sizeof(T) * CHAR_BIT))
  {
    using U = cuda::std::make_unsigned_t<T>;
    assert(result == static_cast<T>(static_cast<U>(v) << cuda::uabs(shift)));
  }
  else
  {
    assert(result == T{0});
  }
}

template <class T, class S>
TEST_FUNC constexpr void test()
{
  constexpr auto tmin = cuda::std::numeric_limits<T>::min();
  constexpr auto tmax = cuda::std::numeric_limits<T>::max();
  constexpr auto smax = cuda::std::numeric_limits<S>::max();

  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::shr(T{}, S{}))>);
  static_assert(noexcept(cuda::std::shr(T{}, S{})));

  const T vs[]         = {tmin, T(-24), T(-1), T{0}, T{1}, T{20}, T{99}, static_cast<T>(1225), tmax};
  const S pos_shifts[] = {S{0}, S{1}, S{7}, S{17}, S{23}, S{32}, S{33}, S{65}, smax};

  // Disable loop unrolling to reduce ptxas compile times.
  _CCCL_PRAGMA_NOUNROLL()
  for (auto v : vs)
  {
    _CCCL_PRAGMA_NOUNROLL()
    for (auto shift : pos_shifts)
    {
      test_pos(v, shift);
    }
  }

  if constexpr (cuda::std::is_signed_v<S>)
  {
    constexpr auto smin = cuda::std::numeric_limits<S>::min();

    const S neg_shifts[] = {smin, S{-65}, S{-33}, S{-32}, S{-15}, S{-7}, S{-1}};

    _CCCL_PRAGMA_NOUNROLL()
    for (auto v : vs)
    {
      _CCCL_PRAGMA_NOUNROLL()
      for (auto shift : neg_shifts)
      {
        test_neg(v, shift);
      }
    }
  }
}

template <class T>
TEST_FUNC constexpr void test()
{
  test<T, signed char>();
  test<T, signed short>();
  test<T, signed>();
  test<T, signed long>();
  test<T, signed long long>();
#if _CCCL_HAS_INT128()
  test<T, __int128_t>();
#endif // _CCCL_HAS_INT128()

  test<T, unsigned char>();
  test<T, unsigned short>();
  test<T, unsigned>();
  test<T, unsigned long>();
  test<T, unsigned long long>();
#if _CCCL_HAS_INT128()
  test<T, __uint128_t>();
#endif // _CCCL_HAS_INT128()
}

TEST_FUNC constexpr bool test()
{
  test<signed char>();
  test<signed short>();
  test<signed>();
  test<signed long>();
  test<signed long long>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test<unsigned char>();
  test<unsigned short>();
  test<unsigned>();
  test<unsigned long>();
  test<unsigned long long>();
#if _CCCL_HAS_INT128()
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
