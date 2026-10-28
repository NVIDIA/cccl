//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class B, class E, class In1, class In2, class Ref>
TEST_FUNC constexpr void test_ipow(In1 base, In2 exp, Ref ref)
{
  if (cuda::std::in_range<B>(base) && cuda::std::in_range<E>(exp) && cuda::std::in_range<B>(ref))
  {
    assert(cuda::ipow(static_cast<B>(base), static_cast<E>(exp)) == static_cast<B>(ref));
  }
}

template <class B, class E>
TEST_FUNC constexpr void test_boundaries()
{
  using U             = cuda::std::make_unsigned_t<B>;
  constexpr int width = cuda::std::numeric_limits<U>::digits;
  constexpr U top_bit = U{1} << (width - 1);

  assert(cuda::ipow(static_cast<B>(top_bit), E{1}) == static_cast<B>(top_bit));
  if (cuda::std::in_range<E>(width - 1))
  {
    assert(cuda::ipow(B{2}, static_cast<E>(width - 1)) == static_cast<B>(top_bit));
    if constexpr (cuda::std::is_signed_v<B>)
    {
      assert(cuda::ipow(B{-2}, static_cast<E>(width - 1)) == cuda::std::numeric_limits<B>::min());
    }
  }
  test_ipow<B, E>(2, width, 0);
  test_ipow<B, E>(2, width + 1, 0);
  test_ipow<B, E>(4, (width - 1) / 2, U{1} << (2 * ((width - 1) / 2)));
  test_ipow<B, E>(4, (width - 1) / 2 + 1, 0);

  constexpr E max_exp = cuda::std::numeric_limits<E>::max();
  constexpr U result  = cuda::std::cmp_less(max_exp, width) ? static_cast<U>(U{1} << static_cast<int>(max_exp)) : U{0};
  assert(cuda::ipow(B{2}, max_exp) == static_cast<B>(result));
  assert(cuda::ipow(B{1}, max_exp) == B{1});
  if constexpr (cuda::std::is_signed_v<B>)
  {
    assert(cuda::ipow(B{-1}, max_exp) == B{-1});
    assert(cuda::ipow(B{-1}, static_cast<E>(max_exp - 1)) == B{1});
  }

  test_ipow<B, E>(2, 1ull << 32, 0);
  test_ipow<B, E>(2, (1ull << 32) + 5, 0);
  test_ipow<B, E>(1ull << 62, (1u << 30) + 1, 0);

  // These products must wrap without overflowing a signed promoted type.
  constexpr B max_base = cuda::std::numeric_limits<B>::max();
  assert(cuda::ipow(max_base, E{2}) == B{1});
  assert(cuda::ipow(max_base, E{3}) == max_base);
  assert(cuda::ipow(max_base, E{4}) == B{1});
}

template <class B, class E>
TEST_FUNC constexpr void test_type()
{
  static_assert(cuda::std::is_same_v<decltype(cuda::ipow(B{}, E{})), B>);
  static_assert(noexcept(cuda::ipow(B{}, E{})));

  test_ipow<B, E>(0, 0, 1);
  test_ipow<B, E>(1, 0, 1);
  test_ipow<B, E>(-1, 0, 1);
  test_ipow<B, E>(127, 0, 1);
  test_ipow<B, E>(-127, 0, 1);

  test_ipow<B, E>(1, 1, 1);
  test_ipow<B, E>(1, 2, 1);
  test_ipow<B, E>(1, -1, 1);
  test_ipow<B, E>(1, -2, 1);
  test_ipow<B, E>(-1, 1, -1);
  test_ipow<B, E>(-1, 2, 1);
  test_ipow<B, E>(-1, 3, -1);
  test_ipow<B, E>(-1, -1, 0);
  test_ipow<B, E>(-1, -2, 0);

  test_ipow<B, E>(2, 2, 4);
  test_ipow<B, E>(2, 12, 4'096);
  test_ipow<B, E>(2, -2, 0);
  test_ipow<B, E>(2, -12, 0);

  test_ipow<B, E>(4, 5, 1'024);
  test_ipow<B, E>(4, 7, 16'384);
  test_ipow<B, E>(4, -5, 0);
  test_ipow<B, E>(128, 5, 34'359'738'368);
  test_ipow<B, E>(1024, 3, 1'073'741'824);

  test_ipow<B, E>(17, 7, 410'338'673);
  test_ipow<B, E>(57, 3, 185'193);
  test_ipow<B, E>(57, -127, 0);
  test_ipow<B, E>(101, 3, 1'030'301);
  test_ipow<B, E>(7891, 2, 62'267'881);

  test_boundaries<B, E>();
}

template <class B>
TEST_FUNC constexpr void test_type()
{
  test_type<B, signed char>();
  test_type<B, signed short>();
  test_type<B, signed int>();
  test_type<B, signed long>();
  test_type<B, signed long long>();
#if _CCCL_HAS_INT128()
  test_type<B, __int128_t>();
#endif // _CCCL_HAS_INT128()

  test_type<B, unsigned char>();
  test_type<B, unsigned short>();
  test_type<B, unsigned int>();
  test_type<B, unsigned long>();
  test_type<B, unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<B, __uint128_t>();
#endif // _CCCL_HAS_INT128()
}

TEST_FUNC constexpr bool test()
{
  test_type<signed char>();
  test_type<signed short>();
  test_type<signed int>();
  test_type<signed long>();
  test_type<signed long long>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_type<unsigned char>();
  test_type<unsigned short>();
  test_type<unsigned int>();
  test_type<unsigned long>();
  test_type<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
