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

// [simd.cassign], basic_vec compound assignment
//
// friend constexpr basic_vec& operator+=(basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec& operator-=(basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec& operator*=(basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec& operator/=(basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec& operator%=(basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec& operator&=(basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec& operator|=(basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec& operator^=(basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec& operator<<=(basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec& operator>>=(basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec& operator<<=(basic_vec&, simd-size-type) noexcept;
// friend constexpr basic_vec& operator>>=(basic_vec&, simd-size-type) noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <typename T, int N>
TEST_FUNC constexpr void test_arithmetic_assign()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec b(T{3});

  // operator+=
  {
    Vec a(T{6});
    static_assert(cuda::std::is_same_v<decltype(a += b), Vec&>);
    static_assert(noexcept(a += b));
    a += b;
    for (int i = 0; i < N; ++i)
    {
      assert(a[i] == static_cast<T>(T{6} + T{3}));
    }
  }
  // operator-=
  {
    Vec a(T{6});
    static_assert(cuda::std::is_same_v<decltype(a -= b), Vec&>);
    static_assert(noexcept(a -= b));
    a -= b;
    for (int i = 0; i < N; ++i)
    {
      assert(a[i] == static_cast<T>(T{6} - T{3}));
    }
  }
  // operator*=
  {
    Vec a(T{6});
    static_assert(cuda::std::is_same_v<decltype(a *= b), Vec&>);
    static_assert(noexcept(a *= b));
    a *= b;
    for (int i = 0; i < N; ++i)
    {
      assert(a[i] == static_cast<T>(T{6} * T{3}));
    }
  }
  // operator/=
  {
    Vec a(T{6});
    static_assert(cuda::std::is_same_v<decltype(a /= b), Vec&>);
    static_assert(noexcept(a /= b));
    a /= b;
    for (int i = 0; i < N; ++i)
    {
      assert(a[i] == static_cast<T>(T{6} / T{3}));
    }
  }
}

template <typename T, int N>
TEST_FUNC constexpr void test_integral_assign()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec b(T{3});

  // operator%=
  {
    Vec a(T{7});
    static_assert(cuda::std::is_same_v<decltype(a %= b), Vec&>);
    static_assert(noexcept(a %= b));
    a %= b;
    for (int i = 0; i < N; ++i)
    {
      assert(a[i] == static_cast<T>(T{7} % T{3}));
    }
  }
  // operator&=
  {
    Vec a(T{7});
    static_assert(cuda::std::is_same_v<decltype(a &= b), Vec&>);
    static_assert(noexcept(a &= b));
    a &= b;
    for (int i = 0; i < N; ++i)
    {
      assert(a[i] == static_cast<T>(T{7} & T{3}));
    }
  }
  // operator|=
  {
    Vec a(T{7});
    static_assert(cuda::std::is_same_v<decltype(a |= b), Vec&>);
    static_assert(noexcept(a |= b));
    a |= b;
    for (int i = 0; i < N; ++i)
    {
      assert(a[i] == static_cast<T>(T{7} | T{3}));
    }
  }
  // operator^=
  {
    Vec a(T{7});
    static_assert(cuda::std::is_same_v<decltype(a ^= b), Vec&>);
    static_assert(noexcept(a ^= b));
    a ^= b;
    for (int i = 0; i < N; ++i)
    {
      assert(a[i] == static_cast<T>(T{7} ^ T{3}));
    }
  }
}

template <typename T, int N>
TEST_FUNC constexpr void test_shift_assign()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec shift(T{1});

  // operator<<=
  {
    Vec a(T{4});
    static_assert(cuda::std::is_same_v<decltype(a <<= shift), Vec&>);
    static_assert(noexcept(a <<= shift));
    a <<= shift;
    for (int i = 0; i < N; ++i)
    {
      assert(a[i] == static_cast<T>(T{4} << T{1}));
    }
  }
  // operator>>=
  {
    Vec a(T{4});
    static_assert(cuda::std::is_same_v<decltype(a >>= shift), Vec&>);
    static_assert(noexcept(a >>= shift));
    a >>= shift;
    for (int i = 0; i < N; ++i)
    {
      assert(a[i] == static_cast<T>(T{4} >> T{1}));
    }
  }
  // operator<<=
  {
    Vec a(T{4});
    static_assert(cuda::std::is_same_v<decltype(a <<= 1), Vec&>);
    static_assert(noexcept(a <<= 1));
    a <<= 1;
    for (int i = 0; i < N; ++i)
    {
      assert(a[i] == static_cast<T>(T{4} << T{1}));
    }
  }
  // operator>>=
  {
    Vec a(T{4});
    static_assert(cuda::std::is_same_v<decltype(a >>= 1), Vec&>);
    static_assert(noexcept(a >>= 1));
    a >>= 1;
    for (int i = 0; i < N; ++i)
    {
      assert(a[i] == static_cast<T>(T{4} >> T{1}));
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_arithmetic_assign<T, N>();
  if constexpr (cuda::std::is_integral_v<T>)
  {
    test_integral_assign<T, N>();
    test_shift_assign<T, N>();
  }
}

DEFINE_BASIC_VEC_TEST()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  static_assert(test());
  assert(test_runtime());
  return 0;
}
