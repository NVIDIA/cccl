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

// [simd.binary], basic_vec binary operators
//
// friend constexpr basic_vec operator+(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec operator-(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec operator*(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec operator/(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec operator%(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec operator&(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec operator|(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec operator^(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec operator<<(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec operator>>(const basic_vec&, const basic_vec&) noexcept;
// friend constexpr basic_vec operator<<(const basic_vec&, simd-size-type) noexcept;
// friend constexpr basic_vec operator>>(const basic_vec&, simd-size-type) noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <typename T, int N>
TEST_FUNC constexpr void test_arithmetic()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec a(T{6});
  Vec b(T{3});

  static_assert(cuda::std::is_same_v<decltype(a + b), Vec>);
  static_assert(cuda::std::is_same_v<decltype(a - b), Vec>);
  static_assert(cuda::std::is_same_v<decltype(a * b), Vec>);
  static_assert(cuda::std::is_same_v<decltype(a / b), Vec>);
  static_assert(noexcept(a + b));
  static_assert(noexcept(a - b));
  static_assert(noexcept(a * b));
  static_assert(noexcept(a / b));

  Vec sum  = a + b;
  Vec diff = a - b;
  Vec prod = a * b;
  Vec quot = a / b;
  for (int i = 0; i < N; ++i)
  {
    assert(sum[i] == static_cast<T>(T{6} + T{3}));
    assert(diff[i] == static_cast<T>(T{6} - T{3}));
    assert(prod[i] == static_cast<T>(T{6} * T{3}));
    assert(quot[i] == static_cast<T>(T{6} / T{3}));
  }
}

template <typename T, int N>
TEST_FUNC constexpr void test_integral_ops()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec a(T{7});
  Vec b(T{3});

  static_assert(cuda::std::is_same_v<decltype(a % b), Vec>);
  static_assert(cuda::std::is_same_v<decltype(a & b), Vec>);
  static_assert(cuda::std::is_same_v<decltype(a | b), Vec>);
  static_assert(cuda::std::is_same_v<decltype(a ^ b), Vec>);
  static_assert(noexcept(a % b));
  static_assert(noexcept(a & b));
  static_assert(noexcept(a | b));
  static_assert(noexcept(a ^ b));

  Vec mod     = a % b;
  Vec bit_and = a & b;
  Vec bit_or  = a | b;
  Vec bit_xor = a ^ b;
  for (int i = 0; i < N; ++i)
  {
    assert(mod[i] == static_cast<T>(T{7} % T{3}));
    assert(bit_and[i] == static_cast<T>(T{7} & T{3}));
    assert(bit_or[i] == static_cast<T>(T{7} | T{3}));
    assert(bit_xor[i] == static_cast<T>(T{7} ^ T{3}));
  }
}

template <typename T, int N>
TEST_FUNC constexpr void test_shifts()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec a(T{4});
  Vec shift_amount(T{1});

  static_assert(cuda::std::is_same_v<decltype(a << shift_amount), Vec>);
  static_assert(cuda::std::is_same_v<decltype(a >> shift_amount), Vec>);
  static_assert(noexcept(a << shift_amount));
  static_assert(noexcept(a >> shift_amount));

  Vec shl = a << shift_amount;
  Vec shr = a >> shift_amount;
  for (int i = 0; i < N; ++i)
  {
    assert(shl[i] == static_cast<T>(T{4} << T{1}));
    assert(shr[i] == static_cast<T>(T{4} >> T{1}));
  }

  static_assert(cuda::std::is_same_v<decltype(a << 1), Vec>);
  static_assert(cuda::std::is_same_v<decltype(a >> 1), Vec>);
  static_assert(noexcept(a << 1));
  static_assert(noexcept(a >> 1));

  Vec shl_n = a << 1;
  Vec shr_n = a >> 1;
  for (int i = 0; i < N; ++i)
  {
    assert(shl_n[i] == static_cast<T>(T{4} << T{1}));
    assert(shr_n[i] == static_cast<T>(T{4} >> T{1}));
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_arithmetic<T, N>();
  if constexpr (cuda::std::is_integral_v<T>)
  {
    test_integral_ops<T, N>();
    test_shifts<T, N>();
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
