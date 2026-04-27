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

// [simd.unary], basic_vec unary operators
//
// constexpr basic_vec& operator++() noexcept;
// constexpr basic_vec  operator++(int) noexcept;
// constexpr basic_vec& operator--() noexcept;
// constexpr basic_vec  operator--(int) noexcept;
// constexpr mask_type  operator!() const noexcept;
// constexpr basic_vec  operator~() const noexcept;
// constexpr basic_vec  operator+() const noexcept;
// constexpr basic_vec  operator-() const noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4146) // unary minus operator applied to unsigned type, result still unsigned

//----------------------------------------------------------------------------------------------------------------------
// operator++ (pre)

template <typename T, int N>
TEST_FUNC constexpr void test_pre_increment()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(T{5});
  static_assert(cuda::std::is_same_v<decltype(++vec), Vec&>);
  static_assert(noexcept(++vec));

  Vec& ref = ++vec;
  assert(&ref == &vec);
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == T{6});
  }
}

//----------------------------------------------------------------------------------------------------------------------
// operator++ (post)

template <typename T, int N>
TEST_FUNC constexpr void test_post_increment()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(T{5});
  static_assert(cuda::std::is_same_v<decltype(vec++), Vec>);
  static_assert(noexcept(vec++));

  Vec old = vec++;
  for (int i = 0; i < N; ++i)
  {
    assert(old[i] == T{5});
    assert(vec[i] == T{6});
  }
}

//----------------------------------------------------------------------------------------------------------------------
// operator-- (pre)

template <typename T, int N>
TEST_FUNC constexpr void test_pre_decrement()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(T{5});
  static_assert(cuda::std::is_same_v<decltype(--vec), Vec&>);
  static_assert(noexcept(--vec));

  Vec& ref = --vec;
  assert(&ref == &vec);
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == T{4});
  }
}

//----------------------------------------------------------------------------------------------------------------------
// operator-- (post)

template <typename T, int N>
TEST_FUNC constexpr void test_post_decrement()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(T{5});
  static_assert(cuda::std::is_same_v<decltype(vec--), Vec>);
  static_assert(noexcept(vec--));

  Vec old = vec--;
  for (int i = 0; i < N; ++i)
  {
    assert(old[i] == T{5});
    assert(vec[i] == T{4});
  }
}

//----------------------------------------------------------------------------------------------------------------------
// operator!

template <typename T, int N>
TEST_FUNC constexpr void test_logical_not()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec    = make_iota_vec<T, N>();
  static_assert(cuda::std::is_same_v<decltype(!vec), Mask>);
  static_assert(noexcept(!vec));

  Mask result = !vec;
  for (int i = 0; i < N; ++i)
  {
    assert(result[i] == (static_cast<T>(i) == T{0}));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// operator~

template <typename T, int N>
TEST_FUNC constexpr void test_bitwise_not()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(T{0});
  static_assert(cuda::std::is_same_v<decltype(~vec), Vec>);
  static_assert(noexcept(~vec));

  Vec result = ~vec;
  for (int i = 0; i < N; ++i)
  {
    assert(result[i] == static_cast<T>(~T{0}));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// operator+

template <typename T, int N>
TEST_FUNC constexpr void test_unary_plus()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(T{42});
  static_assert(cuda::std::is_same_v<decltype(+vec), Vec>);
  static_assert(noexcept(+vec));

  Vec result = +vec;
  for (int i = 0; i < N; ++i)
  {
    assert(result[i] == T{42});
  }
}

//----------------------------------------------------------------------------------------------------------------------
// operator-

template <typename T, int N>
TEST_FUNC constexpr void test_unary_minus()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(T{3});
  static_assert(cuda::std::is_same_v<decltype(-vec), Vec>);
  static_assert(noexcept(-vec));

  Vec result = -vec;
  for (int i = 0; i < N; ++i)
  {
    assert(result[i] == static_cast<T>(-T{3}));
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_pre_increment<T, N>();
  test_post_increment<T, N>();
  test_pre_decrement<T, N>();
  test_post_decrement<T, N>();
  test_logical_not<T, N>();
  test_unary_plus<T, N>();
  test_unary_minus<T, N>();
  if constexpr (cuda::std::is_integral_v<T>)
  {
    test_bitwise_not<T, N>();
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
