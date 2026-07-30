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

// [simd.bit], bit_reverse

#include <cuda/bit>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../simd_test_utils.h"

template <typename T, int N>
struct test_bit_reverse
{
  TEST_FUNC constexpr void operator()() const
  {
    using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

    Vec vec(bit_values<T>{});
    static_assert(cuda::std::is_same_v<decltype(simd::bit_reverse(vec)), Vec>);
    static_assert(noexcept(simd::bit_reverse(vec)));

    Vec result = simd::bit_reverse(vec);
    for (int i = 0; i < N; ++i)
    {
      assert(result[i] == cuda::bit_reverse(vec[i]));
    }

    constexpr T all_bits = cuda::std::numeric_limits<T>::max();
    assert(simd::bit_reverse(Vec{T{0}})[0] == T{0});
    assert(simd::bit_reverse(Vec{T{1}})[0] == static_cast<T>(T{1} << (cuda::std::numeric_limits<T>::digits - 1)));
    assert(simd::bit_reverse(Vec{all_bits})[0] == all_bits);
  }
};

template <typename V, typename = void>
struct has_simd_bit_reverse : cuda::std::false_type
{};

template <typename V>
struct has_simd_bit_reverse<V, cuda::std::void_t<decltype(simd::bit_reverse(cuda::std::declval<V>()))>>
    : cuda::std::true_type
{};

TEST_FUNC constexpr void test_constraints()
{
  using int_vec   = simd::basic_vec<int, simd::fixed_size<4>>;
  using uint_vec  = simd::basic_vec<unsigned, simd::fixed_size<4>>;
  using float_vec = simd::basic_vec<float, simd::fixed_size<4>>;

  static_assert(has_simd_bit_reverse<uint_vec>::value);
  static_assert(!has_simd_bit_reverse<int_vec>::value);
  static_assert(!has_simd_bit_reverse<float_vec>::value);
}

DEFINE_SIMD_BIT_UNSIGNED_TEST(test_bit_reverse)

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
