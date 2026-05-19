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

// [simd.bit], byteswap

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../simd_test_utils.h"

template <typename T, int N>
struct test_byteswap
{
  TEST_FUNC constexpr void operator()() const
  {
    using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
    Vec vec(bit_values<T>{});

    static_assert(cuda::std::is_same_v<decltype(simd::byteswap(vec)), Vec>);
    static_assert(noexcept(simd::byteswap(vec)));

    Vec result = simd::byteswap(vec);
    for (int i = 0; i < N; ++i)
    {
      assert(result[i] == cuda::std::byteswap(vec[i]));
    }
  }
};

template <typename V, typename = void>
struct has_simd_byteswap : cuda::std::false_type
{};

template <typename V>
struct has_simd_byteswap<V, cuda::std::void_t<decltype(simd::byteswap(cuda::std::declval<V>()))>>
    : cuda::std::true_type
{};

TEST_FUNC constexpr void test_constraints()
{
  using IntVec   = simd::basic_vec<int, simd::fixed_size<4>>;
  using FloatVec = simd::basic_vec<float, simd::fixed_size<4>>;

  static_assert(has_simd_byteswap<IntVec>::value);
  static_assert(!has_simd_byteswap<FloatVec>::value);
}

DEFINE_SIMD_BIT_INTEGRAL_TEST(test_byteswap)

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
