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

// [simd.bit], rotl

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../simd_test_utils.h"

template <typename T, int N>
struct test_rotl_vec
{
  TEST_FUNC constexpr void operator()() const
  {
    using Vec      = simd::basic_vec<T, simd::fixed_size<N>>;
    using ShiftVec = simd::rebind_t<cuda::std::make_signed_t<T>, Vec>;

    Vec vec(bit_values<T>{});
    ShiftVec shifts(iota_generator<typename ShiftVec::value_type>{});

    static_assert(cuda::std::is_same_v<decltype(simd::rotl(vec, shifts)), Vec>);
    static_assert(noexcept(simd::rotl(vec, shifts)));

    Vec result = simd::rotl(vec, shifts);
    for (int i = 0; i < N; ++i)
    {
      assert(result[i] == cuda::std::rotl(vec[i], static_cast<int>(shifts[i])));
    }
  }
};

template <typename T, int N>
struct test_rotl_scalar
{
  TEST_FUNC constexpr void operator()() const
  {
    using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
    Vec vec(bit_values<T>{});

    static_assert(cuda::std::is_same_v<decltype(simd::rotl(vec, 1)), Vec>);
    static_assert(noexcept(simd::rotl(vec, 1)));

    Vec result = simd::rotl(vec, -1);
    for (int i = 0; i < N; ++i)
    {
      assert(result[i] == cuda::std::rotl(vec[i], -1));
    }
  }
};

template <typename V0, typename V1, typename = void>
struct has_simd_rotl_vec : cuda::std::false_type
{};

template <typename V0, typename V1>
struct has_simd_rotl_vec<V0,
                         V1,
                         cuda::std::void_t<decltype(simd::rotl(cuda::std::declval<V0>(), cuda::std::declval<V1>()))>>
    : cuda::std::true_type
{};

template <typename V, typename = void>
struct has_simd_rotl_scalar : cuda::std::false_type
{};

template <typename V>
struct has_simd_rotl_scalar<V, cuda::std::void_t<decltype(simd::rotl(cuda::std::declval<V>(), 1))>>
    : cuda::std::true_type
{};

TEST_FUNC constexpr void test_vec_constraints()
{
  using IntVec     = simd::basic_vec<int, simd::fixed_size<4>>;
  using Uint16Vec  = simd::basic_vec<uint16_t, simd::fixed_size<4>>;
  using Uint32Vec  = simd::basic_vec<unsigned, simd::fixed_size<4>>;
  using FloatVec2  = simd::basic_vec<float, simd::fixed_size<2>>;
  using Int32Vec2  = simd::basic_vec<int, simd::fixed_size<2>>;
  using Shift32Vec = simd::basic_vec<int, simd::fixed_size<4>>;

  static_assert(has_simd_rotl_vec<Uint32Vec, Shift32Vec>::value);
  static_assert(!has_simd_rotl_vec<IntVec, Shift32Vec>::value);
  static_assert(!has_simd_rotl_vec<Uint32Vec, Uint16Vec>::value);
  static_assert(!has_simd_rotl_vec<Uint32Vec, Int32Vec2>::value);
  static_assert(!has_simd_rotl_vec<Uint32Vec, FloatVec2>::value);
  static_assert(!has_simd_rotl_vec<FloatVec2, Int32Vec2>::value);
}

TEST_FUNC constexpr void test_scalar_constraints()
{
  using IntVec    = simd::basic_vec<int, simd::fixed_size<4>>;
  using Uint32Vec = simd::basic_vec<unsigned, simd::fixed_size<4>>;
  using FloatVec2 = simd::basic_vec<float, simd::fixed_size<2>>;

  static_assert(has_simd_rotl_scalar<Uint32Vec>::value);
  static_assert(!has_simd_rotl_scalar<IntVec>::value);
  static_assert(!has_simd_rotl_scalar<FloatVec2>::value);
}

TEST_FUNC constexpr bool test()
{
  _SIMD_BIT_TEST_UNSIGNED_TYPES(test_rotl_vec)
  _SIMD_BIT_TEST_UNSIGNED_TYPES(test_rotl_scalar)
  test_vec_constraints();
  test_scalar_constraints();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
