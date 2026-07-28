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

// [simd.bit], shl

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../simd_test_utils.h"

template <typename T>
struct shift_input_values_gen
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I) const
  {
    if constexpr (cuda::std::is_signed_v<T>)
    {
      return I::value % 2 == 0 ? T{-7} : T{5};
    }
    else
    {
      return I::value % 2 == 0 ? T{7} : T{5};
    }
  }
};

template <typename T>
struct shift_count_values_gen
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I) const
  {
    constexpr int digits   = cuda::std::numeric_limits<cuda::std::make_unsigned_t<T>>::digits;
    constexpr int values[] = {-1, 0, digits, digits + 1};
    return static_cast<T>(values[I::value % 4]);
  }
};

template <typename T, int N>
struct test_shl_vec
{
  TEST_FUNC constexpr void operator()() const
  {
    using vec_t     = simd::basic_vec<T, simd::fixed_size<N>>;
    using shift_vec = simd::rebind_t<cuda::std::make_signed_t<T>, vec_t>;

    vec_t vec(shift_input_values_gen<T>{});
    shift_vec shifts(shift_count_values_gen<typename shift_vec::value_type>{});

    static_assert(cuda::std::is_same_v<decltype(simd::shl(vec, shifts)), vec_t>);
    static_assert(noexcept(simd::shl(vec, shifts)));

    vec_t result = simd::shl(vec, shifts);
    for (int i = 0; i < N; ++i)
    {
      assert(result[i] == cuda::std::shl(vec[i], shifts[i]));
    }
  }
};

template <typename T, int N>
struct test_shl_scalar
{
  TEST_FUNC constexpr void operator()() const
  {
    using vec_t = simd::basic_vec<T, simd::fixed_size<N>>;
    vec_t vec(shift_input_values_gen<T>{});

    static_assert(cuda::std::is_same_v<decltype(simd::shl(vec, 1)), vec_t>);
    static_assert(noexcept(simd::shl(vec, 1)));

    constexpr int digits = cuda::std::numeric_limits<cuda::std::make_unsigned_t<T>>::digits;
    const int shifts[]   = {-digits - 1, -1, 0, 1, digits, digits + 1};
    for (int shift : shifts)
    {
      vec_t result = simd::shl(vec, shift);
      for (int i = 0; i < N; ++i)
      {
        assert(result[i] == cuda::std::shl(vec[i], shift));
      }
    }
  }
};

template <typename V0, typename V1, typename = void>
struct has_simd_shl_vec : cuda::std::false_type
{};

template <typename V0, typename V1>
struct has_simd_shl_vec<V0, V1, cuda::std::void_t<decltype(simd::shl(cuda::std::declval<V0>(), cuda::std::declval<V1>()))>>
    : cuda::std::true_type
{};

template <typename V, typename S, typename = void>
struct has_simd_shl_scalar : cuda::std::false_type
{};

template <typename V, typename S>
struct has_simd_shl_scalar<V, S, cuda::std::void_t<decltype(simd::shl(cuda::std::declval<V>(), cuda::std::declval<S>()))>>
    : cuda::std::true_type
{};

TEST_FUNC constexpr void test_constraints()
{
  using int32_vec4  = simd::basic_vec<int32_t, simd::fixed_size<4>>;
  using uint32_vec4 = simd::basic_vec<uint32_t, simd::fixed_size<4>>;
  using int16_vec4  = simd::basic_vec<int16_t, simd::fixed_size<4>>;
  using int32_vec2  = simd::basic_vec<int32_t, simd::fixed_size<2>>;
  using float_vec4  = simd::basic_vec<float, simd::fixed_size<4>>;

  static_assert(has_simd_shl_vec<int32_vec4, uint32_vec4>::value);
  static_assert(!has_simd_shl_vec<int32_vec4, int16_vec4>::value);
  static_assert(!has_simd_shl_vec<int32_vec4, int32_vec2>::value);
  static_assert(!has_simd_shl_vec<int32_vec4, float_vec4>::value);
  static_assert(!has_simd_shl_vec<float_vec4, int32_vec4>::value);

  static_assert(has_simd_shl_scalar<int32_vec4, int>::value);
  static_assert(has_simd_shl_scalar<int32_vec4, unsigned long long>::value);
  static_assert(!has_simd_shl_scalar<int32_vec4, float>::value);
  static_assert(!has_simd_shl_scalar<float_vec4, int>::value);
}

TEST_FUNC constexpr bool test()
{
  _SIMD_BIT_TEST_SIGNED_TYPES(test_shl_vec)
  _SIMD_BIT_TEST_UNSIGNED_TYPES(test_shl_vec)
  _SIMD_BIT_TEST_SIGNED_TYPES(test_shl_scalar)
  _SIMD_BIT_TEST_UNSIGNED_TYPES(test_shl_scalar)
  test_constraints();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
