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

// [simd.math], abs, fabs

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T>
struct signed_values
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I i) const noexcept
  {
    return static_cast<T>(static_cast<int>(i) - 2);
  }
};

// floating point types
template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(math_values<T>{});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::abs(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fabs(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::abs(vec)), Vec>);

  static_assert(noexcept(cuda::std::simd::abs(vec)));
  static_assert(noexcept(cuda::std::simd::fabs(vec)));

  Vec abs_result  = cuda::std::simd::abs(vec);
  Vec fabs_result = cuda::std::simd::fabs(vec);
  for (int i = 0; i < N; ++i)
  {
    assert(abs_result[i] == cuda::std::abs(vec[i]));
    assert(fabs_result[i] == cuda::std::fabs(vec[i]));
  }
}

template <typename V, typename = void>
inline constexpr bool has_simd_fabs = false;

template <typename V>
inline constexpr bool has_simd_fabs<V, cuda::std::void_t<decltype(cuda::std::simd::fabs(cuda::std::declval<V>()))>> =
  true;

template <typename T, int N>
TEST_FUNC void test_signed_integral_type()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(signed_values<T>{});

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::abs(vec)), Vec>);
  static_assert(noexcept(cuda::std::simd::abs(vec)));
  static_assert(!has_simd_fabs<Vec>); // check that fabs is not supported for signed integral types

  Vec result = cuda::std::simd::abs(vec);
  for (int i = 0; i < N; ++i)
  {
    assert(result[i] == (vec[i] < 0 ? -vec[i] : vec[i]));
  }
}

TEST_FUNC void test_signed_integral()
{
  test_signed_integral_type<int8_t, 4>();
  test_signed_integral_type<int16_t, 4>();
  test_signed_integral_type<int32_t, 4>();
  test_signed_integral_type<int64_t, 4>();
#if _CCCL_HAS_INT128()
  test_signed_integral_type<__int128_t, 4>();
#endif // _CCCL_HAS_INT128()
}

DEFINE_SIMD_MATH_FLOATING_TEST()
DEFINE_SIMD_MATH_FLOATING_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  assert(test_runtime());
  test_signed_integral();
  return 0;
}
