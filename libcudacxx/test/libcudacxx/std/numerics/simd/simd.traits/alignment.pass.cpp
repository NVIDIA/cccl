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

// template<class T, class U = typename T::value_type>
// struct alignment;
//
// template<class T, class U = typename T::value_type>
// constexpr size_t alignment_v = alignment<T, U>::value;

#include <cuda/std/__simd_>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <typename T, int N, size_t ExpectedAlign = alignof(T) * N>
TEST_FUNC void test_default_u()
{
  using V = simd::basic_vec<T, simd::fixed_size<N>>;
  static_assert(simd::alignment<V>::value == ExpectedAlign);
  static_assert(simd::alignment_v<V> == ExpectedAlign);
}

template <typename T, int N, typename U, size_t ExpectedAlign = alignof(U) * N>
TEST_FUNC void test_explicit_u()
{
  using V = simd::basic_vec<T, simd::fixed_size<N>>;
  static_assert(simd::alignment<V, U>::value == ExpectedAlign);
  static_assert(simd::alignment_v<V, U> == ExpectedAlign);
}

template <typename T>
TEST_FUNC void test_type()
{
  test_default_u<T, 1>();
  test_default_u<T, 3, alignof(T)>();
  test_default_u<T, 2>();
  test_default_u<T, 4>();
  test_default_u<T, 8>();
}

TEST_FUNC void test()
{
  // default U = value_type
  test_type<char>();
  test_type<short>();
  test_type<int>();
  test_type<long long>();
  test_type<float>();
  test_type<double>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
#endif // _CCCL_HAS_INT128()
#if _LIBCUDACXX_HAS_NVFP16()
  test_type<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_type<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  // explicit U different from value_type
  test_explicit_u<int, 1, float>();
  test_explicit_u<int, 3, float, alignof(float)>();
  test_explicit_u<int, 4, char>();
  test_explicit_u<float, 2, double>();
  test_explicit_u<double, 4, int>();
}

int main(int, char**)
{
  test();
  return 0;
}
