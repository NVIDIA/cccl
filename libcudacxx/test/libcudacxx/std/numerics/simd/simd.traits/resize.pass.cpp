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

// template<simd-size-type N, class V> struct resize;
// template<simd-size-type N, class V> using resize_t = typename resize<N, V>::type;

#include <cuda/std/__simd_>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace simd = cuda::std::simd;

//----------------------------------------------------------------------------------------------------------------------
// resize with basic_vec

template <typename T, int OldN, int NewN>
TEST_FUNC void test_resize_vec()
{
  using OldVec   = simd::basic_vec<T, simd::fixed_size<OldN>>;
  using Result   = simd::resize_t<NewN, OldVec>;
  using Expected = simd::basic_vec<T, simd::fixed_size<NewN>>;
  static_assert(cuda::std::is_same_v<typename Result::value_type, T>);
  static_assert(Result::size() == NewN);
  static_assert(cuda::std::is_same_v<Result, Expected>);
}

template <typename T>
TEST_FUNC void test_resize_vec_all()
{
  test_resize_vec<T, 4, 4>();
  test_resize_vec<T, 4, 2>();
  test_resize_vec<T, 2, 8>();
}

//----------------------------------------------------------------------------------------------------------------------
// resize with basic_mask

template <typename T, int OldN, int NewN>
TEST_FUNC void test_resize_mask()
{
  using OldMask  = simd::basic_mask<sizeof(T), simd::fixed_size<OldN>>;
  using Result   = simd::resize_t<NewN, OldMask>;
  using Expected = simd::basic_mask<sizeof(T), simd::fixed_size<NewN>>;
  static_assert(Result::size() == NewN);
  static_assert(cuda::std::is_same_v<Result, Expected>);
}

template <typename T>
TEST_FUNC void test_resize_mask_all()
{
  test_resize_mask<T, 4, 4>();
  test_resize_mask<T, 4, 2>();
  test_resize_mask<T, 2, 8>();
}

//----------------------------------------------------------------------------------------------------------------------
// resize_t matches resize::type

template <int N, typename V>
TEST_FUNC void test_resize_t_alias()
{
  static_assert(cuda::std::is_same_v<simd::resize_t<N, V>, typename simd::resize<N, V>::type>);
}

TEST_FUNC void test()
{
  // resize basic_vec
  test_resize_vec_all<char>();
  test_resize_vec_all<short>();
  test_resize_vec_all<int>();
  test_resize_vec_all<long long>();
  test_resize_vec_all<float>();
  test_resize_vec_all<double>();

  // resize basic_mask
  test_resize_mask_all<char>();
  test_resize_mask_all<int>();
  test_resize_mask_all<double>();

  // resize_t alias matches resize::type
  test_resize_t_alias<8, simd::vec<int, 4>>();
  test_resize_t_alias<2, simd::vec<float, 8>>();
}

int main(int, char**)
{
  test();
  return 0;
}
