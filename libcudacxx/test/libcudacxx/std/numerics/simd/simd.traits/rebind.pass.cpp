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

// template<class T, class V> struct rebind;
// template<class T, class V> using rebind_t = typename rebind<T, V>::type;

#include <cuda/std/__simd_>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace simd = cuda::std::simd;

//----------------------------------------------------------------------------------------------------------------------
// rebind with basic_vec

template <typename NewT, typename OldT, int N>
TEST_FUNC void test_rebind_vec()
{
  using OldVec = simd::basic_vec<OldT, simd::fixed_size<N>>;
  using Result = simd::rebind_t<NewT, OldVec>;
  static_assert(cuda::std::is_same_v<typename Result::value_type, NewT>);
  static_assert(Result::size() == N);
}

template <typename NewT, typename OldT>
TEST_FUNC void test_rebind_vec_sizes()
{
  test_rebind_vec<NewT, OldT, 1>();
  test_rebind_vec<NewT, OldT, 2>();
  test_rebind_vec<NewT, OldT, 4>();
  test_rebind_vec<NewT, OldT, 8>();
}

//----------------------------------------------------------------------------------------------------------------------
// rebind with basic_mask

template <typename NewT, typename OldT, int N>
TEST_FUNC void test_rebind_mask()
{
  using OldMask = simd::basic_mask<sizeof(OldT), simd::fixed_size<N>>;
  using Result  = simd::rebind_t<NewT, OldMask>;
  static_assert(cuda::std::is_same_v<Result, simd::basic_mask<sizeof(NewT), simd::fixed_size<N>>>);
  static_assert(Result::size() == N);
}

template <typename NewT, typename OldT>
TEST_FUNC void test_rebind_mask_sizes()
{
  test_rebind_mask<NewT, OldT, 1>();
  test_rebind_mask<NewT, OldT, 2>();
  test_rebind_mask<NewT, OldT, 4>();
  test_rebind_mask<NewT, OldT, 8>();
}

//----------------------------------------------------------------------------------------------------------------------
// rebind_t matches rebind::type

template <typename NewT, typename V>
TEST_FUNC void test_rebind_t_alias()
{
  static_assert(cuda::std::is_same_v<simd::rebind_t<NewT, V>, typename simd::rebind<NewT, V>::type>);
}

TEST_FUNC void test()
{
  // rebind basic_vec
  test_rebind_vec_sizes<int, int>();
  test_rebind_vec_sizes<float, float>();
  test_rebind_vec_sizes<float, int>();
  test_rebind_vec_sizes<int, float>();

  // rebind basic_vec: different sizes
  test_rebind_vec_sizes<char, int>();
  test_rebind_vec_sizes<int, char>();
  test_rebind_vec_sizes<short, long long>();
  test_rebind_vec_sizes<double, int>();
  test_rebind_vec_sizes<int, double>();

  // rebind basic_mask
  test_rebind_mask_sizes<int, int>();
  test_rebind_mask_sizes<float, float>();
  test_rebind_mask_sizes<char, int>();
  test_rebind_mask_sizes<double, int>();
  test_rebind_mask_sizes<short, long long>();

  // rebind_t alias matches rebind::type
  test_rebind_t_alias<float, simd::vec<int, 4>>();
  test_rebind_t_alias<int, simd::vec<double, 2>>();
  test_rebind_t_alias<double, simd::mask<int, 4>>();
}

int main(int, char**)
{
  test();
  return 0;
}
