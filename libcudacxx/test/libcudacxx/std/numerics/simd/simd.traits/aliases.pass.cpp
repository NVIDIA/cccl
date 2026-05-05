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

// template<class T, simd-size-type N> using vec  = ...;
// template<class T, simd-size-type N> using mask = ...;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// vec<T, N> resolves to basic_vec<T, deduce-abi-t<T, N>>

template <typename T, int N>
TEST_FUNC constexpr void test_vec_alias()
{
  using Alias  = simd::vec<T, N>;
  using Direct = simd::basic_vec<T, simd::fixed_size<N>>;
  static_assert(cuda::std::is_same_v<Alias, Direct>);
}

//----------------------------------------------------------------------------------------------------------------------
// mask<T, N> resolves to basic_mask<sizeof(T), deduce-abi-t<T, N>>

template <typename T, int N>
TEST_FUNC constexpr void test_mask_alias()
{
  using Alias  = simd::mask<T, N>;
  using Direct = simd::basic_mask<sizeof(T), simd::fixed_size<N>>;
  static_assert(cuda::std::is_same_v<Alias, Direct>);
}

//----------------------------------------------------------------------------------------------------------------------
// default N for vec and mask uses native ABI size

template <typename T>
TEST_FUNC constexpr void test_default_size()
{
  using DefaultVec = simd::vec<T>;
  using NativeVec  = simd::basic_vec<T, simd::native<T>>;
  static_assert(cuda::std::is_same_v<DefaultVec, NativeVec>);

  using DefaultMask = simd::mask<T>;
  using NativeMask  = simd::basic_mask<sizeof(T), simd::native<T>>;
  static_assert(cuda::std::is_same_v<DefaultMask, NativeMask>);
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_vec_alias<T, N>();
  test_mask_alias<T, N>();
  test_default_size<T>();
}

DEFINE_BASIC_VEC_TEST()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  static_assert(test());
  assert(test_runtime());
  return 0;
}
