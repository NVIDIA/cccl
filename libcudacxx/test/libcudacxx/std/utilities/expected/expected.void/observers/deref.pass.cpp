//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr void operator*() const & noexcept;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

// Test noexcept
template <class T, class = void>
constexpr bool DerefNoexcept = false;

template <class T>
constexpr bool DerefNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<T>().operator*())>> =
  noexcept(cuda::std::declval<T>().operator*());

static_assert(!DerefNoexcept<int>, "");

static_assert(DerefNoexcept<cuda::std::expected<void, int>>, "");

__host__ __device__ constexpr bool test()
{
  const cuda::std::expected<void, int> e;
  *e;
  static_assert(cuda::std::is_same_v<decltype(*e), void>, "");

  return true;
}

int main(int, char**)
{
  test();
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  static_assert(test(), "");
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  return 0;
}
