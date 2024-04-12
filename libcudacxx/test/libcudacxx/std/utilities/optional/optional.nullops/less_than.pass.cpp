//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// template <class T> constexpr bool operator<(const optional<T>& x, nullopt_t) noexcept;
// template <class T> constexpr bool operator<(nullopt_t, const optional<T>& x) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/optional>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using cuda::std::nullopt;
  using cuda::std::nullopt_t;
  using cuda::std::optional;

  {
    typedef int T;
    typedef optional<T> O;

    O o1; // disengaged
    O o2{1}; // engaged

    assert(!(nullopt < o1));
    assert((nullopt < o2));
    assert(!(o1 < nullopt));
    assert(!(o2 < nullopt));

    static_assert(noexcept(nullopt < o1), "");
    static_assert(noexcept(o1 < nullopt), "");
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2017
#  if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  static_assert(test());
#  endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
#endif

  return 0;
}
