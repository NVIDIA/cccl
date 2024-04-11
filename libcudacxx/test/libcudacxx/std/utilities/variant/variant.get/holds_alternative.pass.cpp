//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class T, class... Types>
// constexpr bool holds_alternative(const variant<Types...>& v) noexcept;

#include <cuda/std/variant>

#include "test_macros.h"

int main(int, char**)
{
  {
    using V = cuda::std::variant<int>;
    constexpr V v;
    static_assert(cuda::std::holds_alternative<int>(v), "");
  }
  {
    using V = cuda::std::variant<int, long>;
    constexpr V v;
    static_assert(cuda::std::holds_alternative<int>(v), "");
    static_assert(!cuda::std::holds_alternative<long>(v), "");
  }
  { // noexcept test
    using V = cuda::std::variant<int>;
    const V v;
    ASSERT_NOEXCEPT(cuda::std::holds_alternative<int>(v));
  }

  return 0;
}
