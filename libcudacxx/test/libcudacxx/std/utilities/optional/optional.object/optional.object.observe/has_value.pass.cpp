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

// constexpr bool optional<T>::has_value() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using cuda::std::optional;
  {
    const optional<int> opt;
    ((void) opt);
    ASSERT_NOEXCEPT(opt.has_value());
    ASSERT_SAME_TYPE(decltype(opt.has_value()), bool);
  }
  {
    optional<int> opt;
    assert(!opt.has_value());
  }
  {
    optional<int> opt(0);
    assert(opt.has_value());
  }
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    constexpr optional<int> opt;
    static_assert(!opt.has_value(), "");
  }
  {
    constexpr optional<int> opt(0);
    static_assert(opt.has_value(), "");
  }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))

  return 0;
}
