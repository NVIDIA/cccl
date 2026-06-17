//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_FUNC constexpr bool test_always_true()
{
  // Default-constructible
  cuda::always_true at{};

  // Returns true with no arguments
  assert(at() == true);

  // Returns true with a single argument
  assert(at(42) == true);

  // Returns true with multiple arguments of different types
  assert(at(1, 2.0, 'c') == true);

  // Return type is bool
  static_assert(cuda::std::is_same_v<decltype(at()), bool>, "Return type must be bool");
  static_assert(cuda::std::is_same_v<decltype(at(1, 2)), bool>, "Return type must be bool");

  // operator() is noexcept
  static_assert(noexcept(at()), "operator() must be noexcept");
  static_assert(noexcept(at(1, 2, 3)), "operator() must be noexcept");

  // operator() is const-qualified
  const cuda::always_true cat{};
  assert(cat() == true);
  assert(cat(1) == true);

  // constexpr evaluation
  static_assert(cuda::always_true{}() == true, "Must be constexpr");
  static_assert(cuda::always_true{}(1, 2, 3) == true, "Must be constexpr");

  // Type properties
  static_assert(cuda::std::is_empty_v<cuda::always_true>, "always_true must be empty");
  static_assert(cuda::std::is_trivially_copyable_v<cuda::always_true>, "always_true must be trivially copyable");

  return true;
}

TEST_FUNC constexpr bool test_always_false()
{
  // Default-constructible
  cuda::always_false af{};

  // Returns false with no arguments
  assert(af() == false);

  // Returns false with a single argument
  assert(af(42) == false);

  // Returns false with multiple arguments of different types
  assert(af(1, 2.0, 'c') == false);

  // Return type is bool
  static_assert(cuda::std::is_same_v<decltype(af()), bool>, "Return type must be bool");
  static_assert(cuda::std::is_same_v<decltype(af(1, 2)), bool>, "Return type must be bool");

  // operator() is noexcept
  static_assert(noexcept(af()), "operator() must be noexcept");
  static_assert(noexcept(af(1, 2, 3)), "operator() must be noexcept");

  // operator() is const-qualified
  const cuda::always_false caf{};
  assert(caf() == false);
  assert(caf(1) == false);

  // constexpr evaluation
  static_assert(cuda::always_false{}() == false, "Must be constexpr");
  static_assert(cuda::always_false{}(1, 2, 3) == false, "Must be constexpr");

  // Type properties
  static_assert(cuda::std::is_empty_v<cuda::always_false>, "always_false must be empty");
  static_assert(cuda::std::is_trivially_copyable_v<cuda::always_false>, "always_false must be trivially copyable");

  return true;
}

int main(int, char**)
{
  // Types are distinct
  static_assert(!cuda::std::is_same_v<cuda::always_true, cuda::always_false>, "Types must be distinct");

  assert(test_always_true());
  static_assert(test_always_true());

  assert(test_always_false());
  static_assert(test_always_false());

  return 0;
}
