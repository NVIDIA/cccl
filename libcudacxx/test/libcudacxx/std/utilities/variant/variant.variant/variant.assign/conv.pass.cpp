//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16

// <cuda/std/variant>

// template <class ...Types> class variant;

// template <class T>
// variant& operator=(T&&) noexcept(see below);

#include <cuda/std/variant>
// #include <cuda/std/string>
// #include <cuda/std/memory>

#include "variant_test_helpers.h"

int main(int, char**)
{
  static_assert(!cuda::std::is_assignable<cuda::std::variant<int, int>, int>::value, "");
  static_assert(!cuda::std::is_assignable<cuda::std::variant<long, long long>, int>::value, "");
  static_assert(cuda::std::is_assignable<cuda::std::variant<char>, int>::value == VariantAllowsNarrowingConversions, "");

  // static_assert(cuda::std::is_assignable<cuda::std::variant<cuda::std::string, float>, int>::value == VariantAllowsNarrowingConversions, "");
  // static_assert(cuda::std::is_assignable<cuda::std::variant<cuda::std::string, double>, int>::value == VariantAllowsNarrowingConversions, "");
  // static_assert(!cuda::std::is_assignable<cuda::std::variant<cuda::std::string, bool>, int>::value, "");

  static_assert(!cuda::std::is_assignable<cuda::std::variant<int, bool>, decltype("meow")>::value, "");
  static_assert(!cuda::std::is_assignable<cuda::std::variant<int, const bool>, decltype("meow")>::value, "");
  static_assert(!cuda::std::is_assignable<cuda::std::variant<int, const volatile bool>, decltype("meow")>::value, "");

  static_assert(!cuda::std::is_assignable<cuda::std::variant<bool>, cuda::std::true_type>::value, "");
  // static_assert(!cuda::std::is_assignable<cuda::std::variant<bool>, cuda::std::unique_ptr<char> >::value, "");
  static_assert(!cuda::std::is_assignable<cuda::std::variant<bool>, decltype(nullptr)>::value, "");

  return 0;
}
