//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class Err = E>
//   constexpr explicit unexpected(Err&& e);
//
// Constraints:
// - is_same_v<remove_cvref_t<Err>, unexpected> is false; and
// - is_same_v<remove_cvref_t<Err>, in_place_t> is false; and
// - is_constructible_v<E, Err> is true.
//
// Effects: Direct-non-list-initializes unex with cuda::std::forward<Err>(e).
// Throws: Any exception thrown by the initialization of unex.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/utility>

#include "test_macros.h"

// Test Constraints:
static_assert(cuda::std::constructible_from<cuda::std::unexpected<int>, int>, "");

// is_same_v<remove_cvref_t<Err>, unexpected>
struct CstrFromUnexpected {
  TEST_HOST_DEVICE CstrFromUnexpected(CstrFromUnexpected const&) = delete;
  TEST_HOST_DEVICE CstrFromUnexpected(cuda::std::unexpected<CstrFromUnexpected> const&);
};
static_assert(!cuda::std::constructible_from<cuda::std::unexpected<CstrFromUnexpected>, cuda::std::unexpected<CstrFromUnexpected>>, "");

// is_same_v<remove_cvref_t<Err>, in_place_t>
struct CstrFromInplace {
  TEST_HOST_DEVICE CstrFromInplace(cuda::std::in_place_t);
};
static_assert(!cuda::std::constructible_from<cuda::std::unexpected<CstrFromInplace>, cuda::std::in_place_t>, "");

// !is_constructible_v<E, Err>
struct Foo {};
static_assert(!cuda::std::constructible_from<cuda::std::unexpected<Foo>, int>, "");

// test explicit
static_assert(cuda::std::convertible_to<int, int>, "");
static_assert(!cuda::std::convertible_to<int, cuda::std::unexpected<int>>, "");

struct Error {
  int i;
  TEST_HOST_DEVICE constexpr Error(int ii) : i(ii) {}
  TEST_HOST_DEVICE constexpr Error(const Error& other) : i(other.i) {}
  TEST_HOST_DEVICE constexpr Error(Error&& other) : i(other.i) { other.i = 0; }
  TEST_HOST_DEVICE Error(cuda::std::initializer_list<Error>) { assert(false); }
};

TEST_HOST_DEVICE constexpr bool test() {
  // lvalue
  {
    Error e(5);
    cuda::std::unexpected<Error> unex(e);
    assert(unex.error().i == 5);
    assert(e.i == 5);
  }

  // rvalue
  {
    Error e(5);
    cuda::std::unexpected<Error> unex(cuda::std::move(e));
    assert(unex.error().i == 5);
    assert(e.i == 0);
  }

  // Direct-non-list-initializes: does not trigger initializer_list overload
  {
    Error e(5);
    cuda::std::unexpected<Error> unex(e);
    unused(e);
  }

  // Test default template argument.
  // Without it, the template parameter cannot be deduced from an initializer list
  {
    struct Bar {
      int i;
      int j;
      TEST_HOST_DEVICE constexpr Bar(int ii, int jj) : i(ii), j(jj) {}
    };
    cuda::std::unexpected<Bar> ue({5, 6});
    assert(ue.error().i == 5);
    assert(ue.error().j == 6);
  }

  return true;
}

TEST_HOST_DEVICE void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Except {};

  struct Throwing {
    Throwing() = default;
    Throwing(const Throwing&) { throw Except{}; }
  };

  Throwing t;
  try {
    cuda::std::unexpected<Throwing> u(t);
    assert(false);
  } catch (Except) {
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test();
  static_assert(test(), "");
  testException();
  return 0;
}
