//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr const T& value() const &;
// constexpr T& value() &;
// constexpr T&& value() &&;
// constexpr const T&& value() const &&;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test() {
  // non-const &
  {
    cuda::std::expected<int, int> e(5);
    decltype(auto) x = e.value();
    static_assert(cuda::std::same_as<decltype(x), int&>, "");
    assert(&x == &(*e));
    assert(x == 5);
  }

  // const &
  {
    const cuda::std::expected<int, int> e(5);
    decltype(auto) x = e.value();
    static_assert(cuda::std::same_as<decltype(x), const int&>, "");
    assert(&x == &(*e));
    assert(x == 5);
  }

  // non-const &&
  {
    cuda::std::expected<int, int> e(5);
    decltype(auto) x = cuda::std::move(e).value();
    static_assert(cuda::std::same_as<decltype(x), int&&>, "");
    assert(&x == &(*e));
    assert(x == 5);
  }

  // const &&
  {
    const cuda::std::expected<int, int> e(5);
    decltype(auto) x = cuda::std::move(e).value();
    static_assert(cuda::std::same_as<decltype(x), const int&&>, "");
    assert(&x == &(*e));
    assert(x == 5);
  }

  return true;
}

__host__ __device__ void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS

  // int
  {
    const cuda::std::expected<int, int> e(cuda::std::unexpect, 5);
    try {
      (void) e.value();
      assert(false);
    } catch (const cuda::std::bad_expected_access<int>& ex) {
      assert(ex.error() == 5);
    }
  }

  // MoveOnly
  {
    cuda::std::expected<int, MoveOnly> e(cuda::std::unexpect, 5);
    try {
      (void) cuda::std::move(e).value();
      assert(false);
    } catch (const cuda::std::bad_expected_access<MoveOnly>& ex) {
      assert(ex.error() == 5);
    }
  }

#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 17 && defined(_LIBCUDACXX_ADDRESSOF)
  testException();
  return 0;
}
