//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// void reset() noexcept;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

struct X
{
#if !_CCCL_TILE_COMPILATION() // error: a non-__tile__ variable cannot be used in tile code
  STATIC_MEMBER_VAR(dtor_called, bool)
  TEST_FUNC ~X()
  {
    dtor_called() = true;
  }
#endif // !_CCCL_TILE_COMPILATION()
};

template <class T>
TEST_FUNC constexpr void test()
{
  using O = optional<T>;
  cuda::std::remove_reference_t<T> one{1};
  {
    O opt;
    static_assert(noexcept(opt.reset()) == true);
    opt.reset();
    assert(static_cast<bool>(opt) == false);
  }
  {
    O opt(one);
    opt.reset();
    assert(static_cast<bool>(opt) == false);
  }
}

TEST_FUNC constexpr bool test()
{
  test<int>();
  test<int&>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020

#if !_CCCL_TILE_COMPILATION() // error: a non-__tile__ variable cannot be used in tile code
  {
    optional<X> opt{};
    static_assert(noexcept(opt.reset()) == true);
    assert(X::dtor_called() == false);
    opt.reset();
    assert(X::dtor_called() == false);
    assert(static_cast<bool>(opt) == false);
  }
  {
    optional<X> opt(X{});
    X::dtor_called() = false;
    opt.reset();
    assert(X::dtor_called() == true);
    assert(static_cast<bool>(opt) == false);
    X::dtor_called() = false;
  }
#endif // !_CCCL_TILE_COMPILATION()

  return 0;
}
