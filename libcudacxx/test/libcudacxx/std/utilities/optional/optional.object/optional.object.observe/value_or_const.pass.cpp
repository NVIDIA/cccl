//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// template <class U> constexpr T optional<T>::value_or(U&& v) const&;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

struct Y
{
  int i_;

  TEST_FUNC constexpr Y(int i)
      : i_(i)
  {}
};

struct X
{
  int i_;

  TEST_FUNC constexpr X(int i)
      : i_(i)
  {}
  TEST_FUNC constexpr X(const Y& y)
      : i_(y.i_)
  {}
  TEST_FUNC constexpr X(Y&& y)
      : i_(y.i_ + 1)
  {}
  TEST_FUNC friend constexpr bool operator==(const X& x, const X& y)
  {
    return x.i_ == y.i_;
  }
};

TEST_FUNC constexpr bool test()
{
  {
    const optional<X> opt(2);
    const Y y(3);
    assert(opt.value_or(y) == 2);
  }
  {
    const optional<X> opt(2);
    assert(opt.value_or(Y(3)) == 2);
  }
  {
    const optional<X> opt{};
    const Y y(3);
    assert(opt.value_or(y) == 3);
  }
  {
    const optional<X> opt{};
    assert(opt.value_or(Y(3)) == 4);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const optional<X>>().value_or(Y(3))), X>);
  }

  X val{2};
  {
    const optional<X&> opt(val);
    const Y y(3);
    assert(opt.value_or(y) == 2);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const optional<X&>>().value_or(y)), X>);
  }
  {
    const optional<X&> opt(val);
    assert(opt.value_or(Y(3)) == 2);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const optional<X&>>().value_or(Y(3))), X>);
  }
  {
    const optional<X&> opt{};
    const Y y(3);
    assert(opt.value_or(y) == 3);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const optional<X>>().value_or(y)), X>);
  }
  {
    const optional<X&> opt{};
    assert(opt.value_or(Y(3)) == 4);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const optional<X&>>().value_or(Y(3))), X>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
