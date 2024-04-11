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

// void reset() noexcept;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

struct X
{
  STATIC_MEMBER_VAR(dtor_called, bool)
  __host__ __device__ ~X()
  {
    dtor_called() = true;
  }
};

__host__ __device__ constexpr bool check_reset()
{
  {
    optional<int> opt;
    static_assert(noexcept(opt.reset()) == true, "");
    opt.reset();
    assert(static_cast<bool>(opt) == false);
  }
  {
    optional<int> opt(3);
    opt.reset();
    assert(static_cast<bool>(opt) == false);
  }
  return true;
}

int main(int, char**)
{
  check_reset();
#if TEST_STD_VER >= 2020
  static_assert(check_reset());
#endif
  {
    optional<X> opt;
    static_assert(noexcept(opt.reset()) == true, "");
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

  return 0;
}
