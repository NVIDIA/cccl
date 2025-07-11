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
  STATIC_MEMBER_VAR(dtor_called, bool)
  __host__ __device__ ~X()
  {
    dtor_called() = true;
  }
};

template <class T>
__host__ __device__ constexpr void test()
{
  using O = optional<T>;
  cuda::std::remove_reference_t<T> one{1};
  {
    O opt;
    static_assert(noexcept(opt.reset()) == true, "");
    opt.reset();
    assert(static_cast<bool>(opt) == false);
  }
  {
    O opt(one);
    opt.reset();
    assert(static_cast<bool>(opt) == false);
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<int&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020
  {
    optional<X> opt{};
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
