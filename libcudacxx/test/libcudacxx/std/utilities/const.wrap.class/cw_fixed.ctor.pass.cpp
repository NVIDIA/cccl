//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// gcc-10 segfaults with any use of constant_wrapper, gcc-11 fails to evaluate:
//   typename decltype(__cw_fixed_value(_Xp))::type
// UNSUPPORTED: gcc-10 || gcc-11

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
// UNSUPPORTED: nvrtc

// REQUIRES: !c++17

// constant_wrapper

// constexpr cw-fixed-value(type v) noexcept : data(v) {}

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(20094) // a host member cannot be directly read in a __device__/__global__ function
TEST_DIAG_SUPPRESS_NVHPC(expr_has_no_effect)
TEST_DIAG_SUPPRESS_CLANG("-Wunused-value")

template <class T>
using cw_fixed_value = typename cuda::std::__constant_wrapper<T{}>::__cw_fixed_value_type;

struct S
{
  int value;

  TEST_FUNC constexpr S(int v = 0)
      : value(v)
  {}

  TEST_FUNC constexpr bool operator==(const S& other) const
  {
    return value == other.value;
  }
};

TEST_FUNC constexpr bool test()
{
  {
    // int construction
    // the conversion from int to cw-fixed-value<int> uses the constructor
    [[maybe_unused]] cuda::std::__constant_wrapper<42> cw{};
    assert(cw.value == 42);
  }

  {
    // struct construction
    [[maybe_unused]] cuda::std::__constant_wrapper<S{13}> cw{};
    assert(cw.value == S{13});
  }

  {
    // calling the constructor
    [[maybe_unused]] constexpr cw_fixed_value<int> ci{42};
    cuda::std::__constant_wrapper<ci> cw;
    assert(cw == 42);

    static_assert(noexcept(cw_fixed_value<int>{42}));
  }

  {
    // the constructor is implicit
    [[maybe_unused]] constexpr cw_fixed_value<int> ci = 42;
    cuda::std::__constant_wrapper<ci> cw;
    assert(cw == 42);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
