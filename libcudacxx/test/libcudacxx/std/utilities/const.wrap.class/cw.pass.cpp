//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
// UNSUPPORTED: nvrtc

// REQUIRES: !c++17

// constant_wrapper

//   template<auto X>
//    constexpr auto cw = constant_wrapper<X>{};

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/utility>

#include "test_macros.h"

struct S
{
  int value;

  TEST_FUNC constexpr S(int v)
      : value(v)
  {}

  TEST_FUNC constexpr bool operator==(const S& other) const
  {
    return value == other.value;
  }
};

_CCCL_GLOBAL_CONSTANT int arr[3]{1, 2, 3};

TEST_FUNC constexpr bool test()
{
  {
    // int constant
    cuda::std::same_as<const cuda::std::__constant_wrapper<42>> decltype(auto) cw_val = cuda::std::__cw<42>;
    static_assert(cw_val == 42);
  }

  {
    // gcc < 13 fails this test with error:
    //   invalid use of non-static data member ‘S::value’
#if !_CCCL_COMPILER(GCC, <, 13)
    // struct constant
    constexpr S s{13};
    cuda::std::same_as<const cuda::std::__constant_wrapper<s>> decltype(auto) cw_val = cuda::std::__cw<s>;
    static_assert(cw_val == s);
#endif // !_CCCL_COMPILER(GCC, <, 13)
  }

  {
    // array constant
    // gcc complains that cw_val is unused
    [[maybe_unused]] cuda::std::same_as<const cuda::std::__constant_wrapper<arr>> decltype(auto) cw_val =
      cuda::std::__cw<arr>;
    static_assert(cw_val[0] == 1);
    static_assert(cw_val[1] == 2);
    static_assert(cw_val[2] == 3);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
