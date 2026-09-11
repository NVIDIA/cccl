//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo(dabayer): nvrtc doesn't support non-trivial types as static data members without -default-device, fails with:
//   A class static data member with non-const type is considered a host variable, and host variables are not allowed in
//   JIT mode. Consider using -default-device flag to process such data members as __device__ variables in JIT mode

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
    decltype(auto) cw_val = cuda::std::__cw<42>;
    static_assert(cuda::std::same_as<const cuda::std::__constant_wrapper<42>, decltype(cw_val)>);
    static_assert(cw_val == 42);
  }

#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)
  {
    // gcc < 13 fails this test with error:
    //   invalid use of non-static data member 'S::value'
#  if !_CCCL_COMPILER(GCC, <, 13)
    // struct constant
    constexpr S s{13};
    cuda::std::same_as<const cuda::std::__constant_wrapper<s>> decltype(auto) cw_val = cuda::std::__cw<s>;
    static_assert(cw_val == s);
#  endif // !_CCCL_COMPILER(GCC, <, 13)
  }
#endif // TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

  {
    // array constant
    // gcc complains that cw_val is unused
    [[maybe_unused]] decltype(auto) cw_val = cuda::std::__cw<arr>;
    static_assert(cuda::std::same_as<const cuda::std::__constant_wrapper<arr>, decltype(cw_val)>);
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
