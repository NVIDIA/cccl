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

// constexpr operator decltype(value)() const noexcept { return value; }

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(20094) // a host member cannot be directly read in a __device__/__global__ function

struct S
{
  int value;

  TEST_FUNC constexpr S(int v)
      : value(v)
  {}
};

TEST_FUNC constexpr void f1(const S&) {}

TEST_FUNC constexpr bool test()
{
  {
    // int conversion
    cuda::std::__constant_wrapper<6> cw6;
    int result = cw6;
    assert(result == 6);

    static_assert(noexcept(static_cast<int>(cw6)));
  }

  {
    // struct conversion
    constexpr S s{42};
    cuda::std::__constant_wrapper<s> cws;
    const S& result = cws;
    assert(result.value == 42);
    assert(&result == &cws.value);

    static_assert(noexcept(static_cast<const S&>(cws)));
  }

  {
    // gcc < 13 fails this test with:
    //   ‘test()::<lambda(int)>::_FUN’ is not a valid template argument of type ‘int (*)(int)’ because it is not
    //   a variable
#if !_CCCL_COMPILER(GCC, <, 13)
    // function pointer conversion
    constexpr int (*fptr)(int) = [](int x) constexpr {
      return x * 2;
    };
    cuda::std::__constant_wrapper<fptr> cwFptr;
    int (*result)(int) = cwFptr;
    assert(result(5) == 10);

    // nvcc 13.3 fails to produce correct input file for host compiler. NVHPC fails, too. See nvbug 6249821.
#  if _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) || _CCCL_COMPILER(NVHPC)
    static_assert(noexcept(static_cast<int (*)(int)>(decltype(cwFptr)::value)));
#  else // ^^^ _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) vvv
    static_assert(noexcept(static_cast<int (*)(int)>(cwFptr)));
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^
#endif // !_CCCL_COMPILER(GCC, <, 13)
  }

// nvcc < 13.3 fails to generate correct input file for host compiler.
#if !(TEST_CUDA_COMPILER(NVCC, <, 13, 3) && _CCCL_HOST_COMPILATION())
  {
    // conversion is implicit
    cuda::std::__constant_wrapper<S{42}> cws;
    f1(cws);
  }
#endif // !(TEST_CUDA_COMPILER(NVCC, <, 13, 3) && _CCCL_HOST_COMPILATION())

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
