//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// <memory>

// unique_ptr

// Test unique_ptr::pointer type

#include <cuda/std/__memory_>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct Deleter
{
  struct pointer
  {};
};

#if !defined(TEST_COMPILER_GCC) && !defined(TEST_COMPILER_MSVC)
struct D2
{
private:
  typedef void pointer;
};
#endif // !TEST_COMPILER_GCC && !TEST_COMPILER_MSVC

#ifndef TEST_COMPILER_NVRTC // A class static data member with non-const type is considered a host variable
struct D3
{
  static long pointer;
};
#endif // !TEST_COMPILER_NVRTC

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_basic()
{
  typedef typename cuda::std::conditional<IsArray, int[], int>::type VT;
  {
    typedef cuda::std::unique_ptr<VT> P;
    static_assert((cuda::std::is_same<typename P::pointer, int*>::value), "");
  }
  {
    typedef cuda::std::unique_ptr<VT, Deleter> P;
    static_assert((cuda::std::is_same<typename P::pointer, Deleter::pointer>::value), "");
  }
#if !defined(TEST_COMPILER_GCC) && !defined(TEST_COMPILER_MSVC)
  {
    typedef cuda::std::unique_ptr<VT, D2> P;
    static_assert(cuda::std::is_same<typename P::pointer, int*>::value, "");
  }
#endif // !TEST_COMPILER_GCC && !TEST_COMPILER_MSVC
#ifndef TEST_COMPILER_NVRTC
  {
    typedef cuda::std::unique_ptr<VT, D3> P;
    static_assert(cuda::std::is_same<typename P::pointer, int*>::value, "");
  }
#endif // !TEST_COMPILER_NVRTC
}

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  test_basic</*IsArray*/ false>();
  test_basic<true>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}
