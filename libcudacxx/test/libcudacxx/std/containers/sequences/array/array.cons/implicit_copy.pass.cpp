//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// implicitly generated array constructors / assignment operators

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

#if defined(TEST_COMPILER_MSVC_2017)
#  define TEST_CONSTEXPR_CXX14_NOT_MSVC_2017
#else // ^^^ TEST_COMPILER_MSVC_2017 ^^^ / vvv !TEST_COMPILER_MSVC_2017
#  define TEST_CONSTEXPR_CXX14_NOT_MSVC_2017 TEST_CONSTEXPR_CXX14
#endif // !TEST_COMPILER_MSVC_2017

struct NoDefault
{
  __host__ __device__ TEST_CONSTEXPR_CXX14 NoDefault(int) {}
};

struct NonTrivialCopy
{
  __host__ __device__ TEST_CONSTEXPR_CXX14 NonTrivialCopy() {}
  __host__ __device__ TEST_CONSTEXPR_CXX14 NonTrivialCopy(NonTrivialCopy const&) {}
  __host__ __device__ TEST_CONSTEXPR_CXX14 NonTrivialCopy& operator=(NonTrivialCopy const&)
  {
    return *this;
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX14_NOT_MSVC_2017 bool tests()
{
  {
    typedef cuda::std::array<double, 3> Array;
    Array array = {1.1, 2.2, 3.3};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    typedef cuda::std::array<double const, 3> Array;
    Array array = {1.1, 2.2, 3.3};
    Array copy  = array;
    unused(copy);
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(!cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    typedef cuda::std::array<double, 0> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    // const arrays of size 0 should disable the implicit copy assignment operator.
    typedef cuda::std::array<double const, 0> Array;
    Array array = {};
    Array copy  = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(!cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    typedef cuda::std::array<NoDefault, 0> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    typedef cuda::std::array<NoDefault const, 0> Array;
    Array array = {};
    Array copy  = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(!cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }

  // Make sure we can implicitly copy a cuda::std::array of a non-trivially copyable type
  {
    typedef cuda::std::array<NonTrivialCopy, 0> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    unused(copy);
  }

// NVCC believes `copy = array` accesses uninitialized memory
#if defined(TEST_COMPILER_NVCC) || defined(TEST_COMPILER_NVRTC)
  if (!TEST_IS_CONSTANT_EVALUATED())
#endif // TEST_COMPILER_NVCC
  {
    typedef cuda::std::array<NonTrivialCopy, 1> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    unused(copy);
  }
// NVCC believes `copy = array` accesses uninitialized memory
#if defined(TEST_COMPILER_NVCC) || defined(TEST_COMPILER_NVRTC)
  if (!TEST_IS_CONSTANT_EVALUATED())
#endif // TEST_COMPILER_NVCC
  {
    typedef cuda::std::array<NonTrivialCopy, 2> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    unused(copy);
  }

  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER >= 2014 && defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED) && !defined(TEST_COMPILER_MSVC_2017)
  static_assert(tests(), "");
#endif
  return 0;
}
