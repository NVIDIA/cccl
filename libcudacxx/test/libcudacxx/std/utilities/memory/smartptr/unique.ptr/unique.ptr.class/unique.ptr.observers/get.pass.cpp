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

// test get

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "unique_ptr_test_helper.h"

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_basic()
{
  typedef typename cuda::std::conditional<IsArray, int[], int>::type VT;
  typedef const VT CVT;
  {
    typedef cuda::std::unique_ptr<VT> U;
    int* p = newValue<VT>(1);
    U s(p);
    U const& sc = s;
    ASSERT_SAME_TYPE(decltype(s.get()), int*);
    ASSERT_SAME_TYPE(decltype(sc.get()), int*);
    assert(s.get() == p);
    assert(sc.get() == s.get());
  }
  {
    typedef cuda::std::unique_ptr<CVT> U;
    const int* p = newValue<VT>(1);
    U s(p);
    U const& sc = s;
    ASSERT_SAME_TYPE(decltype(s.get()), const int*);
    ASSERT_SAME_TYPE(decltype(sc.get()), const int*);
    assert(s.get() == p);
    assert(sc.get() == s.get());
  }
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
