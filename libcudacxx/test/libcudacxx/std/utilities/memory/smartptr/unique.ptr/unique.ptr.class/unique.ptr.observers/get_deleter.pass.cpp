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

// test get_deleter()

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct Deleter
{
  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter() {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(void*) const {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 int test()
  {
    return 5;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 int test() const
  {
    return 6;
  }
};

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_basic()
{
  typedef typename cuda::std::conditional<IsArray, int[], int>::type VT;
  {
    cuda::std::unique_ptr<int, Deleter> p;
    assert(p.get_deleter().test() == 5);
  }
  {
    const cuda::std::unique_ptr<VT, Deleter> p;
    assert(p.get_deleter().test() == 6);
  }
  {
    typedef cuda::std::unique_ptr<VT, const Deleter&> UPtr;
    const Deleter d;
    UPtr p(nullptr, d);
    const UPtr& cp = p;
    ASSERT_SAME_TYPE(decltype(p.get_deleter()), const Deleter&);
    ASSERT_SAME_TYPE(decltype(cp.get_deleter()), const Deleter&);
    assert(p.get_deleter().test() == 6);
    assert(cp.get_deleter().test() == 6);
  }
  {
    typedef cuda::std::unique_ptr<VT, Deleter&> UPtr;
    Deleter d;
    UPtr p(nullptr, d);
    const UPtr& cp = p;
    ASSERT_SAME_TYPE(decltype(p.get_deleter()), Deleter&);
    ASSERT_SAME_TYPE(decltype(cp.get_deleter()), Deleter&);
    assert(p.get_deleter().test() == 5);
    assert(cp.get_deleter().test() == 5);
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
