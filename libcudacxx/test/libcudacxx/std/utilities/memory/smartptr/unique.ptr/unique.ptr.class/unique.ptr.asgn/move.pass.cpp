//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03

// Self assignement post-conditions are tested.
// ADDITIONAL_COMPILE_FLAGS: -Wno-self-move

// <memory>

// unique_ptr

// Test unique_ptr move assignment

// test move assignment.  Should only require a MoveConstructible deleter, or if
//    deleter is a reference, not even that.

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "deleter_types.h"
#include "test_macros.h"
#include "unique_ptr_test_helper.h"

struct GenericDeleter
{
  __host__ __device__ void operator()(void*) const;
};

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_basic()
{
  typedef typename cuda::std::conditional<IsArray, A[], A>::type VT;
  const int expect_alive = IsArray ? 5 : 1;
  {
    cuda::std::unique_ptr<VT> s1(newValue<VT>(expect_alive));
    A* p = s1.get();
    cuda::std::unique_ptr<VT> s2(newValue<VT>(expect_alive));
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == (expect_alive * 2));
    }
    s2 = cuda::std::move(s1);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == expect_alive);
    }
    assert(s2.get() == p);
    assert(s1.get() == 0);
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
  }
  {
    cuda::std::unique_ptr<VT, Deleter<VT>> s1(newValue<VT>(expect_alive), Deleter<VT>(5));
    A* p = s1.get();
    cuda::std::unique_ptr<VT, Deleter<VT>> s2(newValue<VT>(expect_alive));
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == (expect_alive * 2));
    }
    s2 = cuda::std::move(s1);
    assert(s2.get() == p);
    assert(s1.get() == 0);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == expect_alive);
    }
    assert(s2.get_deleter().state() == 5);
    assert(s1.get_deleter().state() == 0);
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
  }
  {
    CDeleter<VT> d1(5);
    cuda::std::unique_ptr<VT, CDeleter<VT>&> s1(newValue<VT>(expect_alive), d1);
    A* p = s1.get();
    CDeleter<VT> d2(6);
    cuda::std::unique_ptr<VT, CDeleter<VT>&> s2(newValue<VT>(expect_alive), d2);
    s2 = cuda::std::move(s1);
    assert(s2.get() == p);
    assert(s1.get() == 0);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == expect_alive);
    }
    assert(d1.state() == 5);
    assert(d2.state() == 5);
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
  }
  {
    cuda::std::unique_ptr<VT> s(newValue<VT>(expect_alive));
    A* p = s.get();
    s    = cuda::std::move(s);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == expect_alive);
    }
    assert(s.get() == p);
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
  }
}

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_sfinae()
{
  typedef typename cuda::std::conditional<IsArray, int[], int>::type VT;
  {
    typedef cuda::std::unique_ptr<VT> U;
    static_assert(!cuda::std::is_assignable<U, U&>::value, "");
    static_assert(!cuda::std::is_assignable<U, const U&>::value, "");
    static_assert(!cuda::std::is_assignable<U, const U&&>::value, "");
    static_assert(cuda::std::is_nothrow_assignable<U, U&&>::value, "");
  }
  {
    typedef cuda::std::unique_ptr<VT, GenericDeleter> U;
    static_assert(!cuda::std::is_assignable<U, U&>::value, "");
    static_assert(!cuda::std::is_assignable<U, const U&>::value, "");
    static_assert(!cuda::std::is_assignable<U, const U&&>::value, "");
    static_assert(cuda::std::is_nothrow_assignable<U, U&&>::value, "");
  }
  {
    typedef cuda::std::unique_ptr<VT, NCDeleter<VT>&> U;
    static_assert(!cuda::std::is_assignable<U, U&>::value, "");
    static_assert(!cuda::std::is_assignable<U, const U&>::value, "");
    static_assert(!cuda::std::is_assignable<U, const U&&>::value, "");
    static_assert(cuda::std::is_nothrow_assignable<U, U&&>::value, "");
  }
  {
    typedef cuda::std::unique_ptr<VT, const NCDeleter<VT>&> U;
    static_assert(!cuda::std::is_assignable<U, U&>::value, "");
    static_assert(!cuda::std::is_assignable<U, const U&>::value, "");
    static_assert(!cuda::std::is_assignable<U, const U&&>::value, "");
    static_assert(cuda::std::is_nothrow_assignable<U, U&&>::value, "");
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  {
    test_basic</*IsArray*/ false>();
    test_sfinae<false>();
  }
  {
    test_basic</*IsArray*/ true>();
    test_sfinae<true>();
  }

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
