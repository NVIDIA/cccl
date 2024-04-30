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

// Test swap

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "deleter_types.h"
#include "test_macros.h"

#if defined(TEST_COMPILER_NVCC) || defined(TEST_COMPILER_NVRTC)
TEST_NV_DIAG_SUPPRESS(3060) // call to __builtin_is_constant_evaluated appearing in a non-constexpr function
#endif // TEST_COMPILER_NVCC || TEST_COMPILER_NVRTC
#if defined(TEST_COMPILER_GCC)
#  pragma GCC diagnostic ignored "-Wtautological-compare"
#elif defined(TEST_COMPILER_CLANG)
#  pragma clang diagnostic ignored "-Wtautological-compare"
#endif

STATIC_TEST_GLOBAL_VAR int A_count = 0;

struct A
{
  int state_;
  __host__ __device__ TEST_CONSTEXPR_CXX23 A()
      : state_(0)
  {
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      ++A_count;
    }
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit A(int i)
      : state_(i)
  {
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      ++A_count;
    }
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 A(const A& a)
      : state_(a.state_)
  {
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      ++A_count;
    }
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 A& operator=(const A& a)
  {
    state_ = a.state_;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~A()
  {
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      --A_count;
    }
  }

  __host__ __device__ friend TEST_CONSTEXPR_CXX23 bool operator==(const A& x, const A& y)
  {
    return x.state_ == y.state_;
  }
};

template <class T>
struct NonSwappableDeleter
{
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit NonSwappableDeleter(int) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 NonSwappableDeleter& operator=(NonSwappableDeleter const&)
  {
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T*) const {}

private:
  __host__ __device__ NonSwappableDeleter(NonSwappableDeleter const&);
};

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  {
    A* p1 = new A(1);
    cuda::std::unique_ptr<A, Deleter<A>> s1(p1, Deleter<A>(1));
    A* p2 = new A(2);
    cuda::std::unique_ptr<A, Deleter<A>> s2(p2, Deleter<A>(2));
    assert(s1.get() == p1);
    assert(*s1 == A(1));
    assert(s1.get_deleter().state() == 1);
    assert(s2.get() == p2);
    assert(*s2 == A(2));
    assert(s2.get_deleter().state() == 2);
    swap(s1, s2);
    assert(s1.get() == p2);
    assert(*s1 == A(2));
    assert(s1.get_deleter().state() == 2);
    assert(s2.get() == p1);
    assert(*s2 == A(1));
    assert(s2.get_deleter().state() == 1);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == 2);
    }
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
  }
  {
    A* p1 = new A[3];
    cuda::std::unique_ptr<A[], Deleter<A[]>> s1(p1, Deleter<A[]>(1));
    A* p2 = new A[3];
    cuda::std::unique_ptr<A[], Deleter<A[]>> s2(p2, Deleter<A[]>(2));
    assert(s1.get() == p1);
    assert(s1.get_deleter().state() == 1);
    assert(s2.get() == p2);
    assert(s2.get_deleter().state() == 2);
    swap(s1, s2);
    assert(s1.get() == p2);
    assert(s1.get_deleter().state() == 2);
    assert(s2.get() == p1);
    assert(s2.get_deleter().state() == 1);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == 6);
    }
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
  }
  {
    // test that unique_ptr's specialized swap is disabled when the deleter
    // is non-swappable. Instead we should pick up the generic swap(T, T)
    // and perform 3 move constructions.
    typedef NonSwappableDeleter<int> D;
    D d(42);
    int x = 42;
    int y = 43;
    cuda::std::unique_ptr<int, D&> p(&x, d);
    cuda::std::unique_ptr<int, D&> p2(&y, d);
    cuda::std::swap(p, p2);
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
