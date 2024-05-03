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

// test swap

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "unique_ptr_test_helper.h"

STATIC_TEST_GLOBAL_VAR int TT_count = 0;

struct TT
{
  int state_;
  __host__ __device__ TEST_CONSTEXPR_CXX23 TT()
      : state_(-1)
  {
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      ++TT_count;
    }
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit TT(int i)
      : state_(i)
  {
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      ++TT_count;
    }
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 TT(const TT& a)
      : state_(a.state_)
  {
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      ++TT_count;
    }
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 TT& operator=(const TT& a)
  {
    state_ = a.state_;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~TT()
  {
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      --TT_count;
    }
  }

  __host__ __device__ friend TEST_CONSTEXPR_CXX23 bool operator==(const TT& x, const TT& y)
  {
    return x.state_ == y.state_;
  }
};

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX23 typename cuda::std::remove_all_extents<T>::type*
newValueInit(int size, int new_value)
{
  typedef typename cuda::std::remove_all_extents<T>::type VT;
  VT* p = newValue<T>(size);
  for (int i = 0; i < size; ++i)
  {
    (p + i)->state_ = new_value;
  }
  return p;
}

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_basic()
{
  typedef typename cuda::std::conditional<IsArray, TT[], TT>::type VT;
  const int expect_alive = IsArray ? 5 : 1;
  {
    using U = cuda::std::unique_ptr<VT, Deleter<VT>>;
    U u;
    unused(u);
    ASSERT_NOEXCEPT(u.swap(u));
  }
  {
    TT* p1 = newValueInit<VT>(expect_alive, 1);
    cuda::std::unique_ptr<VT, Deleter<VT>> s1(p1, Deleter<VT>(1));
    TT* p2 = newValueInit<VT>(expect_alive, 2);
    cuda::std::unique_ptr<VT, Deleter<VT>> s2(p2, Deleter<VT>(2));
    assert(s1.get() == p1);
    assert(*s1.get() == TT(1));
    assert(s1.get_deleter().state() == 1);
    assert(s2.get() == p2);
    assert(*s2.get() == TT(2));
    assert(s2.get_deleter().state() == 2);
    s1.swap(s2);
    assert(s1.get() == p2);
    assert(*s1.get() == TT(2));
    assert(s1.get_deleter().state() == 2);
    assert(s2.get() == p1);
    assert(*s2.get() == TT(1));
    assert(s2.get_deleter().state() == 1);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(TT_count == (expect_alive * 2));
    }
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(TT_count == 0);
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
