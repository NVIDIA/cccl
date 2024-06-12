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

// test reset

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "unique_ptr_test_helper.h"

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_reset_pointer()
{
  typedef typename cuda::std::conditional<IsArray, A[], A>::type VT;
  const int expect_alive = IsArray ? 3 : 1;
  {
    using U = cuda::std::unique_ptr<VT>;
    U u;
    unused(u);
    ASSERT_NOEXCEPT(u.reset((A*) nullptr));
  }
  {
    cuda::std::unique_ptr<VT> p(newValue<VT>(expect_alive));
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == expect_alive);
    }
    A* i = p.get();
    assert(i != nullptr);
    A* new_value = newValue<VT>(expect_alive);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == (expect_alive * 2));
    }
    p.reset(new_value);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == expect_alive);
    }
    assert(p.get() == new_value);
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
  }
  {
    cuda::std::unique_ptr<const VT> p(newValue<const VT>(expect_alive));
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == expect_alive);
    }
    const A* i = p.get();
    assert(i != nullptr);
    A* new_value = newValue<VT>(expect_alive);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == (expect_alive * 2));
    }
    p.reset(new_value);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == expect_alive);
    }
    assert(p.get() == new_value);
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
  }
}

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_reset_nullptr()
{
  typedef typename cuda::std::conditional<IsArray, A[], A>::type VT;
  const int expect_alive = IsArray ? 3 : 1;
  {
    using U = cuda::std::unique_ptr<VT>;
    U u;
    unused(u);
    ASSERT_NOEXCEPT(u.reset(nullptr));
  }
  {
    cuda::std::unique_ptr<VT> p(newValue<VT>(expect_alive));
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == expect_alive);
    }
    A* i = p.get();
    assert(i != nullptr);
    p.reset(nullptr);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == 0);
    }
    assert(p.get() == nullptr);
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
  }
}

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_reset_no_arg()
{
  typedef typename cuda::std::conditional<IsArray, A[], A>::type VT;
  const int expect_alive = IsArray ? 3 : 1;
  {
    using U = cuda::std::unique_ptr<VT>;
    U u;
    unused(u);
    ASSERT_NOEXCEPT(u.reset());
  }
  {
    cuda::std::unique_ptr<VT> p(newValue<VT>(expect_alive));
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == expect_alive);
    }
    A* i = p.get();
    assert(i != nullptr);
    p.reset();
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == 0);
    }
    assert(p.get() == nullptr);
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  {
    test_reset_pointer</*IsArray*/ false>();
    test_reset_nullptr<false>();
    test_reset_no_arg<false>();
  }
  {
    test_reset_pointer</*IsArray*/ true>();
    test_reset_nullptr<true>();
    test_reset_no_arg<true>();
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
