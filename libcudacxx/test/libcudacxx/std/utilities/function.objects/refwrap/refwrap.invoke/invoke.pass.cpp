//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: a non-__tile__ variable ("count") cannot be used in tile code

// <functional>

// reference_wrapper

// template <class... ArgTypes>
//   requires Callable<T, ArgTypes&&...>
//   Callable<T, ArgTypes&&...>::result_type
//   operator()(ArgTypes&&... args) const;

// #include <cuda/std/functional>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE int count = 0;

// 1 arg, return void

TEST_FUNC void f_void_1(int i)
{
  count += i;
}

struct A_void_1
{
  TEST_FUNC void operator()(int i)
  {
    count += i;
  }

  TEST_FUNC void mem1()
  {
    ++count;
  }
  TEST_FUNC void mem2() const
  {
    ++count;
  }
};

TEST_FUNC void test_void_1()
{
  int save_count = count;
  // function
  {
    cuda::std::reference_wrapper<void(int)> r1(f_void_1);
    int i = 2;
    r1(i);
    assert(count == save_count + 2);
    save_count = count;
  }

#if !_CCCL_TILE_COMPILATION() // error: function-to-pointer decay is unsupported in tile code
  // function pointer
  {
    void (*fp)(int) = f_void_1;
    cuda::std::reference_wrapper<void (*)(int)> r1(fp);
    int i = 3;
    r1(i);
    assert(count == save_count + 3);
    save_count = count;
  }
#endif // !_CCCL_TILE_COMPILATION()
  // functor
  {
    A_void_1 a0;
    cuda::std::reference_wrapper<A_void_1> r1(a0);
    int i = 4;
    r1(i);
    assert(count == save_count + 4);
    save_count = count;
  }
  // member function pointer
  {
    void (A_void_1::*fp)() = &A_void_1::mem1;
    cuda::std::reference_wrapper<void (A_void_1::*)()> r1(fp);
    A_void_1 a;
    r1(a);
    assert(count == save_count + 1);
    save_count   = count;
    A_void_1* ap = &a;
    r1(ap);
    assert(count == save_count + 1);
    save_count = count;
  }
  // const member function pointer
  {
    void (A_void_1::*fp)() const = &A_void_1::mem2;
    cuda::std::reference_wrapper<void (A_void_1::*)() const> r1(fp);
    A_void_1 a;
    r1(a);
    assert(count == save_count + 1);
    save_count   = count;
    A_void_1* ap = &a;
    r1(ap);
    assert(count == save_count + 1);
    save_count = count;
  }
}

// 1 arg, return int

TEST_FUNC int f_int_1(int i)
{
  return i + 1;
}

struct A_int_1
{
  TEST_FUNC A_int_1()
      : data_(5)
  {}
  TEST_FUNC int operator()(int i)
  {
    return i - 1;
  }

  TEST_FUNC int mem1()
  {
    return 3;
  }
  TEST_FUNC int mem2() const
  {
    return 4;
  }
  int data_;
};

TEST_FUNC void test_int_1()
{
  // function
  {
    cuda::std::reference_wrapper<int(int)> r1(f_int_1);
    int i = 2;
    assert(r1(i) == 3);
  }
#if !_CCCL_TILE_COMPILATION() // error: function-to-pointer decay is unsupported in tile code
  // function pointer
  {
    int (*fp)(int) = f_int_1;
    cuda::std::reference_wrapper<int (*)(int)> r1(fp);
    int i = 3;
    assert(r1(i) == 4);
  }
#endif // !_CCCL_TILE_COMPILATION()
  // functor
  {
    A_int_1 a0;
    cuda::std::reference_wrapper<A_int_1> r1(a0);
    int i = 4;
    assert(r1(i) == 3);
  }
  // member function pointer
  {
    int (A_int_1::*fp)() = &A_int_1::mem1;
    cuda::std::reference_wrapper<int (A_int_1::*)()> r1(fp);
    A_int_1 a;
    assert(r1(a) == 3);
    A_int_1* ap = &a;
    assert(r1(ap) == 3);
  }
  // const member function pointer
  {
    int (A_int_1::*fp)() const = &A_int_1::mem2;
    cuda::std::reference_wrapper<int (A_int_1::*)() const> r1(fp);
    A_int_1 a;
    assert(r1(a) == 4);
    A_int_1* ap = &a;
    assert(r1(ap) == 4);
  }
  // member data pointer
  {
    // gcc complains about a non existing parenthesis without the needless alias
    using ptr_to_member = int A_int_1::*;
    ptr_to_member fp    = &A_int_1::data_;
    cuda::std::reference_wrapper<int A_int_1::*> r1(fp);
    A_int_1 a;
    assert(r1(a) == 5);
    r1(a) = 6;
    assert(r1(a) == 6);
    A_int_1* ap = &a;
    assert(r1(ap) == 6);
    r1(ap) = 7;
    assert(r1(ap) == 7);
  }
}

// 2 arg, return void

TEST_FUNC void f_void_2(int i, int j)
{
  count += i + j;
}

struct A_void_2
{
  TEST_FUNC void operator()(int i, int j)
  {
    count += i + j;
  }

  TEST_FUNC void mem1(int i)
  {
    count += i;
  }
  TEST_FUNC void mem2(int i) const
  {
    count += i;
  }
};

TEST_FUNC void test_void_2()
{
  int save_count = count;
  // function
  {
    cuda::std::reference_wrapper<void(int, int)> r1(f_void_2);
    int i = 2;
    int j = 3;
    r1(i, j);
    assert(count == save_count + 5);
    save_count = count;
  }
#if !_CCCL_TILE_COMPILATION() // error: function-to-pointer decay is unsupported in tile code
  // function pointer
  {
    void (*fp)(int, int) = f_void_2;
    cuda::std::reference_wrapper<void (*)(int, int)> r1(fp);
    int i = 3;
    int j = 4;
    r1(i, j);
    assert(count == save_count + 7);
    save_count = count;
  }
#endif // !_CCCL_TILE_COMPILATION()
  // functor
  {
    A_void_2 a0;
    cuda::std::reference_wrapper<A_void_2> r1(a0);
    int i = 4;
    int j = 5;
    r1(i, j);
    assert(count == save_count + 9);
    save_count = count;
  }
  // member function pointer
  {
    void (A_void_2::*fp)(int) = &A_void_2::mem1;
    cuda::std::reference_wrapper<void (A_void_2::*)(int)> r1(fp);
    A_void_2 a;
    int i = 3;
    r1(a, i);
    assert(count == save_count + 3);
    save_count   = count;
    A_void_2* ap = &a;
    r1(ap, i);
    assert(count == save_count + 3);
    save_count = count;
  }
  // const member function pointer
  {
    void (A_void_2::*fp)(int) const = &A_void_2::mem2;
    cuda::std::reference_wrapper<void (A_void_2::*)(int) const> r1(fp);
    A_void_2 a;
    int i = 4;
    r1(a, i);
    assert(count == save_count + 4);
    save_count   = count;
    A_void_2* ap = &a;
    r1(ap, i);
    assert(count == save_count + 4);
    save_count = count;
  }
}

// 2 arg, return int

TEST_FUNC int f_int_2(int i, int j)
{
  return i + j;
}

struct A_int_2
{
  TEST_FUNC int operator()(int i, int j)
  {
    return i + j;
  }

  TEST_FUNC int mem1(int i)
  {
    return i + 1;
  }
  TEST_FUNC int mem2(int i) const
  {
    return i + 2;
  }
};

TEST_FUNC void testint_2()
{
  // function
  {
    cuda::std::reference_wrapper<int(int, int)> r1(f_int_2);
    int i = 2;
    int j = 3;
    assert(r1(i, j) == i + j);
  }
#if !_CCCL_TILE_COMPILATION() // error: function-to-pointer decay is unsupported in tile code
  // function pointer
  {
    int (*fp)(int, int) = f_int_2;
    cuda::std::reference_wrapper<int (*)(int, int)> r1(fp);
    int i = 3;
    int j = 4;
    assert(r1(i, j) == i + j);
  }
#endif // !_CCCL_TILE_COMPILATION()
  // functor
  {
    A_int_2 a0;
    cuda::std::reference_wrapper<A_int_2> r1(a0);
    int i = 4;
    int j = 5;
    assert(r1(i, j) == i + j);
  }
  // member function pointer
  {
    int (A_int_2::*fp)(int) = &A_int_2::mem1;
    cuda::std::reference_wrapper<int (A_int_2::*)(int)> r1(fp);
    A_int_2 a;
    int i = 3;
    assert(r1(a, i) == i + 1);
    A_int_2* ap = &a;
    assert(r1(ap, i) == i + 1);
  }
  // const member function pointer
  {
    int (A_int_2::*fp)(int) const = &A_int_2::mem2;
    cuda::std::reference_wrapper<int (A_int_2::*)(int) const> r1(fp);
    A_int_2 a;
    int i = 4;
    assert(r1(a, i) == i + 2);
    A_int_2* ap = &a;
    assert(r1(ap, i) == i + 2);
  }
}

int main(int, char**)
{
  test_void_1();
  test_int_1();
  test_void_2();
  testint_2();

  return 0;
}
