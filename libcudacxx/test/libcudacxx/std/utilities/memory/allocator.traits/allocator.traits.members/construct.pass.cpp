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

// template <class Alloc>
// struct allocator_traits
// {
//     template <class Ptr, class... Args>
//     static constexpr void construct(allocator_type& a, Ptr p, Args&&... args);
//     ...
// };

#include <cuda/std/__memory_>
#include <cuda/std/__new_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "incomplete_type_helper.h"
#include "test_macros.h"

template <class T>
struct A
{
  typedef T value_type;
};

template <class T>
struct Alloc
{
  typedef T value_type;

  __host__ __device__ TEST_CONSTEXPR_CXX20 Alloc() {}

  __host__ __device__ TEST_CONSTEXPR_CXX20 value_type* allocate(cuda::std::size_t n)
  {
    assert(n == 1);
    return &storage;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 void deallocate(value_type*, cuda::std::size_t) noexcept {}

  value_type storage;
};

template <class T>
struct B
{
  typedef T value_type;

  __host__ __device__ TEST_CONSTEXPR_CXX20 B(int& count)
      : count_(count)
  {}

  template <class U, class... Args>
  __host__ __device__ TEST_CONSTEXPR_CXX20 void construct(U* p, Args&&... args)
  {
    ++count_;
#if TEST_STD_VER >= 2020
    cuda::std::construct_at(p, cuda::std::forward<Args>(args)...);
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
    ::new ((void*) p) U(cuda::std::forward<Args>(args)...);
#endif // TEST_STD_VER >= 2020
  }

  int& count_;
};

struct A0
{
  A0() = default;
  __host__ __device__ TEST_CONSTEXPR_CXX20 A0(int* count)
  {
    ++*count;
  }
};

struct A1
{
  A1() = default;
  __host__ __device__ TEST_CONSTEXPR_CXX20 A1(int* count, char c)
  {
    assert(c == 'c');
    ++*count;
  }
};

struct A2
{
  A2() = default;
  __host__ __device__ TEST_CONSTEXPR_CXX20 A2(int* count, char c, int i)
  {
    assert(c == 'd');
    assert(i == 5);
    ++*count;
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    int A0_count = 0;
    A<A0> a;
    Alloc<A0> alloc;
    A0* a0 = alloc.allocate(1);
    assert(A0_count == 0);
    cuda::std::allocator_traits<A<A0>>::construct(a, a0, &A0_count);
    assert(A0_count == 1);
    alloc.deallocate(a0, 1);
  }
  {
    int A1_count = 0;
    A<A1> a;
    Alloc<A1> alloc;
    A1* a1 = alloc.allocate(1);
    assert(A1_count == 0);
    cuda::std::allocator_traits<A<A1>>::construct(a, a1, &A1_count, 'c');
    assert(A1_count == 1);
    alloc.deallocate(a1, 1);
  }
  {
    int A2_count = 0;
    A<A2> a;
    Alloc<A2> alloc;
    A2* a2 = alloc.allocate(1);
    assert(A2_count == 0);
    cuda::std::allocator_traits<A<A2>>::construct(a, a2, &A2_count, 'd', 5);
    assert(A2_count == 1);
    alloc.deallocate(a2, 1);
  }
  {
    typedef IncompleteHolder* VT;
    typedef A<VT> Alloc2;
    Alloc2 a;
    Alloc<VT> alloc;
    VT* vt = alloc.allocate(1);
    cuda::std::allocator_traits<Alloc2>::construct(a, vt, nullptr);
    alloc.deallocate(vt, 1);
  }

  {
    int A0_count    = 0;
    int b_construct = 0;
    B<A0> b(b_construct);
    Alloc<A0> alloc;
    A0* a0 = alloc.allocate(1);
    assert(A0_count == 0);
    assert(b_construct == 0);
    cuda::std::allocator_traits<B<A0>>::construct(b, a0, &A0_count);
    assert(A0_count == 1);
    assert(b_construct == 1);
    alloc.deallocate(a0, 1);
  }
  {
    int A1_count    = 0;
    int b_construct = 0;
    B<A1> b(b_construct);
    Alloc<A1> alloc;
    A1* a1 = alloc.allocate(1);
    assert(A1_count == 0);
    assert(b_construct == 0);
    cuda::std::allocator_traits<B<A1>>::construct(b, a1, &A1_count, 'c');
    assert(A1_count == 1);
    assert(b_construct == 1);
    alloc.deallocate(a1, 1);
  }
  {
    int A2_count    = 0;
    int b_construct = 0;
    B<A2> b(b_construct);
    Alloc<A2> alloc;
    A2* a2 = alloc.allocate(1);
    assert(A2_count == 0);
    assert(b_construct == 0);
    cuda::std::allocator_traits<B<A2>>::construct(b, a2, &A2_count, 'd', 5);
    assert(A2_count == 1);
    assert(b_construct == 1);
    alloc.deallocate(a2, 1);
  }

  return true;
}

int main(int, char**)
{
  test();

#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020

  return 0;
}
