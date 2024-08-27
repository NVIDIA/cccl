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
//     static constexpr void deallocate(allocator_type& a, pointer p, size_type n) noexcept;
//     ...
// };

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "incomplete_type_helper.h"
#include "test_macros.h"

template <class T>
struct A
{
  typedef T value_type;

  __host__ __device__ TEST_CONSTEXPR_CXX20 A(int& called)
      : called_(called)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX20 void deallocate(value_type* p, cuda::std::size_t n) noexcept
  {
    assert(p == &storage);
    assert(n == 10);
    ++called_;
  }

  int& called_;

  value_type storage;
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    int called = 0;
    A<int> a(called);
    cuda::std::allocator_traits<A<int>>::deallocate(a, &a.storage, 10);
    assert(called == 1);
  }
  {
    int called = 0;
    typedef A<IncompleteHolder*> Alloc;
    Alloc a(called);
    cuda::std::allocator_traits<Alloc>::deallocate(a, &a.storage, 10);
    assert(called == 1);
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
