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
//     static constexpr allocator_type
//         select_on_container_copy_construction(const allocator_type& a);
//     ...
// };

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "incomplete_type_helper.h"
#include "test_macros.h"

template <class T>
struct A
{
  typedef T value_type;
  int id;
  __host__ __device__ TEST_CONSTEXPR_CXX20 explicit A(int i = 0)
      : id(i)
  {}
};

template <class T>
struct B
{
  typedef T value_type;

  int id;
  __host__ __device__ TEST_CONSTEXPR_CXX20 explicit B(int i = 0)
      : id(i)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX20 B select_on_container_copy_construction() const
  {
    return B(100);
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    A<int> a;
    assert(cuda::std::allocator_traits<A<int>>::select_on_container_copy_construction(a).id == 0);
  }
  {
    const A<int> a(0);
    assert(cuda::std::allocator_traits<A<int>>::select_on_container_copy_construction(a).id == 0);
  }
  {
    typedef IncompleteHolder* VT;
    typedef A<VT> Alloc;
    Alloc a;
    assert(cuda::std::allocator_traits<Alloc>::select_on_container_copy_construction(a).id == 0);
  }
  {
    B<int> b;
    assert(cuda::std::allocator_traits<B<int>>::select_on_container_copy_construction(b).id == 100);
  }
  {
    const B<int> b(0);
    assert(cuda::std::allocator_traits<B<int>>::select_on_container_copy_construction(b).id == 100);
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
