//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T, class Alloc> struct uses_allocator;

#include <cuda/std/__memory_>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/vector>
#endif // _LIBCUDACXX_HAS_VECTOR
#include <cuda/std/type_traits>

#include "test_macros.h"

struct A
{};

struct B
{
  typedef int allocator_type;
};

#if !defined(TEST_COMPILER_NVRTC)
struct C
{
  static int allocator_type;
};
#endif // !TEST_COMPILER_NVRTC

struct D
{
  __host__ __device__ static int allocator_type()
  {
    return 0;
  }
};

struct E
{
private:
  typedef int allocator_type;
};

template <bool Expected, class T, class A>
__host__ __device__ void test()
{
  static_assert((cuda::std::uses_allocator<T, A>::value == Expected), "");
  static_assert(
    cuda::std::is_base_of<cuda::std::integral_constant<bool, Expected>, cuda::std::uses_allocator<T, A>>::value, "");
#if TEST_STD_VER >= 2017
  ASSERT_SAME_TYPE(decltype(cuda::std::uses_allocator_v<T, A>), const bool);
  static_assert((cuda::std::uses_allocator_v<T, A> == Expected), "");
#endif // TEST_STD_VER >= 2017
}

int main(int, char**)
{
  test<false, int, cuda::std::allocator<int>>();
#if defined(_LIBCUDACXX_HAS_VECTOR)
  test<true, cuda::std::vector<int>, cuda::std::allocator<int>>();
#endif //_LIBCUDACXX_HAS_VECTOR
  test<false, A, cuda::std::allocator<int>>();
  test<false, B, cuda::std::allocator<int>>();
  test<true, B, double>();
#if !defined(TEST_COMPILER_NVRTC)
  test<false, C, decltype(C::allocator_type)>();
#endif // !TEST_COMPILER_NVRTC
  test<false, D, decltype(D::allocator_type)>();
#if !defined(TEST_COMPILER_GCC) && !defined(TEST_COMPILER_MSVC_2017) // E::allocator_type is private
  test<false, E, int>();
#endif // !TEST_COMPILER_GCC && !TEST_COMPILER_MSVC_2017

  static_assert((!cuda::std::uses_allocator<int, cuda::std::allocator<int>>::value), "");
#if defined(_LIBCUDACXX_HAS_VECTOR)
  static_assert((cuda::std::uses_allocator<cuda::std::vector<int>, cuda::std::allocator<int>>::value), "");
#endif // _LIBCUDACXX_HAS_VECTOR
  static_assert((!cuda::std::uses_allocator<A, cuda::std::allocator<int>>::value), "");
  static_assert((!cuda::std::uses_allocator<B, cuda::std::allocator<int>>::value), "");
  static_assert((cuda::std::uses_allocator<B, double>::value), "");
#if !defined(TEST_COMPILER_NVRTC)
  static_assert((!cuda::std::uses_allocator<C, decltype(C::allocator_type)>::value), "");
  static_assert((!cuda::std::uses_allocator<D, decltype(D::allocator_type)>::value), "");
#endif // !TEST_COMPILER_NVRTC
#if !defined(TEST_COMPILER_GCC) && !defined(TEST_COMPILER_MSVC_2017) // E::allocator_type is private
  static_assert((!cuda::std::uses_allocator<E, int>::value), "");
#endif // !TEST_COMPILER_GCC && !TEST_COMPILER_MSVC_2017

  return 0;
}
