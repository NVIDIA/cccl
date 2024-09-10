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
//     template <class Ptr>
//     static constexpr void destroy(allocator_type& a, Ptr p);
//     ...
// };

// Currently no suppport for std::allocator

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "incomplete_type_helper.h"
#include "test_macros.h"

template <class T>
struct NoDestroy
{
  typedef T value_type;

  __host__ __device__ TEST_CONSTEXPR_CXX20 T* allocate(cuda::std::size_t n)
  {
    return cuda::std::allocator<T>().allocate(n);
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 void deallocate(T* p, cuda::std::size_t n) noexcept
  {
    return cuda::std::allocator<T>().deallocate(p, n);
  }
};

template <class T>
struct CountDestroy
{
  __host__ __device__ TEST_CONSTEXPR explicit CountDestroy(int* counter)
      : counter_(counter)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX20 ~CountDestroy() {}

  typedef T value_type;

  __host__ __device__ TEST_CONSTEXPR_CXX20 T* allocate(cuda::std::size_t n)
  {
    return &storage;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 void deallocate(T* p, cuda::std::size_t n) noexcept {}

  template <class U, class... Args>
  __host__ __device__ TEST_CONSTEXPR_CXX20 void construct(U* p, Args&&... args)
  {
    assert(p == nullptr);
    cuda::std::__construct_at(&storage, cuda::std::forward<Args>(args)...);
  }

  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX20 void destroy(U* p) noexcept
  {
    assert(p == nullptr);
    ++*counter_;
    storage.~U();
  }

  int* counter_;
  union
  {
    char dummy_{};
    value_type storage;
  };
};

struct CountDestructor
{
  __host__ __device__ TEST_CONSTEXPR explicit CountDestructor(int* counter)
      : counter_(counter)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX20 ~CountDestructor()
  {
    ++*counter_;
  }

  int* counter_;
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
#if TEST_STD_VER >= 2020
  if (!TEST_IS_CONSTANT_EVALUATED())
#endif // TEST_STD_VER >= 2020
  {
    using Alloc     = NoDestroy<CountDestructor>;
    int destructors = 0;
    Alloc alloc;
    CountDestructor* pool = cuda::std::allocator_traits<Alloc>::allocate(alloc, 1);

    cuda::std::allocator_traits<Alloc>::construct(alloc, pool, &destructors);
    assert(destructors == 0);

    cuda::std::allocator_traits<Alloc>::destroy(alloc, pool);
    assert(destructors == 1);

    cuda::std::allocator_traits<Alloc>::deallocate(alloc, pool, 1);
  }
#if !defined(TEST_COMPILER_MSVC) && TEST_STD_VER >= 2020 // incomplete type not allowed
  if (!TEST_IS_CONSTANT_EVALUATED())
  {
    typedef IncompleteHolder* T;
    typedef NoDestroy<T> Alloc;
    Alloc alloc;
    T* pool = cuda::std::allocator_traits<Alloc>::allocate(alloc, 1);
    cuda::std::allocator_traits<Alloc>::construct(alloc, pool, nullptr);
    cuda::std::allocator_traits<Alloc>::destroy(alloc, pool);
    cuda::std::allocator_traits<Alloc>::deallocate(alloc, pool, 1);
  }
#endif // !defined(TEST_COMPILER_MSVC) && TEST_STD_VER >= 2020
  {
    using Alloc            = CountDestroy<CountDestructor>;
    int destroys_called    = 0;
    int destructors_called = 0;
    Alloc alloc(&destroys_called);
    CountDestructor* fake_ptr = nullptr;

    cuda::std::allocator_traits<Alloc>::construct(alloc, fake_ptr, &destructors_called);
    assert(destroys_called == 0);
    assert(destructors_called == 0);

    cuda::std::allocator_traits<Alloc>::destroy(alloc, fake_ptr);
    assert(destroys_called == 1);
    assert(destructors_called == 1);

    cuda::std::allocator_traits<Alloc>::deallocate(alloc, fake_ptr, 1);
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
