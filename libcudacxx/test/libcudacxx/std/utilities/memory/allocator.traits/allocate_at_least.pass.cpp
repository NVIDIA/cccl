//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <memory>

// template<class Allocator>
// [[nodiscard]] constexpr allocation_result<typename allocator_traits<Allocator>::pointer>
//   allocate_at_least(Allocator& a, size_t n);

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/cstddef>

// check that cuda::std::allocation_result exists and isn't restricted to pointers
using AllocResult = cuda::std::allocation_result<int>;

template <class T>
struct no_allocate_at_least
{
  using value_type = T;
  T t;

  constexpr T* allocate(cuda::std::size_t)
  {
    return &t;
  }
  constexpr void deallocate(T*, cuda::std::size_t) noexcept {}
};

template <class T>
struct has_allocate_at_least
{
  using value_type = T;
  T t1;
  T t2;

  constexpr T* allocate(cuda::std::size_t)
  {
    return &t1;
  }
  constexpr void deallocate(T*, cuda::std::size_t) noexcept {}
  constexpr cuda::std::allocation_result<T*> allocate_at_least(cuda::std::size_t)
  {
    return {&t2, 2};
  }
};

constexpr bool test()
{
  { // check that cuda::std::allocate_at_least forwards to allocator::allocate if no allocate_at_least exists
    no_allocate_at_least<int> alloc;
    cuda::std::same_as<cuda::std::allocation_result<int*>> decltype(auto) ret = cuda::std::allocate_at_least(alloc, 1);
    assert(ret.count == 1);
    assert(ret.ptr == &alloc.t);
  }

  { // check that cuda::std::allocate_at_least forwards to allocator::allocate_at_least if allocate_at_least exists
    has_allocate_at_least<int> alloc;
    cuda::std::same_as<cuda::std::allocation_result<int*>> decltype(auto) ret = cuda::std::allocate_at_least(alloc, 1);
    assert(ret.count == 2);
    assert(ret.ptr == &alloc.t2);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
