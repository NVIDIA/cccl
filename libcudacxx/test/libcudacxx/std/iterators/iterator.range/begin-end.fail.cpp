//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "test_macros.h"

// <cuda/std/iterator>
// template <class C> auto begin(C& c) -> decltype(c.begin());
// template <class C> auto begin(const C& c) -> decltype(c.begin());
// template <class C> auto end(C& c) -> decltype(c.end());
// template <class C> auto end(const C& c) -> decltype(c.end());
// template <class E> reverse_iterator<const E*> rbegin(initializer_list<E> il);
// template <class E> reverse_iterator<const E*> rend(initializer_list<E> il);

#include <cuda/std/cassert>
#include <cuda/std/iterator>

namespace Foo
{
struct FakeContainer
{};
typedef int FakeIter;

__host__ __device__ FakeIter begin(const FakeContainer&)
{
  return 1;
}
__host__ __device__ FakeIter end(const FakeContainer&)
{
  return 2;
}
__host__ __device__ FakeIter rbegin(const FakeContainer&)
{
  return 3;
}
__host__ __device__ FakeIter rend(const FakeContainer&)
{
  return 4;
}

__host__ __device__ FakeIter cbegin(const FakeContainer&)
{
  return 11;
}
__host__ __device__ FakeIter cend(const FakeContainer&)
{
  return 12;
}
__host__ __device__ FakeIter crbegin(const FakeContainer&)
{
  return 13;
}
__host__ __device__ FakeIter crend(const FakeContainer&)
{
  return 14;
}
} // namespace Foo

int main(int, char**)
{
  // Bug #28927 - shouldn't find these via ADL
  TEST_IGNORE_NODISCARD cuda::std::cbegin(Foo::FakeContainer());
  TEST_IGNORE_NODISCARD cuda::std::cend(Foo::FakeContainer());
  TEST_IGNORE_NODISCARD cuda::std::crbegin(Foo::FakeContainer());
  TEST_IGNORE_NODISCARD cuda::std::crend(Foo::FakeContainer());

  return 0;
}
#endif
