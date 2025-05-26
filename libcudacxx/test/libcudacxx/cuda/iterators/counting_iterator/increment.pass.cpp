//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int) requires incrementable<W>;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::counting_iterator<int> iter1{0};
    cuda::counting_iterator<int> iter2{0};
    assert(iter1 == iter2);
    assert(++iter1 != iter2++);
    assert(iter1 == iter2);

    static_assert(noexcept(++iter2));
    static_assert(noexcept(iter2++));
    static_assert(!cuda::std::is_reference_v<decltype(iter2++)>);
    static_assert(cuda::std::is_reference_v<decltype(++iter2)>);
    static_assert(cuda::std::same_as<cuda::std::remove_reference_t<decltype(++iter2)>, decltype(iter2++)>);
  }
  {
    cuda::counting_iterator<SomeInt> iter1{SomeInt{0}};
    cuda::counting_iterator<SomeInt> iter2{SomeInt{0}};
    assert(iter1 == iter2);
    assert(++iter1 != iter2++);
    assert(iter1 == iter2);

    static_assert(!noexcept(++iter2));
    static_assert(!noexcept(iter2++));
    static_assert(!cuda::std::is_reference_v<decltype(iter2++)>);
    static_assert(cuda::std::is_reference_v<decltype(++iter2)>);
    static_assert(cuda::std::same_as<cuda::std::remove_reference_t<decltype(++iter2)>, decltype(iter2++)>);
  }

  {
    cuda::counting_iterator<NotIncrementable> iter1{NotIncrementable{0}};
    cuda::counting_iterator<NotIncrementable> iter2{NotIncrementable{0}};
    assert(iter1 == iter2);
    assert(++iter1 != iter2);
    iter2++;
    assert(iter1 == iter2);

    static_assert(!noexcept(iter2++));
    static_assert(cuda::std::same_as<decltype(iter2++), void>);
    static_assert(cuda::std::is_reference_v<decltype(++iter2)>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
