//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iterator& operator--() requires decrementable<W>;
// constexpr iterator operator--(int) requires decrementable<W>;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class T>
_CCCL_CONCEPT Decrementable = _CCCL_REQUIRES_EXPR((T), T i)((--i), (i--));

__host__ __device__ constexpr bool test()
{
  {
    cuda::counting_iterator<int> iter1{0};
    cuda::counting_iterator<int> iter2{0};
    assert(iter1 == iter2);
    assert(--iter1 != iter2--);
    assert(iter1 == iter2);

    static_assert(noexcept(--iter2));
    static_assert(noexcept(iter2--));
    static_assert(!cuda::std::is_reference_v<decltype(iter2--)>);
    static_assert(cuda::std::is_reference_v<decltype(--iter2)>);
    static_assert(cuda::std::same_as<cuda::std::remove_reference_t<decltype(--iter2)>, decltype(iter2--)>);
  }

  { // With a decrementable type
    cuda::counting_iterator<SomeInt> iter1{SomeInt{0}};
    cuda::counting_iterator<SomeInt> iter2{SomeInt{0}};
    assert(iter1 == iter2);
    assert(--iter1 != iter2--);
    assert(iter1 == iter2);

    static_assert(!noexcept(--iter2));
    static_assert(!noexcept(iter2--));
    static_assert(!cuda::std::is_reference_v<decltype(iter2--)>);
    static_assert(cuda::std::is_reference_v<decltype(--iter2)>);
    static_assert(cuda::std::same_as<cuda::std::remove_reference_t<decltype(--iter2)>, decltype(iter2--)>);
  }

  static_assert(!Decrementable<cuda::counting_iterator<Int42<ValueCtor>>>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
