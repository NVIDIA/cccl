//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int) requires incrementable<W>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::iota_view<int> io(0);
    auto iter1 = io.begin();
    auto iter2 = io.begin();
    assert(iter1 == iter2);
    assert(++iter1 != iter2++);
    assert(iter1 == iter2);

    static_assert(!cuda::std::is_reference_v<decltype(iter2++)>);
    static_assert(cuda::std::is_reference_v<decltype(++iter2)>);
    static_assert(cuda::std::same_as<cuda::std::remove_reference_t<decltype(++iter2)>, decltype(iter2++)>);
  }
  {
    cuda::std::ranges::iota_view io(SomeInt(0));
    auto iter1 = io.begin();
    auto iter2 = io.begin();
    assert(iter1 == iter2);
    assert(++iter1 != iter2++);
    assert(iter1 == iter2);

    static_assert(!cuda::std::is_reference_v<decltype(iter2++)>);
    static_assert(cuda::std::is_reference_v<decltype(++iter2)>);
    static_assert(cuda::std::same_as<cuda::std::remove_reference_t<decltype(++iter2)>, decltype(iter2++)>);
  }

  {
    cuda::std::ranges::iota_view<NotIncrementable> io(NotIncrementable(0));
    auto iter1 = io.begin();
    auto iter2 = io.begin();
    assert(iter1 == iter2);
    assert(++iter1 != iter2);
    iter2++;
    assert(iter1 == iter2);

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
