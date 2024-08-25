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

// constexpr iterator begin() const;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ constexpr void testType()
{
  {
    cuda::std::ranges::iota_view<T> io(T(0));
    assert(*io.begin() == T(0));
  }
  {
    cuda::std::ranges::iota_view<T> io(T(10));
    assert(*io.begin() == T(10));
    assert(*cuda::std::move(io).begin() == T(10));
  }
  {
    const cuda::std::ranges::iota_view<T> io(T(0));
    assert(*io.begin() == T(0));
  }
  {
    const cuda::std::ranges::iota_view<T> io(T(10));
    assert(*io.begin() == T(10));
  }
}

__host__ __device__ constexpr bool test()
{
  testType<SomeInt>();
  testType<long long>();
  testType<unsigned long long>();
  testType<signed long>();
  testType<unsigned long>();
  testType<int>();
  testType<unsigned>();
  testType<short>();
  testType<unsigned short>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
