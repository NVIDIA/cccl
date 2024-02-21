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

#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wsign-compare"
#elif defined(__GNUC__)
#  pragma GCC diagnostic ignored "-Wsign-compare"
#elif defined(_MSC_VER)
#  pragma warning(disable : 4018 4389) // various "signed/unsigned mismatch"
#endif

// constexpr auto end() const;
// constexpr iterator end() const requires same_as<W, Bound>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

template <class T, class U>
__host__ __device__ constexpr void testType(U u)
{
  {
    cuda::std::ranges::iota_view<T, U> io(T(0), u);
    assert(cuda::std::ranges::next(io.begin(), 10) == io.end());
  }
  {
    cuda::std::ranges::iota_view<T, U> io(T(10), u);
    assert(io.begin() == io.end());
    assert(io.begin() == cuda::std::move(io).end());
  }
  {
    const cuda::std::ranges::iota_view<T, U> io(T(0), u);
    assert(cuda::std::ranges::next(io.begin(), 10) == io.end());
    assert(cuda::std::ranges::next(io.begin(), 10) == cuda::std::move(io).end());
  }
  {
    const cuda::std::ranges::iota_view<T, U> io(T(10), u);
    assert(io.begin() == io.end());
  }

  {
    cuda::std::ranges::iota_view<T> io(T(0), cuda::std::unreachable_sentinel);
    assert(io.begin() != io.end());
    assert(cuda::std::ranges::next(io.begin()) != io.end());
    assert(cuda::std::ranges::next(io.begin(), 10) != io.end());
  }
  {
    const cuda::std::ranges::iota_view<T> io(T(0), cuda::std::unreachable_sentinel);
    assert(io.begin() != io.end());
    assert(cuda::std::ranges::next(io.begin()) != io.end());
    assert(cuda::std::ranges::next(io.begin(), 10) != io.end());
  }
}

__host__ __device__ constexpr bool test()
{
  testType<SomeInt>(SomeInt(10));
  testType<SomeInt>(IntComparableWith(SomeInt(10)));
  testType<signed long>(IntComparableWith<signed long>(10));
  testType<unsigned long>(IntComparableWith<unsigned long>(10));
  testType<int>(IntComparableWith<int>(10));
  testType<int>(int(10));
  testType<int>(unsigned(10));
  testType<unsigned>(unsigned(10));
  testType<unsigned>(int(10));
  testType<unsigned>(IntComparableWith<unsigned>(10));
  testType<short>(short(10));
  testType<short>(IntComparableWith<short>(10));
  testType<unsigned short>(IntComparableWith<unsigned short>(10));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
