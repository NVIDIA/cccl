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

// constexpr explicit iota_view(W value);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

struct SomeIntComparable
{
  using difference_type = int;

  SomeInt value_;
  __host__ __device__ constexpr SomeIntComparable()
      : value_(SomeInt(10))
  {}

  __host__ __device__ friend constexpr bool operator==(SomeIntComparable lhs, SomeIntComparable rhs)
  {
    return lhs.value_ == rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator==(SomeIntComparable lhs, SomeInt rhs)
  {
    return lhs.value_ == rhs;
  }
  __host__ __device__ friend constexpr bool operator==(SomeInt lhs, SomeIntComparable rhs)
  {
    return lhs == rhs.value_;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator!=(SomeIntComparable lhs, SomeIntComparable rhs)
  {
    return lhs.value_ != rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator!=(SomeIntComparable lhs, SomeInt rhs)
  {
    return lhs.value_ != rhs;
  }
  __host__ __device__ friend constexpr bool operator!=(SomeInt lhs, SomeIntComparable rhs)
  {
    return lhs != rhs.value_;
  }
#endif

  __host__ __device__ friend constexpr difference_type operator-(SomeIntComparable lhs, SomeIntComparable rhs)
  {
    return lhs.value_ - rhs.value_;
  }

  __host__ __device__ constexpr SomeIntComparable& operator++()
  {
    ++value_;
    return *this;
  }
  __host__ __device__ constexpr SomeIntComparable operator++(int)
  {
    auto tmp = *this;
    ++value_;
    return tmp;
  }
  __host__ __device__ constexpr SomeIntComparable operator--()
  {
    --value_;
    return *this;
  }
};

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::iota_view<SomeInt> io(SomeInt(42));
    assert((*io.begin()).value_ == 42);
    // Check that end returns cuda::std::unreachable_sentinel.
    assert(io.end() != io.begin());
    static_assert(cuda::std::same_as<decltype(io.end()), cuda::std::unreachable_sentinel_t>);
  }

  {
    cuda::std::ranges::iota_view<SomeInt, SomeIntComparable> io(SomeInt(0));
    assert(cuda::std::ranges::next(io.begin(), 10) == io.end());
  }
  {
    static_assert(!cuda::std::is_convertible_v<cuda::std::ranges::iota_view<SomeInt>, SomeInt>);
    static_assert(cuda::std::is_constructible_v<cuda::std::ranges::iota_view<SomeInt>, SomeInt>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
