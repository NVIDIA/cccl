//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// constexpr W operator*() const noexcept(is_nothrow_copy_constructible_v<W>);

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wsign-compare"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wsign-compare"
#elif defined(_MSC_VER)
#pragma warning (disable: 4018) // various "signed/unsigned mismatch"
#endif

#include <cuda/std/ranges>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../types.h"

struct NotNoexceptCopy {
  using difference_type = int;

  int value_;
  __host__ __device__ constexpr explicit NotNoexceptCopy(int value = 0) : value_(value) {}
  __host__ __device__ constexpr NotNoexceptCopy(const NotNoexceptCopy& other) noexcept(false) : value_(other.value_) {}

#if TEST_STD_VER > 17
  bool operator==(const NotNoexceptCopy&) const = default;
#else
  __host__ __device__ friend constexpr bool operator==(const NotNoexceptCopy &lhs, const NotNoexceptCopy& rhs) {
    return lhs.value_ == rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator!=(const NotNoexceptCopy &lhs, const NotNoexceptCopy& rhs) {
    return lhs.value_ != rhs.value_;
  }
#endif

  __host__ __device__ friend constexpr NotNoexceptCopy& operator+=(NotNoexceptCopy &lhs, const NotNoexceptCopy& rhs) {
    lhs.value_ += rhs.value_; return lhs;
  }
  __host__ __device__ friend constexpr NotNoexceptCopy& operator-=(NotNoexceptCopy &lhs, const NotNoexceptCopy& rhs) {
    lhs.value_ -= rhs.value_; return lhs;
  }

  __host__ __device__ friend constexpr NotNoexceptCopy operator+(NotNoexceptCopy lhs, NotNoexceptCopy rhs) {
    return NotNoexceptCopy{lhs.value_ + rhs.value_};
  }
  __host__ __device__ friend constexpr int operator-(NotNoexceptCopy lhs, NotNoexceptCopy rhs) {
    return lhs.value_ - rhs.value_;
  }

  __host__ __device__ constexpr NotNoexceptCopy& operator++()    { ++value_; return *this; }
  __host__ __device__ constexpr void             operator++(int) { ++value_;               }
};

template<class T>
__host__ __device__ constexpr void testType() {
  {
    cuda::std::ranges::iota_view<T> io(T(0));
    auto iter = io.begin();
    for (int i = 0; i < 100; ++i, ++iter)
      assert(*iter == T(i));

    static_assert(noexcept(*iter) == !cuda::std::same_as<T, NotNoexceptCopy>);
  }
  {
    cuda::std::ranges::iota_view<T> io(T(10));
    auto iter = io.begin();
    for (int i = 10; i < 100; ++i, ++iter)
      assert(*iter == T(i));
  }
  {
    const cuda::std::ranges::iota_view<T> io(T(0));
    auto iter = io.begin();
    for (int i = 0; i < 100; ++i, ++iter)
      assert(*iter == T(i));
  }
  {
    const cuda::std::ranges::iota_view<T> io(T(10));
    auto iter = io.begin();
    for (int i = 10; i < 100; ++i, ++iter)
      assert(*iter == T(i));
  }
}

__host__ __device__ constexpr bool test() {
  testType<SomeInt>();
  testType<NotNoexceptCopy>();
  testType<signed long>();
  testType<unsigned long>();
  testType<int>();
  testType<unsigned>();
  testType<short>();
  testType<unsigned short>();

  // Tests a mix of signed unsigned types.
  {
    const cuda::std::ranges::iota_view<int, unsigned> io(0, 10);
    auto iter = io.begin();
    for (int i = 0; i < 10; ++i, ++iter)
      assert(*iter == i);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
