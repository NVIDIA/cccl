//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr W operator*() const noexcept(is_nothrow_copy_constructible_v<W>);

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"
#include "types.h"

TEST_DIAG_SUPPRESS_CLANG("-Wsign-compare")
TEST_DIAG_SUPPRESS_MSVC(4018) // various "signed/unsigned mismatch"

struct NotNoexceptCopy
{
  using difference_type = int;

  int value_;
  TEST_FUNC constexpr explicit NotNoexceptCopy(int value = 0)
      : value_(value)
  {}
  TEST_FUNC constexpr NotNoexceptCopy(const NotNoexceptCopy& other) noexcept(false)
      : value_(other.value_)
  {}

#if TEST_STD_VER >= 2020
  bool operator==(const NotNoexceptCopy&) const = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  TEST_FUNC friend constexpr bool operator==(const NotNoexceptCopy& lhs, const NotNoexceptCopy& rhs)
  {
    return lhs.value_ == rhs.value_;
  }
  TEST_FUNC friend constexpr bool operator!=(const NotNoexceptCopy& lhs, const NotNoexceptCopy& rhs)
  {
    return lhs.value_ != rhs.value_;
  }
#endif // TEST_STD_VER <= 2017

  TEST_FUNC friend constexpr NotNoexceptCopy& operator+=(NotNoexceptCopy& lhs, const NotNoexceptCopy& rhs)
  {
    lhs.value_ += rhs.value_;
    return lhs;
  }
  TEST_FUNC friend constexpr NotNoexceptCopy& operator-=(NotNoexceptCopy& lhs, const NotNoexceptCopy& rhs)
  {
    lhs.value_ -= rhs.value_;
    return lhs;
  }

  TEST_FUNC friend constexpr NotNoexceptCopy operator+(NotNoexceptCopy lhs, NotNoexceptCopy rhs)
  {
    return NotNoexceptCopy{lhs.value_ + rhs.value_};
  }
  TEST_FUNC friend constexpr int operator-(NotNoexceptCopy lhs, NotNoexceptCopy rhs)
  {
    return lhs.value_ - rhs.value_;
  }

  TEST_FUNC constexpr NotNoexceptCopy& operator++()
  {
    ++value_;
    return *this;
  }
  TEST_FUNC constexpr void operator++(int)
  {
    ++value_;
  }
};

template <class Iter>
TEST_FUNC constexpr void testType()
{
  using T = typename Iter::value_type;
  {
    Iter iter{T{0}};
    for (int i = 0; i < 100; ++i, ++iter)
    {
      assert(*iter == T(i));
    }

    static_assert(noexcept(*iter) == !cuda::std::same_as<T, NotNoexceptCopy>);
    static_assert(cuda::std::is_same_v<decltype(*iter), T>);
  }
  {
    Iter iter{T{10}};
    for (int i = 10; i < 100; ++i, ++iter)
    {
      assert(*iter == T(i));
    }
    static_assert(noexcept(*iter) == !cuda::std::same_as<T, NotNoexceptCopy>);
    static_assert(cuda::std::is_same_v<decltype(*iter), T>);
  }

  {
    const Iter iter{T{0}};
    assert(*iter == T(0));
    static_assert(noexcept(*iter) == !cuda::std::same_as<T, NotNoexceptCopy>);
    static_assert(cuda::std::is_same_v<decltype(*iter), T>);
  }

  {
    const Iter iter{T{42}};
    assert(*iter == T(42));
    static_assert(noexcept(*iter) == !cuda::std::same_as<T, NotNoexceptCopy>);
    static_assert(cuda::std::is_same_v<decltype(*iter), T>);
  }
}

TEST_FUNC constexpr bool test()
{
  testType<cuda::counting_iterator<SomeInt>>();
  testType<cuda::counting_iterator<SomeInt, cuda::std::int16_t>>();

  testType<cuda::counting_iterator<NotNoexceptCopy>>();
  testType<cuda::counting_iterator<NotNoexceptCopy, cuda::std::int16_t>>();

  testType<cuda::counting_iterator<signed long>>();
  testType<cuda::counting_iterator<signed long, cuda::std::int8_t>>();

  testType<cuda::counting_iterator<unsigned long>>();
  testType<cuda::counting_iterator<unsigned long, cuda::std::int8_t>>();

  testType<cuda::counting_iterator<int>>();
  testType<cuda::counting_iterator<int, cuda::std::int8_t>>();

  testType<cuda::counting_iterator<unsigned>>();
  testType<cuda::counting_iterator<unsigned, cuda::std::int8_t>>();

  testType<cuda::counting_iterator<short>>();
  testType<cuda::counting_iterator<short, cuda::std::int8_t>>();

  testType<cuda::counting_iterator<unsigned short>>();
  testType<cuda::counting_iterator<unsigned short, cuda::std::int8_t>>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
