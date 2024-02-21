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

// Test iterator category and iterator concepts.

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

struct Decrementable
{
  using difference_type = int;

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const Decrementable&) const = default;
#else
  __host__ __device__ bool operator==(const Decrementable&) const;
  __host__ __device__ bool operator!=(const Decrementable&) const;

  __host__ __device__ bool operator<(const Decrementable&) const;
  __host__ __device__ bool operator<=(const Decrementable&) const;
  __host__ __device__ bool operator>(const Decrementable&) const;
  __host__ __device__ bool operator>=(const Decrementable&) const;
#endif // !TEST_HAS_NO_SPACESHIP_OPERATOR

  __host__ __device__ constexpr Decrementable& operator++();
  __host__ __device__ constexpr Decrementable operator++(int);
  __host__ __device__ constexpr Decrementable& operator--();
  __host__ __device__ constexpr Decrementable operator--(int);
};

struct Incrementable
{
  using difference_type = int;

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const Incrementable&) const = default;
#else
  __host__ __device__ bool operator==(const Incrementable&) const;
  __host__ __device__ bool operator!=(const Incrementable&) const;

  __host__ __device__ bool operator<(const Incrementable&) const;
  __host__ __device__ bool operator<=(const Incrementable&) const;
  __host__ __device__ bool operator>(const Incrementable&) const;
  __host__ __device__ bool operator>=(const Incrementable&) const;
#endif // !TEST_HAS_NO_SPACESHIP_OPERATOR

  __host__ __device__ constexpr Incrementable& operator++();
  __host__ __device__ constexpr Incrementable operator++(int);
};

struct BigType
{
  char buffer[128];

  using difference_type = int;

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const BigType&) const = default;
#else
  __host__ __device__ bool operator==(const BigType&) const;
  __host__ __device__ bool operator!=(const BigType&) const;

  __host__ __device__ bool operator<(const BigType&) const;
  __host__ __device__ bool operator<=(const BigType&) const;
  __host__ __device__ bool operator>(const BigType&) const;
  __host__ __device__ bool operator>=(const BigType&) const;
#endif // !TEST_HAS_NO_SPACESHIP_OPERATOR

  __host__ __device__ constexpr BigType& operator++();
  __host__ __device__ constexpr BigType operator++(int);
};

struct CharDifferenceType
{
  using difference_type = signed char;

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const CharDifferenceType&) const = default;
#else
  __host__ __device__ bool operator==(const CharDifferenceType&) const;
  __host__ __device__ bool operator!=(const CharDifferenceType&) const;

  __host__ __device__ bool operator<(const CharDifferenceType&) const;
  __host__ __device__ bool operator<=(const CharDifferenceType&) const;
  __host__ __device__ bool operator>(const CharDifferenceType&) const;
  __host__ __device__ bool operator>=(const CharDifferenceType&) const;
#endif // !TEST_HAS_NO_SPACESHIP_OPERATOR

  __host__ __device__ constexpr CharDifferenceType& operator++();
  __host__ __device__ constexpr CharDifferenceType operator++(int);
};

#if TEST_STD_VER >= 2020
template <class T>
concept HasIteratorCategory = requires { typename cuda::std::ranges::iterator_t<T>::iterator_category; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool HasIteratorCategory = false;

template <class T>
inline constexpr bool
  HasIteratorCategory<T, cuda::std::void_t<typename cuda::std::ranges::iterator_t<T>::iterator_category>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ void test()
{
  {
    const cuda::std::ranges::iota_view<char> io(0);
    using Iter = decltype(io.begin());
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, char>);
    static_assert(sizeof(Iter::difference_type) > sizeof(char));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    LIBCPP_STATIC_ASSERT(cuda::std::same_as<Iter::difference_type, int>);
    unused(io);
  }
  {
    const cuda::std::ranges::iota_view<short> io(0);
    using Iter = decltype(io.begin());
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, short>);
    static_assert(sizeof(Iter::difference_type) > sizeof(short));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    LIBCPP_STATIC_ASSERT(cuda::std::same_as<Iter::difference_type, int>);
    unused(io);
  }
  {
    const cuda::std::ranges::iota_view<int> io(0);
    using Iter = decltype(io.begin());
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, int>);
    static_assert(sizeof(Iter::difference_type) > sizeof(int));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    // If we're compiling for 32 bit or windows, int and long are the same size, so long long is the correct difference
    // type.
#if INTPTR_MAX == INT32_MAX || defined(_WIN32)
    LIBCPP_STATIC_ASSERT(cuda::std::same_as<Iter::difference_type, long long>);
#else
    LIBCPP_STATIC_ASSERT(cuda::std::same_as<Iter::difference_type, long>);
#endif
    unused(io);
  }
  {
    const cuda::std::ranges::iota_view<long> io(0);
    using Iter = decltype(io.begin());
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, long>);
    // Same as below, if there is no type larger than long, we can just use that.
    static_assert(sizeof(Iter::difference_type) >= sizeof(long));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    LIBCPP_STATIC_ASSERT(cuda::std::same_as<Iter::difference_type, long long>);
    unused(io);
  }
  {
    const cuda::std::ranges::iota_view<long long> io(0);
    using Iter = decltype(io.begin());
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, long long>);
    // No integer is larger than long long, so it is OK to use long long as the difference type here:
    // https://eel.is/c++draft/range.iota.view#1.3
    static_assert(sizeof(Iter::difference_type) >= sizeof(long long));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    LIBCPP_STATIC_ASSERT(cuda::std::same_as<Iter::difference_type, long long>);
    unused(io);
  }
  {
    const cuda::std::ranges::iota_view<Decrementable> io;
    using Iter = decltype(io.begin());
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, Decrementable>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
    unused(io);
  }
  {
    const cuda::std::ranges::iota_view<Incrementable> io;
    using Iter = decltype(io.begin());
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, Incrementable>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
    unused(io);
  }
  {
    const cuda::std::ranges::iota_view<NotIncrementable> io(NotIncrementable(0));
    using Iter = decltype(io.begin());
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(!HasIteratorCategory<cuda::std::ranges::iota_view<NotIncrementable>>);
    static_assert(cuda::std::same_as<Iter::value_type, NotIncrementable>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
    unused(io);
  }
  {
    const cuda::std::ranges::iota_view<BigType> io;
    using Iter = decltype(io.begin());
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, BigType>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
    unused(io);
  }
  {
    const cuda::std::ranges::iota_view<CharDifferenceType> io;
    using Iter = decltype(io.begin());
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, CharDifferenceType>);
    static_assert(cuda::std::same_as<Iter::difference_type, signed char>);
    unused(io);
  }
}

int main(int, char**)
{
  return 0;
}
