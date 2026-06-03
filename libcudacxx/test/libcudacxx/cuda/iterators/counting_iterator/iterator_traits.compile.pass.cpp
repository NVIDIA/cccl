//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Test iterator category and iterator concepts.

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"
#include "types.h"

#if !TEST_COMPILER(NVRTC)
#  include <iterator>
#endif // !TEST_COMPILER(NVRTC)

struct Decrementable
{
  using difference_type = int;

#if TEST_HAS_SPACESHIP()
  auto operator<=>(const Decrementable&) const = default;
#else
  TEST_FUNC bool operator==(const Decrementable&) const;
  TEST_FUNC bool operator!=(const Decrementable&) const;

  TEST_FUNC bool operator<(const Decrementable&) const;
  TEST_FUNC bool operator<=(const Decrementable&) const;
  TEST_FUNC bool operator>(const Decrementable&) const;
  TEST_FUNC bool operator>=(const Decrementable&) const;
#endif // TEST_HAS_SPACESHIP()

  TEST_FUNC constexpr Decrementable& operator++()
  {
    return *this;
  }
  TEST_FUNC constexpr Decrementable operator++(int)
  {
    return *this;
  }
  TEST_FUNC constexpr Decrementable& operator--()
  {
    return *this;
  }
  TEST_FUNC constexpr Decrementable operator--(int)
  {
    return *this;
  }
};

struct Incrementable
{
  using difference_type = int;

#if TEST_HAS_SPACESHIP()
  auto operator<=>(const Incrementable&) const = default;
#else
  TEST_FUNC bool operator==(const Incrementable&) const;
  TEST_FUNC bool operator!=(const Incrementable&) const;

  TEST_FUNC bool operator<(const Incrementable&) const;
  TEST_FUNC bool operator<=(const Incrementable&) const;
  TEST_FUNC bool operator>(const Incrementable&) const;
  TEST_FUNC bool operator>=(const Incrementable&) const;
#endif // TEST_HAS_SPACESHIP()

  TEST_FUNC constexpr Incrementable& operator++()
  {
    return *this;
  }
  TEST_FUNC constexpr Incrementable operator++(int)
  {
    return *this;
  }
};

struct BigType
{
  char buffer[128];

  using difference_type = int;

#if TEST_HAS_SPACESHIP()
  auto operator<=>(const BigType&) const = default;
#else
  TEST_FUNC bool operator==(const BigType&) const;
  TEST_FUNC bool operator!=(const BigType&) const;

  TEST_FUNC bool operator<(const BigType&) const;
  TEST_FUNC bool operator<=(const BigType&) const;
  TEST_FUNC bool operator>(const BigType&) const;
  TEST_FUNC bool operator>=(const BigType&) const;
#endif // TEST_HAS_SPACESHIP()

  TEST_FUNC constexpr BigType& operator++()
  {
    return *this;
  }
  TEST_FUNC constexpr BigType operator++(int)
  {
    return *this;
  }
};

struct CharDifferenceType
{
  using difference_type = signed char;

#if TEST_HAS_SPACESHIP()
  auto operator<=>(const CharDifferenceType&) const = default;
#else
  TEST_FUNC bool operator==(const CharDifferenceType&) const;
  TEST_FUNC bool operator!=(const CharDifferenceType&) const;

  TEST_FUNC bool operator<(const CharDifferenceType&) const;
  TEST_FUNC bool operator<=(const CharDifferenceType&) const;
  TEST_FUNC bool operator>(const CharDifferenceType&) const;
  TEST_FUNC bool operator>=(const CharDifferenceType&) const;
#endif // TEST_HAS_SPACESHIP()

  TEST_FUNC constexpr CharDifferenceType& operator++()
  {
    return *this;
  }
  TEST_FUNC constexpr CharDifferenceType operator++(int)
  {
    return *this;
  }
};

template <class T>
_CCCL_CONCEPT HasIteratorCategory =
  _CCCL_REQUIRES_EXPR((T))(typename(typename cuda::std::ranges::iterator_t<T>::iterator_category));

template <template <class...> class Traits>
TEST_FUNC void test()
{
#if _CCCL_HAS_INT128()
  using widest_integer = __int128_t;
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
  using widest_integer = long long;
#endif // !_CCCL_HAS_INT128()

  {
    using Iter       = cuda::counting_iterator<char>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, char>);
    static_assert(sizeof(typename IterTraits::difference_type) > sizeof(char));
    static_assert(cuda::std::is_signed_v<typename IterTraits::difference_type>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, int>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  {
    using DiffType   = cuda::std::int64_t;
    using Iter       = cuda::counting_iterator<char, DiffType>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, char>);
    static_assert(sizeof(typename IterTraits::difference_type) == sizeof(DiffType));
    static_assert(cuda::std::is_signed_v<typename IterTraits::difference_type>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, DiffType>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  {
    using Iter       = cuda::counting_iterator<short>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, short>);
    static_assert(sizeof(typename IterTraits::difference_type) > sizeof(short));
    static_assert(cuda::std::is_signed_v<typename IterTraits::difference_type>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, int>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  {
    using DiffType   = cuda::std::int8_t;
    using Iter       = cuda::counting_iterator<short, cuda::std::int8_t>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, short>);
    static_assert(sizeof(typename IterTraits::difference_type) == sizeof(DiffType));
    static_assert(cuda::std::is_signed_v<typename IterTraits::difference_type>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, DiffType>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  {
    using Iter       = cuda::counting_iterator<int>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(sizeof(typename IterTraits::difference_type) > sizeof(int));
    static_assert(cuda::std::is_signed_v<typename IterTraits::difference_type>);
    // If we're compiling for 32 bit or windows, int and long are the same size, so long long is the correct difference
    // type.
#if INTPTR_MAX == INT32_MAX || defined(_WIN32)
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, long long>);
#else
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, long>);
#endif
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  {
    using DiffType   = cuda::std::int8_t;
    using Iter       = cuda::counting_iterator<int, DiffType>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(sizeof(typename IterTraits::difference_type) == sizeof(DiffType));
    static_assert(cuda::std::is_signed_v<typename IterTraits::difference_type>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, DiffType>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  {
    using Iter       = cuda::counting_iterator<long>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, long>);
    // Same as below, if there is no type larger than long, we can just use that.
    static_assert(sizeof(typename IterTraits::difference_type) >= sizeof(long));
    static_assert(cuda::std::is_signed_v<typename IterTraits::difference_type>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, widest_integer>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  {
    using DiffType   = cuda::std::int8_t;
    using Iter       = cuda::counting_iterator<long, DiffType>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, long>);
    static_assert(sizeof(typename IterTraits::difference_type) == sizeof(DiffType));
    static_assert(cuda::std::is_signed_v<typename IterTraits::difference_type>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, DiffType>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  {
    using Iter       = cuda::counting_iterator<long long>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, long long>);
    // No integer is larger than ptrdiff_t, so it is OK to use ptrdiff_t as the difference type here:
    // https://eel.is/c++draft/range.iota.view#1.3
    static_assert(sizeof(typename IterTraits::difference_type) >= sizeof(cuda::std::ptrdiff_t));
    static_assert(cuda::std::is_signed_v<typename IterTraits::difference_type>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, widest_integer>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  {
    using DiffType   = cuda::std::int16_t;
    using Iter       = cuda::counting_iterator<long long, DiffType>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, long long>);
    static_assert(sizeof(typename IterTraits::difference_type) == sizeof(DiffType));
    static_assert(cuda::std::is_signed_v<typename IterTraits::difference_type>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, DiffType>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  {
    using Iter       = cuda::counting_iterator<Decrementable>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, Decrementable>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, int>);
    static_assert(cuda::std::__has_bidirectional_traversal<Iter>);
  }

  {
    using DiffType   = cuda::std::int16_t;
    using Iter       = cuda::counting_iterator<Decrementable, DiffType>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, Decrementable>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, DiffType>);
    static_assert(cuda::std::__has_bidirectional_traversal<Iter>);
  }

  {
    using Iter       = cuda::counting_iterator<Incrementable>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, Incrementable>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, int>);
    static_assert(cuda::std::__has_forward_traversal<Iter>);
  }

  {
    using DiffType   = cuda::std::int8_t;
    using Iter       = cuda::counting_iterator<Incrementable, DiffType>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, Incrementable>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, DiffType>);
    static_assert(cuda::std::__has_forward_traversal<Iter>);
  }

  { // nothing to do here
    using Iter = cuda::counting_iterator<NotIncrementable>;
    static_assert(!HasIteratorCategory<Iter>);
  }

  { // nothing to do here
    using DiffType = cuda::std::int16_t;
    using Iter     = cuda::counting_iterator<NotIncrementable, DiffType>;
    static_assert(!HasIteratorCategory<Iter>);
  }

  {
    using Iter       = cuda::counting_iterator<BigType>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, BigType>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, int>);
    static_assert(cuda::std::__has_forward_traversal<Iter>);
  }

  {
    using DiffType   = cuda::std::int8_t;
    using Iter       = cuda::counting_iterator<BigType, DiffType>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, BigType>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, DiffType>);
    static_assert(cuda::std::__has_forward_traversal<Iter>);
  }

  {
    using Iter       = cuda::counting_iterator<CharDifferenceType>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, CharDifferenceType>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, signed char>);
    static_assert(cuda::std::__has_forward_traversal<Iter>);
  }

  {
    using DiffType   = cuda::std::int16_t;
    using Iter       = cuda::counting_iterator<CharDifferenceType, DiffType>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, CharDifferenceType>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, DiffType>);
    static_assert(cuda::std::__has_forward_traversal<Iter>);
  }
}

TEST_FUNC void test()
{
  test<cuda::std::iterator_traits>();
#if !TEST_COMPILER(NVRTC)
  test<std::iterator_traits>();
#endif // !TEST_COMPILER(NVRTC)
}

int main(int, char**)
{
  return 0;
}
