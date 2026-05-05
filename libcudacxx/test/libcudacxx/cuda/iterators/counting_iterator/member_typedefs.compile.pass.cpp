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
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "types.h"

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

  TEST_FUNC constexpr Decrementable& operator++();
  TEST_FUNC constexpr Decrementable operator++(int);
  TEST_FUNC constexpr Decrementable& operator--();
  TEST_FUNC constexpr Decrementable operator--(int);
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

  TEST_FUNC constexpr Incrementable& operator++();
  TEST_FUNC constexpr Incrementable operator++(int);
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

  TEST_FUNC constexpr BigType& operator++();
  TEST_FUNC constexpr BigType operator++(int);
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

  TEST_FUNC constexpr CharDifferenceType& operator++();
  TEST_FUNC constexpr CharDifferenceType operator++(int);
};

template <class T>
_CCCL_CONCEPT HasIteratorCategory =
  _CCCL_REQUIRES_EXPR((T))(typename(typename cuda::std::ranges::iterator_t<T>::iterator_category));

TEST_FUNC void test()
{
#if _CCCL_HAS_INT128()
  using widest_integer = __int128_t;
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
  using widest_integer = long long;
#endif // !_CCCL_HAS_INT128()
  {
    using Iter = cuda::counting_iterator<char>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, char>);
    static_assert(sizeof(Iter::difference_type) > sizeof(char));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using DiffType = cuda::std::int64_t;
    using Iter     = cuda::counting_iterator<char, DiffType>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, char>);
    static_assert(sizeof(Iter::difference_type) == sizeof(DiffType));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, DiffType>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::counting_iterator<short>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, short>);
    static_assert(sizeof(Iter::difference_type) > sizeof(short));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using DiffType = cuda::std::int8_t;
    using Iter     = cuda::counting_iterator<short, DiffType>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, short>);
    static_assert(sizeof(Iter::difference_type) == sizeof(DiffType));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, DiffType>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::counting_iterator<int>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, int>);
    static_assert(sizeof(Iter::difference_type) > sizeof(int));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    // If we're compiling for 32 bit or windows, int and long are the same size, so long long is the correct difference
    // type.
#if INTPTR_MAX == INT32_MAX || defined(_WIN32)
    static_assert(cuda::std::same_as<Iter::difference_type, long long>);
#else
    static_assert(cuda::std::same_as<Iter::difference_type, long>);
#endif
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using DiffType = cuda::std::int8_t;
    using Iter     = cuda::counting_iterator<int, DiffType>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, int>);
    static_assert(sizeof(Iter::difference_type) == sizeof(DiffType));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, DiffType>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::counting_iterator<long>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, long>);
    // Same as below, if there is no type larger than long, we can just use that.
    static_assert(sizeof(Iter::difference_type) >= sizeof(long));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, widest_integer>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using DiffType = cuda::std::int8_t;
    using Iter     = cuda::counting_iterator<long, DiffType>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, long>);
    static_assert(sizeof(Iter::difference_type) == sizeof(DiffType));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, DiffType>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::counting_iterator<long long>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, long long>);
    // No integer is larger than long long, so it is OK to use long long as the difference type here:
    // https://eel.is/c++draft/range.iota.view#1.3
    static_assert(sizeof(Iter::difference_type) >= sizeof(long long));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, widest_integer>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using DiffType = cuda::std::int16_t;
    using Iter     = cuda::counting_iterator<long long, DiffType>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, long long>);
    static_assert(sizeof(Iter::difference_type) == sizeof(DiffType));
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, DiffType>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::counting_iterator<Decrementable>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, Decrementable>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using DiffType = cuda::std::int16_t;
    using Iter     = cuda::counting_iterator<Decrementable, DiffType>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, Decrementable>);
    static_assert(cuda::std::same_as<Iter::difference_type, DiffType>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::counting_iterator<Incrementable>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, Incrementable>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using DiffType = cuda::std::int8_t;
    using Iter     = cuda::counting_iterator<Incrementable, DiffType>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, Incrementable>);
    static_assert(cuda::std::same_as<Iter::difference_type, DiffType>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::counting_iterator<NotIncrementable>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(!HasIteratorCategory<Iter>);
    static_assert(cuda::std::same_as<Iter::value_type, NotIncrementable>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using DiffType = cuda::std::int16_t;
    using Iter     = cuda::counting_iterator<NotIncrementable, DiffType>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(!HasIteratorCategory<Iter>);
    static_assert(cuda::std::same_as<Iter::value_type, NotIncrementable>);
    static_assert(cuda::std::same_as<Iter::difference_type, DiffType>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::counting_iterator<BigType>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, BigType>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using DiffType = cuda::std::int8_t;
    using Iter     = cuda::counting_iterator<BigType, DiffType>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, BigType>);
    static_assert(cuda::std::same_as<Iter::difference_type, DiffType>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::counting_iterator<CharDifferenceType>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, CharDifferenceType>);
    static_assert(cuda::std::same_as<Iter::difference_type, signed char>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using DiffType = cuda::std::int16_t;
    using Iter     = cuda::counting_iterator<CharDifferenceType, DiffType>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, CharDifferenceType>);
    static_assert(cuda::std::same_as<Iter::difference_type, DiffType>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }
}

int main(int, char**)
{
  return 0;
}
