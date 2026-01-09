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
  __host__ __device__ bool operator==(const Decrementable&) const;
  __host__ __device__ bool operator!=(const Decrementable&) const;

  __host__ __device__ bool operator<(const Decrementable&) const;
  __host__ __device__ bool operator<=(const Decrementable&) const;
  __host__ __device__ bool operator>(const Decrementable&) const;
  __host__ __device__ bool operator>=(const Decrementable&) const;
#endif // TEST_HAS_SPACESHIP()

  __host__ __device__ constexpr Decrementable& operator++();
  __host__ __device__ constexpr Decrementable operator++(int);
  __host__ __device__ constexpr Decrementable& operator--();
  __host__ __device__ constexpr Decrementable operator--(int);
};

struct Incrementable
{
  using difference_type = int;

#if TEST_HAS_SPACESHIP()
  auto operator<=>(const Incrementable&) const = default;
#else
  __host__ __device__ bool operator==(const Incrementable&) const;
  __host__ __device__ bool operator!=(const Incrementable&) const;

  __host__ __device__ bool operator<(const Incrementable&) const;
  __host__ __device__ bool operator<=(const Incrementable&) const;
  __host__ __device__ bool operator>(const Incrementable&) const;
  __host__ __device__ bool operator>=(const Incrementable&) const;
#endif // TEST_HAS_SPACESHIP()

  __host__ __device__ constexpr Incrementable& operator++();
  __host__ __device__ constexpr Incrementable operator++(int);
};

struct BigType
{
  char buffer[128];

  using difference_type = int;

#if TEST_HAS_SPACESHIP()
  auto operator<=>(const BigType&) const = default;
#else
  __host__ __device__ bool operator==(const BigType&) const;
  __host__ __device__ bool operator!=(const BigType&) const;

  __host__ __device__ bool operator<(const BigType&) const;
  __host__ __device__ bool operator<=(const BigType&) const;
  __host__ __device__ bool operator>(const BigType&) const;
  __host__ __device__ bool operator>=(const BigType&) const;
#endif // TEST_HAS_SPACESHIP()

  __host__ __device__ constexpr BigType& operator++();
  __host__ __device__ constexpr BigType operator++(int);
};

struct CharDifferenceType
{
  using difference_type = signed char;

#if TEST_HAS_SPACESHIP()
  auto operator<=>(const CharDifferenceType&) const = default;
#else
  __host__ __device__ bool operator==(const CharDifferenceType&) const;
  __host__ __device__ bool operator!=(const CharDifferenceType&) const;

  __host__ __device__ bool operator<(const CharDifferenceType&) const;
  __host__ __device__ bool operator<=(const CharDifferenceType&) const;
  __host__ __device__ bool operator>(const CharDifferenceType&) const;
  __host__ __device__ bool operator>=(const CharDifferenceType&) const;
#endif // TEST_HAS_SPACESHIP()

  __host__ __device__ constexpr CharDifferenceType& operator++();
  __host__ __device__ constexpr CharDifferenceType operator++(int);
};

template <class T>
_CCCL_CONCEPT HasIteratorCategory =
  _CCCL_REQUIRES_EXPR((T))(typename(typename cuda::std::ranges::iterator_t<T>::iterator_category));

__host__ __device__ void test()
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
    using Iter = cuda::counting_iterator<Decrementable>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, Decrementable>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
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
    using Iter = cuda::counting_iterator<NotIncrementable>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(!HasIteratorCategory<Iter>);
    static_assert(cuda::std::same_as<Iter::value_type, NotIncrementable>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
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
    using Iter = cuda::counting_iterator<CharDifferenceType>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, CharDifferenceType>);
    static_assert(cuda::std::same_as<Iter::difference_type, signed char>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }
}

int main(int, char**)
{
  return 0;
}
