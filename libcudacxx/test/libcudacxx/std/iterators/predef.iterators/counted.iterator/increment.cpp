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

// constexpr counted_iterator& operator++();
// decltype(auto) operator++(int);
// constexpr counted_iterator operator++(int)
//   requires forward_iterator<I>;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
template <class It>
class ThrowsOnInc
{
  It it_;

public:
  typedef cuda::std::input_iterator_tag iterator_category;
  typedef typename cuda::std::iterator_traits<It>::value_type value_type;
  typedef typename cuda::std::iterator_traits<It>::difference_type difference_type;
  typedef It pointer;
  typedef typename cuda::std::iterator_traits<It>::reference reference;

  __host__ __device__ constexpr It base() const
  {
    return it_;
  }

  ThrowsOnInc() = default;
  __host__ __device__ explicit constexpr ThrowsOnInc(It it)
      : it_(it)
  {}

  __host__ __device__ constexpr reference operator*() const
  {
    return *it_;
  }

  __host__ __device__ constexpr ThrowsOnInc& operator++()
  {
    throw 42;
  }
  __host__ __device__ constexpr ThrowsOnInc operator++(int)
  {
    throw 42;
  }
};
#endif // TEST_HAS_NO_EXCEPTIONS

struct InputOrOutputArchetype
{
  using difference_type = int;

  int* ptr;

  __host__ __device__ constexpr int operator*() const
  {
    return *ptr;
  }
  __host__ __device__ constexpr void operator++(int)
  {
    ++ptr;
  }
  __host__ __device__ constexpr InputOrOutputArchetype& operator++()
  {
    ++ptr;
    return *this;
  }
};

#if TEST_STD_VER >= 2020
template <class Iter>
concept PlusEnabled = requires(Iter& iter) {
  iter++;
  ++iter;
};
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class, class = void>
inline constexpr bool PlusEnabled = false;

template <class Iter>
inline constexpr bool
  PlusEnabled<Iter, cuda::std::void_t<decltype(cuda::std::declval<Iter&>()++), decltype(++cuda::std::declval<Iter&>())>> =
    true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    using Counted = cuda::std::counted_iterator<forward_iterator<int*>>;
    cuda::std::counted_iterator iter(forward_iterator<int*>{buffer}, 8);

    assert(iter++ == Counted(forward_iterator<int*>{buffer}, 8));
    assert(++iter == Counted(forward_iterator<int*>{buffer + 2}, 6));

    ASSERT_SAME_TYPE(decltype(iter++), Counted);
    ASSERT_SAME_TYPE(decltype(++iter), Counted&);
  }
  {
    using Counted = cuda::std::counted_iterator<random_access_iterator<int*>>;
    cuda::std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);

    assert(iter++ == Counted(random_access_iterator<int*>{buffer}, 8));
    assert(++iter == Counted(random_access_iterator<int*>{buffer + 2}, 6));

    ASSERT_SAME_TYPE(decltype(iter++), Counted);
    ASSERT_SAME_TYPE(decltype(++iter), Counted&);
  }

  {
    static_assert(PlusEnabled<cuda::std::counted_iterator<random_access_iterator<int*>>>);
    static_assert(!PlusEnabled<const cuda::std::counted_iterator<random_access_iterator<int*>>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    using Counted = cuda::std::counted_iterator<InputOrOutputArchetype>;
    cuda::std::counted_iterator iter(InputOrOutputArchetype{buffer}, 8);

    iter++;
    assert((++iter).base().ptr == buffer + 2);

    ASSERT_SAME_TYPE(decltype(iter++), void);
    ASSERT_SAME_TYPE(decltype(++iter), Counted&);
  }
  {
    using Counted = cuda::std::counted_iterator<cpp20_input_iterator<int*>>;
    cuda::std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);

    iter++;
    assert(++iter == Counted(cpp20_input_iterator<int*>{buffer + 2}, 6));

    ASSERT_SAME_TYPE(decltype(iter++), void);
    ASSERT_SAME_TYPE(decltype(++iter), Counted&);
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    using Counted = cuda::std::counted_iterator<ThrowsOnInc<int*>>;
    cuda::std::counted_iterator iter(ThrowsOnInc<int*>{buffer}, 8);
    try
    {
      (void) iter++;
      assert(false);
    }
    catch (int x)
    {
      assert(x == 42);
      assert(iter.count() == 8);
    }

    ASSERT_SAME_TYPE(decltype(iter++), ThrowsOnInc<int*>);
    ASSERT_SAME_TYPE(decltype(++iter), Counted&);
  }
#endif // TEST_HAS_NO_EXCEPTIONS

  return 0;
}
