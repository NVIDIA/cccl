//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// Iterator traits and member typedefs in zip_view::<iterator>.

#include <cuda/iterator>
#include <cuda/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class T>
_CCCL_CONCEPT HasIterCategory = _CCCL_REQUIRES_EXPR((T))(typename(typename T::iterator_category));

template <class T>
struct DiffTypeIter
{
  using iterator_category = cuda::std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = T;

  TEST_FUNC int operator*() const;
  TEST_FUNC DiffTypeIter& operator++();
  TEST_FUNC void operator++(int);
#if TEST_STD_VER >= 2020
  TEST_FUNC friend constexpr bool operator==(DiffTypeIter, DiffTypeIter) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  TEST_FUNC friend constexpr bool operator==(const DiffTypeIter&, const DiffTypeIter&)
  {
    return true;
  }
  TEST_FUNC friend constexpr bool operator!=(const DiffTypeIter&, const DiffTypeIter&)
  {
    return false;
  }
#endif // TEST_STD_VER <=2017
};

struct Foo
{
  TEST_FUNC constexpr operator int() const noexcept
  {
    return 42;
  }
};

TEST_FUNC void test()
{
  {
    using Iter = cuda::zip_transform_iterator<TimesTwo, int*>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
    static_assert(cuda::std::is_same_v<Iter::reference, int>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  {
    using Iter = cuda::zip_transform_iterator<Plus, int*, Foo*>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
    static_assert(cuda::std::is_same_v<Iter::reference, int>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  { // value type is not a reference
    using Iter = cuda::zip_transform_iterator<ReturnFirstLvalueReference, int*, Foo*>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
    static_assert(cuda::std::is_same_v<Iter::reference, int&>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  { // value type is not a reference
    using Iter = cuda::zip_transform_iterator<ReturnFirstRvalueReference, int*, Foo*>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
    static_assert(cuda::std::is_same_v<Iter::reference, int&&>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  { // If one iterator is not random access then the whole zip_transform_iterator is not random access
    using Iter = cuda::zip_transform_iterator<ReturnFirstRvalueReference, bidirectional_iterator<int*>, Foo*>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
    static_assert(cuda::std::is_same_v<Iter::reference, int&&>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::bidirectional_iterator<Iter>);
  }

  { // If one iterator is not bidirectional_iterator then the whole zip_transform_iterator is not bidirectional_iterator
    using Iter = cuda::zip_transform_iterator<ReturnFirstRvalueReference, forward_iterator<int*>, Foo*>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
    static_assert(cuda::std::is_same_v<Iter::reference, int&&>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::forward_iterator<Iter>);
  }

  { // If one iterator is not forward_iterator then the whole zip_transform_iterator is not forward_iterator
    using Iter =
      cuda::zip_transform_iterator<ReturnFirstRvalueReference, forward_iterator<int*>, cpp20_input_iterator<Foo*>>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
    static_assert(cuda::std::is_same_v<Iter::reference, int&&>);
    static_assert(cuda::std::input_iterator<Iter>);
  }

  { // Takes the difference type from the base iterator
    using Iter = cuda::zip_transform_iterator<TimesTwo, DiffTypeIter<intptr_t>>;
    static_assert(cuda::std::is_same_v<Iter::difference_type, intptr_t>);
  }

  { // Difference type is the common type of the difference types
    using Iter = cuda::zip_transform_iterator<Plus, DiffTypeIter<intptr_t>, DiffTypeIter<cuda::std::ptrdiff_t>>;
    static_assert(
      cuda::std::is_same_v<Iter::difference_type, cuda::std::common_type_t<intptr_t, cuda::std::ptrdiff_t>>);
  }
}

int main(int, char**)
{
  return 0;
}
