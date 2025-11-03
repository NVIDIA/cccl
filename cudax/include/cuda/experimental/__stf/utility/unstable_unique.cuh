//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file
 * @brief Implementation of unstable_unique
 */

#include <cuda/experimental/__stf/utility/unittest.cuh>

#include <iostream>
#include <iterator>
#include <vector>

namespace cuda::experimental::stf
{
/**
 * @brief Removes duplicates from a range using a custom predicate.
 *
 * This function operates from both sides of the range, moving elements from the right-hand side to the left to
 * eliminate duplicates. The order of the elements is not preserved.
 *
 * @tparam iterator The type of the iterator.
 * @tparam BinaryPredicate The type of the predicate.
 * @param first, last The range of elements to remove duplicates from.
 * @param p The predicate to use for comparing elements.
 *
 * @return iterator The new end of the range after duplicates have been removed.
 *
 * @par Example:
 * @code
 * ::std::vector<int> v = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
 * auto new_end = unstable_unique(v.begin(), v.end(), [](int a, int b) { return a == b; });
 * v.erase(new_end, v.end());
 * // v is now {1, 5, 2, 4, 3, 5}
 * @endcode
 */
template <class iterator, class BinaryPredicate>
iterator unstable_unique(iterator first, iterator last, BinaryPredicate p)
{
  if (first == last || ::std::next(first) == last)
  {
    return last; // 0 or 1-element range
  }

  ++first; // position first on the first unknown element (first element will stay by definition)
  bool first_is_known_duplicate = false; // see below for description

  for (; first < last; ++first)
  {
    if (!first_is_known_duplicate)
    {
      if (!p(*first, *::std::prev(first)))
      {
        continue;
      }
    }
    // Here we know that `*first` is a dupe and should be replaced. Also the range is not empty.
    assert(first < last);
    for (--last;; --last)
    {
      if (first == last)
      {
        return first; // just past the last unique element
      }
      assert(first < last);
      if (!p(*last, *::std::prev(last)))
      {
        break;
      }
    }
    assert(!p(*first, *last));
    // Here we know we're good to replace *first with *last.
    // Complicating matter: if we do so, we "forget" whether *::std::next(first) is a duplicate of *first.
    // Maintain `first_is_known_duplicate` to keep track of that.
    first_is_known_duplicate = p(*first, *::std::next(first));
    *first                   = mv(*last);
  }

  return first;
}

/**
 * @brief Removes duplicates from a range using the built-in operator==.
 *
 * This function operates like `unstable_unique` above with `operator==` as the predicate.
 *
 * @tparam iterator The type of the iterator.
 * @param first, last The range of elements to remove duplicates from.
 *
 * @return iterator The new end of the range after duplicates have been removed.
 *
 * @par Example:
 * @code
 * ::std::vector<int> v = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
 * auto new_end = unstable_unique(v.begin(), v.end());
 * v.erase(new_end, v.end());
 * // v is now {1, 5, 2, 4, 3, 5}
 * @endcode
 */
template <class iterator>
iterator unstable_unique(iterator first, iterator last)
{
  return unstable_unique(first, last, [](auto& a, auto& b) {
    return a == b;
  });
}

#ifdef UNITTESTED_FILE
UNITTEST("unstable_unique")
{
  ::std::vector<int> v1;
  auto new_end1 = unstable_unique(v1.begin(), v1.end());
  EXPECT(v1.end() == new_end1);

  ::std::vector<int> v2 = {1, 2, 3, 4, 5};
  auto new_end2         = unstable_unique(v2.begin(), v2.end());
  // ::std::cout << new_end2 - v2.begin() << '\n';
  EXPECT(v2.end() == new_end2);
  // ::std::copy(v2.begin(), new_end2, ::std::ostream_iterator<int>(::std::cout, " "));
  EXPECT(::std::vector<int>({1, 2, 3, 4, 5}) == v2);

  ::std::vector<int> v3 = {1, 1, 2, 3, 4, 5};
  auto new_end3         = unstable_unique(v3.begin(), v3.end());
  // ::std::cerr << new_end3 - v3.begin() << '\n';
  EXPECT(v3.begin() + 5 == new_end3);
  EXPECT(::std::vector<int>({1, 5, 2, 3, 4, 5}) == v3);
  ::std::vector<int> v4 = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  auto new_end4         = unstable_unique(v4.begin(), v4.end());
  // ::std::cerr << new_end4 - v4.begin() << '\n';
  EXPECT(v4.begin() + 5 == new_end4);
  EXPECT(::std::vector<int>({1, 5, 2, 4, 3, 3, 4, 4, 5, 5}) == v4);

  ::std::vector<int> v5 = {1, 1, 1, 1, 1};
  auto new_end5         = unstable_unique(v5.begin(), v5.end());
  EXPECT(1 + v5.begin() == new_end5);
  EXPECT(::std::vector<int>({1, 1, 1, 1, 1}) == v5);

  ::std::vector<int> v6 = {1, 1, 1, 1, 1, 2};
  auto new_end6         = unstable_unique(v6.begin(), v6.end());
  EXPECT(v6.begin() + 2 == new_end6);
  EXPECT(::std::vector<int>({1, 2, 1, 1, 1, 2}) == v6);

  ::std::vector<int> v7 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 5};
  assert(v7.size() == 15);
  auto new_end7 = unstable_unique(v7.begin(), v7.end(), [](int a, int b) {
    return a == b;
  });
  // ::std::cerr << new_end7 - v7.begin() << '\n';
  EXPECT(v7.begin() + 5 == new_end7);
  EXPECT(::std::vector<int>{1, 5, 4, 3, 2, 1, 1, 1, 1, 2, 2, 2, 3, 4, 5} == v7);
};
#endif // UNITTESTED_FILE
} // end namespace cuda::experimental::stf
