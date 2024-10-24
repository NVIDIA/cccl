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

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int) requires forward_range<V>;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

struct EqualTo
{
  int x;
  __host__ __device__ constexpr bool operator()(int y) const
  {
    return x == y;
  }
};

struct TrackingPred : TrackInitialization
{
  using TrackInitialization::TrackInitialization;
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i == 1;
  }
};

template <class Iterator, bool IsForwardRange, bool IsConst>
__host__ __device__ constexpr void test()
{
  using Sentinel       = sentinel_wrapper<Iterator>;
  using View           = minimal_view<Iterator, Sentinel>;
  using FilterView     = cuda::std::ranges::filter_view<View, EqualTo>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;

  // Increment an iterator when it won't find another satisfied value after begin()
  {
    cuda::std::array<int, 5> array{0, 1, 2, 3, 4};
    FilterView view = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, EqualTo{1}};

    FilterIterator it      = view.begin();
    FilterIterator& result = ++it;
    ASSERT_SAME_TYPE(FilterIterator&, decltype(++it));
    assert(&result == &it);
    assert(base(result.base()) == array.end());
  }

  // Increment the iterator and it finds another value after begin()
  {
    cuda::std::array<int, 5> array{99, 1, 99, 1, 99};
    FilterView view = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, EqualTo{1}};

    FilterIterator it = view.begin();
    ++it;
    assert(base(it.base()) == array.begin() + 3);
  }

  // Increment advances all the way to the end of the range
  {
    cuda::std::array<int, 5> array{99, 1, 99, 99, 1};
    FilterView view = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, EqualTo{1}};

    FilterIterator it = view.begin();
    ++it;
    assert(base(it.base()) == array.begin() + 4);
  }

  // Increment an iterator multiple times
  {
    cuda::std::array<int, 10> array{0, 1, 2, 3, 1, 1, 4, 5, 1, 6};
    FilterView view = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, EqualTo{1}};

    FilterIterator it = view.begin();
    assert(base(it.base()) == array.begin() + 1);
    ++it;
    assert(base(it.base()) == array.begin() + 4);
    ++it;
    assert(base(it.base()) == array.begin() + 5);
    ++it;
    assert(base(it.base()) == array.begin() + 8);
    ++it;
    assert(base(it.base()) == array.end());
  }

  // Test with a predicate that takes by non-const reference
  if constexpr (!IsConst)
  {
    cuda::std::array<int, 4> array{99, 1, 99, 1};
    View v{Iterator(array.begin()), Sentinel(Iterator(array.end()))};
    auto pred = [](int& x) {
      return x == 1;
    };
    cuda::std::ranges::filter_view<View, decltype(pred)> view(cuda::std::move(v), pred);
    auto it = view.begin();
    assert(base(it.base()) == array.begin() + 1);
    ++it;
    assert(base(it.base()) == array.begin() + 3);
  }

  // Make sure we do not make a copy of the predicate when we increment
  // (we should be passing it to ranges::find_if using cuda::std::ref)
  {
    bool moved = false, copied = false;
    cuda::std::array<int, 3> array{1, 1, 1};
    View v{Iterator(array.begin()), Sentinel(Iterator(array.end()))};
    cuda::std::ranges::filter_view<View, TrackingPred> view(cuda::std::move(v), TrackingPred(&moved, &copied));
    moved   = false;
    copied  = false;
    auto it = view.begin();
    ++it;
    it++;
    assert(!moved);
    assert(!copied);
  }

  // Check post-increment for input ranges
  if constexpr (!IsForwardRange)
  {
    cuda::std::array<int, 10> array{0, 1, 2, 3, 1, 1, 4, 5, 1, 6};
    FilterView view = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, EqualTo{1}};

    FilterIterator it = view.begin();
    assert(base(it.base()) == array.begin() + 1);
    it++;
    assert(base(it.base()) == array.begin() + 4);
    it++;
    assert(base(it.base()) == array.begin() + 5);
    it++;
    assert(base(it.base()) == array.begin() + 8);
    it++;
    assert(base(it.base()) == array.end());
    static_assert(cuda::std::is_same_v<decltype(it++), void>);
  }

  // Check post-increment for forward ranges
  if constexpr (IsForwardRange)
  {
    cuda::std::array<int, 10> array{0, 1, 2, 3, 1, 1, 4, 5, 1, 6};
    FilterView view = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, EqualTo{1}};

    FilterIterator it     = view.begin();
    FilterIterator result = it++;
    ASSERT_SAME_TYPE(FilterIterator, decltype(it++));
    assert(base(result.base()) == array.begin() + 1);
    assert(base(it.base()) == array.begin() + 4);

    result = it++;
    assert(base(result.base()) == array.begin() + 4);
    assert(base(it.base()) == array.begin() + 5);

    result = it++;
    assert(base(result.base()) == array.begin() + 5);
    assert(base(it.base()) == array.begin() + 8);

    result = it++;
    assert(base(result.base()) == array.begin() + 8);
    assert(base(it.base()) == array.end());
  }
}

__host__ __device__ constexpr bool tests()
{
  test<cpp17_input_iterator<int*>, /* IsForwardRange */ false, /* IsConst */ false>();
  test<cpp20_input_iterator<int*>, /* IsForwardRange */ false, /* IsConst */ false>();
  test<forward_iterator<int*>, /* IsForwardRange */ true, /* IsConst */ false>();
  test<bidirectional_iterator<int*>, /* IsForwardRange */ true, /* IsConst */ false>();
  test<random_access_iterator<int*>, /* IsForwardRange */ true, /* IsConst */ false>();
  test<contiguous_iterator<int*>, /* IsForwardRange */ true, /* IsConst */ false>();
  test<int*, /* IsForwardRange */ true, /* IsConst */ false>();

  test<cpp17_input_iterator<int const*>, /* IsForwardRange */ false, /* IsConst */ true>();
  test<cpp20_input_iterator<int const*>, /* IsForwardRange */ false, /* IsConst */ true>();
  test<forward_iterator<int const*>, /* IsForwardRange */ true, /* IsConst */ true>();
  test<bidirectional_iterator<int const*>, /* IsForwardRange */ true, /* IsConst */ true>();
  test<random_access_iterator<int const*>, /* IsForwardRange */ true, /* IsConst */ true>();
  test<contiguous_iterator<int const*>, /* IsForwardRange */ true, /* IsConst */ true>();
  test<int const*, /* IsForwardRange */ true, /* IsConst */ true>();
  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(tests(), "");
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
