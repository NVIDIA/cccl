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

// constexpr iterator_t<V> const& base() const& noexcept;
// constexpr iterator_t<V> base() &&;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
__host__ __device__ constexpr void test()
{
  using View           = minimal_view<Iterator, Sentinel>;
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;

  cuda::std::array<int, 5> array{0, 1, 2, 3, 4};
  FilterView view = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, AlwaysTrue{}};

  // Test the const& version
  {
    FilterIterator const iter(view, Iterator(array.begin()));
    Iterator const& result = iter.base();
    ASSERT_SAME_TYPE(Iterator const&, decltype(iter.base()));
    ASSERT_NOEXCEPT(iter.base());
    assert(base(result) == array.begin());
  }

  // Test the && version
  {
    FilterIterator iter(view, Iterator(array.begin()));
    Iterator result = cuda::std::move(iter).base();
    ASSERT_SAME_TYPE(Iterator, decltype(cuda::std::move(iter).base()));
    assert(base(result) == array.begin());
  }
}

__host__ __device__ constexpr bool tests()
{
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();
  test<int const*>();
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
