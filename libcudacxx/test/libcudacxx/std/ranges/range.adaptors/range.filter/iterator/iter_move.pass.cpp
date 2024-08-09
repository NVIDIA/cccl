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

// friend constexpr range_rvalue_reference_t<V> iter_move(iterator const& i)
//  noexcept(noexcept(ranges::iter_move(i.current_)));

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iterator, bool HasNoexceptIterMove>
__host__ __device__ constexpr void test()
{
  using Sentinel       = sentinel_wrapper<Iterator>;
  using View           = minimal_view<Iterator, Sentinel>;
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;

  {
    cuda::std::array<int, 5> array{0, 1, 2, 3, 4};
    FilterView view         = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, AlwaysTrue{}};
    FilterIterator const it = view.begin();

    int&& result = iter_move(it);
#if !defined(TEST_COMPILER_ICC) // broken noexcept
    static_assert(noexcept(iter_move(it)) == HasNoexceptIterMove);
#endif // !TEST_COMPILER_ICC
    assert(&result == array.begin());
  }
}

__host__ __device__ constexpr bool tests()
{
  test<cpp17_input_iterator<int*>, /* noexcept */ false>();
  test<cpp20_input_iterator<int*>, /* noexcept */ false>();
  test<forward_iterator<int*>, /* noexcept */ false>();
  test<bidirectional_iterator<int*>, /* noexcept */ false>();
  test<random_access_iterator<int*>, /* noexcept */ false>();
  test<contiguous_iterator<int*>, /* noexcept */ false>();
  test<int*, /* noexcept */ true>();
  test<NoexceptIterMoveInputIterator<true>, /* noexcept */ true>();
  test<NoexceptIterMoveInputIterator<false>, /* noexcept */ false>();
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
