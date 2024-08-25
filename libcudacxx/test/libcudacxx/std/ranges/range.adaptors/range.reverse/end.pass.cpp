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

// constexpr reverse_iterator<iterator_t<V>> end();
// constexpr auto end() const requires common_range<const V>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Common bidirectional range.
  {
    auto rev = cuda::std::ranges::reverse_view(BidirRange{buffer, buffer + 8});
    assert(base(rev.end().base()) == buffer);
    assert(base(cuda::std::move(rev).end().base()) == buffer);

    ASSERT_SAME_TYPE(decltype(rev.end()), cuda::std::reverse_iterator<bidirectional_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(rev).end()), cuda::std::reverse_iterator<bidirectional_iterator<int*>>);
  }
  // Const common bidirectional range.
  {
    const auto rev = cuda::std::ranges::reverse_view(BidirRange{buffer, buffer + 8});
    assert(base(rev.end().base()) == buffer);
    assert(base(cuda::std::move(rev).end().base()) == buffer);

    ASSERT_SAME_TYPE(decltype(rev.end()), cuda::std::reverse_iterator<bidirectional_iterator<const int*>>);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(rev).end()),
                     cuda::std::reverse_iterator<bidirectional_iterator<const int*>>);
  }
  // Non-common, non-const (move only) bidirectional range.
  {
    auto rev = cuda::std::ranges::reverse_view(BidirSentRange<MoveOnly>{buffer, buffer + 8});
    assert(base(cuda::std::move(rev).end().base()) == buffer);

    ASSERT_SAME_TYPE(decltype(cuda::std::move(rev).end()), cuda::std::reverse_iterator<bidirectional_iterator<int*>>);
  }
  // Non-common, const bidirectional range.
  {
    auto rev = cuda::std::ranges::reverse_view(BidirSentRange<Copyable>{buffer, buffer + 8});
    assert(base(rev.end().base()) == buffer);
    assert(base(cuda::std::move(rev).end().base()) == buffer);

    ASSERT_SAME_TYPE(decltype(rev.end()), cuda::std::reverse_iterator<bidirectional_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(rev).end()), cuda::std::reverse_iterator<bidirectional_iterator<int*>>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017
  static_assert(test(), "");
#endif

  return 0;
}
