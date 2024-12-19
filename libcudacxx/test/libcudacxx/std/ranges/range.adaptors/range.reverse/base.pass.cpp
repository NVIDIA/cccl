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

// constexpr V base() const& requires copy_constructible<V> { return base_; }
// constexpr V base() && { return cuda::std::move(base_); }

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test common ranges.
  {
    // Test non-const.
    {
      auto rev = cuda::std::ranges::reverse_view(BidirRange{buffer, buffer + 8});

      decltype(auto) base = rev.base();
      static_assert(cuda::std::same_as<decltype(base), BidirRange>);
      assert(base.begin_ == buffer);
      assert(base.end_ == buffer + 8);

      decltype(auto) moved = cuda::std::move(rev).base();
      static_assert(cuda::std::same_as<decltype(moved), BidirRange>);
      assert(moved.begin_ == buffer);
      assert(moved.end_ == buffer + 8);
    }
    // Test const.
    {
      const auto rev = cuda::std::ranges::reverse_view(BidirRange{buffer, buffer + 8});

      decltype(auto) base = rev.base();
      static_assert(cuda::std::same_as<decltype(base), BidirRange>);
      assert(base.begin_ == buffer);
      assert(base.end_ == buffer + 8);

      decltype(auto) moved = cuda::std::move(rev).base();
      static_assert(cuda::std::same_as<decltype(moved), BidirRange>);
      assert(moved.begin_ == buffer);
      assert(moved.end_ == buffer + 8);
    }
  }
  // Test non-common ranges.
  {
    // Test non-const (also move only).
    {
      auto rev            = cuda::std::ranges::reverse_view(BidirSentRange<MoveOnly>{buffer, buffer + 8});
      decltype(auto) base = cuda::std::move(rev).base();
      static_assert(cuda::std::same_as<decltype(base), BidirSentRange<MoveOnly>>);
      assert(base.begin_ == buffer);
      assert(base.end_ == buffer + 8);
    }
    // Test const.
    {
      const auto rev = cuda::std::ranges::reverse_view(BidirSentRange<Copyable>{buffer, buffer + 8});

      decltype(auto) base = rev.base();
      static_assert(cuda::std::same_as<decltype(base), BidirSentRange<Copyable>>);
      assert(base.begin_ == buffer);
      assert(base.end_ == buffer + 8);

      decltype(auto) moved = cuda::std::move(rev).base();
      static_assert(cuda::std::same_as<decltype(moved), BidirSentRange<Copyable>>);
      assert(moved.begin_ == buffer);
      assert(moved.end_ == buffer + 8);
    }
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
