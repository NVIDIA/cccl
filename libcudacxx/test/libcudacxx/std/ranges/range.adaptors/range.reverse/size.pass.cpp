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

// constexpr auto size() requires sized_range<V>;
// constexpr auto size() const requires sized_range<const V>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

// end -  begin = 8, but size may return something else.
template <CopyCategory CC>
struct BidirSizedRange : cuda::std::ranges::view_base
{
  int* ptr_;
  size_t size_;

  __host__ __device__ constexpr BidirSizedRange(int* ptr, size_t size)
      : ptr_(ptr)
      , size_(size)
  {}

  template <CopyCategory CC2 = CC, cuda::std::enable_if_t<CC2 == Copyable, int> = 0>
  __host__ __device__ constexpr BidirSizedRange(const BidirSizedRange& other) noexcept
      : ptr_(other.ptr_)
      , size_(other.size_)
  {}

  template <CopyCategory CC2 = CC, cuda::std::enable_if_t<CC2 == MoveOnly, int> = 0>
  __host__ __device__ constexpr BidirSizedRange(BidirSizedRange&& other) noexcept
      : ptr_(other.ptr_)
      , size_(other.size_)
  {}

  template <CopyCategory CC2 = CC, cuda::std::enable_if_t<CC2 == Copyable, int> = 0>
  __host__ __device__ constexpr BidirSizedRange& operator=(const BidirSizedRange& other) noexcept
  {
    ptr_  = other.ptr_;
    size_ = other.size_;
    return *this;
  }

  template <CopyCategory CC2 = CC, cuda::std::enable_if_t<CC2 == MoveOnly, int> = 0>
  __host__ __device__ constexpr BidirSizedRange& operator=(BidirSizedRange&& other) noexcept
  {
    ptr_  = other.ptr_;
    size_ = other.size_;
    return *this;
  }

  __host__ __device__ constexpr bidirectional_iterator<int*> begin()
  {
    return bidirectional_iterator<int*>{ptr_};
  }
  __host__ __device__ constexpr bidirectional_iterator<const int*> begin() const
  {
    return bidirectional_iterator<const int*>{ptr_};
  }
  __host__ __device__ constexpr bidirectional_iterator<int*> end()
  {
    return bidirectional_iterator<int*>{ptr_ + 8};
  }
  __host__ __device__ constexpr bidirectional_iterator<const int*> end() const
  {
    return bidirectional_iterator<const int*>{ptr_ + 8};
  }

  __host__ __device__ constexpr size_t size() const
  {
    return size_;
  }
};

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Non-common, non-const bidirectional range.
  {
    auto rev = cuda::std::ranges::reverse_view(BidirSizedRange<Copyable>{buffer, 4});
    assert(cuda::std::ranges::size(rev) == 4);
    assert(rev.size() == 4);
    assert(cuda::std::move(rev).size() == 4);

    ASSERT_SAME_TYPE(decltype(rev.size()), size_t);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(rev).size()), size_t);
  }
  // Non-common, const bidirectional range.
  {
    const auto rev = cuda::std::ranges::reverse_view(BidirSizedRange<Copyable>{buffer, 4});
    assert(cuda::std::ranges::size(rev) == 4);
    assert(rev.size() == 4);
    assert(cuda::std::move(rev).size() == 4);

    ASSERT_SAME_TYPE(decltype(rev.size()), size_t);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(rev).size()), size_t);
  }
  // Non-common, non-const (move only) bidirectional range.
  {
    auto rev = cuda::std::ranges::reverse_view(BidirSizedRange<MoveOnly>{buffer, 4});
    assert(cuda::std::move(rev).size() == 4);

    ASSERT_SAME_TYPE(decltype(cuda::std::move(rev).size()), size_t);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
