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

// constexpr reverse_iterator<iterator_t<V>> begin();
// constexpr reverse_iterator<iterator_t<V>> begin() requires common_range<V>;
// constexpr auto begin() const requires common_range<const V>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

STATIC_TEST_GLOBAL_VAR int globalCount = 0;

struct CountedIter
{
  typedef cuda::std::bidirectional_iterator_tag iterator_category;
  typedef int value_type;
  typedef cuda::std::ptrdiff_t difference_type;
  typedef int* pointer;
  typedef int& reference;
  typedef CountedIter self;

  pointer ptr_;
  __host__ __device__ CountedIter(pointer ptr)
      : ptr_(ptr)
  {}
  CountedIter() = default;

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const self&) const = default;
#else
  __host__ __device__ bool operator==(const self& rhs) const
  {
    return ptr_ == rhs.ptr_;
  }
  __host__ __device__ bool operator!=(const self& rhs) const
  {
    return ptr_ != rhs.ptr_;
  }

  __host__ __device__ bool operator<(const self& rhs) const
  {
    return ptr_ < rhs.ptr_;
  }
  __host__ __device__ bool operator<=(const self& rhs) const
  {
    return ptr_ <= rhs.ptr_;
  }
  __host__ __device__ bool operator>(const self& rhs) const
  {
    return ptr_ > rhs.ptr_;
  }
  __host__ __device__ bool operator>=(const self& rhs) const
  {
    return ptr_ >= rhs.ptr_;
  }
#endif

  __host__ __device__ self& operator++()
  {
    globalCount++;
    ++ptr_;
    return *this;
  }
  __host__ __device__ self operator++(int)
  {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  __host__ __device__ self& operator--();
  __host__ __device__ self operator--(int);
};

struct CountedView : cuda::std::ranges::view_base
{
  int* begin_;
  int* end_;

  __host__ __device__ CountedView(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}

  __host__ __device__ auto begin()
  {
    return CountedIter(begin_);
  }
  __host__ __device__ auto begin() const
  {
    return CountedIter(begin_);
  }
  __host__ __device__ auto end()
  {
    return sentinel_wrapper<CountedIter>(CountedIter(end_));
  }
  __host__ __device__ auto end() const
  {
    return sentinel_wrapper<CountedIter>(CountedIter(end_));
  }
};

struct RASentRange : cuda::std::ranges::view_base
{
  using sent_t       = sentinel_wrapper<random_access_iterator<int*>>;
  using sent_const_t = sentinel_wrapper<random_access_iterator<const int*>>;

  int* begin_;
  int* end_;

  __host__ __device__ constexpr RASentRange(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}

  __host__ __device__ constexpr random_access_iterator<int*> begin()
  {
    return random_access_iterator<int*>{begin_};
  }
  __host__ __device__ constexpr random_access_iterator<const int*> begin() const
  {
    return random_access_iterator<const int*>{begin_};
  }
  __host__ __device__ constexpr sent_t end()
  {
    return sent_t{random_access_iterator<int*>{end_}};
  }
  __host__ __device__ constexpr sent_const_t end() const
  {
    return sent_const_t{random_access_iterator<const int*>{end_}};
  }
};

#if TEST_STD_VER > 2017
template <class T>
concept BeginInvocable = requires(T t) { t.begin(); };
#else
template <class T, class = void>
inline constexpr bool BeginInvocable = false;

template <class T>
inline constexpr bool BeginInvocable<T, cuda::std::void_t<decltype(cuda::std::declval<T>().begin())>> = true;
#endif

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Common bidirectional range.
  {
    auto rev = cuda::std::ranges::reverse_view(BidirRange{buffer, buffer + 8});
    assert(base(rev.begin().base()) == buffer + 8);
    assert(base(cuda::std::move(rev).begin().base()) == buffer + 8);

    ASSERT_SAME_TYPE(decltype(rev.begin()), cuda::std::reverse_iterator<bidirectional_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(rev).begin()), cuda::std::reverse_iterator<bidirectional_iterator<int*>>);
  }
  // Const common bidirectional range.
  {
    const auto rev = cuda::std::ranges::reverse_view(BidirRange{buffer, buffer + 8});
    assert(base(rev.begin().base()) == buffer + 8);
    assert(base(cuda::std::move(rev).begin().base()) == buffer + 8);

    ASSERT_SAME_TYPE(decltype(rev.begin()), cuda::std::reverse_iterator<bidirectional_iterator<const int*>>);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(rev).begin()),
                     cuda::std::reverse_iterator<bidirectional_iterator<const int*>>);
  }
  // Non-common, non-const (move only) bidirectional range.
  {
    auto rev = cuda::std::ranges::reverse_view(BidirSentRange<MoveOnly>{buffer, buffer + 8});
    assert(base(cuda::std::move(rev).begin().base()) == buffer + 8);

    ASSERT_SAME_TYPE(decltype(cuda::std::move(rev).begin()), cuda::std::reverse_iterator<bidirectional_iterator<int*>>);
  }
  // Non-common, non-const bidirectional range.
  {
    auto rev = cuda::std::ranges::reverse_view(BidirSentRange<Copyable>{buffer, buffer + 8});
    assert(base(rev.begin().base()) == buffer + 8);
    assert(base(cuda::std::move(rev).begin().base()) == buffer + 8);

    ASSERT_SAME_TYPE(decltype(rev.begin()), cuda::std::reverse_iterator<bidirectional_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(rev).begin()), cuda::std::reverse_iterator<bidirectional_iterator<int*>>);
  }
  // Non-common random access range.
  // Note: const overload invalid for non-common ranges, though it would not be imposible
  // to implement for random access ranges.
  {
    auto rev = cuda::std::ranges::reverse_view(RASentRange{buffer, buffer + 8});
    assert(base(rev.begin().base()) == buffer + 8);
    assert(base(cuda::std::move(rev).begin().base()) == buffer + 8);

    ASSERT_SAME_TYPE(decltype(rev.begin()), cuda::std::reverse_iterator<random_access_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(rev).begin()), cuda::std::reverse_iterator<random_access_iterator<int*>>);
  }
  {
    static_assert(BeginInvocable<cuda::std::ranges::reverse_view<BidirSentRange<Copyable>>>);
    static_assert(!BeginInvocable<const cuda::std::ranges::reverse_view<BidirSentRange<Copyable>>>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_LIBCUDACXX_ADDRESSOF)

  {
    // Make sure we cache begin.
    int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    CountedView view{buffer, buffer + 8};
    cuda::std::ranges::reverse_view rev(view);
    assert(rev.begin().base().ptr_ == buffer + 8);
    assert(globalCount == 8);
    assert(rev.begin().base().ptr_ == buffer + 8);
    assert(globalCount == 8);
  }

  return 0;
}
