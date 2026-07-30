//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: a non-__tile__ variable cannot be used in tile code

// constexpr reverse_iterator<iterator_t<V>> begin();
// constexpr reverse_iterator<iterator_t<V>> begin() requires common_range<V>;
// constexpr auto begin() const requires common_range<const V>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

TEST_GLOBAL_VARIABLE int globalCount = 0;

struct CountedIter
{
  using iterator_category = cuda::std::bidirectional_iterator_tag;
  using value_type        = int;
  using difference_type   = cuda::std::ptrdiff_t;
  using pointer           = int*;
  using reference         = int&;
  using self              = CountedIter;

  pointer ptr_;
  TEST_FUNC CountedIter(pointer ptr)
      : ptr_(ptr)
  {}
  CountedIter() = default;

  TEST_FUNC reference operator*() const;
  TEST_FUNC pointer operator->() const;
#if TEST_HAS_SPACESHIP()
  auto operator<=>(const self&) const = default;
#else
  TEST_FUNC bool operator==(const self& rhs) const
  {
    return ptr_ == rhs.ptr_;
  }
  TEST_FUNC bool operator!=(const self& rhs) const
  {
    return ptr_ != rhs.ptr_;
  }

  TEST_FUNC bool operator<(const self& rhs) const
  {
    return ptr_ < rhs.ptr_;
  }
  TEST_FUNC bool operator<=(const self& rhs) const
  {
    return ptr_ <= rhs.ptr_;
  }
  TEST_FUNC bool operator>(const self& rhs) const
  {
    return ptr_ > rhs.ptr_;
  }
  TEST_FUNC bool operator>=(const self& rhs) const
  {
    return ptr_ >= rhs.ptr_;
  }
#endif

  TEST_FUNC self& operator++()
  {
    globalCount++;
    ++ptr_;
    return *this;
  }
  TEST_FUNC self operator++(int)
  {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  TEST_FUNC self& operator--();
  TEST_FUNC self operator--(int);
};

struct CountedView : cuda::std::ranges::view_base
{
  int* begin_;
  int* end_;

  TEST_FUNC CountedView(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}

  TEST_FUNC auto begin()
  {
    return CountedIter(begin_);
  }
  TEST_FUNC auto begin() const
  {
    return CountedIter(begin_);
  }
  TEST_FUNC auto end()
  {
    return sentinel_wrapper<CountedIter>(CountedIter(end_));
  }
  TEST_FUNC auto end() const
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

  TEST_FUNC constexpr RASentRange(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}

  TEST_FUNC constexpr random_access_iterator<int*> begin()
  {
    return random_access_iterator<int*>{begin_};
  }
  TEST_FUNC constexpr random_access_iterator<const int*> begin() const
  {
    return random_access_iterator<const int*>{begin_};
  }
  TEST_FUNC constexpr sent_t end()
  {
    return sent_t{random_access_iterator<int*>{end_}};
  }
  TEST_FUNC constexpr sent_const_t end() const
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

TEST_FUNC constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Common bidirectional range.
  {
    auto rev = cuda::std::ranges::reverse_view(BidirRange{buffer, buffer + 8});
    assert(base(rev.begin().base()) == buffer + 8);
    assert(base(cuda::std::move(rev).begin().base()) == buffer + 8);

    static_assert(
      cuda::std::is_same_v<decltype(rev.begin()), cuda::std::reverse_iterator<bidirectional_iterator<int*>>>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(rev).begin()),
                                       cuda::std::reverse_iterator<bidirectional_iterator<int*>>>);
  }
  // Const common bidirectional range.
  {
    const auto rev = cuda::std::ranges::reverse_view(BidirRange{buffer, buffer + 8});
    assert(base(rev.begin().base()) == buffer + 8);
    assert(base(cuda::std::move(rev).begin().base()) == buffer + 8);

    static_assert(
      cuda::std::is_same_v<decltype(rev.begin()), cuda::std::reverse_iterator<bidirectional_iterator<const int*>>>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(rev).begin()),
                                       cuda::std::reverse_iterator<bidirectional_iterator<const int*>>>);
  }
  // Non-common, non-const (move only) bidirectional range.
  {
    auto rev = cuda::std::ranges::reverse_view(BidirSentRange<MoveOnly>{buffer, buffer + 8});
    assert(base(cuda::std::move(rev).begin().base()) == buffer + 8);

    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(rev).begin()),
                                       cuda::std::reverse_iterator<bidirectional_iterator<int*>>>);
  }
  // Non-common, non-const bidirectional range.
  {
    auto rev = cuda::std::ranges::reverse_view(BidirSentRange<Copyable>{buffer, buffer + 8});
    assert(base(rev.begin().base()) == buffer + 8);
    assert(base(cuda::std::move(rev).begin().base()) == buffer + 8);

    static_assert(
      cuda::std::is_same_v<decltype(rev.begin()), cuda::std::reverse_iterator<bidirectional_iterator<int*>>>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(rev).begin()),
                                       cuda::std::reverse_iterator<bidirectional_iterator<int*>>>);
  }
  // Non-common random access range.
  // Note: const overload invalid for non-common ranges, though it would not be impossible
  // to implement for random access ranges.
  {
    auto rev = cuda::std::ranges::reverse_view(RASentRange{buffer, buffer + 8});
    assert(base(rev.begin().base()) == buffer + 8);
    assert(base(cuda::std::move(rev).begin().base()) == buffer + 8);

    static_assert(
      cuda::std::is_same_v<decltype(rev.begin()), cuda::std::reverse_iterator<random_access_iterator<int*>>>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(rev).begin()),
                                       cuda::std::reverse_iterator<random_access_iterator<int*>>>);
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
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)

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
