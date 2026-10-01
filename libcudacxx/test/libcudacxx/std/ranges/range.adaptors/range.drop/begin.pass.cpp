//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: a non-__tile__ variable cannot be used in tile code

// constexpr auto begin()
//   requires (!(simple-view<V> &&
//               random_access_range<const V> && sized_range<const V>));
// constexpr auto begin() const
//   requires random_access_range<const V> && sized_range<const V>;

#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

struct NontrivialDtorIter
{
  using iterator_category = cuda::std::forward_iterator_tag;
  using value_type        = int;
  using difference_type   = cuda::std::ptrdiff_t;
  using pointer           = int*;
  using reference         = int&;

  int* ptr_ = nullptr;

  TEST_HOST_DEVICE_FUNC NontrivialDtorIter() = default;
  TEST_HOST_DEVICE_FUNC explicit NontrivialDtorIter(int* ptr)
      : ptr_(ptr)
  {}
  TEST_HOST_DEVICE_FUNC ~NontrivialDtorIter() {}

  TEST_HOST_DEVICE_FUNC reference operator*() const
  {
    return *ptr_;
  }
  TEST_HOST_DEVICE_FUNC NontrivialDtorIter& operator++()
  {
    ++ptr_;
    return *this;
  }
  TEST_HOST_DEVICE_FUNC NontrivialDtorIter operator++(int)
  {
    NontrivialDtorIter prev = *this;
    ++ptr_;
    return prev;
  }

  TEST_HOST_DEVICE_FUNC friend bool operator==(NontrivialDtorIter lhs, NontrivialDtorIter rhs)
  {
    return lhs.ptr_ == rhs.ptr_;
  }
  TEST_HOST_DEVICE_FUNC friend bool operator!=(NontrivialDtorIter lhs, NontrivialDtorIter rhs)
  {
    return lhs.ptr_ != rhs.ptr_;
  }
};
static_assert(!cuda::std::is_trivially_destructible_v<NontrivialDtorIter>);

struct NontrivialDtorView : cuda::std::ranges::view_base
{
  int* begin_ = nullptr;
  int* end_   = nullptr;

  TEST_HOST_DEVICE_FUNC NontrivialDtorView(int* begin, int* end)
      : begin_(begin)
      , end_(end)
  {}

  TEST_HOST_DEVICE_FUNC NontrivialDtorIter begin() const
  {
    return NontrivialDtorIter{begin_};
  }
  TEST_HOST_DEVICE_FUNC NontrivialDtorIter end() const
  {
    return NontrivialDtorIter{end_};
  }
};
static_assert(cuda::std::ranges::forward_range<NontrivialDtorView>);
static_assert(!cuda::std::ranges::random_access_range<NontrivialDtorView>);
static_assert(!cuda::std::ranges::sized_range<NontrivialDtorView>);

template <class T>
_CCCL_CONCEPT BeginInvocable = _CCCL_REQUIRES_EXPR((T), cuda::std::ranges::drop_view<T> v)((v.begin()));

template <bool IsSimple>
struct MaybeSimpleView : cuda::std::ranges::view_base
{
  int* num_of_non_const_begin_calls;
  int* num_of_const_begin_calls;

  TEST_HOST_DEVICE_FUNC constexpr int* begin()
  {
    ++(*num_of_non_const_begin_calls);
    return nullptr;
  }
  TEST_HOST_DEVICE_FUNC constexpr cuda::std::conditional_t<IsSimple, int*, const int*> begin() const
  {
    ++(*num_of_const_begin_calls);
    return nullptr;
  }
  TEST_HOST_DEVICE_FUNC constexpr int* end() const
  {
    return nullptr;
  }
  TEST_HOST_DEVICE_FUNC constexpr size_t size() const
  {
    return 0;
  }
};

using SimpleView    = MaybeSimpleView<true>;
using NonSimpleView = MaybeSimpleView<false>;

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  // random_access_range<const V> && sized_range<const V>
  cuda::std::ranges::drop_view dropView1(MoveOnlyView(), 4);
  assert(dropView1.begin() == globalBuff + 4);

  // !random_access_range<const V>
  cuda::std::ranges::drop_view dropView2(ForwardView(), 4);
  assert(base(dropView2.begin()) == globalBuff + 4);

  // !random_access_range<const V>
  cuda::std::ranges::drop_view dropView3(InputView(), 4);
  assert(base(dropView3.begin()) == globalBuff + 4);

  // random_access_range<const V> && sized_range<const V>
  cuda::std::ranges::drop_view dropView4(MoveOnlyView(), 8);
  assert(dropView4.begin() == globalBuff + 8);

  // random_access_range<const V> && sized_range<const V>
  cuda::std::ranges::drop_view dropView5(MoveOnlyView(), 0);
  assert(dropView5.begin() == globalBuff);

  // random_access_range<const V> && sized_range<const V>
  const cuda::std::ranges::drop_view dropView6(MoveOnlyView(), 0);
  assert(dropView6.begin() == globalBuff);

  // random_access_range<const V> && sized_range<const V>
  cuda::std::ranges::drop_view dropView7(MoveOnlyView(), 10);
  assert(dropView7.begin() == globalBuff + 8);

  CountedView view8{};
  cuda::std::ranges::drop_view dropView8(view8, 5);
  assert(base(base(dropView8.begin())) == globalBuff + 5);
  assert(dropView8.begin().stride_count() == 5);
  assert(base(base(dropView8.begin())) == globalBuff + 5);
  assert(dropView8.begin().stride_count() == 5);

  static_assert(!BeginInvocable<const ForwardView>);
  {
    // non-common non-simple view,
    // The wording of the standard is:
    // Returns: ranges::next(ranges::begin(base_), count_, ranges::end(base_))
    // Note that "Returns" is used here, meaning that we don't have to do it this way.
    // In fact, this will use ranges::advance that has O(n) on non-common range.
    // but [range.range] requires "amortized constant time" for ranges::begin and ranges::end
    // Here, we test that begin() is indeed constant time, by creating a customized
    // sentinel and counting how many times the sentinel eq function is called.
    // It should be 0 times, but since this test (or any test under libcxx/test/std) is
    // also used by other implementations, we relax the condition to that
    // sentinel_cmp_calls is a constant number.
    int sentinel_cmp_calls_1 = 0;
    int sentinel_cmp_calls_2 = 0;
    using NonCommonView      = MaybeSimpleNonCommonView<false>;
    static_assert(cuda::std::ranges::random_access_range<NonCommonView>);
    static_assert(cuda::std::ranges::sized_range<NonCommonView>);
    cuda::std::ranges::drop_view dropView9_1(NonCommonView{{}, 0, &sentinel_cmp_calls_1}, 4);
    cuda::std::ranges::drop_view dropView9_2(NonCommonView{{}, 0, &sentinel_cmp_calls_2}, 6);
    assert(dropView9_1.begin() == globalBuff + 4);
    assert(dropView9_2.begin() == globalBuff + 6);
    assert(sentinel_cmp_calls_1 == sentinel_cmp_calls_2);
  }

  {
    // non-common simple view, same as above.
    int sentinel_cmp_calls_1 = 0;
    int sentinel_cmp_calls_2 = 0;
    using NonCommonView      = MaybeSimpleNonCommonView<true>;
    static_assert(cuda::std::ranges::random_access_range<NonCommonView>);
    static_assert(cuda::std::ranges::sized_range<NonCommonView>);
    cuda::std::ranges::drop_view dropView10_1(NonCommonView{{}, 0, &sentinel_cmp_calls_1}, 4);
    cuda::std::ranges::drop_view dropView10_2(NonCommonView{{}, 0, &sentinel_cmp_calls_2}, 6);
    assert(dropView10_1.begin() == globalBuff + 4);
    assert(dropView10_2.begin() == globalBuff + 6);
    assert(sentinel_cmp_calls_1 == sentinel_cmp_calls_2);
  }

  {
    static_assert(cuda::std::ranges::random_access_range<const SimpleView>);
    static_assert(cuda::std::ranges::sized_range<const SimpleView>);
    static_assert(cuda::std::ranges::__simple_view<SimpleView>);
    int non_const_calls = 0;
    int const_calls     = 0;
    cuda::std::ranges::drop_view dropView(SimpleView{{}, &non_const_calls, &const_calls}, 4);
    assert(dropView.begin() == nullptr);
    assert(non_const_calls == 0);
    assert(const_calls == 1);
    assert(cuda::std::as_const(dropView).begin() == nullptr);
    assert(non_const_calls == 0);
    assert(const_calls == 2);
  }

  {
    static_assert(cuda::std::ranges::random_access_range<const NonSimpleView>);
    static_assert(cuda::std::ranges::sized_range<const NonSimpleView>);
    static_assert(!cuda::std::ranges::__simple_view<NonSimpleView>);
    int non_const_calls = 0;
    int const_calls     = 0;
    cuda::std::ranges::drop_view dropView(NonSimpleView{{}, &non_const_calls, &const_calls}, 4);
    assert(dropView.begin() == nullptr);
    assert(non_const_calls == 1);
    assert(const_calls == 0);
    assert(cuda::std::as_const(dropView).begin() == nullptr);
    assert(non_const_calls == 1);
    assert(const_calls == 1);
  }

  return true;
}

TEST_HOST_DEVICE_FUNC bool test_nontrivial_iterator_cache()
{
  int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  NontrivialDtorView view(buf, buf + 8);
  cuda::std::ranges::drop_view dropped(view, 3);
  assert(*dropped.begin() == 4);
  assert(*dropped.begin() == 4);
  return true;
}

int main(int, char**)
{
  test();
  test_nontrivial_iterator_cache();
#if TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)

  return 0;
}
