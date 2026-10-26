//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Contiguous iterator adaptors satisfy __is_contiguous_iterator_or_adaptor_v when their stored iterator does.
// __unwrap_iterator_adaptor peels those adaptors down to the contiguous iterator, and __rebase aims the adaptor
// at a different contiguous region. Adaptors that read through a functor can only be rebased when the functor proclaims
// copyable arguments.

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

struct PlusOne
{
  TEST_FUNC constexpr int operator()(int value) const
  {
    return value + 1;
  }
};

struct TimesTwo
{
  TEST_FUNC constexpr int operator()(int value) const
  {
    return value * 2;
  }
};

struct Narrow
{
  TEST_FUNC constexpr char operator()(int value) const
  {
    return static_cast<char>(value);
  }
};

// Contiguous iterator with no base(). Unwrap reaches its pointer through to_address.
struct addressable_iterator
{
  using iterator_category = cuda::std::contiguous_iterator_tag;
  using value_type        = int;
  using difference_type   = cuda::std::ptrdiff_t;
  using pointer           = int*;
  using reference         = int&;

  int* ptr_;

  TEST_FUNC constexpr explicit addressable_iterator(int* ptr)
      : ptr_(ptr)
  {}

  TEST_FUNC constexpr reference operator*() const
  {
    return *ptr_;
  }

  TEST_FUNC constexpr pointer operator->() const
  {
    return ptr_;
  }
};

template <class Iter, bool Expected>
TEST_FUNC constexpr void test_trait()
{
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<Iter> == Expected);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<const Iter> == Expected);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<volatile Iter> == Expected);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<const volatile Iter> == Expected);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<Iter&> == Expected);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<const Iter&> == Expected);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<Iter&&> == Expected);
}

TEST_FUNC constexpr bool test()
{
  int buffer[4] = {1, 2, 3, 4};
  int staged[4] = {10, 20, 30, 40};

  test_trait<int*, true>();
  test_trait<const int*, true>();
  test_trait<contiguous_iterator<int*>, true>();
  test_trait<cuda::counting_iterator<int>, false>();
  test_trait<random_access_iterator<int*>, false>();

  {
    int* input     = buffer;
    auto unwrapped = cuda::__unwrap_iterator_adaptor(input);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<int*>, int*>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<int*>, int>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<const int*>, int>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<volatile int*>, volatile int>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<const volatile int*>, volatile int>);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), int*>);
    assert(unwrapped == buffer);

    cuda::counting_iterator<int> counting{4};
    auto same = cuda::__unwrap_iterator_adaptor(counting);
    static_assert(cuda::std::is_same_v<decltype(same), cuda::counting_iterator<int>>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<cuda::counting_iterator<int>>, int>);
    static_assert(!cuda::__iterator_can_rebase<cuda::counting_iterator<int>>);
    assert(*same == 4);
  }

  {
    using Iter = cuda::transform_iterator<PlusOne, int*>;
    test_trait<Iter, true>();
    static_assert(
      cuda::__is_contiguous_iterator_or_adaptor_v<Iter> == cuda::__is_contiguous_iterator_or_adaptor_v<int*>);

    Iter iter{buffer, PlusOne{}};
    auto advanced  = iter + 2;
    auto unwrapped = cuda::__unwrap_iterator_adaptor(advanced);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<Iter>, int*>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<Iter>, int>);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), int*>);
    assert(unwrapped == buffer + 2);

    auto rebased = iter;
    rebased.__rebase(staged);
    assert(*iter == 2);
    assert(*rebased == 11);
    assert(rebased.base() == staged);
    assert(rebased[3] == 41);
  }

  {
    using Iter = cuda::transform_iterator<PlusOne, const int*>;
    test_trait<Iter, true>();
    const int* input = buffer;
    Iter iter{input, PlusOne{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), const int*>);
    assert(unwrapped == buffer);

    auto rebased = iter;
    rebased.__rebase(staged);
    assert(*iter == 2);
    assert(*rebased == 11);
    assert(rebased.base() == staged);
  }

  {
    using Leaf = contiguous_iterator<int*>;
    using Iter = cuda::transform_iterator<PlusOne, Leaf>;
    test_trait<Iter, true>();
    Iter iter{Leaf{buffer}, PlusOne{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), int*>);
    assert(unwrapped == buffer);

    static_assert(!cuda::__iterator_can_rebase<Iter, int>);
  }

  {
    using Inner = cuda::transform_iterator<PlusOne, int*>;
    using Outer = cuda::transform_iterator<TimesTwo, Inner>;
    test_trait<Outer, true>();
    static_assert(
      cuda::__is_contiguous_iterator_or_adaptor_v<Outer> == cuda::__is_contiguous_iterator_or_adaptor_v<Inner>);

    Outer iter{Inner{buffer, PlusOne{}}, TimesTwo{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), int*>);
    assert(unwrapped == buffer);
    assert(*iter == 4);

    auto rebased = iter;
    rebased.__rebase(staged);
    assert(*iter == 4);
    assert(*rebased == 22);
    assert(rebased.base().base() == staged);
  }

  {
    using Iter = addressable_iterator;
    test_trait<Iter, true>();
    Iter iter{buffer + 2};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<Iter>, int*>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<Iter>, int>);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), int*>);
    assert(unwrapped == buffer + 2);

    using Transformed = cuda::transform_iterator<PlusOne, Iter>;
    test_trait<Transformed, true>();
    Transformed transformed{iter, PlusOne{}};
    auto transformed_unwrapped = cuda::__unwrap_iterator_adaptor(transformed);
    static_assert(cuda::std::is_same_v<decltype(transformed_unwrapped), int*>);
    assert(transformed_unwrapped == buffer + 2);
    assert(*transformed == 4);
    static_assert(!cuda::__iterator_can_rebase<Transformed, int>);
  }

  {
    using Bad = cuda::transform_iterator<PlusOne, cuda::counting_iterator<int>>;
    test_trait<Bad, false>();
    test_trait<cuda::transform_iterator<TimesTwo, Bad>, false>();
    test_trait<cuda::transform_iterator<PlusOne, random_access_iterator<int*>>, false>();
    // A reverse iterator unwraps to a pointer, but does not traverse the storage forward
    test_trait<cuda::transform_iterator<PlusOne, cuda::std::reverse_iterator<int*>>, false>();

    Bad iter{cuda::counting_iterator<int>{5}, PlusOne{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), cuda::counting_iterator<int>>);
    assert(*unwrapped == 5);

    // Without storage behind the adaptor, its own value type is used
    using Narrowed = cuda::transform_iterator<Narrow, cuda::counting_iterator<int>>;
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<Narrowed>, char>);
    static_assert(!cuda::__iterator_can_rebase<Narrowed>);
  }

  {
    using Iter = cuda::transform_iterator<Narrow, int*>;
    test_trait<Iter, true>();
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<Iter>, int>);
    static_assert(cuda::__iterator_can_rebase<Iter>);
    static_assert(cuda::__iterator_can_rebase<cuda::transform_iterator<PlusOne, Iter>>);
    static_assert(
      cuda::__iterator_can_rebase<cuda::transform_iterator<Narrow, cuda::transform_iterator<PlusOne, int*>>>);
  }

  {
    using Iter = cuda::transform_output_iterator<Narrow, char*>;
    test_trait<Iter, true>();

    char output[2]        = {};
    char staged_output[2] = {};
    Iter iter{output, Narrow{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<Iter>, char*>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<Iter>, char>);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), char*>);
    assert(unwrapped == output);

    auto rebased = iter;
    rebased.__rebase(staged_output);
    *iter      = 3;
    rebased[1] = 7;
    assert(output[0] == 3);
    assert(staged_output[1] == 7);
  }

  {
    using Iter = cuda::transform_input_output_iterator<PlusOne, PlusOne, int*>;
    test_trait<Iter, true>();

    Iter iter{buffer, PlusOne{}, PlusOne{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<Iter>, int*>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<Iter>, int>);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), int*>);
    assert(unwrapped == buffer);
    assert(*iter == 2);

    auto rebased = iter;
    rebased.__rebase(staged);
    assert(*rebased == 11);
    *rebased = 5;
    assert(buffer[0] == 1);
    assert(staged[0] == 6);
    assert(rebased.base() == staged);
  }

  {
    using Inner = cuda::transform_input_output_iterator<PlusOne, PlusOne, int*>;
    using Outer = cuda::transform_input_output_iterator<TimesTwo, PlusOne, Inner>;
    test_trait<Outer, true>();
    static_assert(
      cuda::__is_contiguous_iterator_or_adaptor_v<Outer> == cuda::__is_contiguous_iterator_or_adaptor_v<Inner>);

    staged[0] = 10;
    Outer iter{Inner{buffer, PlusOne{}, PlusOne{}}, TimesTwo{}, PlusOne{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), int*>);
    assert(unwrapped == buffer);
    assert(*iter == 4);

    auto rebased = iter;
    rebased.__rebase(staged);
    assert(*iter == 4);
    assert(*rebased == 22);
    *rebased = 3;
    assert(buffer[0] == 1);
    assert(staged[0] == 5);
    assert(rebased.base().base() == staged);
  }

  {
    using Iter = cuda::transform_input_output_iterator<PlusOne, PlusOne, addressable_iterator>;
    test_trait<Iter, true>();
    static_assert(!cuda::__iterator_can_rebase<Iter, int>);
  }

  {
    using Iter = cuda::transform_input_output_iterator<Narrow, PlusOne, int*>;
    test_trait<Iter, true>();
    static_assert(cuda::__iterator_can_rebase<Iter>);
  }

  {
    using Bad = cuda::transform_input_output_iterator<PlusOne, PlusOne, cuda::counting_iterator<int>>;
    test_trait<Bad, false>();
    test_trait<cuda::transform_input_output_iterator<TimesTwo, PlusOne, Bad>, false>();

    Bad iter{cuda::counting_iterator<int>{5}, PlusOne{}, PlusOne{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), cuda::counting_iterator<int>>);
    assert(*unwrapped == 5);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
