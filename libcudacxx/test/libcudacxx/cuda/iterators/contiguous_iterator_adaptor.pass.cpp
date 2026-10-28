//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Transform iterators satisfy __is_contiguous_iterator_or_adaptor_v when their stored iterator does, transform output
// and input output iterators never do. __unwrap_iterator_adaptor peels iterator adaptors down to the contiguous
// iterator, and __rebind_iterator_adaptor rebuilds a transform iterator over a different contiguous region. Transform
// output and input output iterators cannot be rebuilt, so they do not satisfy __iter_can_unwrap.

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/span>
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
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<volatile int*>, int>);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), int*>);
    assert(unwrapped == buffer);

    // A contiguous iterator is rebuilt as a pointer with the qualifiers of its address
    const int* const_input = buffer;
    auto rebound           = cuda::__rebind_iterator_adaptor(const_input, staged);
    static_assert(cuda::std::is_same_v<decltype(rebound), const int*>);
    assert(rebound == staged);
    static_assert(!cuda::__iter_can_unwrap<int*>);

    cuda::counting_iterator<int> counting{4};
    auto same = cuda::__unwrap_iterator_adaptor(counting);
    static_assert(cuda::std::is_same_v<decltype(same), cuda::counting_iterator<int>>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<cuda::counting_iterator<int>>, int>);
    static_assert(!cuda::__iter_can_rebind<cuda::counting_iterator<int>, int>);
    static_assert(!cuda::__iter_can_unwrap<cuda::counting_iterator<int>>);
    assert(*same == 4);
  }

  {
    using Iter = cuda::transform_iterator<PlusOne, int*>;
    test_trait<Iter, true>();
    static_assert(cuda::__iter_can_unwrap<Iter>);
    static_assert(
      cuda::__is_contiguous_iterator_or_adaptor_v<Iter> == cuda::__is_contiguous_iterator_or_adaptor_v<int*>);

    Iter iter{buffer, PlusOne{}};
    auto advanced  = iter + 2;
    auto unwrapped = cuda::__unwrap_iterator_adaptor(advanced);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<Iter>, int*>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<Iter>, int>);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), int*>);
    assert(unwrapped == buffer + 2);

    auto rebound = iter.__rebind(staged);
    static_assert(cuda::std::is_same_v<decltype(rebound), Iter>);
    assert(*iter == 2);
    assert(*rebound == 11);
    assert(rebound.base() == staged);
    assert(rebound[3] == 41);
  }

  {
    using Iter = cuda::transform_iterator<PlusOne, const int*>;
    test_trait<Iter, true>();
    static_assert(cuda::__iter_can_unwrap<Iter>);
    const int* input = buffer;
    Iter iter{input, PlusOne{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), const int*>);
    assert(unwrapped == buffer);

    auto rebound = iter.__rebind(staged);
    static_assert(cuda::std::is_same_v<decltype(rebound), Iter>);
    assert(*iter == 2);
    assert(*rebound == 11);
    assert(rebound.base() == staged);
  }

  {
    // The rebuilt adaptor reads through a pointer instead of the contiguous iterator
    using Leaf = contiguous_iterator<int*>;
    using Iter = cuda::transform_iterator<PlusOne, Leaf>;
    test_trait<Iter, true>();
    static_assert(cuda::__iter_can_unwrap<Iter>);
    Iter iter{Leaf{buffer}, PlusOne{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), int*>);
    assert(unwrapped == buffer);

    auto rebound = iter.__rebind(staged);
    static_assert(cuda::std::is_same_v<decltype(rebound), cuda::transform_iterator<PlusOne, int*>>);
    assert(*iter == 2);
    assert(*rebound == 11);
    assert(rebound.base() == staged);
  }

  {
    using Iter = cuda::transform_iterator<PlusOne, cuda::std::span<int>::iterator>;
    static_assert(cuda::__iter_can_unwrap<Iter>);
    Iter iter{cuda::std::span<int>{buffer}.begin(), PlusOne{}};
    assert(cuda::__unwrap_iterator_adaptor(iter) == buffer);

    auto rebound = iter.__rebind(staged);
    static_assert(cuda::std::is_same_v<decltype(rebound), cuda::transform_iterator<PlusOne, int*>>);
    assert(*rebound == 11);
    assert(rebound.base() == staged);
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

    auto rebound = iter.__rebind(staged);
    static_assert(cuda::std::is_same_v<decltype(rebound), Outer>);
    assert(*iter == 4);
    assert(*rebound == 22);
    assert(rebound.base().base() == staged);
  }

  {
    using Inner  = cuda::transform_iterator<PlusOne, contiguous_iterator<int*>>;
    using Middle = cuda::transform_iterator<TimesTwo, Inner>;
    using Outer  = cuda::transform_iterator<PlusOne, Middle>;
    test_trait<Outer, true>();
    static_assert(cuda::__iter_can_unwrap<Middle>);
    static_assert(cuda::__iter_can_unwrap<Outer>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<Outer>, int*>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<Outer>, int>);

    Outer iter{Middle{Inner{contiguous_iterator<int*>{buffer}, PlusOne{}}, TimesTwo{}}, PlusOne{}};
    assert(cuda::__unwrap_iterator_adaptor(iter + 1) == buffer + 1);
    assert(*iter == 5);

    // Only the innermost contiguous iterator is replaced, the adaptors around it are kept
    using Rebound =
      cuda::transform_iterator<PlusOne, cuda::transform_iterator<TimesTwo, cuda::transform_iterator<PlusOne, int*>>>;
    auto rebound = cuda::__rebind_iterator_adaptor(iter, staged);
    static_assert(cuda::std::is_same_v<decltype(rebound), Rebound>);
    assert(*iter == 5);
    assert(*rebound == 23);
    assert(rebound.base().base().base() == staged);
  }

  {
    // Nested adaptors are only rebuilt when the innermost iterator is contiguous
    using OverCounting =
      cuda::transform_iterator<TimesTwo, cuda::transform_iterator<PlusOne, cuda::counting_iterator<int>>>;
    using OverReverse =
      cuda::transform_iterator<TimesTwo, cuda::transform_iterator<PlusOne, cuda::std::reverse_iterator<int*>>>;
    static_assert(!cuda::__iter_can_unwrap<OverCounting>);
    static_assert(!cuda::__iter_can_unwrap<OverReverse>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<OverCounting>, cuda::counting_iterator<int>>);

    // The storage is described by the innermost pointer, not by the value type of a nested adaptor
    using OverNarrow = cuda::transform_iterator<PlusOne, cuda::transform_iterator<Narrow, int*>>;
    static_assert(cuda::__iter_can_unwrap<OverNarrow>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<OverNarrow>, int*>);
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<OverNarrow>, int>);
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
    static_assert(cuda::__iter_can_unwrap<Transformed>);
    Transformed transformed{iter, PlusOne{}};
    assert(cuda::__unwrap_iterator_adaptor(transformed) == buffer + 2);
    assert(*transformed == 4);

    auto rebound = transformed.__rebind(staged);
    static_assert(cuda::std::is_same_v<decltype(rebound), cuda::transform_iterator<PlusOne, int*>>);
    assert(*rebound == 11);
  }

  {
    using Bad = cuda::transform_iterator<PlusOne, cuda::counting_iterator<int>>;
    test_trait<Bad, false>();
    test_trait<cuda::transform_iterator<TimesTwo, Bad>, false>();
    test_trait<cuda::transform_iterator<PlusOne, random_access_iterator<int*>>, false>();
    // A reverse iterator unwraps to a pointer, but does not traverse the storage forward
    test_trait<cuda::transform_iterator<PlusOne, cuda::std::reverse_iterator<int*>>, false>();
    static_assert(!cuda::__iter_can_unwrap<cuda::transform_iterator<PlusOne, cuda::std::reverse_iterator<int*>>>);

    Bad iter{cuda::counting_iterator<int>{5}, PlusOne{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), cuda::counting_iterator<int>>);
    static_assert(!cuda::__iter_can_unwrap<Bad>);
    assert(*unwrapped == 5);
  }

  {
    using Iter = cuda::transform_iterator<Narrow, int*>;
    test_trait<Iter, true>();
    static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<Iter>, int>);
    static_assert(cuda::__iter_can_unwrap<Iter>);
  }

  {
    using Iter = cuda::transform_output_iterator<Narrow, char*>;
    test_trait<Iter, false>();
    static_assert(!cuda::__iter_can_unwrap<Iter>);

    char output[2] = {};
    Iter iter{output, Narrow{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), char*>);
    assert(unwrapped == output);
  }

  {
    using Iter = cuda::transform_input_output_iterator<PlusOne, PlusOne, int*>;
    test_trait<Iter, false>();
    static_assert(!cuda::__iter_can_unwrap<Iter>);

    Iter iter{buffer, PlusOne{}, PlusOne{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), int*>);
    assert(unwrapped == buffer);
    assert(*iter == 2);
  }

  {
    using Inner = cuda::transform_input_output_iterator<PlusOne, PlusOne, int*>;
    using Outer = cuda::transform_input_output_iterator<TimesTwo, PlusOne, Inner>;
    test_trait<Outer, false>();
    static_assert(!cuda::__iter_can_unwrap<Outer>);

    Outer iter{Inner{buffer, PlusOne{}, PlusOne{}}, TimesTwo{}, PlusOne{}};
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    static_assert(cuda::std::is_same_v<decltype(unwrapped), int*>);
    assert(unwrapped == buffer);
    assert(*iter == 4);
  }

  {
    using Iter = cuda::transform_input_output_iterator<PlusOne, PlusOne, addressable_iterator>;
    test_trait<Iter, false>();
    static_assert(!cuda::__iter_can_unwrap<Iter>);
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
