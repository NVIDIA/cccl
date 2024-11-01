//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11

#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/inplace_vector>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
#  include <stdexcept>
#endif // !TEST_HAS_NO_EXCEPTIONS

_CCCL_DIAG_SUPPRESS_GCC("-Wmissing-braces")
_CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-braces")
_CCCL_DIAG_SUPPRESS_MSVC(5246)

#if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
template <class T, template <class, size_t> class Range>
__host__ __device__ constexpr void test_ranges()
{
  { // inplace_vector<T, 0>::assign_range with an empty input
    cuda::std::inplace_vector<T, 0> vec{};
    vec.assign_range(Range<T, 0>{});
    assert(vec.empty());
  }

  using inplace_vector = cuda::std::inplace_vector<T, 42>;
  { // inplace_vector<T, N>::assign_range with an empty input
    inplace_vector vec{};
    vec.assign_range(Range<T, 0>{});
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::assign_range with an empty input, shrinking
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign_range(Range<T, 0>{});
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::assign_range with a non-empty input, shrinking
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign_range(Range<T, 2>{T(42), T(42)});
    assert(!vec.empty());
    assert(equal_range(vec, cuda::std::array<T, 2>{T(42), T(42)}));
  }

  { // inplace_vector<T, N>::assign_range with a non-empty input, growing
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign_range(Range<T, 6>{T(42), T(1), T(42), T(1337), T(0), T(42)});
    assert(!vec.empty());
    assert(equal_range(vec, cuda::std::array<T, 6>{T(42), T(1), T(42), T(1337), T(0), T(42)}));
  }
}
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)

template <class T>
__host__ __device__ constexpr void test()
{
  { // inplace_vector<T, 0>::assign(count, const T&)
    cuda::std::inplace_vector<T, 0> vec{};
    vec.assign(0, T(42));
    assert(vec.empty());
  }

  { // inplace_vector<T, 0>::assign(iter, iter), with input iterators
    using iter = cpp17_input_iterator<const T*>;
    cuda::std::initializer_list<T> input{};
    cuda::std::inplace_vector<T, 0> vec{};
    vec.assign(iter{input.begin()}, iter{input.end()});
    assert(vec.empty());
  }

  { // inplace_vector<T, 0>::assign(iter, iter), with forward iterators
    cuda::std::initializer_list<T> input{};
    cuda::std::inplace_vector<T, 0> vec{};
    vec.assign(input.begin(), input.end());
    assert(vec.empty());
  }

  { // inplace_vector<T, 0>::assign(initializer_list)
    cuda::std::initializer_list<T> input{};
    cuda::std::inplace_vector<T, 0> vec{};
    vec.assign(input);
    assert(vec.empty());
  }

  using inplace_vector = cuda::std::inplace_vector<T, 42>;
  { // inplace_vector<T, N>::assign(count, const T&), zero count from empty
    inplace_vector vec{};
    vec.assign(0, T(42));
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::assign(count, const T&), shrinking to empty
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign(0, T(42));
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::assign(count, const T&), shrinking
    const cuda::std::array<T, 2> expected = {T(42), T(42)};
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign(2, T(42));
    assert(!vec.empty());
    assert(equal_range(vec, expected));
  }

  { // inplace_vector<T, N>::assign(count, const T&), growing
    const cuda::std::array<T, 6> expected = {T(42), T(42), T(42), T(42), T(42), T(42)};
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign(6, T(42));
    assert(!vec.empty());
    assert(equal_range(vec, expected));
  }

  { // inplace_vector<T, N>::assign(iter, iter), with input iterators empty range
    using iter                            = cpp17_input_iterator<const T*>;
    const cuda::std::array<T, 0> expected = {};
    inplace_vector vec{};
    vec.assign(iter{expected.begin()}, iter{expected.end()});
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::assign(iter, iter), with input iterators shrink to empty range
    using iter                            = cpp17_input_iterator<const T*>;
    const cuda::std::array<T, 0> expected = {};
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign(iter{expected.begin()}, iter{expected.end()});
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::assign(iter, iter), with input iterators shrinking
    using iter                            = cpp17_input_iterator<const T*>;
    const cuda::std::array<T, 2> expected = {T(42), T(42)};
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign(iter{expected.begin()}, iter{expected.end()});
    assert(!vec.empty());
    assert(equal_range(vec, expected));
  }

  { // inplace_vector<T, N>::assign(iter, iter), with input iterators growing
    using iter                            = cpp17_input_iterator<const T*>;
    const cuda::std::array<T, 6> expected = {T(42), T(1), T(42), T(1337), T(0), T(42)};
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign(iter{expected.begin()}, iter{expected.end()});
    assert(!vec.empty());
    assert(equal_range(vec, expected));
  }

  { // inplace_vector<T, N>::assign(iter, iter), with forward iterators empty range
    const cuda::std::array<T, 0> expected = {};
    inplace_vector vec{};
    vec.assign(expected.begin(), expected.end());
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::assign(iter, iter), with forward iterators shrinking to empty
    const cuda::std::array<T, 0> expected = {};
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign(expected.begin(), expected.end());
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::assign(iter, iter), with forward iterators shrinking
    const cuda::std::array<T, 2> expected = {T(42), T(42)};
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign(expected.begin(), expected.end());
    assert(!vec.empty());
    assert(equal_range(vec, expected));
  }

  { // inplace_vector<T, N>::assign(iter, iter), with forward iterators growing
    const cuda::std::array<T, 6> expected = {T(42), T(1), T(42), T(1337), T(0), T(42)};
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign(expected.begin(), expected.end());
    assert(!vec.empty());
    assert(equal_range(vec, expected));
  }

  { // inplace_vector<T, N>::assign(initializer_list), empty range
    cuda::std::initializer_list<T> expected = {};
    inplace_vector vec{};
    vec.assign(expected);
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::assign(initializer_list), shrinking to empty
    cuda::std::initializer_list<T> expected = {};
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign(expected);
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::assign(initializer_list), shrinking
    cuda::std::array<T, 2> expected{T(42), T(42)};
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign({T(42), T(42)});
    assert(!vec.empty());
    assert(equal_range(vec, expected));
  }

  { // inplace_vector<T, N>::assign(initializer_list), growing
    cuda::std::array<T, 6> expected{T(42), T(1), T(42), T(1337), T(0), T(42)};
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec.assign({T(42), T(1), T(42), T(1337), T(0), T(42)});
    assert(!vec.empty());
    assert(equal_range(vec, expected));
  }

#if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
  test_ranges<T, input_range>();
  test_ranges<T, uncommon_range>();
  test_ranges<T, sized_uncommon_range>();
  test_ranges<T, cuda::std::array>();
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<Trivial>();

  if (!cuda::std::__libcpp_is_constant_evaluated())
  {
    test<NonTrivial>();
    test<NonTrivialDestructor>();
    test<ThrowingDefaultConstruct>();
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
#  if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
template <template <class, size_t> class Range>
void test_exceptions()
{ // assign_range throws std::bad_alloc
  constexpr size_t capacity = 4;
  using inplace_vector      = cuda::std::inplace_vector<int, capacity>;
  inplace_vector too_small{};
  try
  {
    too_small.assign_range(Range<int, 2 + capacity>{0, 1, 2, 3, 4, 5});
    assert(false);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }
}
#  endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)

void test_exceptions()
{ // assign throws std::bad_alloc
  constexpr size_t capacity = 4;
  using inplace_vector      = cuda::std::inplace_vector<int, capacity>;
  inplace_vector too_small{};
  const cuda::std::array<int, 7> input{0, 1, 2, 3, 4, 5, 6};

  try
  {
    too_small.assign(2 * capacity, 42);
    assert(false);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    using iter = cpp17_input_iterator<const int*>;
    too_small.assign(iter{input.begin()}, iter{input.end()});
    assert(false);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    too_small.assign(input.begin(), input.end());
    assert(false);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    too_small.assign(cuda::std::initializer_list<int>{0, 1, 2, 3, 4, 5, 6});
    assert(false);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

#  if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
  test_exceptions<input_range>();
  test_exceptions<uncommon_range>();
  test_exceptions<sized_uncommon_range>();
  test_exceptions<cuda::std::array>();
#  endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
