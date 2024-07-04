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
  {
    cuda::std::inplace_vector<T, 42> input_range_empty{};
    input_range_empty.assign_range(Range<T, 0>{});
    assert(input_range_empty.empty());
  }

  {
    cuda::std::inplace_vector<T, 42> input_range_shrink_empty{T(1), T(42), T(1337), T(0)};
    input_range_shrink_empty.assign_range(Range<T, 0>{});
    assert(input_range_shrink_empty.empty());
  }

  {
    const cuda::std::array<T, 2> expected = {T(42), T(42)};
    cuda::std::inplace_vector<T, 42> input_range_shrink{T(1), T(42), T(1337), T(0)};
    input_range_shrink.assign_range(Range<T, 2>{T(42), T(42)});
    assert(!input_range_shrink.empty());
    assert(cuda::std::equal(input_range_shrink.begin(), input_range_shrink.end(), expected.begin(), expected.end()));
  }

  {
    const cuda::std::array<T, 6> expected = {T(42), T(42), T(42), T(42), T(42), T(42)};
    cuda::std::inplace_vector<T, 42> input_range_grow{T(1), T(42), T(1337), T(0)};
    input_range_grow.assign_range(Range<T, 6>{T(42), T(42), T(42), T(42), T(42), T(42)});
    assert(!input_range_grow.empty());
    assert(cuda::std::equal(input_range_grow.begin(), input_range_grow.end(), expected.begin(), expected.end()));
  }
}
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)

template <class T>
__host__ __device__ constexpr void test()
{
  {
    cuda::std::inplace_vector<T, 0> no_capacity_size_value{};
    no_capacity_size_value.assign(0, T(42));
    assert(no_capacity_size_value.empty());
  }

  {
    using iter = cpp17_input_iterator<const T*>;
    const cuda::std::initializer_list<T> input{};
    cuda::std::inplace_vector<T, 0> no_capacity_input_iter_iter{};
    no_capacity_input_iter_iter.assign(iter{input.begin()}, iter{input.end()});
    assert(no_capacity_input_iter_iter.empty());
  }

  {
    const cuda::std::initializer_list<T> input{};
    cuda::std::inplace_vector<T, 0> no_capacity_iter_iter{};
    no_capacity_iter_iter.assign(input.begin(), input.end());
    assert(no_capacity_iter_iter.empty());
  }

  {
    const cuda::std::initializer_list<T> input{};
    cuda::std::inplace_vector<T, 0> no_capacity_init_list{};
    no_capacity_init_list.assign(input);
    assert(no_capacity_init_list.empty());
  }

  {
    cuda::std::inplace_vector<T, 42> size_value_empty{};
    size_value_empty.assign(0, T(42));
    assert(size_value_empty.empty());
  }

  {
    cuda::std::inplace_vector<T, 42> size_value_shrink_empty{T(1), T(42), T(1337), T(0)};
    size_value_shrink_empty.assign(0, T(42));
    assert(size_value_shrink_empty.empty());
  }

  {
    const cuda::std::array<T, 2> expected = {T(42), T(42)};
    cuda::std::inplace_vector<T, 42> size_value_shrink{T(1), T(42), T(1337), T(0)};
    size_value_shrink.assign(2, T(42));
    assert(!size_value_shrink.empty());
    assert(cuda::std::equal(size_value_shrink.begin(), size_value_shrink.end(), expected.begin(), expected.end()));
  }

  {
    const cuda::std::array<T, 6> expected = {T(42), T(42), T(42), T(42), T(42), T(42)};
    cuda::std::inplace_vector<T, 42> size_value_grow{T(1), T(42), T(1337), T(0)};
    size_value_grow.assign(6, T(42));
    assert(!size_value_grow.empty());
    assert(cuda::std::equal(size_value_grow.begin(), size_value_grow.end(), expected.begin(), expected.end()));
  }

  {
    using iter                            = cpp17_input_iterator<const T*>;
    const cuda::std::array<T, 0> expected = {};
    cuda::std::inplace_vector<T, 42> input_iter_iter_empty{};
    input_iter_iter_empty.assign(iter{expected.begin()}, iter{expected.end()});
    assert(input_iter_iter_empty.empty());
  }

  {
    using iter                            = cpp17_input_iterator<const T*>;
    const cuda::std::array<T, 0> expected = {};
    cuda::std::inplace_vector<T, 42> input_iter_iter_shrink_empty{T(1), T(42), T(1337), T(0)};
    input_iter_iter_shrink_empty.assign(iter{expected.begin()}, iter{expected.end()});
    assert(input_iter_iter_shrink_empty.empty());
  }

  {
    using iter                            = cpp17_input_iterator<const T*>;
    const cuda::std::array<T, 2> expected = {T(42), T(42)};
    cuda::std::inplace_vector<T, 42> input_iter_iter_shrink{T(1), T(42), T(1337), T(0)};
    input_iter_iter_shrink.assign(iter{expected.begin()}, iter{expected.end()});
    assert(!input_iter_iter_shrink.empty());
    assert(
      cuda::std::equal(input_iter_iter_shrink.begin(), input_iter_iter_shrink.end(), expected.begin(), expected.end()));
  }

  {
    using iter                            = cpp17_input_iterator<const T*>;
    const cuda::std::array<T, 6> expected = {T(42), T(42), T(42), T(42), T(42), T(42)};
    cuda::std::inplace_vector<T, 42> input_iter_iter_grow{T(1), T(42), T(1337), T(0)};
    input_iter_iter_grow.assign(iter{expected.begin()}, iter{expected.end()});
    assert(!input_iter_iter_grow.empty());
    assert(
      cuda::std::equal(input_iter_iter_grow.begin(), input_iter_iter_grow.end(), expected.begin(), expected.end()));
  }

  {
    const cuda::std::array<T, 0> expected = {};
    cuda::std::inplace_vector<T, 42> iter_iter_empty{};
    iter_iter_empty.assign(expected.begin(), expected.end());
    assert(iter_iter_empty.empty());
  }

  {
    const cuda::std::array<T, 0> expected = {};
    cuda::std::inplace_vector<T, 42> iter_iter_shrink_empty{T(1), T(42), T(1337), T(0)};
    iter_iter_shrink_empty.assign(expected.begin(), expected.end());
    assert(iter_iter_shrink_empty.empty());
  }

  {
    const cuda::std::array<T, 2> expected = {T(42), T(42)};
    cuda::std::inplace_vector<T, 42> iter_iter_shrink{T(1), T(42), T(1337), T(0)};
    iter_iter_shrink.assign(expected.begin(), expected.end());
    assert(!iter_iter_shrink.empty());
    assert(cuda::std::equal(iter_iter_shrink.begin(), iter_iter_shrink.end(), expected.begin(), expected.end()));
  }

  {
    const cuda::std::array<T, 6> expected = {T(42), T(42), T(42), T(42), T(42), T(42)};
    cuda::std::inplace_vector<T, 42> iter_iter_grow{T(1), T(42), T(1337), T(0)};
    iter_iter_grow.assign(expected.begin(), expected.end());
    assert(!iter_iter_grow.empty());
    assert(cuda::std::equal(iter_iter_grow.begin(), iter_iter_grow.end(), expected.begin(), expected.end()));
  }

  {
    const cuda::std::initializer_list<T> expected = {};
    cuda::std::inplace_vector<T, 42> init_list_empty{};
    init_list_empty.assign(expected);
    assert(init_list_empty.empty());
  }

  {
    const cuda::std::initializer_list<T> expected = {};
    cuda::std::inplace_vector<T, 42> init_list_shrink_empty{T(1), T(42), T(1337), T(0)};
    init_list_shrink_empty.assign(expected);
    assert(init_list_shrink_empty.empty());
  }

  {
    const cuda::std::initializer_list<T> expected = {T(42), T(42)};
    cuda::std::inplace_vector<T, 42> init_list_shrink{T(1), T(42), T(1337), T(0)};
    init_list_shrink.assign(expected);
    assert(!init_list_shrink.empty());
    assert(cuda::std::equal(init_list_shrink.begin(), init_list_shrink.end(), expected.begin(), expected.end()));
  }

  {
    const cuda::std::initializer_list<T> expected = {T(42), T(42), T(42), T(42), T(42), T(42)};
    cuda::std::inplace_vector<T, 42> init_list_grow{T(1), T(42), T(1337), T(0)};
    init_list_grow.assign(expected);
    assert(!init_list_grow.empty());
    assert(cuda::std::equal(init_list_grow.begin(), init_list_grow.end(), expected.begin(), expected.end()));
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

  if (!cuda::std::__libcpp_is_constant_evaluated())
  {
    test<NonTrivial>();
    test<NonTrivialDestructor>();
    test<ThrowingDefaultConstruct>();
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{ // assign throws std::bad_alloc
  constexpr size_t capacity = 4;
  using vec                 = cuda::std::inplace_vector<int, capacity>;

  try
  {
    vec too_small{};
    too_small.assign(2 * capacity, 42);
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
    cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
    vec too_small{};
    too_small.assign(iter{input.begin()}, iter{input.end()});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
    vec too_small{};
    too_small.assign(input.begin(), input.end());
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    cuda::std::initializer_list<int> input{0, 1, 2, 3, 4, 5, 6};
    vec too_small{};
    too_small.assign(input);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

#  if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
  try
  {
    input_range<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
    vec too_small{};
    too_small.assign_range(input);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    uncommon_range<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
    vec too_small{};
    too_small.assign_range(input);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    sized_uncommon_range<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
    vec too_small{};
    too_small.assign_range(input);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }
#  endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // _LIBCUDACXX_IS_CONSTANT_EVALUATED

#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
