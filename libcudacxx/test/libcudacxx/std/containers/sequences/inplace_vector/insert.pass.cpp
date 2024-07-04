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
template <class T, class Range>
__host__ __device__ constexpr void test_range()
{
  constexpr size_t max_capacity = 5ull;
  using vec                     = cuda::std::inplace_vector<T, max_capacity>;
  {
    const cuda::std::array<T, 5> expected_insert_range = {T(0), T(42), T(3), T(1337), T(5)};

    vec insert_range            = {T(0), T(5)};
    const auto res_insert_range = insert_range.insert_range(insert_range.begin() + 1, Range{T(42), T(3), T(1337)});
    static_assert(cuda::std::is_same<decltype(res_insert_range), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(
      insert_range.begin(), insert_range.end(), expected_insert_range.begin(), expected_insert_range.end()));
    assert(res_insert_range == insert_range.begin() + 1);

    vec insert_range_const = {T(0), T(5)};
    const auto res_insert_range_const =
      insert_range_const.insert_range(insert_range_const.cbegin() + 1, Range{T(42), T(3), T(1337)});
    static_assert(cuda::std::is_same<decltype(res_insert_range_const), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(
      insert_range_const.begin(), insert_range_const.end(), expected_insert_range.begin(), expected_insert_range.end()));
    assert(res_insert_range_const == insert_range_const.cbegin() + 1);
  }

  {
    const cuda::std::initializer_list<T> expected_append_range = {T(0), T(5), T(42), T(3), T(1337)};
    vec append_range                                           = {T(0), T(5)};
    append_range.append_range(Range{T(42), T(3), T(1337)});
    assert(cuda::std::equal(
      append_range.begin(), append_range.end(), expected_append_range.begin(), expected_append_range.end()));

    Range try_input{T(42), T(3), T(1337)};
    vec try_append_range      = {T(0), T(5)};
    auto res_try_append_range = try_append_range.try_append_range(try_input);
    static_assert(cuda::std::is_same<decltype(res_try_append_range), cuda::std::ranges::iterator_t<Range>>::value, "");
    assert(cuda::std::equal(
      try_append_range.begin(), try_append_range.end(), expected_append_range.begin(), expected_append_range.end()));
    assert(res_try_append_range == try_input.end());

    Range try_input_partial{T(3), T(1337), T(1)};
    vec try_append_range_partial      = {T(0), T(5), T(42)};
    auto res_try_append_range_partial = try_append_range_partial.try_append_range(try_input_partial);
    static_assert(
      cuda::std::is_same<decltype(res_try_append_range_partial), cuda::std::ranges::iterator_t<Range>>::value, "");
    assert(cuda::std::equal(
      try_append_range_partial.begin(),
      try_append_range_partial.end(),
      expected_append_range.begin(),
      expected_append_range.end()));
    assert(++res_try_append_range_partial == try_input_partial.end());
  }
}
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)

template <class T>
__host__ __device__ constexpr void test()
{
  constexpr size_t max_capacity = 42ull;
  using vec                     = cuda::std::inplace_vector<T, max_capacity>;

  {
    const cuda::std::initializer_list<T> expected_insert_value = {T(0), T(3), T(5)};
    const T to_be_inserted                                     = 3;

    vec insert_lvalue            = {T(0), T(5)};
    const auto res_insert_lvalue = insert_lvalue.insert(insert_lvalue.begin() + 1, to_be_inserted);
    static_assert(cuda::std::is_same<decltype(res_insert_lvalue), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(
      insert_lvalue.begin(), insert_lvalue.end(), expected_insert_value.begin(), expected_insert_value.end()));
    assert(res_insert_lvalue == insert_lvalue.begin() + 1);

    vec insert_lvalue_const            = {T(0), T(5)};
    const auto res_insert_lvalue_const = insert_lvalue_const.insert(insert_lvalue_const.cbegin() + 1, to_be_inserted);
    static_assert(cuda::std::is_same<decltype(res_insert_lvalue_const), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(
      insert_lvalue_const.begin(),
      insert_lvalue_const.end(),
      expected_insert_value.begin(),
      expected_insert_value.end()));
    assert(res_insert_lvalue_const == insert_lvalue_const.cbegin() + 1);

    vec insert_rvalue            = {T(0), T(5)};
    const auto res_insert_rvalue = insert_rvalue.insert(insert_rvalue.begin() + 1, 3);
    static_assert(cuda::std::is_same<decltype(res_insert_rvalue), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(
      insert_rvalue.begin(), insert_rvalue.end(), expected_insert_value.begin(), expected_insert_value.end()));
    assert(res_insert_rvalue == insert_rvalue.begin() + 1);

    vec insert_rvalue_const            = {T(0), T(5)};
    const auto res_insert_rvalue_const = insert_rvalue_const.insert(insert_rvalue_const.cbegin() + 1, 3);
    static_assert(cuda::std::is_same<decltype(res_insert_rvalue_const), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(
      insert_rvalue_const.begin(),
      insert_rvalue_const.end(),
      expected_insert_value.begin(),
      expected_insert_value.end()));
    assert(res_insert_rvalue_const == insert_rvalue_const.cbegin() + 1);
  }

  {
    using iter                                                = cpp17_input_iterator<const T*>;
    const cuda::std::initializer_list<T> expected_insert_iter = {T(0), T(42), T(3), T(1337), T(5)};
    const cuda::std::initializer_list<T> to_be_inserted       = {T(42), T(3), T(1337)};

    vec insert_iter_range = {T(0), T(5)};
    const auto res_insert_iter_range =
      insert_iter_range.insert(insert_iter_range.begin() + 1, iter{to_be_inserted.begin()}, iter{to_be_inserted.end()});
    static_assert(cuda::std::is_same<decltype(res_insert_iter_range), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(
      insert_iter_range.begin(), insert_iter_range.end(), expected_insert_iter.begin(), expected_insert_iter.end()));
    assert(res_insert_iter_range == insert_iter_range.begin() + 1);

    vec insert_iter_range_const            = {T(0), T(5)};
    const auto res_insert_iter_range_const = insert_iter_range_const.insert(
      insert_iter_range_const.cbegin() + 1, iter{to_be_inserted.begin()}, iter{to_be_inserted.end()});
    static_assert(cuda::std::is_same<decltype(res_insert_iter_range_const), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(
      insert_iter_range_const.begin(),
      insert_iter_range_const.end(),
      expected_insert_iter.begin(),
      expected_insert_iter.end()));
    assert(res_insert_iter_range_const == insert_iter_range_const.cbegin() + 1);
  }

  {
    const cuda::std::initializer_list<T> expected_insert_iter = {T(0), T(42), T(3), T(1337), T(5)};
    const cuda::std::initializer_list<T> to_be_inserted       = {T(42), T(3), T(1337)};

    vec insert_iter_range = {T(0), T(5)};
    const auto res_insert_iter_range =
      insert_iter_range.insert(insert_iter_range.begin() + 1, to_be_inserted.begin(), to_be_inserted.end());
    static_assert(cuda::std::is_same<decltype(res_insert_iter_range), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(
      insert_iter_range.begin(), insert_iter_range.end(), expected_insert_iter.begin(), expected_insert_iter.end()));
    assert(res_insert_iter_range == insert_iter_range.begin() + 1);

    vec insert_iter_range_const            = {T(0), T(5)};
    const auto res_insert_iter_range_const = insert_iter_range_const.insert(
      insert_iter_range_const.cbegin() + 1, to_be_inserted.begin(), to_be_inserted.end());
    static_assert(cuda::std::is_same<decltype(res_insert_iter_range_const), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(
      insert_iter_range_const.begin(),
      insert_iter_range_const.end(),
      expected_insert_iter.begin(),
      expected_insert_iter.end()));
    assert(res_insert_iter_range_const == insert_iter_range_const.cbegin() + 1);
  }

  {
    const cuda::std::initializer_list<T> expected_insert_iter = {T(0), T(42), T(3), T(1337), T(5)};
    const cuda::std::initializer_list<T> to_be_inserted       = {T(42), T(3), T(1337)};

    vec insert_initializer            = {T(0), T(5)};
    const auto res_insert_initializer = insert_initializer.insert(insert_initializer.begin() + 1, to_be_inserted);
    static_assert(cuda::std::is_same<decltype(res_insert_initializer), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(
      insert_initializer.begin(), insert_initializer.end(), expected_insert_iter.begin(), expected_insert_iter.end()));
    assert(res_insert_initializer == insert_initializer.begin() + 1);

    vec insert_initializer_const = {T(0), T(5)};
    const auto res_insert_initializer_const =
      insert_initializer_const.insert(insert_initializer_const.cbegin() + 1, to_be_inserted);
    static_assert(cuda::std::is_same<decltype(res_insert_initializer_const), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(
      insert_initializer_const.begin(),
      insert_initializer_const.end(),
      expected_insert_iter.begin(),
      expected_insert_iter.end()));
    assert(res_insert_initializer_const == insert_initializer_const.cbegin() + 1);
  }

#if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
  test_range<T, input_range<T, 3>>();
  test_range<T, uncommon_range<T, 3>>();
  test_range<T, sized_uncommon_range<T, 3>>();
  test_range<T, cuda::std::array<T, 3>>();
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
}

__host__ __device__ constexpr bool test()
{
  test<int>();

  if (!cuda::std::is_constant_evaluated())
  {
    test<NonTrivial>();
    test<NonTrivialDestructor>();
    test<ThrowingDefaultConstruct>();
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{ // insert throws std::bad_alloc
  using vec = cuda::std::inplace_vector<int, 2>;
  vec too_small{1, 2};

  try
  {
    const int input = 5;
    too_small.insert(too_small.begin(), input);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    too_small.insert(too_small.begin(), 1);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    too_small.insert(too_small.begin(), 5, 42);
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
    cuda::std::array<int, 3> input{42, 3, 1337};
    too_small.insert(too_small.begin(), iter{input.begin()}, iter{input.end()});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    cuda::std::array<int, 3> input{42, 3, 1337};
    too_small.insert(too_small.begin(), input.begin(), input.end());
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    too_small.insert(too_small.begin(), {42, 3, 1337});
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
    too_small.insert_range(too_small.begin(), input_range<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    too_small.insert_range(too_small.begin(), uncommon_range<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    too_small.insert_range(too_small.begin(), sized_uncommon_range<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    too_small.insert_range(too_small.begin(), cuda::std::array<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    too_small.append_range(input_range<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    too_small.append_range(uncommon_range<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    too_small.append_range(sized_uncommon_range<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    too_small.append_range(cuda::std::array<int, 3>{42, 3, 1337});
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
