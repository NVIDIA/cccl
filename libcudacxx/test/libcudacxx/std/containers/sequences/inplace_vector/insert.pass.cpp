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
__host__ __device__ constexpr void test_range()
{
  constexpr size_t max_capacity = 5ull;
  using inplace_vector          = cuda::std::inplace_vector<T, max_capacity>;

  { // inplace_vector<T, 0>::insert_range(iter, range)
    cuda::std::inplace_vector<T, 0> vec{};
    const auto res = vec.insert_range(vec.begin(), Range<T, 0>{});
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(vec.empty());
    assert(res == vec.begin());
  }

  { // inplace_vector<T, N>::insert_range(iter, range)
    inplace_vector vec = {T(0), T(5)};
    const auto res     = vec.insert_range(vec.begin() + 1, Range<T, 3>{T(42), T(3), T(1337)});
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, cuda::std::array<T, 5>{T(0), T(42), T(3), T(1337), T(5)}));
    assert(res == vec.begin() + 1);
  }

  { // inplace_vector<T, 0>::insert_range(const iter, range)
    cuda::std::inplace_vector<T, 0> vec{};
    const auto res = vec.insert_range(vec.cbegin(), Range<T, 0>{});
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(vec.empty());
    assert(res == vec.begin());
  }

  { // inplace_vector<T, N>::insert_range(const_iter, range)
    inplace_vector vec = {T(0), T(5)};
    const auto res     = vec.insert_range(vec.cbegin() + 1, Range<T, 3>{T(42), T(3), T(1337)});
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, cuda::std::array<T, 5>{T(0), T(42), T(3), T(1337), T(5)}));
    assert(res == vec.cbegin() + 1);
  }

  { // inplace_vector<T, 0>::append_range(range)
    cuda::std::inplace_vector<T, 0> vec{};
    vec.append_range(Range<T, 0>{});
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::append_range(range)
    inplace_vector vec = {T(0), T(5)};
    vec.append_range(Range<T, 3>{T(42), T(3), T(1337)});
    assert(equal_range(vec, cuda::std::array<T, 5>{T(0), T(5), T(42), T(3), T(1337)}));
  }

  { // inplace_vector<T, 0>::try_append_range(range)
    Range<T, 3> input{T(42), T(3), T(1337)};
    cuda::std::inplace_vector<T, 0> vec{};
    auto res = vec.try_append_range(input);
    static_assert(cuda::std::is_same<decltype(res), cuda::std::ranges::iterator_t<Range<T, 3>>>::value, "");
    assert(vec.empty());
    assert(res == input.begin());
  }

  { // inplace_vector<T, N>::try_append_range(range)
    Range<T, 3> input{T(42), T(3), T(1337)};
    inplace_vector vec{T(0), T(5)};
    auto res = vec.try_append_range(input);
    static_assert(cuda::std::is_same<decltype(res), cuda::std::ranges::iterator_t<Range<T, 3>>>::value, "");
    assert(equal_range(vec, cuda::std::array<T, 5>{T(0), T(5), T(42), T(3), T(1337)}));
    assert(res == input.end());
  }

  { // inplace_vector<T, N>::try_append_range(range), beyond capacity
    Range<T, 4> input{T(42), T(3), T(1337), T(1)};
    inplace_vector vec{T(0), T(5)};
    auto res = vec.try_append_range(input);
    static_assert(cuda::std::is_same<decltype(res), cuda::std::ranges::iterator_t<Range<T, 3>>>::value, "");
    assert(equal_range(vec, cuda::std::array<T, 5>{T(0), T(5), T(42), T(3), T(1337)}));
    assert(++res == input.end());
  }
}
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)

template <class T>
__host__ __device__ constexpr void test()
{
  constexpr size_t max_capacity = 42ull;
  using inplace_vector          = cuda::std::inplace_vector<T, max_capacity>;

  { // inplace_vector<T, N>::insert(iter, const T&)
    const T to_be_inserted = 3;
    inplace_vector vec     = {T(0), T(5)};
    const auto res         = vec.insert(vec.begin() + 1, to_be_inserted);
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, cuda::std::array<T, 3>{T(0), T(3), T(5)}));
    assert(res == vec.begin() + 1);
  }

  { // inplace_vector<T, N>::insert(const_iter, const T&)
    const T to_be_inserted = 3;
    inplace_vector vec     = {T(0), T(5)};
    const auto res         = vec.insert(vec.cbegin() + 1, to_be_inserted);
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, cuda::std::array<T, 3>{T(0), T(3), T(5)}));
    assert(res == vec.begin() + 1);
  }

  { // inplace_vector<T, N>::insert(iter, T&&)
    inplace_vector vec = {T(0), T(5)};
    const auto res     = vec.insert(vec.begin() + 1, T(3));
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, cuda::std::array<T, 3>{T(0), T(3), T(5)}));
    assert(res == vec.begin() + 1);
  }

  { // inplace_vector<T, N>::insert(const_iter, T&&)
    inplace_vector vec = {T(0), T(5)};
    const auto res     = vec.insert(vec.cbegin() + 1, T(3));
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, cuda::std::array<T, 3>{T(0), T(3), T(5)}));
    assert(res == vec.begin() + 1);
  }

  const cuda::std::array<T, 5> expected{T(0), T(42), T(3), T(1337), T(5)};
  cuda::std::initializer_list<T> input{T(42), T(3), T(1337)};
  { // inplace_vector<T, N>::insert(iter, iter, iter), input iterators
    using iter         = cpp17_input_iterator<const T*>;
    inplace_vector vec = {T(0), T(5)};
    const auto res     = vec.insert(vec.begin() + 1, iter{input.begin()}, iter{input.end()});
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, expected));
    assert(res == vec.begin() + 1);
  }

  { // inplace_vector<T, N>::insert(const_iter, iter, iter), input iterators
    using iter         = cpp17_input_iterator<const T*>;
    inplace_vector vec = {T(0), T(5)};
    const auto res     = vec.insert(vec.cbegin() + 1, iter{input.begin()}, iter{input.end()});
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, expected));
    assert(res == vec.cbegin() + 1);
  }

  { // inplace_vector<T, N>::insert(iter, iter, iter), forward iterators
    inplace_vector vec = {T(0), T(5)};
    const auto res     = vec.insert(vec.begin() + 1, input.begin(), input.end());
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, expected));
    assert(res == vec.begin() + 1);
  }

  { // inplace_vector<T, N>::insert(const_iter, iter, iter), forward iterators
    inplace_vector vec = {T(0), T(5)};
    const auto res     = vec.insert(vec.cbegin() + 1, input.begin(), input.end());
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, expected));
    assert(res == vec.cbegin() + 1);
  }

  { // inplace_vector<T, N>::insert(iter, init_list)
    inplace_vector vec = {T(0), T(5)};
    const auto res     = vec.insert(vec.begin() + 1, input);
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, expected));
    assert(res == vec.begin() + 1);
  }

  { // inplace_vector<T, N>::insert(const_iter, init_list)
    inplace_vector vec = {T(0), T(5)};
    const auto res     = vec.insert(vec.cbegin() + 1, input);
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, expected));
    assert(res == vec.cbegin() + 1);
  }

#if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
  test_range<T, input_range>();
  test_range<T, uncommon_range>();
  test_range<T, sized_uncommon_range>();
  test_range<T, cuda::std::array>();
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
void test_exceptions()
{ // insert throws std::bad_alloc
  using inplace_vector = cuda::std::inplace_vector<int, 2>;
  inplace_vector too_small{1, 2};

  try
  {
    const int input = 5;
    too_small.insert(too_small.begin(), input);
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
    too_small.insert(too_small.begin(), 1);
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
    too_small.insert(too_small.begin(), 5, 42);
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
    cuda::std::array<int, 3> input{42, 3, 1337};
    too_small.insert(too_small.begin(), iter{input.begin()}, iter{input.end()});
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
    cuda::std::array<int, 3> input{42, 3, 1337};
    too_small.insert(too_small.begin(), input.begin(), input.end());
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
    too_small.insert(too_small.begin(), {42, 3, 1337});
    assert(false);
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
    too_small.insert_range(too_small.begin(), uncommon_range<int, 3>{42, 3, 1337});
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
    too_small.insert_range(too_small.begin(), sized_uncommon_range<int, 3>{42, 3, 1337});
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
    too_small.insert_range(too_small.begin(), cuda::std::array<int, 3>{42, 3, 1337});
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
    too_small.append_range(input_range<int, 3>{42, 3, 1337});
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
    too_small.append_range(uncommon_range<int, 3>{42, 3, 1337});
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
    too_small.append_range(sized_uncommon_range<int, 3>{42, 3, 1337});
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
    too_small.append_range(cuda::std::array<int, 3>{42, 3, 1337});
    assert(false);
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
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
