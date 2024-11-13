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

#include "cuda/std/__type_traits/is_nothrow_default_constructible.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
#  include <stdexcept>
#endif // !TEST_HAS_NO_EXCEPTIONS

_CCCL_DIAG_SUPPRESS_GCC("-Wmissing-braces")
_CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-braces")
_CCCL_DIAG_SUPPRESS_MSVC(5246)

template <class T>
__host__ __device__ constexpr void test_default()
{
  { // inplace_vecto<T, 0> is default_constructible
    cuda::std::inplace_vector<T, 0> vec{};
    assert(vec.empty());
    static_assert(cuda::std::is_nothrow_default_constructible<cuda::std::inplace_vector<T, 0>>::value, "");
  }

  { // inplace_vecto<T, N> is default_constructible
    cuda::std::inplace_vector<T, 42> vec{};
    assert(vec.empty());
    static_assert(cuda::std::is_nothrow_default_constructible<cuda::std::inplace_vector<T, 42>>::value, "");
  }
}

template <class T>
__host__ __device__ constexpr void test_copy_move()
{
  // Zero capacity inplace_vector is trivial
  static_assert(cuda::std::is_nothrow_copy_constructible<cuda::std::inplace_vector<T, 0>>::value, "");
  static_assert(cuda::std::is_nothrow_move_constructible<cuda::std::inplace_vector<T, 0>>::value, "");
  static_assert(cuda::std::is_nothrow_copy_constructible<cuda::std::inplace_vector<T, 42>>::value
                  == cuda::std::is_nothrow_copy_constructible<T>::value,
                "");
  static_assert(cuda::std::is_nothrow_move_constructible<cuda::std::inplace_vector<T, 42>>::value
                  == cuda::std::is_nothrow_move_constructible<T>::value,
                "");
  { // inplace_vector<T, 0> can be copy constructed
    cuda::std::inplace_vector<T, 0> input{};
    cuda::std::inplace_vector<T, 0> vec(input);
    assert(vec.empty());
  }

  { // inplace_vector<T, 0> can be move constructed
    cuda::std::inplace_vector<T, 0> input{};
    cuda::std::inplace_vector<T, 0> vec(cuda::std::move(input));
    assert(input.empty());
    assert(vec.empty());
  }

  using inplace_vector = cuda::std::inplace_vector<T, 42>;
  { // inplace_vector<T, N> can be copy constructed from empty input
    const inplace_vector input{};
    inplace_vector vec(input);
    assert(vec.empty());
  }

  { // inplace_vector<T, N> can be move constructed with empty input
    inplace_vector input{};
    inplace_vector vec(cuda::std::move(input));
    assert(vec.empty());
    assert(input.empty());
  }

  { // inplace_vector<T, N> can be copy constructed from non-empty input
    inplace_vector input{T(1), T(42), T(1337), T(0)};
    inplace_vector vec(input);
    assert(!vec.empty());
    assert(equal_range(vec, input));
  }

  { // inplace_vector<T, N> can be move constructed from non-empty input
    inplace_vector input{T(1), T(42), T(1337), T(0)};
    inplace_vector vec(cuda::std::move(input));
    assert(!vec.empty());
    assert(input.size() == 4);
    assert(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
  }
}

template <class T>
__host__ __device__ constexpr void test_size()
{
  { // inplace_vector<T, 0> can be constructed from a size
    cuda::std::inplace_vector<T, 0> vec(0);
    assert(vec.empty());
#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 10) && !defined(TEST_COMPILER_MSVC)
    static_assert(!noexcept(cuda::std::inplace_vector<T, 0>(0)), "");
#endif // !TEST_COMPILER_GCC < 10 && !TEST_COMPILER_MSVC
  }

  using inplace_vector = cuda::std::inplace_vector<T, 42>;
  { // inplace_vector<T, N> can be constructed from a size, is empty if zero
    inplace_vector vec(0);
    assert(vec.empty());
#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 10) && !defined(TEST_COMPILER_MSVC)
    static_assert(!noexcept(inplace_vector(0)), "");
#endif // !TEST_COMPILER_GCC < 10 && !TEST_COMPILER_MSVC
  }

  { // inplace_vector<T, N> can be constructed from a size, elements are value initialized
    constexpr size_t size{3};
    inplace_vector vec(size);
    assert(!vec.empty());
    assert(equal_range(vec, cuda::std::array<T, size>{T(0), T(0), T(0)}));
#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 10) && !defined(TEST_COMPILER_MSVC)
    static_assert(!noexcept(inplace_vector(3)), "");
#endif // !TEST_COMPILER_GCC < 10 && !TEST_COMPILER_MSVC
  }
}

template <class T>
__host__ __device__ constexpr void test_size_value()
{
  { // inplace_vector<T, 0> can be constructed from a size and a const T&
    cuda::std::inplace_vector<T, 0> vec(0, T(42));
    assert(vec.empty());
#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 10) && !defined(TEST_COMPILER_MSVC)
    static_assert(!noexcept(cuda::std::inplace_vector<T, 0>(0, T(42))), "");
#endif // !TEST_COMPILER_GCC < 10 && !TEST_COMPILER_MSVC
  }

  using inplace_vector = cuda::std::inplace_vector<T, 42>;
  { // inplace_vector<T, N> can be constructed from a size and a const T&, is empty if zero
    inplace_vector vec(0, T(42));
    assert(vec.empty());
#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 10) && !defined(TEST_COMPILER_MSVC)
    static_assert(!noexcept(inplace_vector(0, T(42))), "");
#endif // !TEST_COMPILER_GCC < 10 && !TEST_COMPILER_MSVC
  }

  { // inplace_vector<T, N> can be constructed from a size and a const T&, elements are copied
    constexpr size_t size{3};
    inplace_vector vec(size, T(42));
    assert(!vec.empty());
    assert(equal_range(vec, cuda::std::array<T, size>{T(42), T(42), T(42)}));
#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 10) && !defined(TEST_COMPILER_MSVC)
    static_assert(!noexcept(inplace_vector(3, T(42))), "");
#endif // !TEST_COMPILER_GCC < 10 && !TEST_COMPILER_MSVC
  }
}

template <class T>
__host__ __device__ constexpr void test_iter()
{
  const cuda::std::array<T, 4> input{T(1), T(42), T(1337), T(0)};
  { // inplace_vector<T, 0> can be constructed from two equal input iterators
    using iter = cpp17_input_iterator<const T*>;
    cuda::std::inplace_vector<T, 0> vec(iter{input.begin()}, iter{input.begin()});
    assert(vec.empty());
  }

  { // inplace_vector<T, 0> can be constructed from two equal forward iterators
    using iter = forward_iterator<const T*>;
    cuda::std::inplace_vector<T, 0> vec(iter{input.begin()}, iter{input.begin()});
    assert(vec.empty());
  }

  using inplace_vector = cuda::std::inplace_vector<T, 42>;
  { // inplace_vector<T, N> can be constructed from two equal input iterators
    using iter = cpp17_input_iterator<const T*>;
    inplace_vector vec(iter{input.begin()}, iter{input.begin()});
    assert(vec.empty());
  }

  { // inplace_vector<T, N> can be constructed from two equal forward iterators
    using iter = forward_iterator<const T*>;
    inplace_vector vec(iter{input.begin()}, iter{input.begin()});
    assert(vec.empty());
  }

  { // inplace_vector<T, N> can be constructed from two input iterators
    using iter = cpp17_input_iterator<const T*>;
    inplace_vector vec(iter{input.begin()}, iter{input.end()});
    assert(!vec.empty());
    assert(equal_range(vec, input));
  }

  { // inplace_vector<T, N> can be constructed from two forward iterators
    using iter = forward_iterator<const T*>;
    inplace_vector vec(iter{input.begin()}, iter{input.end()});
    assert(!vec.empty());
    assert(equal_range(vec, input));
  }
}

template <class T>
__host__ __device__ constexpr void test_init_list()
{
  { // inplace_vector<T, 0> can be constructed from an empty initializer_list
    cuda::std::initializer_list<T> input{};
    cuda::std::inplace_vector<T, 0> vec(input);
    assert(vec.empty());
  }

  using inplace_vector = cuda::std::inplace_vector<T, 42>;
  { // inplace_vector<T, N> can be constructed from an empty initializer_list
    cuda::std::initializer_list<T> input{};
    inplace_vector vec(input);
    assert(vec.empty());
  }

  { // inplace_vector<T, N> can be constructed from a non-empty initializer_list
    cuda::std::array<T, 4> expected{T(1), T(42), T(1337), T(0)};
    inplace_vector vec({T(1), T(42), T(1337), T(0)});
    assert(!vec.empty());
    assert(equal_range(vec, expected));
  }
}

#if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC)
template <class T, template <class, size_t> class Range>
__host__ __device__ constexpr void test_range()
{
  { // inplace_vector<T, 0> can be constructed from an empty range
    cuda::std::inplace_vector<T, 0> vec(cuda::std::from_range, Range<T, 0>{});
    assert(vec.empty());
  }

  using inplace_vector = cuda::std::inplace_vector<T, 42>;
  { // inplace_vector<T, N> can be constructed from an empty range
    inplace_vector vec(cuda::std::from_range, Range<T, 0>{});
    assert(vec.empty());
  }

  { // inplace_vector<T, N> can be constructed from a non-empty range
    inplace_vector vec(cuda::std::from_range, Range<T, 4>{T(1), T(42), T(1337), T(0)});
    assert(!vec.empty());
    assert(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
  }
}

template <class T>
__host__ __device__ constexpr void test_range()
{
#  if !defined(TEST_COMPILER_GCC) || __GNUC__ >= 8
  test_range<T, input_range>();
  test_range<T, uncommon_range>();
  test_range<T, sized_uncommon_range>();
#  endif // !TEST_COMPILER_GCC < 8
  test_range<T, cuda::std::array>();
}
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC)

template <class T, cuda::std::enable_if_t<cuda::std::is_trivial<T>::value, int> = 0>
__host__ __device__ constexpr void test()
{
  test_default<T>();
  test_copy_move<T>();
  test_size<T>();
  test_size_value<T>();
  test_iter<T>();
  test_init_list<T>();
#if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC)
  test_range<T>();
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC)
}

template <class T, cuda::std::enable_if_t<!cuda::std::is_trivial<T>::value, int> = 0>
__host__ __device__ constexpr void test()
{
  test_default<T>();

  if (!cuda::std::__libcpp_is_constant_evaluated())
  {
    test_copy_move<T>();
    test_size<T>();
    test_size_value<T>();
    test_iter<T>();
    test_init_list<T>();
#if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC)
    test_range<T>();
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC)
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<Trivial>();
  test<NonTrivial>();
  test<ThrowingDefaultConstruct>();
  test<ThrowingCopyConstructor>();
  test<ThrowingMoveConstructor>();
  test<ThrowingCopyAssignment>();
  test<ThrowingMoveAssignment>();

  // Due to reinterpret_cast within the destructor a on trivially destructible type cannot be constexpr at all
  if (!cuda::std::__libcpp_is_constant_evaluated())
  {
    test<NonTrivialDestructor>();
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{ // constructors throw std::bad_alloc
  constexpr size_t capacity = 4;
  using inplace_vector      = cuda::std::inplace_vector<int, capacity>;

  try
  {
    inplace_vector too_small(2 * capacity);
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
    inplace_vector too_small(2 * capacity, 42);
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
    cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
    inplace_vector too_small(iter{input.begin()}, iter{input.end()});
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
    cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
    inplace_vector too_small(input.begin(), input.end());
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
    cuda::std::initializer_list<int> input{0, 1, 2, 3, 4, 5, 6};
    inplace_vector too_small(input);
    assert(false);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

#  if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC)
  try
  {
    input_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
    inplace_vector too_small(cuda::std::from_range, input);
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
    uncommon_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
    inplace_vector too_small(cuda::std::from_range, input);
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
    sized_uncommon_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
    inplace_vector too_small(cuda::std::from_range, input);
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
    cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
    inplace_vector too_small(cuda::std::from_range, input);
    assert(false);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }
#  endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC)
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
