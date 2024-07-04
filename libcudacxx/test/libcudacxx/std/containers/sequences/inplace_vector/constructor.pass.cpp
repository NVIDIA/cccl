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
  {
    cuda::std::inplace_vector<T, 0> no_capacity{};
    assert(no_capacity.empty());
    static_assert(cuda::std::is_nothrow_default_constructible<cuda::std::inplace_vector<T, 0>>::value, "");
  }

  {
    cuda::std::inplace_vector<T, 42> with_capacity{};
    assert(with_capacity.empty());
    static_assert(cuda::std::is_nothrow_default_constructible<cuda::std::inplace_vector<T, 42>>::value, "");
  }
}

template <class T>
__host__ __device__ constexpr void test_copy_move()
{
  static_assert(cuda::std::is_nothrow_copy_constructible<cuda::std::inplace_vector<T, 0>>::value, "");
  static_assert(cuda::std::is_nothrow_move_constructible<cuda::std::inplace_vector<T, 0>>::value, "");

  static_assert(cuda::std::is_nothrow_copy_constructible<cuda::std::inplace_vector<T, 42>>::value
                  == cuda::std::is_nothrow_copy_constructible<T>::value,
                "");
  static_assert(cuda::std::is_nothrow_move_constructible<cuda::std::inplace_vector<T, 42>>::value
                  == cuda::std::is_nothrow_move_constructible<T>::value,
                "");
  {
    cuda::std::inplace_vector<T, 0> input{};
    cuda::std::inplace_vector<T, 0> no_capacity_copy(input);
    assert(no_capacity_copy.empty());

    cuda::std::inplace_vector<T, 0> no_capacity_move(cuda::std::move(input));
    assert(no_capacity_move.empty());
  }

  {
    cuda::std::inplace_vector<T, 42> input{};
    cuda::std::inplace_vector<T, 42> with_capacity_empty_copy(input);
    assert(with_capacity_empty_copy.empty());

    cuda::std::inplace_vector<T, 42> with_capacity_empty_move(cuda::std::move(input));
    assert(with_capacity_empty_move.empty());
  }

  {
    cuda::std::inplace_vector<T, 42> input{T(1), T(42), T(1337), T(0)};
    cuda::std::inplace_vector<T, 42> with_capacity_copy(input);
    assert(!with_capacity_copy.empty());
    assert(with_capacity_copy.size() == input.size());
    assert(cuda::std::equal(with_capacity_copy.begin(), with_capacity_copy.end(), input.begin(), input.end()));

    cuda::std::inplace_vector<T, 42> with_capacity_move(cuda::std::move(input));
    assert(input.empty());
    assert(!with_capacity_move.empty());
    assert(with_capacity_copy.size() == with_capacity_copy.size());
    assert(cuda::std::equal(
      with_capacity_copy.begin(), with_capacity_copy.end(), with_capacity_copy.begin(), with_capacity_copy.end()));
  }
}

template <class T>
__host__ __device__ constexpr void test_size()
{
  {
    cuda::std::inplace_vector<T, 0> no_capacity(0);
    assert(no_capacity.empty());
    static_assert(!noexcept(cuda::std::inplace_vector<T, 0>(0)), "");
  }

  {
    cuda::std::inplace_vector<T, 42> with_capacity_empty(0);
    assert(with_capacity_empty.empty());
    static_assert(!noexcept(cuda::std::inplace_vector<T, 42>(0)), "");
  }

  {
    cuda::std::inplace_vector<T, 42> with_capacity_non_empty(3);
    assert(!with_capacity_non_empty.empty());
    assert(with_capacity_non_empty.size() == 3);
    static_assert(!noexcept(cuda::std::inplace_vector<T, 42>(3)), "");

    const T expected[] = {T(), T(), T()};
    assert(cuda::std::equal(with_capacity_non_empty.begin(), with_capacity_non_empty.end(), expected, expected + 3));
  }
}

template <class T>
__host__ __device__ constexpr void test_size_value()
{
  {
    cuda::std::inplace_vector<T, 0> no_capacity(0, T(42));
    assert(no_capacity.empty());
    static_assert(!noexcept(cuda::std::inplace_vector<T, 0>(0, T(42))), "");
  }

  {
    cuda::std::inplace_vector<T, 42> with_capacity_empty(0, T(42));
    assert(with_capacity_empty.empty());
    static_assert(!noexcept(cuda::std::inplace_vector<T, 42>(0, T(42))), "");
  }

  {
    cuda::std::inplace_vector<T, 42> with_capacity_non_empty(3, T(42));
    assert(!with_capacity_non_empty.empty());
    assert(with_capacity_non_empty.size() == 3);
    static_assert(!noexcept(cuda::std::inplace_vector<T, 42>(3, T(42))), "");

    const T expected[] = {T(42), T(42), T(42)};
    assert(cuda::std::equal(with_capacity_non_empty.begin(), with_capacity_non_empty.end(), expected, expected + 3));
  }
}

template <class T>
__host__ __device__ constexpr void test_iter()
{
  const T input[] = {T(1), T(42), T(1337), T(0)};
  {
    using iter = cpp17_input_iterator<const T*>;
    cuda::std::inplace_vector<T, 0> from_input_iter_no_capacity(iter{input}, iter{input});
    assert(from_input_iter_no_capacity.empty());
  }

  {
    using iter = forward_iterator<const T*>;
    cuda::std::inplace_vector<T, 0> from_forward_iter_no_capacity(iter{input}, iter{input});
    assert(from_forward_iter_no_capacity.empty());
  }

  {
    using iter = cpp17_input_iterator<const T*>;
    cuda::std::inplace_vector<T, 42> from_input_iter_empty(iter{input}, iter{input});
    assert(from_input_iter_empty.empty());
  }

  {
    using iter = forward_iterator<const T*>;
    cuda::std::inplace_vector<T, 42> from_forward_iter_empty(iter{input}, iter{input});
    assert(from_forward_iter_empty.empty());
  }

  {
    using iter = cpp17_input_iterator<const T*>;
    cuda::std::inplace_vector<T, 42> from_input_iter_non_empty(iter{input}, iter{input + 3});
    assert(!from_input_iter_non_empty.empty());
    assert(from_input_iter_non_empty.size() == 3);
    assert(cuda::std::equal(from_input_iter_non_empty.begin(), from_input_iter_non_empty.end(), input, input + 3));
  }

  {
    using iter = forward_iterator<const T*>;
    cuda::std::inplace_vector<T, 42> from_forward_iter_non_empty(iter{input}, iter{input + 3});
    assert(!from_forward_iter_non_empty.empty());
    assert(from_forward_iter_non_empty.size() == 3);
    assert(cuda::std::equal(from_forward_iter_non_empty.begin(), from_forward_iter_non_empty.end(), input, input + 3));
  }
}

template <class T>
__host__ __device__ constexpr void test_init_list()
{
  {
    cuda::std::initializer_list<T> input{};
    cuda::std::inplace_vector<T, 0> no_capacity(input);
    assert(no_capacity.empty());
  }

  {
    cuda::std::initializer_list<T> input{};
    cuda::std::inplace_vector<T, 42> from_empty(input);
    assert(from_empty.empty());
  }

  {
    cuda::std::initializer_list<T> input{T(1), T(42), T(1337), T(0)};
    cuda::std::inplace_vector<T, 42> from_non_empty(input);
    assert(!from_non_empty.empty());
    assert(from_non_empty.size() == input.size());
    assert(cuda::std::equal(from_non_empty.begin(), from_non_empty.end(), input.begin(), input.end()));
  }
}

#if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
template <class T, template <class, size_t> class Range>
__host__ __device__ constexpr void test_range()
{
  {
    cuda::std::inplace_vector<T, 0> no_capacity(Range<T, 0>{});
    assert(no_capacity.empty());
  }

  {
    cuda::std::inplace_vector<T, 42> from_empty(Range<T, 0>{});
    assert(from_empty.empty());
  }

  {
    const cuda::std::array<T, 4> expected{T(1), T(42), T(1337), T(0)};
    cuda::std::inplace_vector<T, 42> from_non_empty(Range<T, 4>{T(1), T(42), T(1337), T(0)});
    assert(!from_non_empty.empty());
    assert(from_non_empty.size() == expected.size());
    assert(cuda::std::equal(from_non_empty.begin(), from_non_empty.end(), expected.begin(), expected.end()));
  }
}

template <class T>
__host__ __device__ constexpr void test_range()
{
  test_range<T, input_range>();
  test_range<T, uncommon_range>();
  test_range<T, sized_uncommon_range>();
  test_range<T, cuda::std::array>();
}
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)

template <class T, cuda::std::enable_if_t<cuda::std::is_trivial<T>::value, int> = 0>
__host__ __device__ constexpr void test()
{
  test_default<T>();
  test_copy_move<T>();
  test_size<T>();
  test_size_value<T>();
  test_iter<T>();
  test_init_list<T>();
#if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
  test_range<T>();
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
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
#if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
    test_range<T>();
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<NonTrivial>();
  test<ThrowingDefaultConstruct>();
  test<ThrowingCopyConstructor>();
  test<ThrowingMoveConstructor>();

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
  using vec                 = cuda::std::inplace_vector<int, capacity>;

  try
  {
    vec too_small(2 * capacity);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    vec too_small(2 * capacity, 42);
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
    vec too_small(iter{input.begin()}, iter{input.end()});
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
    vec too_small(input.begin(), input.end());
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
    vec too_small(input);
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
    input_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
    vec too_small(input);
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
    vec too_small(input);
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
    vec too_small(input);
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
    vec too_small(input);
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
