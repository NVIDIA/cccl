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

template <class T>
__host__ __device__ constexpr void test_copy()
{
  static_assert(cuda::std::is_nothrow_copy_assignable<cuda::std::inplace_vector<T, 0>>::value, "");
  static_assert(cuda::std::is_nothrow_copy_assignable<cuda::std::inplace_vector<T, 42>>::value
                    == cuda::std::is_nothrow_copy_constructible<T>::value
                  && cuda::std::is_nothrow_copy_assignable<T>::value,
                "");

  {
    const cuda::std::inplace_vector<T, 0> input{};
    cuda::std::inplace_vector<T, 0> no_capacity{};
    no_capacity = input;
    assert(no_capacity.empty());
  }

  {
    const cuda::std::inplace_vector<T, 42> input{};
    cuda::std::inplace_vector<T, 42> empty_to_empty{};
    empty_to_empty = input;
    assert(empty_to_empty.empty());
  }

  {
    const cuda::std::inplace_vector<T, 42> input{};
    cuda::std::inplace_vector<T, 42> empty_to_non_empty{T(1), T(42), T(1337), T(0)};
    empty_to_non_empty = input;
    assert(empty_to_non_empty.empty());
  }

  {
    const cuda::std::inplace_vector<T, 42> input{T(1), T(42), T(1337), T(0)};
    cuda::std::inplace_vector<T, 42> non_empty_to_empty{};
    non_empty_to_empty = input;
    assert(!non_empty_to_empty.empty());
    assert(cuda::std::equal(non_empty_to_empty.begin(), non_empty_to_empty.end(), input.begin(), input.end()));
  }

  {
    const cuda::std::inplace_vector<T, 42> input{T(1), T(42), T(1337), T(0)};
    cuda::std::inplace_vector<T, 42> non_empty_to_non_empty_shrink{T(0), T(42), T(1337), T(42), T(5)};
    non_empty_to_non_empty_shrink = input;
    assert(!non_empty_to_non_empty_shrink.empty());
    assert(cuda::std::equal(
      non_empty_to_non_empty_shrink.begin(), non_empty_to_non_empty_shrink.end(), input.begin(), input.end()));
  }

  {
    const cuda::std::inplace_vector<T, 42> input{T(1), T(42), T(1337), T(0)};
    cuda::std::inplace_vector<T, 42> non_empty_to_non_empty_grow{T(0), T(42)};
    non_empty_to_non_empty_grow = input;
    assert(!non_empty_to_non_empty_grow.empty());
    assert(cuda::std::equal(
      non_empty_to_non_empty_grow.begin(), non_empty_to_non_empty_grow.end(), input.begin(), input.end()));
  }
}

template <class T>
__host__ __device__ constexpr void test_move()
{
  static_assert(cuda::std::is_nothrow_move_assignable<cuda::std::inplace_vector<T, 0>>::value, "");
  static_assert(cuda::std::is_nothrow_move_assignable<cuda::std::inplace_vector<T, 42>>::value
                    == cuda::std::is_nothrow_move_constructible<T>::value
                  && cuda::std::is_nothrow_move_assignable<T>::value,
                "");

  {
    cuda::std::inplace_vector<T, 0> input{};
    cuda::std::inplace_vector<T, 0> no_capacity{};
    no_capacity = cuda::std::move(input);
    assert(no_capacity.empty());
    assert(input.empty());
  }

  {
    cuda::std::inplace_vector<T, 42> input{};
    cuda::std::inplace_vector<T, 42> empty_to_empty{};
    empty_to_empty = cuda::std::move(input);
    assert(empty_to_empty.empty());
    assert(input.empty());
  }

  {
    cuda::std::inplace_vector<T, 42> input{};
    cuda::std::inplace_vector<T, 42> empty_to_non_empty{T(1), T(42), T(1337), T(0)};
    empty_to_non_empty = cuda::std::move(input);
    assert(empty_to_non_empty.empty());
    assert(input.empty());
  }

  const cuda::std::initializer_list<T> expected{T(1), T(42), T(1337), T(0)};
  {
    cuda::std::inplace_vector<T, 42> input{T(1), T(42), T(1337), T(0)};
    cuda::std::inplace_vector<T, 42> non_empty_to_empty{};
    non_empty_to_empty = cuda::std::move(input);
    assert(!non_empty_to_empty.empty());
    assert(cuda::std::equal(non_empty_to_empty.begin(), non_empty_to_empty.end(), expected.begin(), expected.end()));
    assert(input.empty());
  }

  {
    cuda::std::inplace_vector<T, 42> input{T(1), T(42), T(1337), T(0)};
    cuda::std::inplace_vector<T, 42> non_empty_to_non_empty_shrink{T(0), T(42), T(1337), T(42), T(5)};
    non_empty_to_non_empty_shrink = cuda::std::move(input);
    assert(!non_empty_to_non_empty_shrink.empty());
    assert(cuda::std::equal(
      non_empty_to_non_empty_shrink.begin(), non_empty_to_non_empty_shrink.end(), expected.begin(), expected.end()));
    assert(input.empty());
  }

  {
    cuda::std::inplace_vector<T, 42> input{T(1), T(42), T(1337), T(0)};
    cuda::std::inplace_vector<T, 42> non_empty_to_non_empty_grow{T(0), T(42)};
    non_empty_to_non_empty_grow = cuda::std::move(input);
    assert(!non_empty_to_non_empty_grow.empty());
    assert(cuda::std::equal(
      non_empty_to_non_empty_grow.begin(), non_empty_to_non_empty_grow.end(), expected.begin(), expected.end()));
    assert(input.empty());
  }
}

template <class T>
__host__ __device__ constexpr void test_init_list()
{
  {
    const cuda::std::initializer_list<T> input{};
    cuda::std::inplace_vector<T, 0> no_capacity{};
    no_capacity = input;
    assert(no_capacity.empty());
  }

  {
    const cuda::std::initializer_list<T> input{};
    cuda::std::inplace_vector<T, 42> empty_to_empty{};
    empty_to_empty = input;
    assert(empty_to_empty.empty());
  }

  {
    const cuda::std::initializer_list<T> input{};
    cuda::std::inplace_vector<T, 42> empty_to_non_empty{T(1), T(42), T(1337), T(0)};
    empty_to_non_empty = input;
    assert(empty_to_non_empty.empty());
  }

  {
    const cuda::std::initializer_list<T> input{T(1), T(42), T(1337), T(0)};
    cuda::std::inplace_vector<T, 42> non_empty_to_empty{};
    non_empty_to_empty = input;
    assert(!non_empty_to_empty.empty());
    assert(cuda::std::equal(non_empty_to_empty.begin(), non_empty_to_empty.end(), input.begin(), input.end()));
  }

  {
    const cuda::std::initializer_list<T> input{T(1), T(42), T(1337), T(0)};
    cuda::std::inplace_vector<T, 42> non_empty_to_non_empty_shrink{T(0), T(42), T(1337), T(42), T(5)};
    non_empty_to_non_empty_shrink = input;
    assert(!non_empty_to_non_empty_shrink.empty());
    assert(cuda::std::equal(
      non_empty_to_non_empty_shrink.begin(), non_empty_to_non_empty_shrink.end(), input.begin(), input.end()));
  }

  {
    const cuda::std::initializer_list<T> input{T(1), T(42), T(1337), T(0)};
    cuda::std::inplace_vector<T, 42> non_empty_to_non_empty_grow{T(0), T(42)};
    non_empty_to_non_empty_grow = input;
    assert(!non_empty_to_non_empty_grow.empty());
    assert(cuda::std::equal(
      non_empty_to_non_empty_grow.begin(), non_empty_to_non_empty_grow.end(), input.begin(), input.end()));
  }
}

template <class T>
__host__ __device__ constexpr void test()
{
  test_copy<T>();
  test_move<T>();
  test_init_list<T>();
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
{ // assignment throws std::bad_alloc
  constexpr size_t capacity = 4;
  using vec                 = cuda::std::inplace_vector<int, capacity>;

  try
  {
    cuda::std::initializer_list<int> input{0, 1, 2, 3, 4, 5, 6};
    vec too_small{};
    too_small = input;
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }
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
