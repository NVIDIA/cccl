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
  // Zero capacity inplace_vector is nothrow_copy_assignable
  static_assert(cuda::std::is_nothrow_copy_assignable<cuda::std::inplace_vector<T, 0>>::value, "");
  static_assert(cuda::std::is_nothrow_copy_assignable<cuda::std::inplace_vector<T, 42>>::value
                    == cuda::std::is_nothrow_copy_constructible<T>::value
                  && cuda::std::is_nothrow_copy_assignable<T>::value,
                "");

  { // inplace_vector<T, 0> can be copy assigned
    const cuda::std::inplace_vector<T, 0> input{};
    cuda::std::inplace_vector<T, 0> no_capacity{};
    no_capacity = input;
    assert(no_capacity.empty());
  }

  using inplace_vector = cuda::std::inplace_vector<T, 42>;
  { // inplace_vector<T, N> can be copy assigned an empty input
    const inplace_vector input{};
    inplace_vector vec{};
    vec = input;
    assert(vec.empty());
  }

  { // inplace_vector<T, N> can be copy assigned an empty input, shrinking
    const inplace_vector input{};
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec = input;
    assert(vec.empty());
  }

  { // inplace_vector<T, N> can be copy assigned a non-empty input, growing from empty
    const inplace_vector input{T(1), T(42), T(1337), T(0)};
    inplace_vector vec{};
    vec = input;
    assert(!vec.empty());
    assert(equal_range(vec, input));
  }

  { // inplace_vector<T, N> can be copy assigned a non-empty input, shrinking
    const inplace_vector input{T(1), T(42), T(1337), T(0)};
    inplace_vector vec{T(0), T(42), T(1337), T(42), T(5)};
    vec = input;
    assert(!vec.empty());
    assert(equal_range(vec, input));
  }

  { // inplace_vector<T, N> can be copy assigned a non-empty input, growing
    const inplace_vector input{T(1), T(42), T(1337), T(0)};
    inplace_vector vec{T(0), T(42)};
    vec = input;
    assert(!vec.empty());
    assert(equal_range(vec, input));
  }
}

template <class T>
__host__ __device__ constexpr void test_move()
{
  // Zero capacity inplace_vector is nothrow_move_assignable
  static_assert(cuda::std::is_nothrow_move_assignable<cuda::std::inplace_vector<T, 0>>::value, "");
  static_assert(cuda::std::is_nothrow_move_assignable<cuda::std::inplace_vector<T, 42>>::value
                    == cuda::std::is_nothrow_move_constructible<T>::value
                  && cuda::std::is_nothrow_move_assignable<T>::value,
                "");

  { // inplace_vector<T, 0> can be move assigned
    cuda::std::inplace_vector<T, 0> input{};
    cuda::std::inplace_vector<T, 0> no_capacity{};
    no_capacity = cuda::std::move(input);
    assert(no_capacity.empty());
    assert(input.empty());
  }

  using inplace_vector = cuda::std::inplace_vector<T, 42>;
  { // inplace_vector<T, N> can be move assigned an empty input
    inplace_vector input{};
    inplace_vector vec{};
    vec = cuda::std::move(input);
    assert(vec.empty());
    assert(input.empty());
  }

  { // inplace_vector<T, N> can be move assigned an empty input, shrinking
    inplace_vector input{};
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec = cuda::std::move(input);
    assert(vec.empty());
    assert(input.empty());
  }

  const cuda::std::array<T, 4> expected{T(1), T(42), T(1337), T(0)};
  { // inplace_vector<T, N> can be move assigned a non-empty input, growing from empty. clears input
    inplace_vector input{T(1), T(42), T(1337), T(0)};
    inplace_vector vec{};
    vec = cuda::std::move(input);
    assert(!vec.empty());
    assert(input.empty());
    assert(equal_range(vec, expected));
  }

  { // inplace_vector<T, N> can be move assigned a non-empty input, shrinking. clears input
    inplace_vector input{T(1), T(42), T(1337), T(0)};
    inplace_vector vec{T(0), T(42), T(1337), T(42), T(5)};
    vec = cuda::std::move(input);
    assert(!vec.empty());
    assert(input.empty());
    assert(equal_range(vec, expected));
  }

  { // inplace_vector<T, N> can be move assigned a non-empty input, growing. clears input
    inplace_vector input{T(1), T(42), T(1337), T(0)};
    inplace_vector vec{T(0), T(42)};
    vec = cuda::std::move(input);
    assert(!vec.empty());
    assert(input.empty());
    assert(equal_range(vec, expected));
  }
}

template <class T>
__host__ __device__ constexpr void test_init_list()
{
  { // inplace_vector<T, 0> can be assigned an empty initializer_list
    const cuda::std::initializer_list<T> input{};
    cuda::std::inplace_vector<T, 0> vec{};
    vec = input;
    assert(vec.empty());
  }

  using inplace_vector = cuda::std::inplace_vector<T, 42>;
  const cuda::std::initializer_list<T> empty_input{};
  { // inplace_vector<T, N> can be assigned an empty initializer_list
    inplace_vector vec{};
    vec = empty_input;
    assert(vec.empty());
  }

  { // inplace_vector<T, N> can be assigned an empty initializer_list, shrinking
    inplace_vector vec{T(1), T(42), T(1337), T(0)};
    vec = empty_input;
    assert(vec.empty());
  }

  const cuda::std::initializer_list<T> input{T(1), T(42), T(1337), T(0)};
  { // inplace_vector<T, N> can be assigned a non-empty initializer_list, from empty
    inplace_vector vec{};
    vec = input;
    assert(!vec.empty());
    assert(equal_range(vec, input));
  }

  { // inplace_vector<T, N> can be assigned a non-empty initializer_list, shrinking
    inplace_vector vec{T(0), T(42), T(1337), T(42), T(5)};
    vec = input;
    assert(!vec.empty());
    assert(equal_range(vec, input));
  }

  { // inplace_vector<T, N> can be assigned a non-empty initializer_list, growing from non empty
    inplace_vector vec{T(0), T(42)};
    vec = input;
    assert(!vec.empty());
    assert(equal_range(vec, input));
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
{ // assignment throws std::bad_alloc
  constexpr size_t capacity = 4;
  using inplace_vector      = cuda::std::inplace_vector<int, capacity>;
  inplace_vector too_small{};

  try
  {
    cuda::std::initializer_list<int> input{0, 1, 2, 3, 4, 5, 6};
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
