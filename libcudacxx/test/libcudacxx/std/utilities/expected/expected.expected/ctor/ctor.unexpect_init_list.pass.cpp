//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Older Clangs do not support the C++20 feature to constrain destructors

// template<class U, class... Args>
//   constexpr explicit expected(unexpect_t, initializer_list<U> il, Args&&... args);
//
// Constraints: is_constructible_v<E, initializer_list<U>&, Args...> is true.
//
// Effects: Direct-non-list-initializes unex with il, cuda::std::forward<Args>(args)....
//
// Postconditions: has_value() is false.
//
// Throws: Any exception thrown by the initialization of unex.

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/inplace_vector>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

// Test Constraints:
static_assert(cuda::std::is_constructible_v<cuda::std::expected<int, cuda::std::inplace_vector<int, 3>>,
                                            cuda::std::unexpect_t,
                                            cuda::std::initializer_list<int>>,
              "");

// !is_constructible_v<T, initializer_list<U>&, Args...>
static_assert(
  !cuda::std::is_constructible_v<cuda::std::expected<int, int>, cuda::std::unexpect_t, cuda::std::initializer_list<int>>,
  "");

// test explicit
template <class T>
__host__ __device__ void conversion_test(T);

template <class T, class... Args>
_CCCL_CONCEPT ImplicitlyConstructible = _CCCL_REQUIRES_EXPR((T, variadic Args), T t, Args&&... args)(
  (conversion_test<T>({cuda::std::forward<Args>(args)...})));
static_assert(ImplicitlyConstructible<int, int>, "");

static_assert(!ImplicitlyConstructible<cuda::std::expected<int, cuda::std::inplace_vector<int, 3>>,
                                       cuda::std::unexpect_t,
                                       cuda::std::initializer_list<int>>,
              "");

template <size_t N, class... Ts>
struct Data
{
  int vec_[N]{};
  cuda::std::tuple<Ts...> tuple_;

  _CCCL_TEMPLATE(class... Us)
  _CCCL_REQUIRES(cuda::std::is_constructible<cuda::std::tuple<Ts...>, Us&&...>::value)
  __host__ __device__ constexpr Data(cuda::std::initializer_list<int> il, Us&&... us)
      : tuple_(cuda::std::forward<Us>(us)...)
  {
    auto ibegin = il.begin();
    for (cuda::std::size_t i = 0; ibegin != il.end(); ++ibegin, ++i)
    {
      vec_[i] = *ibegin;
    }
  }
};

template <class Range1, class Range2>
__host__ __device__ constexpr bool equal(Range1&& lhs, Range2&& rhs)
{
  auto* left  = lhs + 0;
  auto* right = rhs.begin();

  for (; right != rhs.end(); ++left, ++right)
  {
    assert(*left == *right);
  }
  assert(right == rhs.end());
  return true;
}

__host__ __device__ constexpr bool test()
{
  // no arg
  {
    cuda::std::expected<int, Data<3>> e(cuda::std::unexpect, {1, 2, 3});
    assert(!e.has_value());
    auto expectedList = {1, 2, 3};
    assert(equal(e.error().vec_, expectedList));
  }

  // one arg
  {
    cuda::std::expected<int, Data<3, MoveOnly>> e(cuda::std::unexpect, {4, 5, 6}, MoveOnly(5));
    assert(!e.has_value());
    auto expectedList = {4, 5, 6};
    assert((equal(e.error().vec_, expectedList)));
    assert(cuda::std::get<0>(e.error().tuple_) == 5);
  }

  // multi args
  {
    int i = 5;
    int j = 6;
    MoveOnly m(7);
    cuda::std::expected<int, Data<2, int&, int&&, MoveOnly>> e(
      cuda::std::unexpect, {1, 2}, i, cuda::std::move(j), cuda::std::move(m));
    assert(!e.has_value());
    auto expectedList = {1, 2};
    assert((equal(e.error().vec_, expectedList)));
    assert(&cuda::std::get<0>(e.error().tuple_) == &i);
    assert(&cuda::std::get<1>(e.error().tuple_) == &j);
    assert(cuda::std::get<2>(e.error().tuple_) == 7);
    assert(m.get() == 0);
  }

  return true;
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{
  struct Except
  {};

  struct Throwing
  {
    Throwing(cuda::std::initializer_list<int>, int)
    {
      throw Except{};
    };
  };

  try
  {
    cuda::std::expected<int, Throwing> u(cuda::std::unexpect, {1, 2}, 5);
    assert(false);
  }
  catch (Except)
  {}
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // defined(_CCCL_BUILTIN_ADDRESSOF)
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
