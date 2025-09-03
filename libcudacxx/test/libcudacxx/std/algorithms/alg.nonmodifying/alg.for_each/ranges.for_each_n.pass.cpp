//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<input_iterator I, class Proj = identity,
//          indirectly_unary_invocable<projected<I, Proj>> Fun>
//   constexpr ranges::for_each_n_result<I, Fun>
//     ranges::for_each_n(I first, iter_difference_t<I> n, Fun f, Proj proj = {});

#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct Callable
{
  __host__ __device__ void operator()(int);
};

template <class Iter, class = void>
inline constexpr bool HasForEachN = false;
template <class Iter>
inline constexpr bool
  HasForEachN<Iter,
              cuda::std::void_t<decltype(cuda::std::ranges::for_each_n(cuda::std::declval<Iter>(), 0, Callable{}))>> =
    true;

static_assert(HasForEachN<int*>);
static_assert(!HasForEachN<InputIteratorNotDerivedFrom>);
static_assert(!HasForEachN<InputIteratorNotIndirectlyReadable>);
static_assert(!HasForEachN<InputIteratorNotInputOrOutputIterator>);

template <class Func, class = void>
inline constexpr bool HasForEachItFunc = false;
template <class Func>
inline constexpr bool HasForEachItFunc<
  Func,
  cuda::std::void_t<decltype(cuda::std::ranges::for_each_n(static_cast<int*>(nullptr), 0, cuda::std::declval<Func>()))>> =
  true;

static_assert(HasForEachItFunc<Callable>);
static_assert(!HasForEachItFunc<IndirectUnaryPredicateNotPredicate>);
static_assert(!HasForEachItFunc<IndirectUnaryPredicateNotCopyConstructible>);

struct with_mutable_arg
{
  int i = 0;
  __host__ __device__ constexpr void operator()(int& a) noexcept
  {
    a += i++;
  }
};

struct Always_false
{
  __host__ __device__ constexpr Always_false(const bool val) noexcept
  {
    assert(val);
  }
};

struct should_not_be_called
{
  template <class T>
  __host__ __device__ constexpr void operator()(T&) const noexcept
  {
    Always_false{false};
  }
};

struct zeroes_out
{
  __host__ __device__ constexpr void operator()(int& a) noexcept
  {
    a = 0;
  }
};

template <class Iter>
__host__ __device__ constexpr void test_iterator()
{
  { // simple test
    int a[]            = {1, 6, 3, 4};
    decltype(auto) ret = cuda::std::ranges::for_each_n(Iter(a), 4, with_mutable_arg{});
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::for_each_result<Iter, with_mutable_arg>>);

    assert(a[0] == 1);
    assert(a[1] == 7);
    assert(a[2] == 5);
    assert(a[3] == 7);
    assert(base(ret.in) == a + 4);
    int i = 0;
    ret.fun(i);
    assert(i == 4);
  }

  { // check that an empty range works
    cuda::std::array<int, 0> a = {};
    cuda::std::ranges::for_each_n(Iter(a.data()), 0, should_not_be_called{});
  }
}

__host__ __device__ constexpr bool test()
{
  test_iterator<cpp17_input_iterator<int*>>();
  test_iterator<cpp20_input_iterator<int*>>();
  test_iterator<forward_iterator<int*>>();
  test_iterator<bidirectional_iterator<int*>>();
  test_iterator<random_access_iterator<int*>>();
  test_iterator<contiguous_iterator<int*>>();
  test_iterator<int*>();

  { // check that cuda::std::invoke is used
    struct S
    {
      int check;
      int other;
    };

    S a[] = {{1, 2}, {3, 4}, {5, 6}};
    cuda::std::ranges::for_each_n(a, 3, zeroes_out{}, &S::check);
    assert(a[0].check == 0);
    assert(a[0].other == 2);
    assert(a[1].check == 0);
    assert(a[1].other == 4);
    assert(a[2].check == 0);
    assert(a[2].other == 6);
  }

  return true;
}

int main(int, char**)
{
  test();
#if !TEST_COMPILER(GCC, <, 10) // accessing value of ‘a’ through a ‘int’ glvalue in a constant expression
  static_assert(test());
#endif // !TEST_COMPILER(GCC, <, 10)

  return 0;
}
