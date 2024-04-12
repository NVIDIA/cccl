//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class I>
// unspecified iter_move;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/utility>

#include "../unqualified_lookup_wrapper.h"
#include "test_macros.h"

using IterMoveT = decltype(cuda::std::ranges::iter_move);

// Wrapper around an iterator for testing `iter_move` when an unqualified call to `iter_move` isn't
// possible.
template <typename I>
class iterator_wrapper
{
public:
  iterator_wrapper() = default;

  __host__ __device__ constexpr explicit iterator_wrapper(I i) noexcept
      : base_(cuda::std::move(i))
  {}

  // `noexcept(false)` is used to check that this operator is called.
  __host__ __device__ constexpr decltype(auto) operator*() const& noexcept(false)
  {
    return *base_;
  }

  // `noexcept` is used to check that this operator is called.
  __host__ __device__ constexpr auto&& operator*() && noexcept
  {
    return cuda::std::move(*base_);
  }

  __host__ __device__ constexpr iterator_wrapper& operator++() noexcept
  {
    ++base_;
    return *this;
  }

  __host__ __device__ constexpr void operator++(int) noexcept
  {
    ++base_;
  }

  __host__ __device__ constexpr bool operator==(iterator_wrapper const& other) const noexcept
  {
    return base_ == other.base_;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ constexpr bool operator!=(iterator_wrapper const& other) const noexcept
  {
    return base_ != other.base_;
  }
#endif

private:
  I base_ = I{};
};

template <typename It, typename Out>
__host__ __device__ constexpr void unqualified_lookup_move(It first_, It last_, Out result_first_, Out result_last_)
{
  auto first        = ::check_unqualified_lookup::unqualified_lookup_wrapper<It>{cuda::std::move(first_)};
  auto last         = ::check_unqualified_lookup::unqualified_lookup_wrapper<It>{cuda::std::move(last_)};
  auto result_first = ::check_unqualified_lookup::unqualified_lookup_wrapper<It>{cuda::std::move(result_first_)};
  auto result_last  = ::check_unqualified_lookup::unqualified_lookup_wrapper<It>{cuda::std::move(result_last_)};

#ifndef TEST_COMPILER_ICC
  static_assert(!noexcept(cuda::std::ranges::iter_move(first)), "unqualified-lookup case not being chosen");
#endif // TEST_COMPILER_ICC

  for (; first != last && result_first != result_last; (void) ++first, ++result_first)
  {
    *result_first = cuda::std::ranges::iter_move(first);
  }
}

template <typename It, typename Out>
__host__ __device__ constexpr void lvalue_move(It first_, It last_, Out result_first_, Out result_last_)
{
  auto first        = iterator_wrapper<It>{cuda::std::move(first_)};
  auto last         = ::iterator_wrapper<It>{cuda::std::move(last_)};
  auto result_first = iterator_wrapper<It>{cuda::std::move(result_first_)};
  auto result_last  = iterator_wrapper<It>{cuda::std::move(result_last_)};

#ifndef TEST_COMPILER_ICC
  static_assert(!noexcept(cuda::std::ranges::iter_move(first)),
                "`operator*() const&` is not noexcept, and there's no hidden "
                "friend iter_move.");
#endif // TEST_COMPILER_ICC

  for (; first != last && result_first != result_last; (void) ++first, ++result_first)
  {
    *result_first = cuda::std::ranges::iter_move(first);
  }
}

template <typename It, typename Out>
__host__ __device__ constexpr void rvalue_move(It first_, It last_, Out result_first_, Out result_last_)
{
  auto first        = iterator_wrapper<It>{cuda::std::move(first_)};
  auto last         = iterator_wrapper<It>{cuda::std::move(last_)};
  auto result_first = iterator_wrapper<It>{cuda::std::move(result_first_)};
  auto result_last  = iterator_wrapper<It>{cuda::std::move(result_last_)};

  static_assert(noexcept(cuda::std::ranges::iter_move(cuda::std::move(first))),
                "`operator*() &&` is noexcept, and there's no hidden friend iter_move.");

  for (; first != last && result_first != result_last; (void) ++first, ++result_first)
  {
    auto i        = first;
    *result_first = cuda::std::ranges::iter_move(cuda::std::move(i));
  }
}

template <bool NoExcept>
struct WithADL
{
  WithADL() = default;
  __host__ __device__ constexpr int operator*() const
  {
    return 0;
  }
  __host__ __device__ constexpr WithADL& operator++();
  __host__ __device__ constexpr void operator++(int);
  __host__ __device__ constexpr bool operator==(WithADL const&) const;
  __host__ __device__ friend constexpr int iter_move(WithADL&&) noexcept(NoExcept)
  {
    return 0;
  }
};

template <bool NoExcept>
struct WithoutADL
{
  WithoutADL() = default;
  __host__ __device__ constexpr int operator*() const noexcept(NoExcept)
  {
    return 0;
  }
  __host__ __device__ constexpr WithoutADL& operator++();
  __host__ __device__ constexpr void operator++(int);
  __host__ __device__ constexpr bool operator==(WithoutADL const&) const;
};

template <class It, class Pred>
__host__ __device__ constexpr bool all_of(It first, It last, Pred pred)
{
  for (; first != last; ++first)
  {
    if (!pred(*first))
    {
      return false;
    }
  }
  return true;
}

__host__ __device__ constexpr bool test()
{
  constexpr int full_size = 100;
  constexpr int half_size = full_size / 2;
  constexpr int reset     = 0;
  move_tracker v1[full_size];

  struct move_counter_is
  {
    __host__ __device__ constexpr move_counter_is(const int counter)
        : _counter(counter)
    {}

    __host__ __device__ constexpr bool operator()(move_tracker const& x)
    {
      return x.moves() == _counter;
    }

    const int _counter;
  };

  move_tracker v2[half_size];
  unqualified_lookup_move(cuda::std::begin(v1), cuda::std::end(v1), cuda::std::begin(v2), cuda::std::end(v2));
  assert(all_of(cuda::std::cbegin(v1), cuda::std::cend(v1), move_counter_is(reset)));
  assert(all_of(cuda::std::cbegin(v2), cuda::std::cend(v2), move_counter_is(1)));

  move_tracker v3[half_size];
  unqualified_lookup_move(
    cuda::std::begin(v1) + half_size, cuda::std::end(v1), cuda::std::begin(v3), cuda::std::end(v3));
  assert(all_of(cuda::std::cbegin(v1), cuda::std::cend(v1), move_counter_is(reset)));
  assert(all_of(cuda::std::cbegin(v3), cuda::std::cend(v3), move_counter_is(1)));

  move_tracker v4[half_size];
  unqualified_lookup_move(cuda::std::begin(v3), cuda::std::end(v3), cuda::std::begin(v4), cuda::std::end(v4));
  assert(all_of(cuda::std::cbegin(v3), cuda::std::cend(v3), move_counter_is(reset)));
  assert(all_of(cuda::std::cbegin(v4), cuda::std::cend(v4), move_counter_is(2)));

  lvalue_move(cuda::std::begin(v2), cuda::std::end(v2), cuda::std::begin(v1) + half_size, cuda::std::end(v1));
  assert(all_of(cuda::std::cbegin(v2), cuda::std::cend(v2), move_counter_is(reset)));
  assert(all_of(cuda::std::cbegin(v1) + half_size, cuda::std::cend(v1), move_counter_is(2)));

  lvalue_move(cuda::std::begin(v4), cuda::std::end(v4), cuda::std::begin(v1), cuda::std::end(v1));
  assert(all_of(cuda::std::cbegin(v4), cuda::std::cend(v4), move_counter_is(reset)));
  assert(all_of(cuda::std::cbegin(v1), cuda::std::cbegin(v1) + half_size, move_counter_is(3)));

  rvalue_move(cuda::std::begin(v1), cuda::std::end(v1), cuda::std::begin(v2), cuda::std::end(v2));
  assert(all_of(cuda::std::cbegin(v1), cuda::std::cbegin(v1) + half_size, move_counter_is(reset)));
  assert(all_of(cuda::std::cbegin(v2), cuda::std::cend(v2), move_counter_is(4)));

  rvalue_move(cuda::std::begin(v1) + half_size, cuda::std::end(v1), cuda::std::begin(v3), cuda::std::end(v3));
  assert(all_of(cuda::std::cbegin(v1), cuda::std::cend(v1), move_counter_is(reset)));
  assert(all_of(cuda::std::cbegin(v3), cuda::std::cend(v3), move_counter_is(3)));

  auto unscoped = check_unqualified_lookup::unscoped_enum::a;
  assert(cuda::std::ranges::iter_move(unscoped) == check_unqualified_lookup::unscoped_enum::a);
#ifndef TEST_COMPILER_ICC
  assert(!noexcept(cuda::std::ranges::iter_move(unscoped)));
#endif // TEST_COMPILER_ICC

  auto scoped = check_unqualified_lookup::scoped_enum::a;
  assert(cuda::std::ranges::iter_move(scoped) == nullptr);
  assert(noexcept(cuda::std::ranges::iter_move(scoped)));

  auto some_union = check_unqualified_lookup::some_union{0};
  assert(cuda::std::ranges::iter_move(some_union) == 0);
#ifndef TEST_COMPILER_ICC
  assert(!noexcept(cuda::std::ranges::iter_move(some_union)));

  // Check noexcept-correctness
  static_assert(noexcept(cuda::std::ranges::iter_move(cuda::std::declval<WithADL<true>>())));
  static_assert(noexcept(cuda::std::ranges::iter_move(cuda::std::declval<WithoutADL<true>>())));
// old GCC seems to fall over the chaining of the noexcept clauses here
#  if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 9)
  static_assert(!noexcept(cuda::std::ranges::iter_move(cuda::std::declval<WithADL<false>>())));
  static_assert(!noexcept(cuda::std::ranges::iter_move(cuda::std::declval<WithoutADL<false>>())));
#  endif
#endif // TEST_COMPILER_ICC

  return true;
}

#ifndef _CCCL_CUDACC_BELOW_11_3 // nvcc segfaults here
static_assert(!cuda::std::is_invocable_v<IterMoveT, int*, int*>); // too many arguments
static_assert(!cuda::std::is_invocable_v<IterMoveT, int>);
#endif // _CCCL_CUDACC_BELOW_11_3

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};
static_assert(cuda::std::is_invocable_v<IterMoveT, Holder<Incomplete>**>);
static_assert(cuda::std::is_invocable_v<IterMoveT, Holder<Incomplete>**&>);
#endif

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
