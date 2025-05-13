//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr auto begin() requires (!simple-view<V>)
// { return ranges::begin(base_); }
//
// constexpr auto begin() const
//   requires range<const V> &&
//            indirect_unary_predicate<const Pred, iterator_t<const V>>
// { return ranges::begin(base_); }

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

// Test Constraints
template <class T>
_CCCL_CONCEPT HasConstBegin = _CCCL_REQUIRES_EXPR((T), const T& ct)((unused(ct.begin())));

template <class T>
_CCCL_CONCEPT HasBegin = _CCCL_REQUIRES_EXPR((T), T& t)((unused(t.begin())));

template <class T>
_CCCL_CONCEPT HasConstAndNonConstBegin = _CCCL_REQUIRES_EXPR((T), T& t, const T& ct)(
  requires(HasConstBegin<T>), requires(!cuda::std::same_as<decltype(t.begin()), decltype(ct.begin())>));

template <class T>
_CCCL_CONCEPT HasOnlyNonConstBegin = HasBegin<T> && !HasConstBegin<T>;

template <class T>
_CCCL_CONCEPT HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;

struct Pred
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i > 5;
  }
};

static_assert(HasOnlyConstBegin<cuda::std::ranges::take_while_view<SimpleView, Pred>>);

static_assert(HasOnlyNonConstBegin<cuda::std::ranges::take_while_view<ConstNotRange, Pred>>);

static_assert(HasConstAndNonConstBegin<cuda::std::ranges::take_while_view<NonSimple, Pred>>);

struct NotPredForConst
{
  __host__ __device__ constexpr bool operator()(int& i) const
  {
    return i > 5;
  }
};
static_assert(HasOnlyNonConstBegin<cuda::std::ranges::take_while_view<NonSimple, NotPredForConst>>);

__host__ __device__ constexpr bool test()
{
  // simple-view
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    SimpleView v{buffer};
    cuda::std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.begin();
    static_assert(cuda::std::same_as<decltype(it1), int*>);
    assert(it1 == buffer);
    decltype(auto) it2 = cuda::std::as_const(twv).begin();
    static_assert(cuda::std::same_as<decltype(it2), int*>);
    assert(it2 == buffer);
  }

  // const not range
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    ConstNotRange v{buffer};
    cuda::std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.begin();
    static_assert(cuda::std::same_as<decltype(it1), int*>);
    assert(it1 == buffer);
  }

  // NonSimple
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    NonSimple v{buffer};
    cuda::std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.begin();
    static_assert(cuda::std::same_as<decltype(it1), int*>);
    assert(it1 == buffer);
    decltype(auto) it2 = cuda::std::as_const(twv).begin();
    static_assert(cuda::std::same_as<decltype(it2), const int*>);
    assert(it2 == buffer);
  }

  // NotPredForConst
  // LWG 3450: The const overloads of `take_while_view::begin/end` are underconstrained
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    NonSimple v{buffer};
    cuda::std::ranges::take_while_view twv(v, NotPredForConst{});
    decltype(auto) it1 = twv.begin();
    static_assert(cuda::std::same_as<decltype(it1), int*>);
    assert(it1 == buffer);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
