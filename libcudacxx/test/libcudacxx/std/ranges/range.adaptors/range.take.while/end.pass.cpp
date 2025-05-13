//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr auto end() requires (!simple-view<V>)
// { return sentinel<false>(ranges::end(base_), addressof(*pred_)); }
// constexpr auto end() const
//   requires range<const V> &&
//            indirect_unary_predicate<const Pred, iterator_t<const V>>
// { return sentinel<true>(ranges::end(base_), addressof(*pred_)); }

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

// Test Constraints
template <class T>
_CCCL_CONCEPT HasConstEnd = _CCCL_REQUIRES_EXPR((T), const T& ct)((unused(ct.end())));

template <class T>
_CCCL_CONCEPT HasEnd = _CCCL_REQUIRES_EXPR((T), T& t)((unused(t.end())));

template <class T>
_CCCL_CONCEPT HasConstAndNonConstEnd = _CCCL_REQUIRES_EXPR((T), T& t, const T& ct)(
  requires(HasConstEnd<T>), requires(!cuda::std::same_as<decltype(t.end()), decltype(ct.end())>));

template <class T>
_CCCL_CONCEPT HasOnlyNonConstEnd = HasEnd<T> && !HasConstEnd<T>;

template <class T>
_CCCL_CONCEPT HasOnlyConstEnd = HasConstEnd<T> && !HasConstAndNonConstEnd<T>;

struct Pred
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i < 5;
  }
};

static_assert(HasOnlyConstEnd<cuda::std::ranges::take_while_view<SimpleView, Pred>>);

static_assert(HasOnlyNonConstEnd<cuda::std::ranges::take_while_view<ConstNotRange, Pred>>);

static_assert(HasConstAndNonConstEnd<cuda::std::ranges::take_while_view<NonSimple, Pred>>);

struct NotPredForConst
{
  __host__ __device__ constexpr bool operator()(int& i) const
  {
    return i > 5;
  }
};
static_assert(HasOnlyNonConstEnd<cuda::std::ranges::take_while_view<NonSimple, NotPredForConst>>);

__host__ __device__ constexpr bool test()
{
  // simple-view
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    SimpleView v{buffer};
    cuda::std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.end();
    assert(it1 == buffer + 4);
    decltype(auto) it2 = cuda::std::as_const(twv).end();
    assert(it2 == buffer + 4);

    static_assert(cuda::std::same_as<decltype(it1), decltype(it2)>);
  }

  // const not range
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    ConstNotRange v{buffer};
    cuda::std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.end();
    assert(it1 == buffer + 4);
  }

  // NonSimple
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    NonSimple v{buffer};
    cuda::std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.end();
    assert(it1 == buffer + 4);
    decltype(auto) it2 = cuda::std::as_const(twv).end();
    assert(it2 == buffer + 4);

    static_assert(!cuda::std::same_as<decltype(it1), decltype(it2)>);
  }

  // NotPredForConst
  // LWG 3450: The const overloads of `take_while_view::begin/end` are underconstrained
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    NonSimple v{buffer};
    cuda::std::ranges::take_while_view twv(v, NotPredForConst{});
    decltype(auto) it1 = twv.end();
    assert(it1 == buffer);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_ADDRESSOF

  return 0;
}
