//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// test_allocator and the uses-allocator constructors in copy_move_types.h are host-only.
// UNSUPPORTED: nvrtc

// <cuda/std/tuple>

// template <class... Types>
// template <class Alloc, class U1, class U2>
// constexpr explicit(see below)
//   tuple<Types...>::tuple(allocator_arg_t, const Alloc& a, const pair<U1, U2>&& u);

// Constraints:
// - sizeof...(Types) is 2 and
// - is_constructible_v<T0, decltype(get<0>(FWD(u)))> is true and
// - is_constructible_v<T1, decltype(get<1>(FWD(u)))> is true.

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "copy_move_types.h"
#include "test_allocator.h"
#include "test_macros.h"

// test constraints
// sizeof...(Types) == 2
static_assert(cuda::std::is_constructible_v<cuda::std::tuple<ConstMove, int>,
                                            cuda::std::allocator_arg_t,
                                            test_allocator<int>,
                                            const cuda::std::pair<ConstMove, int>&&>);

static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<ConstMove>,
                                             cuda::std::allocator_arg_t,
                                             test_allocator<int>,
                                             const cuda::std::pair<ConstMove, int>&&>);

static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<ConstMove, int, int>,
                                             cuda::std::allocator_arg_t,
                                             test_allocator<int>,
                                             const cuda::std::pair<ConstMove, int>&&>);

// test constraints
// is_constructible_v<T0, decltype(get<0>(FWD(u)))> is true and
// is_constructible_v<T1, decltype(get<1>(FWD(u)))> is true.
static_assert(cuda::std::is_constructible_v<cuda::std::tuple<int, int>,
                                            cuda::std::allocator_arg_t,
                                            test_allocator<int>,
                                            const cuda::std::pair<int, int>&&>);

static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<NoConstructorFromInt, int>,
                                             cuda::std::allocator_arg_t,
                                             test_allocator<int>,
                                             const cuda::std::pair<int, int>&&>);

static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<int, NoConstructorFromInt>,
                                             cuda::std::allocator_arg_t,
                                             test_allocator<int>,
                                             const cuda::std::pair<int, int>&&>);

static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<NoConstructorFromInt, NoConstructorFromInt>,
                                             cuda::std::allocator_arg_t,
                                             test_allocator<int>,
                                             const cuda::std::pair<int, int>&&>);

// test: The expression inside explicit is equivalent to:
// !is_convertible_v<decltype(get<0>(FWD(u))), T0> ||
// !is_convertible_v<decltype(get<1>(FWD(u))), T1>
static_assert(ImplicitlyConstructible<cuda::std::tuple<ConvertibleFrom<ConstMove>, ConvertibleFrom<ConstMove>>,
                                      cuda::std::allocator_arg_t,
                                      test_allocator<int>,
                                      const cuda::std::pair<ConstMove, ConstMove>&&>);

static_assert(
  !ImplicitlyConstructible<cuda::std::tuple<ConvertibleFrom<ConstMove>, ExplicitConstructibleFrom<ConstMove>>,
                           cuda::std::allocator_arg_t,
                           test_allocator<int>,
                           const cuda::std::pair<ConstMove, ConstMove>&&>);

static_assert(
  !ImplicitlyConstructible<cuda::std::tuple<ExplicitConstructibleFrom<ConstMove>, ConvertibleFrom<ConstMove>>,
                           cuda::std::allocator_arg_t,
                           test_allocator<int>,
                           const cuda::std::pair<ConstMove, ConstMove>&&>);

bool test()
{
  // test implicit conversions.
  {
    const cuda::std::pair<ConstMove, int> p{1, 2};
    cuda::std::tuple<ConvertibleFrom<ConstMove>, ConvertibleFrom<int>> t = {
      cuda::std::allocator_arg, test_allocator<int>{}, cuda::std::move(p)};
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v == 2);
    assert(cuda::std::get<0>(t).alloc_constructed);
    assert(cuda::std::get<1>(t).alloc_constructed);
  }

  // test explicit conversions.
  {
    const cuda::std::pair<ConstMove, int> p{1, 2};
    cuda::std::tuple<ExplicitConstructibleFrom<ConstMove>, ExplicitConstructibleFrom<int>> t{
      cuda::std::allocator_arg, test_allocator<int>{}, cuda::std::move(p)};
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v == 2);
    assert(cuda::std::get<0>(t).alloc_constructed);
    assert(cuda::std::get<1>(t).alloc_constructed);
  }

  // const rvalue overload should be called
  {
    const cuda::std::pair<TracedCopyMove, TracedCopyMove> p;
    cuda::std::tuple<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> t = {
      cuda::std::allocator_arg, test_allocator<int>{}, cuda::std::move(p)};
    assert(constMoveCtrCalled(cuda::std::get<0>(t).v));
    assert(constMoveCtrCalled(cuda::std::get<1>(t)));
    assert(cuda::std::get<0>(t).alloc_constructed);
    assert(cuda::std::get<1>(t).alloc_constructed);
  }

  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, ({ test(); }))
  return 0;
}
