//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Older Clangs do not support the C++20 feature to constrain destructors

// template<class G>
//   constexpr expected& operator=(const unexpected<G>& e);
//
// Let GF be const G&
// Constraints:
// - is_constructible_v<E, GF> is true; and
// - is_assignable_v<E&, GF> is true; and
// - is_nothrow_constructible_v<E, GF> || is_nothrow_move_constructible_v<T> ||
//   is_nothrow_move_constructible_v<E> is true.
//
// Effects:
// - If has_value() is true, equivalent to:
//   reinit-expected(unex, val, cuda::std::forward<GF>(e.error()));
//   has_val = false;
// - Otherwise, equivalent to: unex = cuda::std::forward<GF>(e.error());
// Returns: *this.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../../types.h"
#include "test_macros.h"

struct NotCopyConstructible
{
  NotCopyConstructible(const NotCopyConstructible&)            = delete;
  NotCopyConstructible& operator=(const NotCopyConstructible&) = default;
};

struct NotCopyAssignable
{
  NotCopyAssignable(const NotCopyAssignable&)            = default;
  NotCopyAssignable& operator=(const NotCopyAssignable&) = delete;
};

struct MoveMayThrow
{
  MoveMayThrow(MoveMayThrow const&)            = default;
  MoveMayThrow& operator=(const MoveMayThrow&) = default;
  __host__ __device__ MoveMayThrow(MoveMayThrow&&) noexcept(false) {}
  __host__ __device__ MoveMayThrow& operator=(MoveMayThrow&&) noexcept(false)
  {
    return *this;
  }
};

// Test constraints
static_assert(cuda::std::is_assignable_v<cuda::std::expected<int, int>&, const cuda::std::unexpected<int>&>, "");

#ifndef TEST_COMPILER_MSVC_2017
// !is_constructible_v<E, GF>
static_assert(!cuda::std::is_assignable_v<cuda::std::expected<int, NotCopyConstructible>&,
                                          const cuda::std::unexpected<NotCopyConstructible>&>,
              "");

// !is_assignable_v<E&, GF>
static_assert(!cuda::std::is_assignable_v<cuda::std::expected<int, NotCopyAssignable>&,
                                          const cuda::std::unexpected<NotCopyAssignable>&>,
              "");
#endif // TEST_COMPILER_MSVC_2017

template <bool moveNoexcept, bool convertNoexcept>
struct MaybeNoexcept
{
  __host__ __device__ explicit MaybeNoexcept(int) noexcept(convertNoexcept);
  __host__ __device__ MaybeNoexcept(MaybeNoexcept&&) noexcept(moveNoexcept);
  MaybeNoexcept& operator=(MaybeNoexcept&&) = default;
  __host__ __device__ MaybeNoexcept& operator=(int);
};

// !is_nothrow_constructible_v<E, GF> && !is_nothrow_move_constructible_v<T> &&
// is_nothrow_move_constructible_v<E>
static_assert(cuda::std::is_assignable_v<cuda::std::expected<MaybeNoexcept<false, false>, MaybeNoexcept<true, false>>&,
                                         const cuda::std::unexpected<int>&>,
              "");

// is_nothrow_constructible_v<E, GF> && !is_nothrow_move_constructible_v<T> &&
// !is_nothrow_move_constructible_v<E>
static_assert(cuda::std::is_assignable_v<cuda::std::expected<MaybeNoexcept<false, false>, MaybeNoexcept<false, true>>&,
                                         const cuda::std::unexpected<int>&>,
              "");

// !is_nothrow_constructible_v<E, GF> && is_nothrow_move_constructible_v<T> &&
// !is_nothrow_move_constructible_v<E>
static_assert(cuda::std::is_assignable_v<cuda::std::expected<MaybeNoexcept<true, true>, MaybeNoexcept<false, false>>&,
                                         const cuda::std::unexpected<int>&>,
              "");

#ifndef TEST_COMPILER_MSVC_2017
#  ifndef TEST_COMPILER_ICC
// !is_nothrow_constructible_v<E, GF> && !is_nothrow_move_constructible_v<T> &&
// !is_nothrow_move_constructible_v<E>
static_assert(!cuda::std::is_assignable_v<cuda::std::expected<MaybeNoexcept<false, false>, MaybeNoexcept<false, false>>&,
                                          const cuda::std::unexpected<int>&>,
              "");
#  endif // TEST_COMPILER_ICC
#endif // TEST_COMPILER_MSVC_2017

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // - If has_value() is true, equivalent to:
  //   reinit-expected(unex, val, cuda::std::forward<GF>(e.error()));
  // is_nothrow_constructible_v<E, GF>
  //
  //  In this case, it should call the branch
  //    destroy_at(addressof(oldval));
  //    construct_at(addressof(newval), cuda::std::forward<Args>(args)...);
  {
    BothNoexcept::state oldState{};
    cuda::std::expected<BothNoexcept, BothNoexcept> e(cuda::std::in_place, oldState, 5);
    const cuda::std::unexpected<int> un(10);
    decltype(auto) x = (e = un);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<BothNoexcept, BothNoexcept>&>, "");
    assert(&x == &e);

    assert(!oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(e.error().copiedFromInt);
  }

  // - If has_value() is true, equivalent to:
  //   reinit-expected(unex, val, cuda::std::forward<GF>(e.error()));
  // !is_nothrow_constructible_v<E, GF> && is_nothrow_move_constructible_v<E>
  //
  //  In this case, it should call the branch
  //  T tmp(cuda::std::forward<Args>(args)...);
  //  destroy_at(addressof(oldval));
  //  construct_at(addressof(newval), cuda::std::move(tmp));
  {
    BothNoexcept::state oldState{};
    cuda::std::expected<BothNoexcept, MoveNoexceptConvThrow> e(cuda::std::in_place, oldState, 5);
    const cuda::std::unexpected<int> un(10);
    decltype(auto) x = (e = un);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<BothNoexcept, MoveNoexceptConvThrow>&>, "");
    assert(&x == &e);

    assert(!oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(!e.error().copiedFromInt);
    assert(e.error().movedFromTmp);
  }

  // - If has_value() is true, equivalent to:
  //   reinit-expected(unex, val, cuda::std::forward<GF>(e.error()));
  // !is_nothrow_constructible_v<E, GF> && !is_nothrow_move_constructible_v<E>
  // is_nothrow_move_constructible_v<T>
  //
  //  In this case, it should call the branch
  //  U tmp(cuda::std::move(oldval));
  //  destroy_at(addressof(oldval));
  //  try {
  //    construct_at(addressof(newval), cuda::std::forward<Args>(args)...);
  //  } catch (...) {
  //    construct_at(addressof(oldval), cuda::std::move(tmp));
  //    throw;
  //  }
  {
    BothNoexcept::state oldState{};
    cuda::std::expected<BothNoexcept, BothMayThrow> e(cuda::std::in_place, oldState, 5);
    const cuda::std::unexpected<int> un(10);
    decltype(auto) x = (e = un);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<BothNoexcept, BothMayThrow>&>, "");
    assert(&x == &e);

    assert(oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(e.error().copiedFromInt);
  }

  // Otherwise, equivalent to: unex = cuda::std::forward<GF>(e.error());
  {
    Traced::state oldState{};
    Traced::state newState{};
    cuda::std::expected<int, Traced> e1(cuda::std::unexpect, oldState, 5);
    const cuda::std::unexpected<Traced> e(cuda::std::in_place, newState, 10);
    decltype(auto) x = (e1 = e);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<int, Traced>&>, "");
    assert(&x == &e1);

    assert(!e1.has_value());
    assert(e1.error().data_ == 10);
    assert(oldState.copyAssignCalled);
  }
  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{
  cuda::std::expected<int, ThrowOnConvert> e1(cuda::std::in_place, 5);
  const cuda::std::unexpected<int> un(10);
  try
  {
    e1 = un;
    assert(false);
  }
  catch (Except)
  {
    assert(e1.has_value());
    assert(*e1 == 5);
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
