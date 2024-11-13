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

//  template<class U = T>
//   constexpr expected& operator=(U&& v);
//
// Constraints:
// - is_same_v<expected, remove_cvref_t<U>> is false; and
// - remove_cvref_t<U> is not a specialization of unexpected; and
// - is_constructible_v<T, U> is true; and
// - is_assignable_v<T&, U> is true; and
// - is_nothrow_constructible_v<T, U> || is_nothrow_move_constructible_v<T> ||
//   is_nothrow_move_constructible_v<E> is true.
//
// Effects:
// - If has_value() is true, equivalent to: val = cuda::std::forward<U>(v);
// - Otherwise, equivalent to:
//   reinit-expected(val, unex, cuda::std::forward<U>(v));
//   has_val = true;
// - Returns: *this.

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

// Test constraints
static_assert(cuda::std::is_assignable_v<cuda::std::expected<int, int>&, int>, "");

// is_same_v<expected, remove_cvref_t<U>>
// it is true because it covered by the copy assignment
static_assert(cuda::std::is_assignable_v<cuda::std::expected<int, int>&, cuda::std::expected<int, int>>, "");

// remove_cvref_t<U> is a specialization of unexpected
// it is true because it covered the unepxected overload
static_assert(cuda::std::is_assignable_v<cuda::std::expected<int, int>&, cuda::std::unexpected<int>>, "");

// !is_constructible_v<T, U>
struct NoCtorFromInt
{
  __host__ __device__ NoCtorFromInt(int) = delete;
  __host__ __device__ NoCtorFromInt& operator=(int);
};
static_assert(!cuda::std::is_assignable_v<cuda::std::expected<NoCtorFromInt, int>&, int>, "");

// !is_assignable_v<T&, U>
struct NoAssignFromInt
{
  __host__ __device__ explicit NoAssignFromInt(int);
  __host__ __device__ NoAssignFromInt& operator=(int) = delete;
};
static_assert(!cuda::std::is_assignable_v<cuda::std::expected<NoAssignFromInt, int>&, int>, "");

template <bool moveNoexcept, bool convertNoexcept>
struct MaybeNoexcept
{
  __host__ __device__ explicit MaybeNoexcept(int) noexcept(convertNoexcept);
  __host__ __device__ MaybeNoexcept(MaybeNoexcept&&) noexcept(moveNoexcept);
  MaybeNoexcept& operator=(MaybeNoexcept&&) = default;
  __host__ __device__ MaybeNoexcept& operator=(int);
};

// !is_nothrow_constructible_v<T, U> && !is_nothrow_move_constructible_v<T> &&
// is_nothrow_move_constructible_v<E>
static_assert(cuda::std::is_assignable_v<cuda::std::expected<MaybeNoexcept<false, false>, int>&, int>, "");

// is_nothrow_constructible_v<T, U> && !is_nothrow_move_constructible_v<T> &&
// !is_nothrow_move_constructible_v<E>
static_assert(
  cuda::std::is_assignable_v<cuda::std::expected<MaybeNoexcept<false, true>, MaybeNoexcept<false, false>>&, int>, "");

// !is_nothrow_constructible_v<T, U> && is_nothrow_move_constructible_v<T> &&
// !is_nothrow_move_constructible_v<E>
static_assert(
  cuda::std::is_assignable_v<cuda::std::expected<MaybeNoexcept<true, false>, MaybeNoexcept<false, false>>&, int>, "");

#ifndef TEST_COMPILER_ICC
// !is_nothrow_constructible_v<T, U> && !is_nothrow_move_constructible_v<T> &&
// !is_nothrow_move_constructible_v<E>
static_assert(
  !cuda::std::is_assignable_v<cuda::std::expected<MaybeNoexcept<false, false>, MaybeNoexcept<false, false>>&, int>, "");
#endif // TEST_COMPILER_ICC

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // If has_value() is true, equivalent to: val = cuda::std::forward<U>(v);
  // Copy
  {
    Traced::state oldState{};
    Traced::state newState{};
    cuda::std::expected<Traced, int> e1(cuda::std::in_place, oldState, 5);
    Traced u(newState, 10);
    decltype(auto) x = (e1 = u);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<Traced, int>&>, "");
    assert(&x == &e1);

    assert(e1.has_value());
    assert(e1.value().data_ == 10);
    assert(oldState.copyAssignCalled);
  }

  // If has_value() is true, equivalent to: val = cuda::std::forward<U>(v);
  // Move
  {
    Traced::state oldState{};
    Traced::state newState{};
    cuda::std::expected<Traced, int> e1(cuda::std::in_place, oldState, 5);
    Traced u(newState, 10);
    decltype(auto) x = (e1 = cuda::std::move(u));
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<Traced, int>&>, "");
    assert(&x == &e1);

    assert(e1.has_value());
    assert(e1.value().data_ == 10);
    assert(oldState.moveAssignCalled);
  }

  // Otherwise, equivalent to:
  //   reinit-expected(val, unex, cuda::std::forward<U>(v));
  // is_nothrow_constructible_v<T, U> && !is_nothrow_move_constructible_v<T> &&
  // !is_nothrow_move_constructible_v<E>
  // copy
  //
  //  In this case, it should call the branch
  //    destroy_at(addressof(oldval));
  //    construct_at(addressof(newval), cuda::std::forward<Args>(args)...);
  {
    BothMayThrow::state oldState{};
    cuda::std::expected<MoveThrowConvNoexcept, BothMayThrow> e1(cuda::std::unexpect, oldState, 5);
    const int i      = 10;
    decltype(auto) x = (e1 = i);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<MoveThrowConvNoexcept, BothMayThrow>&>, "");
    assert(&x == &e1);

    assert(e1.has_value());
    assert(e1.value().data_ == 10);

    assert(!oldState.copyCtorCalled);
    assert(!oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(e1.value().copiedFromInt);
  }

  // Otherwise, equivalent to:
  //   reinit-expected(val, unex, cuda::std::forward<U>(v));
  // is_nothrow_constructible_v<T, U> && !is_nothrow_move_constructible_v<T> &&
  // !is_nothrow_move_constructible_v<E>
  // move
  //
  //  In this case, it should call the branch
  //    destroy_at(addressof(oldval));
  //    construct_at(addressof(newval), cuda::std::forward<Args>(args)...);
  {
    BothMayThrow::state oldState{};
    cuda::std::expected<MoveThrowConvNoexcept, BothMayThrow> e1(cuda::std::unexpect, oldState, 5);
    decltype(auto) x = (e1 = 10);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<MoveThrowConvNoexcept, BothMayThrow>&>, "");
    assert(&x == &e1);

    assert(e1.has_value());
    assert(e1.value().data_ == 10);

    assert(!oldState.copyCtorCalled);
    assert(!oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(e1.value().movedFromInt);
  }

  // Otherwise, equivalent to:
  //   reinit-expected(val, unex, cuda::std::forward<U>(v));
  // !is_nothrow_constructible_v<T, U> && is_nothrow_move_constructible_v<T> &&
  // !is_nothrow_move_constructible_v<E>
  // copy
  //
  //  In this case, it should call the branch
  //  T tmp(cuda::std::forward<Args>(args)...);
  //  destroy_at(addressof(oldval));
  //  construct_at(addressof(newval), cuda::std::move(tmp));
  {
    BothMayThrow::state oldState{};
    cuda::std::expected<MoveNoexceptConvThrow, BothMayThrow> e1(cuda::std::unexpect, oldState, 5);
    const int i      = 10;
    decltype(auto) x = (e1 = i);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<MoveNoexceptConvThrow, BothMayThrow>&>, "");
    assert(&x == &e1);

    assert(e1.has_value());
    assert(e1.value().data_ == 10);

    assert(!oldState.copyCtorCalled);
    assert(!oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(!e1.value().copiedFromInt);
    assert(e1.value().movedFromTmp);
  }

  // Otherwise, equivalent to:
  //   reinit-expected(val, unex, cuda::std::forward<U>(v));
  // !is_nothrow_constructible_v<T, U> && is_nothrow_move_constructible_v<T> &&
  // !is_nothrow_move_constructible_v<E>
  // move
  //
  //  In this case, it should call the branch
  //  T tmp(cuda::std::forward<Args>(args)...);
  //  destroy_at(addressof(oldval));
  //  construct_at(addressof(newval), cuda::std::move(tmp));
  {
    BothMayThrow::state oldState{};
    cuda::std::expected<MoveNoexceptConvThrow, BothMayThrow> e1(cuda::std::unexpect, oldState, 5);
    decltype(auto) x = (e1 = 10);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<MoveNoexceptConvThrow, BothMayThrow>&>, "");
    assert(&x == &e1);

    assert(e1.has_value());
    assert(e1.value().data_ == 10);

    assert(!oldState.copyCtorCalled);
    assert(!oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(!e1.value().copiedFromInt);
    assert(e1.value().movedFromTmp);
  }

  // Otherwise, equivalent to:
  //   reinit-expected(val, unex, cuda::std::forward<U>(v));
  // !is_nothrow_constructible_v<T, U> && !is_nothrow_move_constructible_v<T> &&
  // is_nothrow_move_constructible_v<E>
  // copy
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
    TracedNoexcept::state oldState{};
    cuda::std::expected<BothMayThrow, TracedNoexcept> e1(cuda::std::unexpect, oldState, 5);
    const int i      = 10;
    decltype(auto) x = (e1 = i);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<BothMayThrow, TracedNoexcept>&>, "");
    assert(&x == &e1);

    assert(e1.has_value());
    assert(e1.value().data_ == 10);

    assert(!oldState.copyCtorCalled);
    assert(oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(e1.value().copiedFromInt);
  }

  // Otherwise, equivalent to:
  //   reinit-expected(val, unex, cuda::std::forward<U>(v));
  // !is_nothrow_constructible_v<T, U> && !is_nothrow_move_constructible_v<T> &&
  // is_nothrow_move_constructible_v<E>
  // move
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
    TracedNoexcept::state oldState{};
    cuda::std::expected<BothMayThrow, TracedNoexcept> e1(cuda::std::unexpect, oldState, 5);
    decltype(auto) x = (e1 = 10);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<BothMayThrow, TracedNoexcept>&>, "");
    assert(&x == &e1);

    assert(e1.has_value());
    assert(e1.value().data_ == 10);

    assert(!oldState.copyCtorCalled);
    assert(oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(e1.value().movedFromInt);
  }

  // Test default template argument.
  // Without it, the template parameter cannot be deduced from an initializer list
  {
    struct Bar
    {
      int i;
      int j;
      __host__ __device__ constexpr Bar(int ii, int jj)
          : i(ii)
          , j(jj)
      {}
    };

    cuda::std::expected<Bar, int> e({5, 6});
    e = {7, 8};
    assert(e.value().i == 7);
    assert(e.value().j == 8);
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{
  cuda::std::expected<ThrowOnConvert, int> e1(cuda::std::unexpect, 5);
  try
  {
    e1 = 10;
    assert(false);
  }
  catch (Except)
  {
    assert(!e1.has_value());
    assert(e1.error() == 5);
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
