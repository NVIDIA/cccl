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

// constexpr expected& operator=(const expected& rhs);
//
// Effects:
// - If this->has_value() && rhs.has_value() is true, no effects.
// - Otherwise, if this->has_value() is true, equivalent to: construct_at(addressof(unex), rhs.unex); has_val = false;
// - Otherwise, if rhs.has_value() is true, destroys unex and sets has_val to true.
// - Otherwise, equivalent to unex = rhs.error().
//
// Returns: *this.
//
// Remarks: This operator is defined as deleted unless is_copy_assignable_v<E> is true and is_copy_constructible_v<E> is
// true.

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
static_assert(cuda::std::is_copy_assignable_v<cuda::std::expected<void, int>>, "");

// !is_copy_assignable_v<E>
static_assert(!cuda::std::is_copy_assignable_v<cuda::std::expected<void, NotCopyAssignable>>, "");

// !is_copy_constructible_v<E>
static_assert(!cuda::std::is_copy_assignable_v<cuda::std::expected<void, NotCopyConstructible>>, "");

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // If this->has_value() && rhs.has_value() is true, no effects.
  {
    cuda::std::expected<void, int> e1;
    cuda::std::expected<void, int> e2;
    decltype(auto) x = (e1 = e2);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<void, int>&>, "");
    assert(&x == &e1);
    assert(e1.has_value());
  }

  // Otherwise, if this->has_value() is true, equivalent to: construct_at(addressof(unex), rhs.unex); has_val = false;
  {
    Traced::state state{};
    cuda::std::expected<void, Traced> e1;
    cuda::std::expected<void, Traced> e2(cuda::std::unexpect, state, 5);
    decltype(auto) x = (e1 = e2);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<void, Traced>&>, "");
    assert(&x == &e1);
    assert(!e1.has_value());
    assert(e1.error().data_ == 5);

    assert(state.copyCtorCalled);
  }

  // Otherwise, if rhs.has_value() is true, destroys unex and sets has_val to true.
  {
    Traced::state state{};
    cuda::std::expected<void, Traced> e1(cuda::std::unexpect, state, 5);
    cuda::std::expected<void, Traced> e2;
    decltype(auto) x = (e1 = e2);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<void, Traced>&>, "");
    assert(&x == &e1);
    assert(e1.has_value());

    assert(state.dtorCalled);
  }

  // Otherwise, equivalent to unex = rhs.error().
  {
    Traced::state state{};
    cuda::std::expected<void, Traced> e1(cuda::std::unexpect, state, 5);
    cuda::std::expected<void, Traced> e2(cuda::std::unexpect, state, 10);
    decltype(auto) x = (e1 = e2);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<void, Traced>&>, "");
    assert(&x == &e1);
    assert(!e1.has_value());
    assert(e1.error().data_ == 10);

    assert(state.copyAssignCalled);
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{
  cuda::std::expected<void, ThrowOnCopyConstruct> e1(cuda::std::in_place);
  cuda::std::expected<void, ThrowOnCopyConstruct> e2(cuda::std::unexpect);
  try
  {
    e1 = e2;
    assert(false);
  }
  catch (Except)
  {
    assert(e1.has_value());
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
