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
//
// Constraints: is_constructible_v<E, GF> is true and is_assignable_v<E&, GF> is true.
//
// Effects:
// - If has_value() is true, equivalent to:
//   construct_at(addressof(unex), cuda::std::forward<GF>(e.error()));
//   has_val = false;
// - Otherwise, equivalent to: unex = cuda::std::forward<GF>(e.error());
//
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
static_assert(cuda::std::is_assignable_v<cuda::std::expected<void, int>&, const cuda::std::unexpected<int>&>, "");

// !is_constructible_v<E, GF>
static_assert(!cuda::std::is_assignable_v<cuda::std::expected<void, NotCopyConstructible>&,
                                          const cuda::std::unexpected<NotCopyConstructible>&>,
              "");

// !is_assignable_v<E&, GF>
static_assert(!cuda::std::is_assignable_v<cuda::std::expected<void, NotCopyAssignable>&,
                                          const cuda::std::unexpected<NotCopyAssignable>&>,
              "");

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // - If has_value() is true, equivalent to:
  //   construct_at(addressof(unex), cuda::std::forward<GF>(e.error()));
  //   has_val = false;
  {
    Traced::state state{};
    cuda::std::expected<void, Traced> e;
    cuda::std::unexpected<Traced> un(cuda::std::in_place, state, 5);
    decltype(auto) x = (e = un);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<void, Traced>&>, "");
    assert(&x == &e);
    assert(!e.has_value());
    assert(e.error().data_ == 5);

    assert(state.copyCtorCalled);
  }

  // - Otherwise, equivalent to: unex = cuda::std::forward<GF>(e.error());
  {
    Traced::state state1{};
    Traced::state state2{};
    cuda::std::expected<void, Traced> e(cuda::std::unexpect, state1, 5);
    cuda::std::unexpected<Traced> un(cuda::std::in_place, state2, 10);
    decltype(auto) x = (e = un);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<void, Traced>&>, "");
    assert(&x == &e);
    assert(!e.has_value());
    assert(e.error().data_ == 10);

    assert(state1.copyAssignCalled);
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{
  cuda::std::expected<void, ThrowOnCopyConstruct> e1(cuda::std::in_place);
  cuda::std::unexpected<ThrowOnCopyConstruct> un(cuda::std::in_place);
  try
  {
    e1 = un;
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
