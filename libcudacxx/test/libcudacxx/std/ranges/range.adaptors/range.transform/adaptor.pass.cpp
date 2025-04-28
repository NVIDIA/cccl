//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// cuda::std::views::transform

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

template <class View, class T>
_CCCL_CONCEPT CanBePiped =
  _CCCL_REQUIRES_EXPR((View, T), View&& view, T&& t)((cuda::std::forward<View>(view) | cuda::std::forward<T>(t)));

struct NonCopyableFunction
{
  NonCopyableFunction(NonCopyableFunction const&) = delete;
  template <class T>
  __host__ __device__ constexpr T operator()(T x) const
  {
    return x;
  }
};

__host__ __device__ constexpr bool test()
{
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  // Test `views::transform(f)(v)`
  {
    {
      using Result          = cuda::std::ranges::transform_view<MoveOnlyView, PlusOne>;
      decltype(auto) result = cuda::std::views::transform(PlusOne{})(MoveOnlyView{buff});
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(result.begin().base() == buff);
      assert(result[0] == 1);
      assert(result[1] == 2);
      assert(result[2] == 3);
    }
    {
      auto const partial    = cuda::std::views::transform(PlusOne{});
      using Result          = cuda::std::ranges::transform_view<MoveOnlyView, PlusOne>;
      decltype(auto) result = partial(MoveOnlyView{buff});
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(result.begin().base() == buff);
      assert(result[0] == 1);
      assert(result[1] == 2);
      assert(result[2] == 3);
    }
  }

  // Test `v | views::transform(f)`
  {
    {
      using Result          = cuda::std::ranges::transform_view<MoveOnlyView, PlusOne>;
      decltype(auto) result = MoveOnlyView{buff} | cuda::std::views::transform(PlusOne{});
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(result.begin().base() == buff);
      assert(result[0] == 1);
      assert(result[1] == 2);
      assert(result[2] == 3);
    }
    {
      auto const partial    = cuda::std::views::transform(PlusOne{});
      using Result          = cuda::std::ranges::transform_view<MoveOnlyView, PlusOne>;
      decltype(auto) result = MoveOnlyView{buff} | partial;
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(result.begin().base() == buff);
      assert(result[0] == 1);
      assert(result[1] == 2);
      assert(result[2] == 3);
    }
  }

  // Test `views::transform(v, f)`
  {
    using Result          = cuda::std::ranges::transform_view<MoveOnlyView, PlusOne>;
    decltype(auto) result = cuda::std::views::transform(MoveOnlyView{buff}, PlusOne{});
    static_assert(cuda::std::same_as<decltype(result), Result>);
    assert(result.begin().base() == buff);
    assert(result[0] == 1);
    assert(result[1] == 2);
    assert(result[2] == 3);
  }

  // Test that one can call cuda::std::views::transform with arbitrary stuff, as long as we
  // don't try to actually complete the call by passing it a range.
  //
  // That makes no sense and we can't do anything with the result, but it's valid.
  {
    struct X
    {};
    auto partial = cuda::std::views::transform(X{});
    unused(partial);
  }

  // Test `adaptor | views::transform(f)`
  {
    {
      using Result =
        cuda::std::ranges::transform_view<cuda::std::ranges::transform_view<MoveOnlyView, PlusOne>, TimesTwo>;
      decltype(auto) result =
        MoveOnlyView{buff} | cuda::std::views::transform(PlusOne{}) | cuda::std::views::transform(TimesTwo{});
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(result.begin().base().base() == buff);
      assert(result[0] == 2);
      assert(result[1] == 4);
      assert(result[2] == 6);
    }

    {
      auto const partial = cuda::std::views::transform(PlusOne{}) | cuda::std::views::transform(TimesTwo{});
      using Result =
        cuda::std::ranges::transform_view<cuda::std::ranges::transform_view<MoveOnlyView, PlusOne>, TimesTwo>;
      decltype(auto) result = MoveOnlyView{buff} | partial;
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(result.begin().base().base() == buff);
      assert(result[0] == 2);
      assert(result[1] == 4);
      assert(result[2] == 6);
    }
  }

  // Test SFINAE friendliness
  {
    struct NotAView
    {};
    struct NotInvocable
    {};

    static_assert(!CanBePiped<MoveOnlyView, decltype(cuda::std::views::transform)>);
    static_assert(CanBePiped<MoveOnlyView, decltype(cuda::std::views::transform(PlusOne{}))>);
    static_assert(!CanBePiped<NotAView, decltype(cuda::std::views::transform(PlusOne{}))>);
    static_assert(!CanBePiped<MoveOnlyView, decltype(cuda::std::views::transform(NotInvocable{}))>);

    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::transform)>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::transform), PlusOne, MoveOnlyView>);
    static_assert(cuda::std::is_invocable_v<decltype(cuda::std::views::transform), MoveOnlyView, PlusOne>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::transform), MoveOnlyView, PlusOne, PlusOne>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::transform), NonCopyableFunction>);
  }

  {
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::ranges::views::transform), decltype(cuda::std::views::transform)>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // _CCCL_BUILTIN_ADDRESSOF

  return 0;
}
