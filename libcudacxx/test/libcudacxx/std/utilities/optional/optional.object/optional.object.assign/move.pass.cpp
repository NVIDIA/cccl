//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// constexpr optional<T>& operator=(optional<T>&& rhs)
//     noexcept(is_nothrow_move_assignable<T>::value &&
//              is_nothrow_move_constructible<T>::value);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::optional;

struct Y
{};
static_assert(cuda::std::is_nothrow_move_assignable<optional<Y>>::value);

struct ThrowsMove
{
  TEST_FUNC ThrowsMove() noexcept {}
  TEST_FUNC ThrowsMove(ThrowsMove const&) noexcept {}
  TEST_FUNC ThrowsMove(ThrowsMove&&) noexcept(false) {}
  TEST_FUNC ThrowsMove& operator=(ThrowsMove const&) noexcept
  {
    return *this;
  }
  TEST_FUNC ThrowsMove& operator=(ThrowsMove&&) noexcept
  {
    return *this;
  }
};
static_assert(!cuda::std::is_nothrow_move_assignable<optional<ThrowsMove>>::value);

struct ThrowsMoveAssign
{
  TEST_FUNC ThrowsMoveAssign() noexcept {}
  TEST_FUNC ThrowsMoveAssign(ThrowsMoveAssign const&) noexcept {}
  TEST_FUNC ThrowsMoveAssign(ThrowsMoveAssign&&) noexcept {}
  TEST_FUNC ThrowsMoveAssign& operator=(ThrowsMoveAssign const&) noexcept
  {
    return *this;
  }
  TEST_FUNC ThrowsMoveAssign& operator=(ThrowsMoveAssign&&) noexcept(false)
  {
    return *this;
  }
};

static_assert(!cuda::std::is_nothrow_move_assignable<optional<ThrowsMoveAssign>>::value);

struct NoThrowMove
{
  TEST_FUNC NoThrowMove() noexcept(false) {}
  TEST_FUNC NoThrowMove(NoThrowMove const&) noexcept(false) {}
  TEST_FUNC NoThrowMove(NoThrowMove&&) noexcept {}
  TEST_FUNC NoThrowMove& operator=(NoThrowMove const&) noexcept
  {
    return *this;
  }
  TEST_FUNC NoThrowMove& operator=(NoThrowMove&&) noexcept
  {
    return *this;
  }
};
static_assert(cuda::std::is_nothrow_move_assignable<optional<NoThrowMove>>::value);

#if TEST_HAS_EXCEPTIONS()
struct X
{
  STATIC_MEMBER_VAR(throw_now, bool)
  STATIC_MEMBER_VAR(alive, int)

  X()
  {
    ++alive();
  }
  X(X&&)
  {
    if (throw_now())
    {
      TEST_THROW(6);
    }
    ++alive();
  }

  X& operator=(X&&)
  {
    if (throw_now())
    {
      TEST_THROW(42);
    }
    return *this;
  }

  ~X()
  {
    assert(alive() > 0);
    --alive();
  }
};

void test_exceptions()
{
  {
    static_assert(!cuda::std::is_nothrow_move_assignable<optional<X>>::value);
    X::alive()     = 0;
    X::throw_now() = false;
    optional<X> opt{};
    optional<X> input(X{});
    assert(X::alive() == 1);
    assert(static_cast<bool>(input) == true);
    try
    {
      X::throw_now() = true;
      opt            = cuda::std::move(input);
      assert(false);
    }
    catch (int i)
    {
      assert(i == 6);
      assert(static_cast<bool>(opt) == false);
    }
    assert(X::alive() == 1);
  }
  assert(X::alive() == 0);
  {
    static_assert(!cuda::std::is_nothrow_move_assignable<optional<X>>::value);
    X::throw_now() = false;
    optional<X> opt(X{});
    optional<X> input(X{});
    assert(X::alive() == 2);
    assert(static_cast<bool>(input) == true);
    try
    {
      X::throw_now() = true;
      opt            = cuda::std::move(input);
      assert(false);
    }
    catch (int i)
    {
      assert(i == 42);
      assert(static_cast<bool>(opt) == true);
    }
    assert(X::alive() == 2);
  }
  assert(X::alive() == 0);
}
#endif // TEST_HAS_EXCEPTIONS()

template <class T>
TEST_FUNC constexpr void test()
{
  cuda::std::remove_reference_t<T> val{42};
  cuda::std::remove_reference_t<T> other_val{1337};

  static_assert(cuda::std::is_nothrow_move_assignable<optional<T>>::value);
  // empty move assigned to empty
  {
    optional<T> opt{};
    optional<T> input{};
    opt = cuda::std::move(input);
    assert(!opt.has_value());
    assert(!input.has_value());
  }
  // empty move assigned to non-empty
  {
    optional<T> opt{val};
    optional<T> input{};
    opt = cuda::std::move(input);
    assert(!opt.has_value());
    assert(!input.has_value());
  }
  // non-empty move assigned to empty
  {
    optional<T> opt{};
    optional<T> input{val};
    opt = cuda::std::move(input);
    assert(opt.has_value());
    assert(input.has_value()); // input still holds a moved from value
    assert(*opt == val);
    if constexpr (cuda::std::is_reference_v<T>)
    {
      assert(input.operator->() == opt.operator->());
    }
  }
  // non-empty move assigned to empty
  {
    optional<T> opt{other_val};
    optional<T> input{val};
    opt = cuda::std::move(input);
    assert(opt.has_value());
    assert(input.has_value()); // input still holds a moved from value
    assert(*opt == val);
    if constexpr (cuda::std::is_reference_v<T>)
    {
      assert(input.operator->() == opt.operator->());
    }
  }
}

TEST_FUNC constexpr bool test()
{
  test<int>();
  test<int&>();

  test<TrivialTestTypes::TestType>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  {
    using T = TestTypes::TestType;
    T::reset();
    optional<T> opt(3);
    optional<T> input{};
    assert(T::alive() == 1);
    opt = cuda::std::move(input);
    assert(T::alive() == 0);
    assert(static_cast<bool>(input) == false);
    assert(static_cast<bool>(opt) == static_cast<bool>(input));
  }

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
