//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
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

template <class Tp>
__host__ __device__ constexpr bool assign_empty(optional<Tp>&& lhs)
{
  optional<Tp> rhs;
  lhs = cuda::std::move(rhs);
  return !lhs.has_value() && !rhs.has_value();
}

template <class Tp>
__host__ __device__ constexpr bool assign_value(optional<Tp>&& lhs)
{
  optional<Tp> rhs(101);
  lhs = cuda::std::move(rhs);
  return lhs.has_value() && rhs.has_value() && *lhs == Tp{101};
}
#ifndef TEST_HAS_NO_EXCEPTIONS
struct X
{
  STATIC_MEMBER_VAR(throw_now, bool);
  STATIC_MEMBER_VAR(alive, int);

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
    static_assert(!cuda::std::is_nothrow_move_assignable<optional<X>>::value, "");
    X::alive()     = 0;
    X::throw_now() = false;
    optional<X> opt;
    optional<X> opt2(X{});
    assert(X::alive() == 1);
    assert(static_cast<bool>(opt2) == true);
    try
    {
      X::throw_now() = true;
      opt            = cuda::std::move(opt2);
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
    static_assert(!cuda::std::is_nothrow_move_assignable<optional<X>>::value, "");
    X::throw_now() = false;
    optional<X> opt(X{});
    optional<X> opt2(X{});
    assert(X::alive() == 2);
    assert(static_cast<bool>(opt2) == true);
    try
    {
      X::throw_now() = true;
      opt            = cuda::std::move(opt2);
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
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  {
    static_assert(cuda::std::is_nothrow_move_assignable<optional<int>>::value, "");
    optional<int> opt;
    constexpr optional<int> opt2;
    opt = cuda::std::move(opt2);
    static_assert(static_cast<bool>(opt2) == false, "");
    assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
  }
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    optional<int> opt;
    constexpr optional<int> opt2(2);
    opt = cuda::std::move(opt2);
    static_assert(static_cast<bool>(opt2) == true, "");
    static_assert(*opt2 == 2, "");
    assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
    assert(*opt == *opt2);
  }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    optional<int> opt(3);
    constexpr optional<int> opt2;
    opt = cuda::std::move(opt2);
    static_assert(static_cast<bool>(opt2) == false, "");
    assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
  }
  {
    using T = TestTypes::TestType;
    T::reset();
    optional<T> opt(3);
    optional<T> opt2;
    assert(T::alive() == 1);
    opt = cuda::std::move(opt2);
    assert(T::alive() == 0);
    assert(static_cast<bool>(opt2) == false);
    assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
  }
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    optional<int> opt(3);
    constexpr optional<int> opt2(2);
    opt = cuda::std::move(opt2);
    static_assert(static_cast<bool>(opt2) == true, "");
    static_assert(*opt2 == 2, "");
    assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
    assert(*opt == *opt2);
  }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    using O = optional<int>;
#if !defined(TEST_COMPILER_GCC) || __GNUC__ > 6
#  if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
    static_assert(assign_empty(O{42}), "");
    static_assert(assign_value(O{42}), "");
#  endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
#endif // !defined(TEST_COMPILER_GCC) || __GNUC__ > 6
    assert(assign_empty(O{42}));
    assert(assign_value(O{42}));
  }
  {
    using O = optional<TrivialTestTypes::TestType>;
#if !defined(TEST_COMPILER_GCC) || __GNUC__ > 6
#  if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
    static_assert(assign_empty(O{42}), "");
    static_assert(assign_value(O{42}), "");
#  endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
#endif // !defined(TEST_COMPILER_GCC) || __GNUC__ > 6
    assert(assign_empty(O{42}));
    assert(assign_value(O{42}));
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  {
    static_assert(cuda::std::is_nothrow_move_assignable<optional<Y>>::value, "");
  }
  {
#ifndef TEST_COMPILER_ICC
    struct ThrowsMove
    {
      __host__ __device__ ThrowsMove() noexcept {}
      __host__ __device__ ThrowsMove(ThrowsMove const&) noexcept {}
      __host__ __device__ ThrowsMove(ThrowsMove&&) noexcept(false) {}
      __host__ __device__ ThrowsMove& operator=(ThrowsMove const&) noexcept
      {
        return *this;
      }
      __host__ __device__ ThrowsMove& operator=(ThrowsMove&&) noexcept
      {
        return *this;
      }
    };
    static_assert(!cuda::std::is_nothrow_move_assignable<optional<ThrowsMove>>::value, "");
    struct ThrowsMoveAssign
    {
      __host__ __device__ ThrowsMoveAssign() noexcept {}
      __host__ __device__ ThrowsMoveAssign(ThrowsMoveAssign const&) noexcept {}
      __host__ __device__ ThrowsMoveAssign(ThrowsMoveAssign&&) noexcept {}
      __host__ __device__ ThrowsMoveAssign& operator=(ThrowsMoveAssign const&) noexcept
      {
        return *this;
      }
      __host__ __device__ ThrowsMoveAssign& operator=(ThrowsMoveAssign&&) noexcept(false)
      {
        return *this;
      }
    };
    static_assert(!cuda::std::is_nothrow_move_assignable<optional<ThrowsMoveAssign>>::value, "");
#endif // TEST_COMPILER_ICC
    struct NoThrowMove
    {
      __host__ __device__ NoThrowMove() noexcept(false) {}
      __host__ __device__ NoThrowMove(NoThrowMove const&) noexcept(false) {}
      __host__ __device__ NoThrowMove(NoThrowMove&&) noexcept {}
      __host__ __device__ NoThrowMove& operator=(NoThrowMove const&) noexcept
      {
        return *this;
      }
      __host__ __device__ NoThrowMove& operator=(NoThrowMove&&) noexcept
      {
        return *this;
      }
    };
    static_assert(cuda::std::is_nothrow_move_assignable<optional<NoThrowMove>>::value, "");
  }
  return 0;
}
