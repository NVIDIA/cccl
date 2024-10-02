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

// From LWG2451:
// template <class U>
// optional<T>& operator=(optional<U>&& rhs);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/optional>
#ifdef _LIBCUDACXX_HAS_MEMORY
#  include <cuda/std/memory>
#endif
#include <cuda/std/type_traits>

#include "archetypes.h"
#include "MoveOnly.h"
#include "test_macros.h"

using cuda::std::optional;

struct Y1
{
  Y1() = default;
  __host__ __device__ Y1(const int&) {}
  Y1& operator=(const Y1&) = delete;
};

struct Y2
{
  Y2()           = default;
  Y2(const int&) = delete;
  __host__ __device__ Y2& operator=(const int&)
  {
    return *this;
  }
};

struct B
{
  virtual ~B() = default;
};
class D : public B
{};

template <class T>
struct AssignableFrom
{
  STATIC_MEMBER_VAR(type_constructed, int);
  STATIC_MEMBER_VAR(type_assigned, int);
  STATIC_MEMBER_VAR(int_constructed, int);
  STATIC_MEMBER_VAR(int_assigned, int);

  __host__ __device__ static void reset()
  {
    type_constructed() = int_constructed() = 0;
    type_assigned() = int_assigned() = 0;
  }

  AssignableFrom() = default;

  __host__ __device__ explicit AssignableFrom(T)
  {
    ++type_constructed();
  }
  __host__ __device__ AssignableFrom& operator=(T)
  {
    ++type_assigned();
    return *this;
  }

  __host__ __device__ AssignableFrom(int)
  {
    ++int_constructed();
  }
  __host__ __device__ AssignableFrom& operator=(int)
  {
    ++int_assigned();
    return *this;
  }

private:
  AssignableFrom(AssignableFrom const&)            = delete;
  AssignableFrom& operator=(AssignableFrom const&) = delete;
};

__host__ __device__ void test_with_test_type()
{
  using T = TestTypes::TestType;
  T::reset();
  { // non-empty to empty
    T::reset_constructors();
    optional<T> opt;
    optional<int> other(42);
    opt = cuda::std::move(other);
    assert(T::alive() == 1);
    assert(T::constructed() == 1);
    assert(T::value_constructed() == 1);
    assert(T::assigned() == 0);
    assert(T::destroyed() == 0);
    assert(static_cast<bool>(other) == true);
    assert(*other == 42);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T(42));
  }
  assert(T::alive() == 0);
  { // non-empty to non-empty
    optional<T> opt(101);
    optional<int> other(42);
    T::reset_constructors();
    opt = cuda::std::move(other);
    assert(T::alive() == 1);
    assert(T::constructed() == 0);
    assert(T::assigned() == 1);
    assert(T::value_assigned() == 1);
    assert(T::destroyed() == 0);
    assert(static_cast<bool>(other) == true);
    assert(*other == 42);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T(42));
  }
  assert(T::alive() == 0);
  { // empty to non-empty
    optional<T> opt(101);
    optional<int> other;
    T::reset_constructors();
    opt = cuda::std::move(other);
    assert(T::alive() == 0);
    assert(T::constructed() == 0);
    assert(T::assigned() == 0);
    assert(T::destroyed() == 1);
    assert(static_cast<bool>(other) == false);
    assert(static_cast<bool>(opt) == false);
  }
  assert(T::alive() == 0);
  { // empty to empty
    optional<T> opt;
    optional<int> other;
    T::reset_constructors();
    opt = cuda::std::move(other);
    assert(T::alive() == 0);
    assert(T::constructed() == 0);
    assert(T::assigned() == 0);
    assert(T::destroyed() == 0);
    assert(static_cast<bool>(other) == false);
    assert(static_cast<bool>(opt) == false);
  }
  assert(T::alive() == 0);
}

__host__ __device__ void test_ambiguous_assign()
{
  using OptInt = cuda::std::optional<int>;
  {
    using T = AssignableFrom<OptInt&&>;
    T::reset();
    {
      OptInt a(42);
      cuda::std::optional<T> t;
      t = cuda::std::move(a);
      assert(T::type_constructed() == 1);
      assert(T::type_assigned() == 0);
      assert(T::int_constructed() == 0);
      assert(T::int_assigned() == 0);
    }
    {
      using Opt = cuda::std::optional<T>;
      static_assert(!cuda::std::is_assignable<Opt&, const OptInt&&>::value, "");
      static_assert(!cuda::std::is_assignable<Opt&, const OptInt&>::value, "");
      static_assert(!cuda::std::is_assignable<Opt&, OptInt&>::value, "");
    }
  }
  {
    using T = AssignableFrom<OptInt const&&>;
    T::reset();
    {
      const OptInt a(42);
      cuda::std::optional<T> t;
      t = cuda::std::move(a);
      assert(T::type_constructed() == 1);
      assert(T::type_assigned() == 0);
      assert(T::int_constructed() == 0);
      assert(T::int_assigned() == 0);
    }
    T::reset();
    {
      OptInt a(42);
      cuda::std::optional<T> t;
      t = cuda::std::move(a);
      assert(T::type_constructed() == 1);
      assert(T::type_assigned() == 0);
      assert(T::int_constructed() == 0);
      assert(T::int_assigned() == 0);
    }
    {
      using Opt = cuda::std::optional<T>;
      static_assert(cuda::std::is_assignable<Opt&, OptInt&&>::value, "");
      static_assert(!cuda::std::is_assignable<Opt&, const OptInt&>::value, "");
      static_assert(!cuda::std::is_assignable<Opt&, OptInt&>::value, "");
    }
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    optional<int> opt;
    optional<short> opt2;
    opt = cuda::std::move(opt2);
    assert(static_cast<bool>(opt2) == false);
    assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
  }
  {
    optional<int> opt;
    optional<short> opt2(short{2});
    opt = cuda::std::move(opt2);
    assert(static_cast<bool>(opt2) == true);
    assert(*opt2 == 2);
    assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
    assert(*opt == *opt2);
  }
  {
    optional<int> opt(3);
    optional<short> opt2;
    opt = cuda::std::move(opt2);
    assert(static_cast<bool>(opt2) == false);
    assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
  }
  {
    optional<int> opt(3);
    optional<short> opt2(short{2});
    opt = cuda::std::move(opt2);
    assert(static_cast<bool>(opt2) == true);
    assert(*opt2 == 2);
    assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
    assert(*opt == *opt2);
  }

  enum class state_t
  {
    inactive,
    constructed,
    copy_assigned,
    move_assigned
  };
  class StateTracker
  {
  public:
    __host__ __device__ constexpr StateTracker(state_t& s)
        : state_(&s)
    {
      *state_ = state_t::constructed;
    }

    StateTracker(StateTracker&&)      = default;
    StateTracker(StateTracker const&) = default;

    __host__ __device__ constexpr StateTracker& operator=(StateTracker&& other) noexcept
    {
      *state_      = state_t::inactive;
      state_       = other.state_;
      *state_      = state_t::move_assigned;
      other.state_ = nullptr;
      return *this;
    }

    __host__ __device__ constexpr StateTracker& operator=(StateTracker const& other) noexcept
    {
      *state_ = state_t::inactive;
      state_  = other.state_;
      *state_ = state_t::copy_assigned;
      return *this;
    }

  private:
    state_t* state_;
  };
  {
    auto state = cuda::std::array<state_t, 2>{state_t::inactive, state_t::inactive};
    auto opt1  = cuda::std::optional<StateTracker>(state[0]);
    assert(state[0] == state_t::constructed);

    auto opt2 = cuda::std::optional<StateTracker>(state[1]);
    assert(state[1] == state_t::constructed);

    opt1 = cuda::std::move(opt2);
    assert(state[0] == state_t::inactive);
    assert(state[1] == state_t::move_assigned);
  }
  {
    auto state = cuda::std::array<state_t, 2>{state_t::inactive, state_t::inactive};
    auto opt1  = cuda::std::optional<StateTracker>(state[0]);
    assert(state[0] == state_t::constructed);

    auto opt2 = cuda::std::optional<StateTracker>(state[1]);
    assert(state[1] == state_t::constructed);

    opt1 = opt2;
    assert(state[0] == state_t::inactive);
    assert(state[1] == state_t::copy_assigned);
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct X
{
  STATIC_MEMBER_VAR(throw_now, bool);

  X() = default;
  X(int&&)
  {
    if (throw_now())
    {
      TEST_THROW(6);
    }
  }
};

void test_exceptions()
{
  optional<X> opt;
  optional<int> opt2(42);
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
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif
  test_with_test_type();
  test_ambiguous_assign();
  test();
#ifdef _LIBCUDACXX_HAS_MEMORY
  {
    optional<cuda::std::unique_ptr<B>> opt;
    optional<cuda::std::unique_ptr<D>> other(new D());
    opt = cuda::std::move(other);
    assert(static_cast<bool>(opt) == true);
    assert(static_cast<bool>(other) == true);
    assert(opt->get() != nullptr);
    assert(other->get() == nullptr);
  }
#endif // _LIBCUDACXX_HAS_MEMORY

#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
