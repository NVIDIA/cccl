//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++17

// <memory>

// template <class T, class ...Args>
// constexpr T* construct_at(T* location, Args&& ...args);

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/memory>

#include "test_iterators.h"
#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4244) // conversion possible loss of data

struct Foo
{
  TEST_FUNC constexpr Foo() {}
  TEST_FUNC constexpr Foo(int a, char b, double c)
      : a_(a)
      , b_(b)
      , c_(c)
  {}
  TEST_FUNC constexpr Foo(int a, char b, double c, int* count)
      : Foo(a, b, c)
  {
    *count += 1;
  }
  TEST_FUNC constexpr bool operator==(Foo const& other) const
  {
    return a_ == other.a_ && b_ == other.b_ && c_ == other.c_;
  }

private:
  int a_;
  char b_;
  double c_;
};

struct Counted
{
  int& count_;
  TEST_FUNC constexpr Counted(int& count)
      : count_(count)
  {
    ++count;
  }
  TEST_FUNC constexpr Counted(Counted const& that)
      : count_(that.count_)
  {
    ++count_;
  }
  TEST_FUNC constexpr ~Counted()
  {
    --count_;
  }
};

union union_t
{
  int first{42};
  double second;
};

struct NotAssignable
{
  NotAssignable()                     = default;
  NotAssignable(const NotAssignable&) = default;
  NotAssignable(NotAssignable&&)      = default;

  NotAssignable& operator=(const NotAssignable&) = delete;
  NotAssignable& operator=(NotAssignable&&)      = delete;
};

constexpr bool move_assignment_called = false;
struct Always_false
{
  TEST_FUNC constexpr Always_false(const bool val) noexcept
  {
    assert(val);
  }
};

struct Dest
{
  struct tag
  {};
  TEST_FUNC constexpr Dest(tag) {}
};
struct ConvertibleToDest
{
  TEST_FUNC constexpr operator Dest() const noexcept
  {
    return Dest{Dest::tag{}};
  }
};

struct WithSpecialMoveAssignment
{
  WithSpecialMoveAssignment()                                            = default;
  WithSpecialMoveAssignment(const WithSpecialMoveAssignment&)            = default;
  WithSpecialMoveAssignment(WithSpecialMoveAssignment&&)                 = default;
  WithSpecialMoveAssignment& operator=(const WithSpecialMoveAssignment&) = default;
  TEST_FUNC constexpr WithSpecialMoveAssignment& operator=(WithSpecialMoveAssignment&&) noexcept
  {
    Always_false invalid{move_assignment_called};
    unused(invalid);
    return *this;
  };
};
static_assert(cuda::std::is_trivially_constructible_v<WithSpecialMoveAssignment>);

TEST_FUNC constexpr bool test()
{
  {
    int i    = 99;
    int* res = cuda::std::construct_at(&i);
    assert(res == &i);
    assert(*res == 0);
  }

  {
    int i    = 0;
    int* res = cuda::std::construct_at(&i, 42);
    assert(res == &i);
    assert(*res == 42);
  }

  {
    Foo foo   = {};
    int count = 0;
    Foo* res  = cuda::std::construct_at(&foo, 42, 'x', 123.89, &count);
    assert(res == &foo);
    assert(*res == Foo(42, 'x', 123.89));
    assert(count == 1);
  }

  // switching of the active member of a union must work
  {
    union_t with_int{};
    double* res = cuda::std::construct_at(&with_int.second, 123.89);
    assert(res == &with_int.second);
    assert(*res == 123.89);
  }

  // ensure that we can construct trivially constructible types with a deleted move assignment
  {
    NotAssignable not_assignable{};
    NotAssignable* res = cuda::std::construct_at(&not_assignable);
    assert(res == &not_assignable);
  }

  // ensure that we can construct trivially constructible types with a nefarious move assignment
  {
    WithSpecialMoveAssignment with_special_move_assignment{};
    WithSpecialMoveAssignment* res = cuda::std::construct_at(&with_special_move_assignment);
    assert(res == &with_special_move_assignment);
  }

  // ensure that we can construct despite narrowing conversions
  {
    int i    = 0;
    int* res = cuda::std::construct_at(&i, 2.0);
    assert(res == &i);
    assert(*res == 2);
  }

#if !TEST_COMPILER(NVHPC, <, 25, 5)
  // ensure that we can construct despite only through conversion operator
  {
    Dest i{Dest::tag{}};
    const ConvertibleToDest conv{};
    Dest* res = cuda::std::construct_at(&i, conv);
    assert(res == &i);
  }
#endif // !TEST_COMPILER(NVHPC, <, 25, 5)
#if 0 // we do not support std::allocator
    {
        cuda::std::allocator<Counted> a;
        Counted* p = a.allocate(2);
        int count = 0;
        cuda::std::construct_at(p, count);
        assert(count == 1);
        cuda::std::construct_at(p+1, count);
        assert(count == 2);
        (p+1)->~Counted();
        assert(count == 1);
        p->~Counted();
        assert(count == 0);
        a.deallocate(p, 2);
    }
    {
        cuda::std::allocator<Counted const> a;
        Counted const* p = a.allocate(2);
        int count = 0;
        cuda::std::construct_at(p, count);
        assert(count == 1);
        cuda::std::construct_at(p+1, count);
        assert(count == 2);
        (p+1)->~Counted();
        assert(count == 1);
        p->~Counted();
        assert(count == 0);
        a.deallocate(p, 2);
    }
#endif

  return true;
}

template <class... Args, class = decltype(cuda::std::construct_at(cuda::std::declval<Args>()...))>
TEST_FUNC constexpr auto can_construct_at_impl(Args&&...) -> cuda::std::true_type;

template <class... Args>
TEST_FUNC constexpr auto can_construct_at_impl(...) -> cuda::std::false_type;

template <class... Args>
inline constexpr bool can_construct_at = decltype(can_construct_at_impl(cuda::std::declval<Args>()...))::value;

// Check that SFINAE works.
static_assert(can_construct_at<int*, int>);
static_assert(can_construct_at<Foo*, int, char, double>);
static_assert(!can_construct_at<Foo*, int, char>);
static_assert(!can_construct_at<Foo*, int, char, double, int>);
static_assert(!can_construct_at<void*, int, char, double>);
static_assert(!can_construct_at<int*, int, char, double>);
#if 0 // We do not support ranges yet
static_assert(!can_construct_at<contiguous_iterator<Foo*>, int, char, double>);
#endif
// Can't construct function pointers.

#if !TEST_COMPILER(MSVC) // nvbug 4075886
static_assert(!can_construct_at<int (*)()>);
static_assert(!can_construct_at<int (*)(), cuda::std::nullptr_t>);
#endif // TEST_COMPILER(MSVC)

int main(int, char**)
{
  test();
#if !TEST_COMPILER(MSVC2019)
  static_assert(test());
#endif // !TEST_COMPILER(MSVC2019)
  return 0;
}
