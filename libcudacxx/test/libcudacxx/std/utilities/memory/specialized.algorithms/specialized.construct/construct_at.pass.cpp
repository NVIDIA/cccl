//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// template <class T, class ...Args>
// constexpr T* construct_at(T* location, Args&& ...args);

// #include <cuda/std/memory>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

#if defined(TEST_COMPILER_MSVC)
#  pragma warning(disable : 4244)
#endif // TEST_COMPILER_MSVC

struct Foo
{
  __host__ __device__ constexpr Foo() {}
  __host__ __device__ constexpr Foo(int a, char b, double c)
      : a_(a)
      , b_(b)
      , c_(c)
  {}
  __host__ __device__ constexpr Foo(int a, char b, double c, int* count)
      : Foo(a, b, c)
  {
    *count += 1;
  }
  __host__ __device__ constexpr bool operator==(Foo const& other) const
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
  __host__ __device__ constexpr Counted(int& count)
      : count_(count)
  {
    ++count;
  }
  __host__ __device__ constexpr Counted(Counted const& that)
      : count_(that.count_)
  {
    ++count_;
  }
  __host__ __device__ constexpr ~Counted()
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
  __host__ __device__ constexpr Always_false(const bool val) noexcept
  {
    assert(val);
  }
};

struct WithSpecialMoveAssignment
{
  WithSpecialMoveAssignment()                                            = default;
  WithSpecialMoveAssignment(const WithSpecialMoveAssignment&)            = default;
  WithSpecialMoveAssignment(WithSpecialMoveAssignment&&)                 = default;
  WithSpecialMoveAssignment& operator=(const WithSpecialMoveAssignment&) = default;
  __host__ __device__ constexpr WithSpecialMoveAssignment& operator=(WithSpecialMoveAssignment&&) noexcept
  {
    Always_false invalid{move_assignment_called};
    unused(invalid);
    return *this;
  };
};
static_assert(cuda::std::is_trivially_constructible_v<WithSpecialMoveAssignment>);

__host__ __device__ constexpr bool test()
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
__host__ __device__ constexpr bool can_construct_at(Args&&...)
{
  return true;
}

template <class... Args>
__host__ __device__ constexpr bool can_construct_at(...)
{
  return false;
}

// Check that SFINAE works.
static_assert(can_construct_at((int*) nullptr, 42));
static_assert(can_construct_at((Foo*) nullptr, 1, '2', 3.0));
static_assert(!can_construct_at((Foo*) nullptr, 1, '2'));
static_assert(!can_construct_at((Foo*) nullptr, 1, '2', 3.0, 4));
static_assert(!can_construct_at(nullptr, 1, '2', 3.0));
static_assert(!can_construct_at((int*) nullptr, 1, '2', 3.0));
#if 0 // We do not support ranges yet
static_assert(!can_construct_at(contiguous_iterator<Foo*>(), 1, '2', 3.0));
#endif
// Can't construct function pointers.

#ifndef TEST_COMPILER_MSVC // nvbug 4075886
static_assert(!can_construct_at((int (*)()) nullptr));
static_assert(!can_construct_at((int (*)()) nullptr, nullptr));
#endif // TEST_COMPILER_MSVC

int main(int, char**)
{
  test();
#if !(defined(TEST_COMPILER_CLANG) && __clang_major__ <= 10) && !defined(TEST_COMPILER_MSVC_2017) \
  && !defined(TEST_COMPILER_MSVC_2019)
  static_assert(test());
#endif
  return 0;
}
