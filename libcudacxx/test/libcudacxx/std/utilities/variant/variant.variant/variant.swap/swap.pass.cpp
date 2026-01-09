//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class ...Types> class variant;

// void swap(variant& rhs) noexcept(see below)

#include <cuda/std/cassert>
// #include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_convertible.h"
#include "test_macros.h"
#include "variant_test_helpers.h"

struct NotSwappable
{};
__host__ __device__ void swap(NotSwappable&, NotSwappable&) = delete;

struct NotCopyable
{
  NotCopyable()                              = default;
  NotCopyable(const NotCopyable&)            = delete;
  NotCopyable& operator=(const NotCopyable&) = delete;
};

struct NotCopyableWithSwap
{
  NotCopyableWithSwap()                                      = default;
  NotCopyableWithSwap(const NotCopyableWithSwap&)            = delete;
  NotCopyableWithSwap& operator=(const NotCopyableWithSwap&) = delete;
};
__host__ __device__ void swap(NotCopyableWithSwap&, NotCopyableWithSwap) {}

struct NotMoveAssignable
{
  NotMoveAssignable()                               = default;
  NotMoveAssignable(NotMoveAssignable&&)            = default;
  NotMoveAssignable& operator=(NotMoveAssignable&&) = delete;
};

struct NotMoveAssignableWithSwap
{
  NotMoveAssignableWithSwap()                                       = default;
  NotMoveAssignableWithSwap(NotMoveAssignableWithSwap&&)            = default;
  NotMoveAssignableWithSwap& operator=(NotMoveAssignableWithSwap&&) = delete;
};
__host__ __device__ void swap(NotMoveAssignableWithSwap&, NotMoveAssignableWithSwap&) noexcept {}

template <bool Throws>
__host__ __device__ void do_throw()
{}

template <>
__host__ __device__ void do_throw<true>()
{
#if TEST_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw 42;), (cuda::std::terminate();))
#else
  cuda::std::terminate();
#endif
}

template <bool NT_Copy, bool NT_Move, bool NT_CopyAssign, bool NT_MoveAssign, bool NT_Swap, bool EnableSwap = true>
struct NothrowTypeImp
{
  STATIC_MEMBER_VAR(move_called, int)
  STATIC_MEMBER_VAR(move_assign_called, int)
  STATIC_MEMBER_VAR(swap_called, int)
  __host__ __device__ static void reset()
  {
    move_called() = move_assign_called() = swap_called() = 0;
  }
  NothrowTypeImp() = default;
  __host__ __device__ explicit NothrowTypeImp(int v)
      : value(v)
  {}
  __host__ __device__ NothrowTypeImp(const NothrowTypeImp& o) noexcept(NT_Copy)
      : value(o.value)
  {
    assert(false);
  } // never called by test
  __host__ __device__ NothrowTypeImp(NothrowTypeImp&& o) noexcept(NT_Move)
      : value(o.value)
  {
    ++move_called();
    do_throw<!NT_Move>();
    o.value = -1;
  }
  __host__ __device__ NothrowTypeImp& operator=(const NothrowTypeImp&) noexcept(NT_CopyAssign)
  {
    assert(false);
    return *this;
  } // never called by the tests
  __host__ __device__ NothrowTypeImp& operator=(NothrowTypeImp&& o) noexcept(NT_MoveAssign)
  {
    ++move_assign_called();
    do_throw<!NT_MoveAssign>();
    value   = o.value;
    o.value = -1;
    return *this;
  }
  int value;
};

template <bool NT_Copy, bool NT_Move, bool NT_CopyAssign, bool NT_MoveAssign, bool NT_Swap>
__host__ __device__ void
swap(NothrowTypeImp<NT_Copy, NT_Move, NT_CopyAssign, NT_MoveAssign, NT_Swap, true>& lhs,
     NothrowTypeImp<NT_Copy, NT_Move, NT_CopyAssign, NT_MoveAssign, NT_Swap, true>& rhs) noexcept(NT_Swap)
{
  lhs.swap_called()++;
  do_throw<!NT_Swap>();
  int tmp   = lhs.value;
  lhs.value = rhs.value;
  rhs.value = tmp;
}

// throwing copy, nothrow move ctor/assign, no swap provided
using NothrowMoveable = NothrowTypeImp<false, true, false, true, false, false>;
// throwing copy and move assign, nothrow move ctor, no swap provided
using NothrowMoveCtor = NothrowTypeImp<false, true, false, false, false, false>;
// nothrow move ctor, throwing move assignment, swap provided
using NothrowMoveCtorWithThrowingSwap = NothrowTypeImp<false, true, false, false, false, true>;
// throwing move ctor, nothrow move assignment, no swap provided
using ThrowingMoveCtor = NothrowTypeImp<false, false, false, true, false, false>;
// throwing special members, nothrowing swap
using ThrowingTypeWithNothrowSwap = NothrowTypeImp<false, false, false, false, true, true>;
using NothrowTypeWithThrowingSwap = NothrowTypeImp<true, true, true, true, false, true>;
// throwing move assign with nothrow move and nothrow swap
using ThrowingMoveAssignNothrowMoveCtorWithSwap = NothrowTypeImp<false, true, false, false, true, true>;
// throwing move assign with nothrow move but no swap.
using ThrowingMoveAssignNothrowMoveCtor = NothrowTypeImp<false, true, false, false, false, false>;

struct NonThrowingNonNoexceptType
{
  STATIC_MEMBER_VAR(move_called, int)
  __host__ __device__ static void reset()
  {
    move_called() = 0;
  }
  NonThrowingNonNoexceptType() = default;
  __host__ __device__ NonThrowingNonNoexceptType(int v)
      : value(v)
  {}
  __host__ __device__ NonThrowingNonNoexceptType(NonThrowingNonNoexceptType&& o) noexcept(false)
      : value(o.value)
  {
    ++move_called();
    o.value = -1;
  }
  __host__ __device__ NonThrowingNonNoexceptType& operator=(NonThrowingNonNoexceptType&&) noexcept(false)
  {
    assert(false); // never called by the tests.
    return *this;
  }
  int value;
};

#if TEST_HAS_EXCEPTIONS()
struct ThrowsOnSecondMove
{
  int value;
  int move_count;
  ThrowsOnSecondMove(int v)
      : value(v)
      , move_count(0)
  {}
  ThrowsOnSecondMove(ThrowsOnSecondMove&& o) noexcept(false)
      : value(o.value)
      , move_count(o.move_count + 1)
  {
    if (move_count == 2)
    {
      do_throw<true>();
    }
    o.value = -1;
  }
  ThrowsOnSecondMove& operator=(ThrowsOnSecondMove&&)
  {
    assert(false); // not called by test
    return *this;
  }
};

void test_swap_valueless_by_exception()
{
  using V = cuda::std::variant<int, MakeEmptyT>;
  { // both empty
    V v1;
    makeEmpty(v1);
    V v2;
    makeEmpty(v2);
    assert(MakeEmptyT::alive == 0);
    { // member swap
      v1.swap(v2);
      assert(v1.valueless_by_exception());
      assert(v2.valueless_by_exception());
      assert(MakeEmptyT::alive == 0);
    }
    { // non-member swap
      swap(v1, v2);
      assert(v1.valueless_by_exception());
      assert(v2.valueless_by_exception());
      assert(MakeEmptyT::alive == 0);
    }
  }
  { // only one empty
    V v1(42);
    V v2;
    makeEmpty(v2);
    { // member swap
      v1.swap(v2);
      assert(v1.valueless_by_exception());
      assert(cuda::std::get<0>(v2) == 42);
      // swap again
      v2.swap(v1);
      assert(v2.valueless_by_exception());
      assert(cuda::std::get<0>(v1) == 42);
    }
    { // non-member swap
      swap(v1, v2);
      assert(v1.valueless_by_exception());
      assert(cuda::std::get<0>(v2) == 42);
      // swap again
      swap(v1, v2);
      assert(v2.valueless_by_exception());
      assert(cuda::std::get<0>(v1) == 42);
    }
  }
}
#endif // TEST_HAS_EXCEPTIONS()

__host__ __device__ void test_swap_same_alternative()
{
  {
    using T = ThrowingTypeWithNothrowSwap;
    using V = cuda::std::variant<T, int>;
    T::reset();
    V v1(cuda::std::in_place_index<0>, 42);
    V v2(cuda::std::in_place_index<0>, 100);
    v1.swap(v2);
    assert(T::swap_called() == 1);
    assert(cuda::std::get<0>(v1).value == 100);
    assert(cuda::std::get<0>(v2).value == 42);
    swap(v1, v2);
    assert(T::swap_called() == 2);
    assert(cuda::std::get<0>(v1).value == 42);
    assert(cuda::std::get<0>(v2).value == 100);
  }
  {
    using T = NothrowMoveable;
    using V = cuda::std::variant<T, int>;
    T::reset();
    V v1(cuda::std::in_place_index<0>, 42);
    V v2(cuda::std::in_place_index<0>, 100);
    v1.swap(v2);
    assert(T::swap_called() == 0);
    assert(T::move_called() == 1);
    assert(T::move_assign_called() == 2);
    assert(cuda::std::get<0>(v1).value == 100);
    assert(cuda::std::get<0>(v2).value == 42);
    T::reset();
    swap(v1, v2);
    assert(T::swap_called() == 0);
    assert(T::move_called() == 1);
    assert(T::move_assign_called() == 2);
    assert(cuda::std::get<0>(v1).value == 42);
    assert(cuda::std::get<0>(v2).value == 100);
  }
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions_same_alternative()
{
  {
    using T = NothrowTypeWithThrowingSwap;
    using V = cuda::std::variant<T, int>;
    T::reset();
    V v1(cuda::std::in_place_index<0>, 42);
    V v2(cuda::std::in_place_index<0>, 100);
    try
    {
      v1.swap(v2);
      assert(false);
    }
    catch (int)
    {}
    assert(T::swap_called() == 1);
    assert(T::move_called() == 0);
    assert(T::move_assign_called() == 0);
    assert(cuda::std::get<0>(v1).value == 42);
    assert(cuda::std::get<0>(v2).value == 100);
  }
  {
    using T = ThrowingMoveCtor;
    using V = cuda::std::variant<T, int>;
    T::reset();
    V v1(cuda::std::in_place_index<0>, 42);
    V v2(cuda::std::in_place_index<0>, 100);
    try
    {
      v1.swap(v2);
      assert(false);
    }
    catch (int)
    {}
    assert(T::move_called() == 1); // call threw
    assert(T::move_assign_called() == 0);
    assert(cuda::std::get<0>(v1).value == 42); // throw happened before v1 was moved from
    assert(cuda::std::get<0>(v2).value == 100);
  }
  {
    using T = ThrowingMoveAssignNothrowMoveCtor;
    using V = cuda::std::variant<T, int>;
    T::reset();
    V v1(cuda::std::in_place_index<0>, 42);
    V v2(cuda::std::in_place_index<0>, 100);
    try
    {
      v1.swap(v2);
      assert(false);
    }
    catch (int)
    {}
    assert(T::move_called() == 1);
    assert(T::move_assign_called() == 1); // call threw and didn't complete
    assert(cuda::std::get<0>(v1).value == -1); // v1 was moved from
    assert(cuda::std::get<0>(v2).value == 100);
  }
}
#endif // TEST_HAS_EXCEPTIONS()

__host__ __device__ void test_swap_different_alternatives()
{
  {
    using T = NothrowMoveCtorWithThrowingSwap;
    using V = cuda::std::variant<T, int>;
    T::reset();
    V v1(cuda::std::in_place_index<0>, 42);
    V v2(cuda::std::in_place_index<1>, 100);
    v1.swap(v2);
    assert(T::swap_called() == 0);
    // The libc++ implementation double copies the argument, and not
    // the variant swap is called on.
    assert(T::move_called() == 1);
    assert(T::move_called() <= 2);
    assert(T::move_assign_called() == 0);
    assert(cuda::std::get<1>(v1) == 100);
    assert(cuda::std::get<0>(v2).value == 42);
    T::reset();
    swap(v1, v2);
    assert(T::swap_called() == 0);
    assert(T::move_called() == 2);
    assert(T::move_called() <= 2);
    assert(T::move_assign_called() == 0);
    assert(cuda::std::get<0>(v1).value == 42);
    assert(cuda::std::get<1>(v2) == 100);
  }
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions_different_alternatives()
{
  {
    using T1 = ThrowingTypeWithNothrowSwap;
    using T2 = NonThrowingNonNoexceptType;
    using V  = cuda::std::variant<T1, T2>;
    T1::reset();
    T2::reset();
    V v1(cuda::std::in_place_index<0>, 42);
    V v2(cuda::std::in_place_index<1>, 100);
    try
    {
      v1.swap(v2);
      assert(false);
    }
    catch (int)
    {}
    assert(T1::swap_called() == 0);
    assert(T1::move_called() == 1); // throws
    assert(T1::move_assign_called() == 0);
    // FIXME: libc++ shouldn't move from T2 here.
    assert(T2::move_called() == 1);
    assert(T2::move_called() <= 1);
    assert(cuda::std::get<0>(v1).value == 42);
    if (T2::move_called() != 0)
    {
      assert(v2.valueless_by_exception());
    }
    else
    {
      assert(cuda::std::get<1>(v2).value == 100);
    }
  }
  {
    using T1 = NonThrowingNonNoexceptType;
    using T2 = ThrowingTypeWithNothrowSwap;
    using V  = cuda::std::variant<T1, T2>;
    T1::reset();
    T2::reset();
    V v1(cuda::std::in_place_index<0>, 42);
    V v2(cuda::std::in_place_index<1>, 100);
    try
    {
      v1.swap(v2);
      assert(false);
    }
    catch (int)
    {}
    assert(T1::move_called() == 0);
    assert(T1::move_called() <= 1);
    assert(T2::swap_called() == 0);
    assert(T2::move_called() == 1); // throws
    assert(T2::move_assign_called() == 0);
    if (T1::move_called() != 0)
    {
      assert(v1.valueless_by_exception());
    }
    else
    {
      assert(cuda::std::get<0>(v1).value == 42);
    }
    assert(cuda::std::get<1>(v2).value == 100);
  }
// FIXME: The tests below are just very libc++ specific
#  ifdef _CUDA_STD_VERSION
  {
    using T1 = ThrowsOnSecondMove;
    using T2 = NonThrowingNonNoexceptType;
    using V  = cuda::std::variant<T1, T2>;
    T2::reset();
    V v1(cuda::std::in_place_index<0>, 42);
    V v2(cuda::std::in_place_index<1>, 100);
    v1.swap(v2);
    assert(T2::move_called() == 2);
    assert(cuda::std::get<1>(v1).value == 100);
    assert(cuda::std::get<0>(v2).value == 42);
    assert(cuda::std::get<0>(v2).move_count == 1);
  }
  {
    using T1 = NonThrowingNonNoexceptType;
    using T2 = ThrowsOnSecondMove;
    using V  = cuda::std::variant<T1, T2>;
    T1::reset();
    V v1(cuda::std::in_place_index<0>, 42);
    V v2(cuda::std::in_place_index<1>, 100);
    try
    {
      v1.swap(v2);
      assert(false);
    }
    catch (int)
    {}
    assert(T1::move_called() == 1);
    assert(v1.valueless_by_exception());
    assert(cuda::std::get<0>(v2).value == 42);
  }
#  endif // _CUDA_STD_VERSION
}
#endif // TEST_HAS_EXCEPTIONS()

template <class Var>
__host__ __device__ constexpr auto has_swap_member_imp(int)
  -> decltype(cuda::std::declval<Var&>().swap(cuda::std::declval<Var&>()), true)
{
  return true;
}

template <class Var>
__host__ __device__ constexpr auto has_swap_member_imp(long) -> bool
{
  return false;
}

template <class Var>
__host__ __device__ constexpr bool has_swap_member()
{
  return has_swap_member_imp<Var>(0);
}

__host__ __device__ void test_swap_sfinae()
{
  {
    // This variant type does not provide either a member or non-member swap
    // but is still swappable via the generic swap algorithm, since the
    // variant is move constructible and move assignable.
    using V = cuda::std::variant<int, NotSwappable>;
    static_assert(!has_swap_member<V>(), "");
    static_assert(cuda::std::is_swappable_v<V>, "");
  }
  {
    using V = cuda::std::variant<int, NotCopyable>;
    static_assert(!has_swap_member<V>(), "");
    static_assert(!cuda::std::is_swappable_v<V>, "");
  }
  {
    using V = cuda::std::variant<int, NotCopyableWithSwap>;
    static_assert(!has_swap_member<V>(), "");
    static_assert(!cuda::std::is_swappable_v<V>, "");
  }
  {
    using V = cuda::std::variant<int, NotMoveAssignable>;
    static_assert(!has_swap_member<V>(), "");
    static_assert(!cuda::std::is_swappable_v<V>, "");
  }
}

__host__ __device__ void test_swap_noexcept()
{
  {
    using V = cuda::std::variant<int, NothrowMoveable>;
    static_assert(cuda::std::is_swappable_v<V> && has_swap_member<V>(), "");
    static_assert(cuda::std::is_nothrow_swappable_v<V>, "");
    // instantiate swap
    V v1, v2;
    v1.swap(v2);
    swap(v1, v2);
  }
  {
    using V = cuda::std::variant<int, NothrowMoveCtor>;
    static_assert(cuda::std::is_swappable_v<V> && has_swap_member<V>(), "");
    static_assert(!cuda::std::is_nothrow_swappable_v<V>, "");
    // instantiate swap
    V v1, v2;
    v1.swap(v2);
    swap(v1, v2);
  }
  {
    using V = cuda::std::variant<int, ThrowingTypeWithNothrowSwap>;
    static_assert(cuda::std::is_swappable_v<V> && has_swap_member<V>(), "");
    static_assert(!cuda::std::is_nothrow_swappable_v<V>, "");
    // instantiate swap
    V v1, v2;
    v1.swap(v2);
    swap(v1, v2);
  }
  {
    using V = cuda::std::variant<int, ThrowingMoveAssignNothrowMoveCtor>;
    static_assert(cuda::std::is_swappable_v<V> && has_swap_member<V>(), "");
    static_assert(!cuda::std::is_nothrow_swappable_v<V>, "");
    // instantiate swap
    V v1, v2;
    v1.swap(v2);
    swap(v1, v2);
  }
  {
    using V = cuda::std::variant<int, ThrowingMoveAssignNothrowMoveCtorWithSwap>;
    static_assert(cuda::std::is_swappable_v<V> && has_swap_member<V>(), "");
    static_assert(cuda::std::is_nothrow_swappable_v<V>, "");
    // instantiate swap
    V v1, v2;
    v1.swap(v2);
    swap(v1, v2);
  }
  {
    using V = cuda::std::variant<int, NotMoveAssignableWithSwap>;
    static_assert(cuda::std::is_swappable_v<V> && has_swap_member<V>(), "");
    static_assert(cuda::std::is_nothrow_swappable_v<V>, "");
    // instantiate swap
    V v1, v2;
    v1.swap(v2);
    swap(v1, v2);
  }
  {
    // This variant type does not provide either a member or non-member swap
    // but is still swappable via the generic swap algorithm, since the
    // variant is move constructible and move assignable.
    using V = cuda::std::variant<int, NotSwappable>;
    static_assert(!has_swap_member<V>(), "");
    static_assert(cuda::std::is_swappable_v<V>, "");
    static_assert(cuda::std::is_nothrow_swappable_v<V>, "");
    V v1, v2;
    swap(v1, v2);
  }
}

#ifdef _CUDA_STD_VERSION
// This is why variant should SFINAE member swap. :-)
template class cuda::std::variant<int, NotSwappable>;
#endif // _CUDA_STD_VERSION

int main(int, char**)
{
  test_swap_same_alternative();
  test_swap_different_alternatives();
  test_swap_sfinae();
  test_swap_noexcept();

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_swap_valueless_by_exception();))
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions_same_alternative();))
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions_different_alternatives();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
