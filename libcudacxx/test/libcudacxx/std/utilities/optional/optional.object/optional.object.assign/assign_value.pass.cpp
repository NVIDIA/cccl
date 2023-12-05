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

// template <class U> optional<T>& operator=(U&& v);

#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>
#ifdef _LIBCUDACXX_HAS_MEMORY
#include <cuda/std/memory>
#else
#include "MoveOnly.h"
#endif

#include "test_macros.h"
#include "archetypes.h"

using cuda::std::optional;

struct ThrowAssign {
  STATIC_MEMBER_VAR(dtor_called, unsigned);
  ThrowAssign() = default;
  __host__ __device__
  ThrowAssign(int) { TEST_THROW(42); }
  __host__ __device__
  ThrowAssign& operator=(int) {
      TEST_THROW(42);
      return *this;
  }
  __host__ __device__
  ~ThrowAssign() { ++dtor_called(); }
};

template <class T, class Arg = T, bool Expect = true>
__host__ __device__
void assert_assignable() {
    static_assert(cuda::std::is_assignable<optional<T>&, Arg>::value == Expect, "");
    static_assert(!cuda::std::is_assignable<const optional<T>&, Arg>::value, "");
}

struct MismatchType {
  __host__ __device__
  explicit MismatchType(int) {}
  __host__ __device__
  explicit MismatchType(char*) {}
  explicit MismatchType(int*) = delete;
  __host__ __device__
  MismatchType& operator=(int) { return *this; }
  __host__ __device__
  MismatchType& operator=(int*) { return *this; }
  MismatchType& operator=(char*) = delete;
};

struct FromOptionalType {
  using Opt = cuda::std::optional<FromOptionalType>;
  FromOptionalType() = default;
  FromOptionalType(FromOptionalType const&) = delete;
  template <class Dummy = void>
  __host__ __device__
  constexpr FromOptionalType(Opt&) { Dummy::BARK; }
  template <class Dummy = void>
  __host__ __device__
  constexpr FromOptionalType& operator=(Opt&) { Dummy::BARK; return *this; }
};

__host__ __device__
void test_sfinae() {
    using I = TestTypes::TestType;
    using E = ExplicitTestTypes::TestType;
    assert_assignable<int>();
    assert_assignable<int, int&>();
    assert_assignable<int, int const&>();
    // Implicit test type
    assert_assignable<I, I const&>();
    assert_assignable<I, I&&>();
    assert_assignable<I, int>();
    assert_assignable<I, void*, false>();
    // Explicit test type
    assert_assignable<E, E const&>();
    assert_assignable<E, E &&>();
    assert_assignable<E, int>();
    assert_assignable<E, void*, false>();
    // Mismatch type
    assert_assignable<MismatchType, int>();
    assert_assignable<MismatchType, int*, false>();
    assert_assignable<MismatchType, char*, false>();
    // Type constructible from optional
    assert_assignable<FromOptionalType, cuda::std::optional<FromOptionalType>&, false>();
}

__host__ __device__
void test_with_test_type()
{
    using T = TestTypes::TestType;
    T::reset();
    { // to empty
        optional<T> opt;
        opt = 3;
        assert(T::alive() == 1);
        assert(T::constructed() == 1);
        assert(T::value_constructed() == 1);
        assert(T::assigned() == 0);
        assert(T::destroyed() == 0);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(3));
    }
    { // to existing
        optional<T> opt(42);
        T::reset_constructors();
        opt = 3;
        assert(T::alive() == 1);
        assert(T::constructed() == 0);
        assert(T::assigned() == 1);
        assert(T::value_assigned() == 1);
        assert(T::destroyed() == 0);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(3));
    }
    { // test default argument
        optional<T> opt;
        T::reset_constructors();
        opt = {1, 2};
        assert(T::alive() == 1);
        assert(T::constructed() == 2);
        assert(T::value_constructed() == 1);
        assert(T::move_constructed() == 1);
        assert(T::assigned() == 0);
        assert(T::destroyed() == 1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(1, 2));
    }
    { // test default argument
        optional<T> opt(42);
        T::reset_constructors();
        opt = {1, 2};
        assert(T::alive() == 1);
        assert(T::constructed() == 1);
        assert(T::value_constructed() == 1);
        assert(T::assigned() == 1);
        assert(T::move_assigned() == 1);
        assert(T::destroyed() == 1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(1, 2));
    }
    { // test default argument
        optional<T> opt;
        T::reset_constructors();
        opt = {1};
        assert(T::alive() == 1);
        assert(T::constructed() == 2);
        assert(T::value_constructed() == 1);
        assert(T::move_constructed() == 1);
        assert(T::assigned() == 0);
        assert(T::destroyed() == 1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(1));
    }
    { // test default argument
        optional<T> opt(42);
        T::reset_constructors();
        opt = {};
        assert(static_cast<bool>(opt) == false);
        assert(T::alive() == 0);
        assert(T::constructed() == 0);
        assert(T::assigned() == 0);
        assert(T::destroyed() == 1);
    }
}

template <class T, class Value = int>
__host__ __device__
void test_with_type() {
    { // to empty
        optional<T> opt;
        opt = Value(3);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(3));
    }
    { // to existing
        optional<T> opt(Value(42));
        opt = Value(3);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(3));
    }
    { // test const
        optional<T> opt(Value(42));
        const T t(Value(3));
        opt = t;
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(3));
    }
    { // test default argument
        optional<T> opt;
        opt = {Value(1)};
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(1));
    }
    { // test default argument
        optional<T> opt(Value(42));
        opt = {};
        assert(static_cast<bool>(opt) == false);
    }
}

template <class T>
__host__ __device__
void test_with_type_multi() {
    test_with_type<T>();
    { // test default argument
        optional<T> opt;
        opt = {1, 2};
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(1, 2));
    }
    { // test default argument
        optional<T> opt(42);
        opt = {1, 2};
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(1, 2));
    }
}

__host__ __device__
void test_throws()
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    using T = ThrowAssign;
    {
        optional<T> opt;
        try {
            opt = 42;
            assert(false);
        } catch (int) {}
        assert(static_cast<bool>(opt) == false);
    }
    assert(T::dtor_called() == 0);
    {
        T::dtor_called() = 0;
        optional<T> opt(cuda::std::in_place);
        try {
            opt = 42;
            assert(false);
        } catch (int) {}
        assert(static_cast<bool>(opt) == true);
        assert(T::dtor_called() == 0);
    }
    assert(T::dtor_called() == 1);
#endif
}

enum MyEnum { Zero, One, Two, Three, FortyTwo = 42 };

using Fn = void(*)();

// https://llvm.org/PR38638
template <class T>
__host__ __device__
constexpr T pr38638(T v)
{
  cuda::std::optional<T> o;
  o = v;
  return *o + 2;
}


int main(int, char**)
{
    test_sfinae();
    // Test with instrumented type
    test_with_test_type();
    // Test with various scalar types
    test_with_type<int>();
    test_with_type<MyEnum, MyEnum>();
    test_with_type<int, MyEnum>();
    test_with_type<Fn, Fn>();
    // Test types with multi argument constructors
    test_with_type_multi<ConstexprTestTypes::TestType>();
    test_with_type_multi<TrivialTestTypes::TestType>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    {
        optional<cuda::std::unique_ptr<int>> opt;
        opt = cuda::std::unique_ptr<int>(new int(3));
        assert(static_cast<bool>(opt) == true);
        assert(**opt == 3);
    }
    {
        optional<cuda::std::unique_ptr<int>> opt(cuda::std::unique_ptr<int>(new int(2)));
        opt = cuda::std::unique_ptr<int>(new int(3));
        assert(static_cast<bool>(opt) == true);
        assert(**opt == 3);
    }
#else
    {
        optional<MoveOnly> opt;
        opt = MoveOnly(3);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == 3);
    }
    {
        optional<MoveOnly> opt(MoveOnly(2));
        opt = MoveOnly(3);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == 3);
    }
#endif
    test_throws();

#if !defined(TEST_COMPILER_GCC) || __GNUC__ > 6
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
    static_assert(pr38638(3) == 5, "");
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
#endif // !defined(TEST_COMPILER_GCC) || __GNUC__ > 6

  return 0;
}
