//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class From, class To>
// concept convertible_to;

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::convertible_to;

namespace
{
enum ClassicEnum
{
  a,
  b
};
enum class ScopedEnum
{
  x,
  y
};
struct Empty
{};
using nullptr_t = decltype(nullptr);

template <class T, class U>
__host__ __device__ void CheckConvertibleTo()
{
  static_assert(convertible_to<T, U>, "");
  static_assert(convertible_to<const T, U>, "");
  static_assert(convertible_to<T, const U>, "");
  static_assert(convertible_to<const T, const U>, "");
}

template <class T, class U>
__host__ __device__ void CheckNotConvertibleTo()
{
  static_assert(!convertible_to<T, U>, "");
  static_assert(!convertible_to<const T, U>, "");
  static_assert(!convertible_to<T, const U>, "");
  static_assert(!convertible_to<const T, const U>, "");
}

template <class T, class U>
__host__ __device__ void CheckIsConvertibleButNotConvertibleTo()
{
  // Sanity check T is either implicitly xor explicitly convertible to U.
  static_assert(cuda::std::is_convertible_v<T, U>, "");
  static_assert(cuda::std::is_convertible_v<const T, U>, "");
  static_assert(cuda::std::is_convertible_v<T, const U>, "");
  static_assert(cuda::std::is_convertible_v<const T, const U>, "");
  CheckNotConvertibleTo<T, U>();
}

// Tests that should objectively return false (except for bool and nullptr_t)
#if TEST_STD_VER > 2017
template <class T>
#else
_LIBCUDACXX_TEMPLATE(class T)
_LIBCUDACXX_REQUIRES((!(cuda::std::same_as<T, bool> || cuda::std::same_as<T, nullptr_t>) ))
#endif
__host__ __device__ constexpr void CommonlyNotConvertibleTo()
{
  CheckNotConvertibleTo<T, void>();
  CheckNotConvertibleTo<T, nullptr_t>();
  CheckNotConvertibleTo<T, T*>();
  CheckNotConvertibleTo<T, T Empty::*>();
  CheckNotConvertibleTo<T, T (Empty::*)()>();
  CheckNotConvertibleTo<T, T[sizeof(T)]>();
  CheckNotConvertibleTo<T, T (*)()>();
  CheckNotConvertibleTo<T, T (&)()>();
  CheckNotConvertibleTo<T, T (&&)()>();
}

_LIBCUDACXX_TEMPLATE(class T)
_LIBCUDACXX_REQUIRES(cuda::std::same_as<T, bool>)
__host__ __device__ constexpr void CommonlyNotConvertibleTo()
{
  CheckNotConvertibleTo<bool, void>();
  CheckNotConvertibleTo<bool, nullptr_t>();
  CheckConvertibleTo<bool Empty::*, bool>();
  CheckConvertibleTo<bool (Empty::*)(), bool>();
  CheckConvertibleTo<bool[2], bool>();
  CheckConvertibleTo<bool (*)(), bool>();
  CheckConvertibleTo<bool (&)(), bool>();
  CheckConvertibleTo<bool (&&)(), bool>();
}

_LIBCUDACXX_TEMPLATE(class T)
_LIBCUDACXX_REQUIRES(cuda::std::same_as<T, nullptr_t>)
__host__ __device__ constexpr void CommonlyNotConvertibleTo()
{
  CheckNotConvertibleTo<nullptr_t, void>();
  CheckConvertibleTo<nullptr_t, nullptr_t>();
  CheckConvertibleTo<nullptr_t, void*>();
  CheckConvertibleTo<nullptr_t, int Empty::*>();
  CheckConvertibleTo<nullptr_t, void (Empty::*)()>();
  CheckNotConvertibleTo<nullptr_t, int[2]>();
  CheckConvertibleTo<nullptr_t, void (*)()>();
  CheckNotConvertibleTo<nullptr_t, void (&)()>();
  CheckNotConvertibleTo<nullptr_t, void (&&)()>();
}
} // namespace

using Function = void();
#if TEST_STD_VER > 2014
using NoexceptFunction = void() noexcept;
#endif
using ConstFunction = void() const;
using Array         = char[1];

struct StringType
{
  __host__ __device__ StringType(const char*) {}
};

class NonCopyable
{
  __host__ __device__ NonCopyable(NonCopyable&);
};

template <typename T>
class CannotInstantiate
{
  enum
  {
    X = T::ThisExpressionWillBlowUp
  };
};

struct abstract
{
  __host__ __device__ virtual int f() = 0;
};

struct ExplicitlyConvertible;
struct ImplicitlyConvertible;

struct ExplicitlyConstructible
{
  __host__ __device__ explicit ExplicitlyConstructible(int);
  __host__ __device__ explicit ExplicitlyConstructible(ExplicitlyConvertible);
  explicit ExplicitlyConstructible(ImplicitlyConvertible) = delete;
};

struct ExplicitlyConvertible
{
  __host__ __device__ explicit operator ExplicitlyConstructible() const
  {
    return ExplicitlyConstructible(0);
  }
};

struct ImplicitlyConstructible;

struct ImplicitlyConvertible
{
  __host__ __device__ operator ExplicitlyConstructible() const;
  operator ImplicitlyConstructible() const = delete;
};

struct ImplicitlyConstructible
{
  __host__ __device__ ImplicitlyConstructible(ImplicitlyConvertible);
};

int main(int, char**)
{
  // void
  CheckConvertibleTo<void, void>();
  CheckNotConvertibleTo<void, Function>();
  CheckNotConvertibleTo<void, Function&>();
  CheckNotConvertibleTo<void, Function*>();
#if TEST_STD_VER > 2014
  CheckNotConvertibleTo<void, NoexceptFunction>();
  CheckNotConvertibleTo<void, NoexceptFunction&>();
  CheckNotConvertibleTo<void, NoexceptFunction*>();
#endif // TEST_STD_VER > 2014
  CheckNotConvertibleTo<void, Array>();
  CheckNotConvertibleTo<void, Array&>();
  CheckNotConvertibleTo<void, char>();
  CheckNotConvertibleTo<void, char&>();
  CheckNotConvertibleTo<void, char*>();
  CheckNotConvertibleTo<char, void>();

  // Function
  CheckNotConvertibleTo<Function, void>();
  CheckNotConvertibleTo<Function, Function>();
  // CheckNotConvertibleTo<Function, NoexceptFunction>();
  // CheckNotConvertibleTo<Function, NoexceptFunction&>();
  // CheckNotConvertibleTo<Function, NoexceptFunction*>();
  // CheckNotConvertibleTo<Function, NoexceptFunction* const>();
  CheckConvertibleTo<Function, Function&>();
  CheckConvertibleTo<Function, Function*>();
  CheckConvertibleTo<Function, Function* const>();

  static_assert(convertible_to<Function, Function&&>, "");
#if TEST_STD_VER > 2014
  // static_assert(!convertible_to<Function, NoexceptFunction&&>, "");
#endif

  CheckNotConvertibleTo<Function, Array>();
  CheckNotConvertibleTo<Function, Array&>();
  CheckNotConvertibleTo<Function, char>();
  CheckNotConvertibleTo<Function, char&>();
  CheckNotConvertibleTo<Function, char*>();

  // Function&
  CheckNotConvertibleTo<Function&, void>();
  CheckNotConvertibleTo<Function&, Function>();
  CheckConvertibleTo<Function&, Function&>();

  CheckConvertibleTo<Function&, Function*>();
  CheckNotConvertibleTo<Function&, Array>();
  CheckNotConvertibleTo<Function&, Array&>();
  CheckNotConvertibleTo<Function&, char>();
  CheckNotConvertibleTo<Function&, char&>();
  CheckNotConvertibleTo<Function&, char*>();

  // Function*
  CheckNotConvertibleTo<Function*, void>();
  CheckNotConvertibleTo<Function*, Function>();
  CheckNotConvertibleTo<Function*, Function&>();
  CheckConvertibleTo<Function*, Function*>();

  CheckNotConvertibleTo<Function*, Array>();
  CheckNotConvertibleTo<Function*, Array&>();
  CheckNotConvertibleTo<Function*, char>();
  CheckNotConvertibleTo<Function*, char&>();
  CheckNotConvertibleTo<Function*, char*>();

  // Non-referencable function type
  static_assert(!convertible_to<ConstFunction, Function>, "");
  static_assert(!convertible_to<ConstFunction, Function*>, "");
  static_assert(!convertible_to<ConstFunction, Function&>, "");
  static_assert(!convertible_to<ConstFunction, Function&&>, "");
  static_assert(!convertible_to<Function*, ConstFunction>, "");
  static_assert(!convertible_to<Function&, ConstFunction>, "");
  static_assert(!convertible_to<ConstFunction, ConstFunction>, "");
  static_assert(!convertible_to<ConstFunction, void>, "");

#if TEST_STD_VER > 2014
  // NoexceptFunction
  CheckNotConvertibleTo<NoexceptFunction, void>();
  CheckNotConvertibleTo<NoexceptFunction, Function>();
  CheckNotConvertibleTo<NoexceptFunction, NoexceptFunction>();
  CheckConvertibleTo<NoexceptFunction, NoexceptFunction&>();
  CheckConvertibleTo<NoexceptFunction, NoexceptFunction*>();
  CheckConvertibleTo<NoexceptFunction, NoexceptFunction* const>();
#  ifndef TEST_COMPILER_MSVC_2017
  CheckConvertibleTo<NoexceptFunction, Function&>();
#  endif // !TEST_COMPILER_MSVC_2017
  CheckConvertibleTo<NoexceptFunction, Function*>();
  CheckConvertibleTo<NoexceptFunction, Function* const>();

#  ifndef TEST_COMPILER_MSVC_2017
  static_assert(convertible_to<NoexceptFunction, Function&&>, "");
#  endif // !TEST_COMPILER_MSVC_2017
  static_assert(convertible_to<NoexceptFunction, NoexceptFunction&&>, "");

  CheckNotConvertibleTo<NoexceptFunction, Array>();
  CheckNotConvertibleTo<NoexceptFunction, Array&>();
  CheckNotConvertibleTo<NoexceptFunction, char>();
  CheckNotConvertibleTo<NoexceptFunction, char&>();
  CheckNotConvertibleTo<NoexceptFunction, char*>();

  // NoexceptFunction&
  CheckNotConvertibleTo<NoexceptFunction&, void>();
#  ifndef TEST_COMPILER_MSVC_2017
  CheckNotConvertibleTo<NoexceptFunction&, Function>();
#  endif // !TEST_COMPILER_MSVC_2017
  CheckNotConvertibleTo<NoexceptFunction&, NoexceptFunction>();
#  ifndef TEST_COMPILER_MSVC_2017
  CheckConvertibleTo<NoexceptFunction&, Function&>();
#  endif // !TEST_COMPILER_MSVC_2017
  CheckConvertibleTo<NoexceptFunction&, NoexceptFunction&>();

  CheckConvertibleTo<NoexceptFunction&, Function*>();
  CheckConvertibleTo<NoexceptFunction&, NoexceptFunction*>();
  CheckNotConvertibleTo<NoexceptFunction&, Array>();
  CheckNotConvertibleTo<NoexceptFunction&, Array&>();
  CheckNotConvertibleTo<NoexceptFunction&, char>();
  CheckNotConvertibleTo<NoexceptFunction&, char&>();
  CheckNotConvertibleTo<NoexceptFunction&, char*>();

  // NoexceptFunction*
  CheckNotConvertibleTo<NoexceptFunction*, void>();
  CheckNotConvertibleTo<NoexceptFunction*, Function>();
  CheckNotConvertibleTo<NoexceptFunction*, Function&>();
  CheckNotConvertibleTo<NoexceptFunction*, NoexceptFunction>();
  CheckNotConvertibleTo<NoexceptFunction*, NoexceptFunction&>();
  CheckConvertibleTo<NoexceptFunction*, Function*>();
  CheckConvertibleTo<NoexceptFunction*, NoexceptFunction*>();

  CheckNotConvertibleTo<NoexceptFunction*, Array>();
  CheckNotConvertibleTo<NoexceptFunction*, Array&>();
  CheckNotConvertibleTo<NoexceptFunction*, char>();
  CheckNotConvertibleTo<NoexceptFunction*, char&>();
  CheckNotConvertibleTo<NoexceptFunction*, char*>();
#endif // TEST_STD_VER > 2014

  // Array
  CheckNotConvertibleTo<Array, void>();
  CheckNotConvertibleTo<Array, Function>();
  CheckNotConvertibleTo<Array, Function&>();
  CheckNotConvertibleTo<Array, Function*>();
#if TEST_STD_VER > 2014
  CheckNotConvertibleTo<Array, NoexceptFunction>();
  CheckNotConvertibleTo<Array, NoexceptFunction&>();
  CheckNotConvertibleTo<Array, NoexceptFunction*>();
#endif // TEST_STD_VER > 2014
  CheckNotConvertibleTo<Array, Array>();

  static_assert(!convertible_to<Array, Array&>, "");
  static_assert(convertible_to<Array, const Array&>, "");

  static_assert(!convertible_to<const Array, Array&>, "");
  static_assert(convertible_to<const Array, const Array&>, "");
#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC has a bug where lets the conversion happen
  static_assert(!convertible_to<Array, volatile Array&>, "");
  static_assert(!convertible_to<Array, const volatile Array&>, "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017

  static_assert(convertible_to<Array, Array&&>, "");
  static_assert(convertible_to<Array, const Array&&>, "");
  static_assert(convertible_to<Array, volatile Array&&>, "");
  static_assert(convertible_to<Array, const volatile Array&&>, "");
  static_assert(convertible_to<const Array, const Array&&>, "");
  static_assert(!convertible_to<Array&, Array&&>, "");
  static_assert(!convertible_to<Array&&, Array&>, "");

  CheckNotConvertibleTo<Array, char>();
  CheckNotConvertibleTo<Array, char&>();

  static_assert(convertible_to<Array, char*>, "");
  static_assert(convertible_to<Array, const char*>, "");
  static_assert(convertible_to<Array, char* const>, "");
  static_assert(convertible_to<Array, char* const volatile>, "");

  static_assert(!convertible_to<const Array, char*>, "");
  static_assert(convertible_to<const Array, const char*>, "");

  static_assert(!convertible_to<char[42][42], char*>, "");
  static_assert(!convertible_to<char[][1], char*>, "");

  // Array&
  CheckNotConvertibleTo<Array&, void>();
  CheckNotConvertibleTo<Array&, Function>();
  CheckNotConvertibleTo<Array&, Function&>();
  CheckNotConvertibleTo<Array&, Function*>();
#if TEST_STD_VER > 2014
  CheckNotConvertibleTo<Array&, NoexceptFunction>();
  CheckNotConvertibleTo<Array&, NoexceptFunction&>();
  CheckNotConvertibleTo<Array&, NoexceptFunction*>();
#endif // TEST_STD_VER > 2014
  CheckNotConvertibleTo<Array&, Array>();

  static_assert(convertible_to<Array&, Array&>, "");
  static_assert(convertible_to<Array&, const Array&>, "");
  static_assert(!convertible_to<const Array&, Array&>, "");
  static_assert(convertible_to<const Array&, const Array&>, "");

  CheckNotConvertibleTo<Array&, char>();
  CheckNotConvertibleTo<Array&, char&>();

  static_assert(convertible_to<Array&, char*>, "");
  static_assert(convertible_to<Array&, const char*>, "");
  static_assert(!convertible_to<const Array&, char*>, "");
  static_assert(convertible_to<const Array&, const char*>, "");

  static_assert(convertible_to<Array, StringType>, "");
  static_assert(convertible_to<char(&)[], StringType>, "");

  // char
  CheckNotConvertibleTo<char, void>();
  CheckNotConvertibleTo<char, Function>();
  CheckNotConvertibleTo<char, Function&>();
  CheckNotConvertibleTo<char, Function*>();
#if TEST_STD_VER > 2014
  CheckNotConvertibleTo<char, NoexceptFunction>();
  CheckNotConvertibleTo<char, NoexceptFunction&>();
  CheckNotConvertibleTo<char, NoexceptFunction*>();
#endif // TEST_STD_VER > 2014
  CheckNotConvertibleTo<char, Array>();
  CheckNotConvertibleTo<char, Array&>();

  CheckConvertibleTo<char, char>();

  static_assert(!convertible_to<char, char&>, "");
  static_assert(convertible_to<char, const char&>, "");
  static_assert(!convertible_to<const char, char&>, "");
  static_assert(convertible_to<const char, const char&>, "");

  CheckNotConvertibleTo<char, char*>();

  // char&
  CheckNotConvertibleTo<char&, void>();
  CheckNotConvertibleTo<char&, Function>();
  CheckNotConvertibleTo<char&, Function&>();
  CheckNotConvertibleTo<char&, Function*>();
#if TEST_STD_VER > 2014
  CheckNotConvertibleTo<char&, NoexceptFunction>();
  CheckNotConvertibleTo<char&, NoexceptFunction&>();
  CheckNotConvertibleTo<char&, NoexceptFunction*>();
#endif // TEST_STD_VER > 2014
  CheckNotConvertibleTo<char&, Array>();
  CheckNotConvertibleTo<char&, Array&>();

  CheckConvertibleTo<char&, char>();

  static_assert(convertible_to<char&, char&>, "");
  static_assert(convertible_to<char&, const char&>, "");
  static_assert(!convertible_to<const char&, char&>, "");
  static_assert(convertible_to<const char&, const char&>, "");

  CheckNotConvertibleTo<char&, char*>();

  // char*
  CheckNotConvertibleTo<char*, void>();
  CheckNotConvertibleTo<char*, Function>();
  CheckNotConvertibleTo<char*, Function&>();
  CheckNotConvertibleTo<char*, Function*>();
#if TEST_STD_VER > 2014
  CheckNotConvertibleTo<char*, NoexceptFunction>();
  CheckNotConvertibleTo<char*, NoexceptFunction&>();
  CheckNotConvertibleTo<char*, NoexceptFunction*>();
#endif // TEST_STD_VER > 2014
  CheckNotConvertibleTo<char*, Array>();
  CheckNotConvertibleTo<char*, Array&>();

  CheckNotConvertibleTo<char*, char>();
  CheckNotConvertibleTo<char*, char&>();

  static_assert(convertible_to<char*, char*>, "");
  static_assert(convertible_to<char*, const char*>, "");
  static_assert(!convertible_to<const char*, char*>, "");
  static_assert(convertible_to<const char*, const char*>, "");

  // NonCopyable
  static_assert(convertible_to<NonCopyable&, NonCopyable&>, "");
  static_assert(convertible_to<NonCopyable&, const NonCopyable&>, "");
  static_assert(convertible_to<NonCopyable&, const volatile NonCopyable&>, "");
  static_assert(convertible_to<NonCopyable&, volatile NonCopyable&>, "");
  static_assert(convertible_to<const NonCopyable&, const NonCopyable&>, "");
  static_assert(convertible_to<const NonCopyable&, const volatile NonCopyable&>, "");
  static_assert(convertible_to<volatile NonCopyable&, const volatile NonCopyable&>, "");
  static_assert(convertible_to<const volatile NonCopyable&, const volatile NonCopyable&>, "");
  static_assert(!convertible_to<const NonCopyable&, NonCopyable&>, "");

  // This test requires Access control SFINAE which we only have in C++11 or when
  // we are using the compiler builtin for convertible_to.
  CheckNotConvertibleTo<NonCopyable&, NonCopyable>();

  // Ensure that CannotInstantiate is not instantiated by convertible_to when it is not needed.
  // For example CannotInstantiate is instantiated as a part of ADL lookup for arguments of type CannotInstantiate*.
  // static_assert(
  //    convertible_to<CannotInstantiate<int>*, CannotInstantiate<int>*>, "");

  // Test for PR13592
  static_assert(!convertible_to<abstract, abstract>, "");

  CommonlyNotConvertibleTo<int>();
  CommonlyNotConvertibleTo<bool>();
  CommonlyNotConvertibleTo<nullptr_t>();

  CheckNotConvertibleTo<int, ExplicitlyConstructible>();
  CheckNotConvertibleTo<ExplicitlyConvertible, ExplicitlyConstructible>();
  CheckNotConvertibleTo<ExplicitlyConstructible, ExplicitlyConvertible>();
  CheckIsConvertibleButNotConvertibleTo<ImplicitlyConvertible, ExplicitlyConstructible>();
  CheckNotConvertibleTo<ImplicitlyConstructible, ImplicitlyConvertible>();

  return 0;
}
