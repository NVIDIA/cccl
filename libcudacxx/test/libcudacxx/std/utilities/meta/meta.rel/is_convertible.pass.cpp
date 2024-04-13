//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_convertible
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_is_convertible()
{
  static_assert((cuda::std::is_convertible<T, U>::value), "");
  static_assert((cuda::std::is_convertible<const T, U>::value), "");
  static_assert((cuda::std::is_convertible<T, const U>::value), "");
  static_assert((cuda::std::is_convertible<const T, const U>::value), "");
#if TEST_STD_VER > 2011
  static_assert((cuda::std::is_convertible_v<T, U>), "");
  static_assert((cuda::std::is_convertible_v<const T, U>), "");
  static_assert((cuda::std::is_convertible_v<T, const U>), "");
  static_assert((cuda::std::is_convertible_v<const T, const U>), "");
#endif
}

template <class T, class U>
__host__ __device__ void test_is_not_convertible()
{
  static_assert((!cuda::std::is_convertible<T, U>::value), "");
  static_assert((!cuda::std::is_convertible<const T, U>::value), "");
  static_assert((!cuda::std::is_convertible<T, const U>::value), "");
  static_assert((!cuda::std::is_convertible<const T, const U>::value), "");
#if TEST_STD_VER > 2011
  static_assert((!cuda::std::is_convertible_v<T, U>), "");
  static_assert((!cuda::std::is_convertible_v<const T, U>), "");
  static_assert((!cuda::std::is_convertible_v<T, const U>), "");
  static_assert((!cuda::std::is_convertible_v<const T, const U>), "");
#endif
}

typedef void Function();
typedef void ConstFunction() const;
typedef char Array[1];

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

int main(int, char**)
{
  // void
  test_is_convertible<void, void>();
  test_is_not_convertible<void, Function>();
  test_is_not_convertible<void, Function&>();
  test_is_not_convertible<void, Function*>();
  test_is_not_convertible<void, Array>();
  test_is_not_convertible<void, Array&>();
  test_is_not_convertible<void, char>();
  test_is_not_convertible<void, char&>();
  test_is_not_convertible<void, char*>();
  test_is_not_convertible<char, void>();

  // Function
  test_is_not_convertible<Function, void>();
  test_is_not_convertible<Function, Function>();
  test_is_convertible<Function, Function&>();
  test_is_convertible<Function, Function*>();
  test_is_convertible<Function, Function* const>();

  static_assert((cuda::std::is_convertible<Function, Function&&>::value), "");

  test_is_not_convertible<Function, Array>();
  test_is_not_convertible<Function, Array&>();
  test_is_not_convertible<Function, char>();
  test_is_not_convertible<Function, char&>();
  test_is_not_convertible<Function, char*>();

  // Function&
  test_is_not_convertible<Function&, void>();
  test_is_not_convertible<Function&, Function>();
  test_is_convertible<Function&, Function&>();

  test_is_convertible<Function&, Function*>();
  test_is_not_convertible<Function&, Array>();
  test_is_not_convertible<Function&, Array&>();
  test_is_not_convertible<Function&, char>();
  test_is_not_convertible<Function&, char&>();
  test_is_not_convertible<Function&, char*>();

  // Function*
  test_is_not_convertible<Function*, void>();
  test_is_not_convertible<Function*, Function>();
  test_is_not_convertible<Function*, Function&>();
  test_is_convertible<Function*, Function*>();

  test_is_not_convertible<Function*, Array>();
  test_is_not_convertible<Function*, Array&>();
  test_is_not_convertible<Function*, char>();
  test_is_not_convertible<Function*, char&>();
  test_is_not_convertible<Function*, char*>();

  // Non-referencable function type
  static_assert((!cuda::std::is_convertible<ConstFunction, Function>::value), "");
  static_assert((!cuda::std::is_convertible<ConstFunction, Function*>::value), "");
  static_assert((!cuda::std::is_convertible<ConstFunction, Function&>::value), "");
  static_assert((!cuda::std::is_convertible<ConstFunction, Function&&>::value), "");
  static_assert((!cuda::std::is_convertible<Function*, ConstFunction>::value), "");
  static_assert((!cuda::std::is_convertible<Function&, ConstFunction>::value), "");
  static_assert((!cuda::std::is_convertible<ConstFunction, ConstFunction>::value), "");
  static_assert((!cuda::std::is_convertible<ConstFunction, void>::value), "");

  // Array
  test_is_not_convertible<Array, void>();
  test_is_not_convertible<Array, Function>();
  test_is_not_convertible<Array, Function&>();
  test_is_not_convertible<Array, Function*>();
  test_is_not_convertible<Array, Array>();

#if !defined(TEST_COMPILER_MSVC_2017) && !defined(_LIBCUDACXX_USE_IS_CONVERTIBLE_FALLBACK)
  static_assert((!cuda::std::is_convertible<Array, Array&>::value), "");
#endif // !TEST_COMPILER_MSVC_2017 && !_LIBCUDACXX_USE_IS_CONVERTIBLE_FALLBACK
  static_assert((cuda::std::is_convertible<Array, const Array&>::value), "");
#ifndef TEST_COMPILER_MSVC
  // TODO: Unclear why this fails.
  static_assert((!cuda::std::is_convertible<Array, const volatile Array&>::value), "");
#endif

  static_assert((!cuda::std::is_convertible<const Array, Array&>::value), "");
  static_assert((cuda::std::is_convertible<const Array, const Array&>::value), "");
#if !defined(TEST_COMPILER_MSVC_2017) && !defined(_LIBCUDACXX_USE_IS_CONVERTIBLE_FALLBACK)
  static_assert((!cuda::std::is_convertible<Array, volatile Array&>::value), "");
#endif // !TEST_COMPILER_MSVC_2017 && !_LIBCUDACXX_USE_IS_CONVERTIBLE_FALLBACK

  static_assert((cuda::std::is_convertible<Array, Array&&>::value), "");
  static_assert((cuda::std::is_convertible<Array, const Array&&>::value), "");
#if !defined(TEST_COMPILER_NVRTC) && !defined(TEST_COMPILER_MSVC_2017)
  // No idea why this fails under NVRTC.
  // TODO: File a compiler bug
  static_assert((cuda::std::is_convertible<Array, volatile Array&&>::value), "");
#endif
  static_assert((cuda::std::is_convertible<Array, const volatile Array&&>::value), "");
  static_assert((cuda::std::is_convertible<const Array, const Array&&>::value), "");
#if !defined(TEST_COMPILER_MSVC_2017) && !defined(_LIBCUDACXX_USE_IS_CONVERTIBLE_FALLBACK)
  static_assert((!cuda::std::is_convertible<Array&, Array&&>::value), "");
  static_assert((!cuda::std::is_convertible<Array&&, Array&>::value), "");
#endif // !TEST_COMPILER_MSVC_2017 && !_LIBCUDACXX_USE_IS_CONVERTIBLE_FALLBACK

  test_is_not_convertible<Array, char>();
  test_is_not_convertible<Array, char&>();

  static_assert((cuda::std::is_convertible<Array, char*>::value), "");
  static_assert((cuda::std::is_convertible<Array, const char*>::value), "");
  static_assert((cuda::std::is_convertible<Array, char* const>::value), "");
  static_assert((cuda::std::is_convertible<Array, char* const volatile>::value), "");

  static_assert((!cuda::std::is_convertible<const Array, char*>::value), "");
  static_assert((cuda::std::is_convertible<const Array, const char*>::value), "");

  static_assert((!cuda::std::is_convertible<char[42][42], char*>::value), "");
  static_assert((!cuda::std::is_convertible<char[][1], char*>::value), "");

  // Array&
  test_is_not_convertible<Array&, void>();
  test_is_not_convertible<Array&, Function>();
  test_is_not_convertible<Array&, Function&>();
  test_is_not_convertible<Array&, Function*>();
  test_is_not_convertible<Array&, Array>();

  static_assert((cuda::std::is_convertible<Array&, Array&>::value), "");
  static_assert((cuda::std::is_convertible<Array&, const Array&>::value), "");
  static_assert((!cuda::std::is_convertible<const Array&, Array&>::value), "");
  static_assert((cuda::std::is_convertible<const Array&, const Array&>::value), "");

  test_is_not_convertible<Array&, char>();
  test_is_not_convertible<Array&, char&>();

  static_assert((cuda::std::is_convertible<Array&, char*>::value), "");
  static_assert((cuda::std::is_convertible<Array&, const char*>::value), "");
  static_assert((!cuda::std::is_convertible<const Array&, char*>::value), "");
  static_assert((cuda::std::is_convertible<const Array&, const char*>::value), "");

  static_assert((cuda::std::is_convertible<Array, StringType>::value), "");
#if !defined(TEST_COMPILER_MSVC) && !defined(TEST_COMPILER_NVRTC)
  // TODO: Investigate why this is failing.
  static_assert((cuda::std::is_convertible<char(&)[], StringType>::value), "");
#endif

  // char
  test_is_not_convertible<char, void>();
  test_is_not_convertible<char, Function>();
  test_is_not_convertible<char, Function&>();
  test_is_not_convertible<char, Function*>();
  test_is_not_convertible<char, Array>();
  test_is_not_convertible<char, Array&>();

  test_is_convertible<char, char>();

  static_assert((!cuda::std::is_convertible<char, char&>::value), "");
  static_assert((cuda::std::is_convertible<char, const char&>::value), "");
  static_assert((!cuda::std::is_convertible<const char, char&>::value), "");
  static_assert((cuda::std::is_convertible<const char, const char&>::value), "");

  test_is_not_convertible<char, char*>();

  // char&
  test_is_not_convertible<char&, void>();
  test_is_not_convertible<char&, Function>();
  test_is_not_convertible<char&, Function&>();
  test_is_not_convertible<char&, Function*>();
  test_is_not_convertible<char&, Array>();
  test_is_not_convertible<char&, Array&>();

  test_is_convertible<char&, char>();

  static_assert((cuda::std::is_convertible<char&, char&>::value), "");
  static_assert((cuda::std::is_convertible<char&, const char&>::value), "");
  static_assert((!cuda::std::is_convertible<const char&, char&>::value), "");
  static_assert((cuda::std::is_convertible<const char&, const char&>::value), "");

  test_is_not_convertible<char&, char*>();

  // char*
  test_is_not_convertible<char*, void>();
  test_is_not_convertible<char*, Function>();
  test_is_not_convertible<char*, Function&>();
  test_is_not_convertible<char*, Function*>();
  test_is_not_convertible<char*, Array>();
  test_is_not_convertible<char*, Array&>();

  test_is_not_convertible<char*, char>();
  test_is_not_convertible<char*, char&>();

  static_assert((cuda::std::is_convertible<char*, char*>::value), "");
  static_assert((cuda::std::is_convertible<char*, const char*>::value), "");
  static_assert((!cuda::std::is_convertible<const char*, char*>::value), "");
  static_assert((cuda::std::is_convertible<const char*, const char*>::value), "");

  // NonCopyable
  static_assert((cuda::std::is_convertible<NonCopyable&, NonCopyable&>::value), "");
  static_assert((cuda::std::is_convertible<NonCopyable&, const NonCopyable&>::value), "");
  static_assert((cuda::std::is_convertible<NonCopyable&, const volatile NonCopyable&>::value), "");
  static_assert((cuda::std::is_convertible<NonCopyable&, volatile NonCopyable&>::value), "");
  static_assert((cuda::std::is_convertible<const NonCopyable&, const NonCopyable&>::value), "");
  static_assert((cuda::std::is_convertible<const NonCopyable&, const volatile NonCopyable&>::value), "");
  static_assert((cuda::std::is_convertible<volatile NonCopyable&, const volatile NonCopyable&>::value), "");
  static_assert((cuda::std::is_convertible<const volatile NonCopyable&, const volatile NonCopyable&>::value), "");
  static_assert((!cuda::std::is_convertible<const NonCopyable&, NonCopyable&>::value), "");
  // This test requires access control SFINAE which we only have on non-MSVC
  // compilers or when we are using the compiler builtin for
  // is_convertible.
#if !defined(TEST_COMPILER_MSVC) || !defined(_LIBCUDACXX_USE_IS_CONVERTIBLE_FALLBACK)
  test_is_not_convertible<NonCopyable&, NonCopyable>();
#endif

  // Ensure that CannotInstantiate is not instantiated by is_convertible when it is not needed.
  // For example CannotInstantiate is instatiated as a part of ADL lookup for arguments of type CannotInstantiate*.
  static_assert((cuda::std::is_convertible<CannotInstantiate<int>*, CannotInstantiate<int>*>::value), "");

  return 0;
}
