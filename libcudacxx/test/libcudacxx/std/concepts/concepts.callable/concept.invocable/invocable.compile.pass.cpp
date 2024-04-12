//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class T, class U>
// concept invocable;

#include <cuda/std/chrono>
#include <cuda/std/concepts>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::invocable;

template <class R, class... Args>
__host__ __device__ constexpr bool check_invocable()
{
  constexpr bool result = invocable<R(Args...), Args...>;
  static_assert(invocable<R(Args...) noexcept, Args...> == result, "");
  static_assert(invocable<R (*)(Args...), Args...> == result, "");
  static_assert(invocable<R (*)(Args...) noexcept, Args...> == result, "");
  static_assert(invocable<R (&)(Args...), Args...> == result, "");
  static_assert(invocable<R (&)(Args...) noexcept, Args...> == result, "");

  return result;
}

static_assert(check_invocable<void>(), "");
static_assert(check_invocable<void, int>(), "");
static_assert(check_invocable<void, int&>(), "");
static_assert(check_invocable<void, int*, double>(), "");
static_assert(check_invocable<int>(), "");
static_assert(check_invocable<int, int[]>(), "");

struct S;
static_assert(check_invocable<int, int S::*, cuda::std::nullptr_t>(), "");
static_assert(check_invocable<int, int (S::*)(), int (S::*)(int), int>(), "");
static_assert(invocable<void (*)(int const&), int&>, "");
static_assert(invocable<void (*)(int const&), int&&>, "");
static_assert(invocable<void (*)(int volatile&), int&>, "");
static_assert(invocable<void (*)(int const volatile&), int&>, "");

static_assert(!invocable<void(), int>, "");
static_assert(!invocable<void(int)>, "");
static_assert(!invocable<void(int*), double*>, "");
static_assert(!invocable<void (*)(int&), double*>, "");
static_assert(!invocable<void (*)(int&&), int&>, "");
static_assert(!invocable<void (*)(int&&), int const&>, "");

static_assert(!invocable<void>, "");
static_assert(!invocable<void*>, "");
static_assert(!invocable<int>, "");
static_assert(!invocable<int&>, "");
static_assert(!invocable<int&&>, "");

namespace function_objects
{
struct function_object
{
  __host__ __device__ void operator()();
};
static_assert(invocable<function_object>, "");
static_assert(!invocable<function_object const>, "");
static_assert(!invocable<function_object volatile>, "");
static_assert(!invocable<function_object const volatile>, "");
static_assert(invocable<function_object&>, "");
static_assert(!invocable<function_object const&>, "");
static_assert(!invocable<function_object volatile&>, "");
static_assert(!invocable<function_object const volatile&>, "");

struct const_function_object
{
  __host__ __device__ void operator()(int) const;
};
static_assert(invocable<const_function_object, int>, "");
static_assert(invocable<const_function_object const, int>, "");
static_assert(!invocable<const_function_object volatile, int>, "");
static_assert(!invocable<const_function_object const volatile, int>, "");
static_assert(invocable<const_function_object&, int>, "");
static_assert(invocable<const_function_object const&, int>, "");
static_assert(!invocable<const_function_object volatile&, int>, "");
static_assert(!invocable<const_function_object const volatile&, int>, "");

struct volatile_function_object
{
  __host__ __device__ void operator()(int, int) volatile;
};
static_assert(invocable<volatile_function_object, int, int>, "");
static_assert(!invocable<volatile_function_object const, int, int>, "");
static_assert(invocable<volatile_function_object volatile, int, int>, "");
static_assert(!invocable<volatile_function_object const volatile, int, int>, "");
static_assert(invocable<volatile_function_object&, int, int>, "");
static_assert(!invocable<volatile_function_object const&, int, int>, "");
static_assert(invocable<volatile_function_object volatile&, int, int>, "");
static_assert(!invocable<volatile_function_object const volatile&, int, int>, "");

struct cv_function_object
{
  __host__ __device__ void operator()(int[]) const volatile;
};
static_assert(invocable<cv_function_object, int*>, "");
static_assert(invocable<cv_function_object const, int*>, "");
static_assert(invocable<cv_function_object volatile, int*>, "");
static_assert(invocable<cv_function_object const volatile, int*>, "");
static_assert(invocable<cv_function_object&, int*>, "");
static_assert(invocable<cv_function_object const&, int*>, "");
static_assert(invocable<cv_function_object volatile&, int*>, "");
static_assert(invocable<cv_function_object const volatile&, int*>, "");

struct lvalue_function_object
{
  __host__ __device__ void operator()() &;
};
static_assert(!invocable<lvalue_function_object>, "");
static_assert(!invocable<lvalue_function_object const>, "");
static_assert(!invocable<lvalue_function_object volatile>, "");
static_assert(!invocable<lvalue_function_object const volatile>, "");
static_assert(invocable<lvalue_function_object&>, "");
static_assert(!invocable<lvalue_function_object const&>, "");
static_assert(!invocable<lvalue_function_object volatile&>, "");
static_assert(!invocable<lvalue_function_object const volatile&>, "");

struct lvalue_const_function_object
{
  __host__ __device__ void operator()(int) const&;
};
static_assert(invocable<lvalue_const_function_object, int>, "");
static_assert(invocable<lvalue_const_function_object const, int>, "");
static_assert(!invocable<lvalue_const_function_object volatile, int>, "");
static_assert(!invocable<lvalue_const_function_object const volatile, int>, "");
static_assert(invocable<lvalue_const_function_object&, int>, "");
static_assert(invocable<lvalue_const_function_object const&, int>, "");
static_assert(!invocable<lvalue_const_function_object volatile&, int>, "");
static_assert(!invocable<lvalue_const_function_object const volatile&, int>, "");

struct lvalue_volatile_function_object
{
  __host__ __device__ void operator()(int, int) volatile&;
};
static_assert(!invocable<lvalue_volatile_function_object, int, int>, "");
static_assert(!invocable<lvalue_volatile_function_object const, int, int>, "");
static_assert(!invocable<lvalue_volatile_function_object volatile, int, int>, "");
static_assert(!invocable<lvalue_volatile_function_object const volatile, int, int>, "");
static_assert(invocable<lvalue_volatile_function_object&, int, int>, "");
static_assert(!invocable<lvalue_volatile_function_object const&, int, int>, "");
static_assert(invocable<lvalue_volatile_function_object volatile&, int, int>, "");
static_assert(!invocable<lvalue_volatile_function_object const volatile&, int, int>, "");

struct lvalue_cv_function_object
{
  __host__ __device__ void operator()(int[]) const volatile&;
};
static_assert(!invocable<lvalue_cv_function_object, int*>, "");
static_assert(!invocable<lvalue_cv_function_object const, int*>, "");
static_assert(!invocable<lvalue_cv_function_object volatile, int*>, "");
static_assert(!invocable<lvalue_cv_function_object const volatile, int*>, "");
static_assert(invocable<lvalue_cv_function_object&, int*>, "");
static_assert(invocable<lvalue_cv_function_object const&, int*>, "");
static_assert(invocable<lvalue_cv_function_object volatile&, int*>, "");
static_assert(invocable<lvalue_cv_function_object const volatile&, int*>, "");
//
struct rvalue_function_object
{
  __host__ __device__ void operator()() &&;
};
static_assert(invocable<rvalue_function_object>, "");
static_assert(!invocable<rvalue_function_object const>, "");
static_assert(!invocable<rvalue_function_object volatile>, "");
static_assert(!invocable<rvalue_function_object const volatile>, "");
static_assert(!invocable<rvalue_function_object&>, "");
static_assert(!invocable<rvalue_function_object const&>, "");
static_assert(!invocable<rvalue_function_object volatile&>, "");
static_assert(!invocable<rvalue_function_object const volatile&>, "");

struct rvalue_const_function_object
{
  __host__ __device__ void operator()(int) const&&;
};
static_assert(invocable<rvalue_const_function_object, int>, "");
static_assert(invocable<rvalue_const_function_object const, int>, "");
static_assert(!invocable<rvalue_const_function_object volatile, int>, "");
static_assert(!invocable<rvalue_const_function_object const volatile, int>, "");
static_assert(!invocable<rvalue_const_function_object&, int>, "");
static_assert(!invocable<rvalue_const_function_object const&, int>, "");
static_assert(!invocable<rvalue_const_function_object volatile&, int>, "");
static_assert(!invocable<rvalue_const_function_object const volatile&, int>, "");

struct rvalue_volatile_function_object
{
  __host__ __device__ void operator()(int, int) volatile&&;
};
static_assert(invocable<rvalue_volatile_function_object, int, int>, "");
static_assert(!invocable<rvalue_volatile_function_object const, int, int>, "");
static_assert(invocable<rvalue_volatile_function_object volatile, int, int>, "");
static_assert(!invocable<rvalue_volatile_function_object const volatile, int, int>, "");
static_assert(!invocable<rvalue_volatile_function_object&, int, int>, "");
static_assert(!invocable<rvalue_volatile_function_object const&, int, int>, "");
static_assert(!invocable<rvalue_volatile_function_object volatile&, int, int>, "");
static_assert(!invocable<rvalue_volatile_function_object const volatile&, int, int>, "");

struct rvalue_cv_function_object
{
  __host__ __device__ void operator()(int[]) const volatile&&;
};
static_assert(invocable<rvalue_cv_function_object, int*>, "");
static_assert(invocable<rvalue_cv_function_object const, int*>, "");
static_assert(invocable<rvalue_cv_function_object volatile, int*>, "");
static_assert(invocable<rvalue_cv_function_object const volatile, int*>, "");
static_assert(!invocable<rvalue_cv_function_object&, int*>, "");
static_assert(!invocable<rvalue_cv_function_object const&, int*>, "");
static_assert(!invocable<rvalue_cv_function_object volatile&, int*>, "");
static_assert(!invocable<rvalue_cv_function_object const volatile&, int*>, "");

struct multiple_overloads
{
  struct A
  {};
  struct B
  {
    __host__ __device__ B(int);
  };
  struct AB
      : A
      , B
  {};
  struct O
  {};
  __host__ __device__ void operator()(A) const;
  __host__ __device__ void operator()(B) const;
};
static_assert(invocable<multiple_overloads, multiple_overloads::A>, "");
static_assert(invocable<multiple_overloads, multiple_overloads::B>, "");
static_assert(invocable<multiple_overloads, int>, "");
static_assert(!invocable<multiple_overloads, multiple_overloads::AB>, "");
static_assert(!invocable<multiple_overloads, multiple_overloads::O>, "");
} // namespace function_objects

namespace pointer_to_member_functions
{
template <class Member, class T, class... Args>
__host__ __device__ constexpr bool check_member_is_invocable()
{
  constexpr bool result = invocable<Member, T&&, Args...>;
  using uncv_t          = cuda::std::remove_cvref_t<T>;
  static_assert(invocable<Member, uncv_t*, Args...> == result, "");
  static_assert(invocable<Member, cuda::std::reference_wrapper<uncv_t>, Args...> == result, "");
  static_assert(!invocable<Member, cuda::std::nullptr_t, Args...>, "");
  static_assert(!invocable<Member, int, Args...>, "");
  static_assert(!invocable<Member, int*, Args...>, "");
  static_assert(!invocable<Member, double*, Args...>, "");
  struct S2
  {};
  static_assert(!invocable<Member, S2*, Args...>, "");
  return result;
}

static_assert(check_member_is_invocable<int S::*, S>(), "");
static_assert(invocable<int S::*, S&>, "");
static_assert(invocable<int S::*, S const&>, "");
static_assert(invocable<int S::*, S volatile&>, "");
static_assert(invocable<int S::*, S const volatile&>, "");
static_assert(invocable<int S::*, S&&>, "");
static_assert(invocable<int S::*, S const&&>, "");
static_assert(invocable<int S::*, S volatile&&>, "");
static_assert(invocable<int S::*, S const volatile&&>, "");

static_assert(check_member_is_invocable<int (S::*)(int), S, int>(), "");
static_assert(!check_member_is_invocable<int (S::*)(int), S>(), "");
using unqualified = void (S::*)();
static_assert(invocable<unqualified, S&>, "");
static_assert(!invocable<unqualified, S const&>, "");
static_assert(!invocable<unqualified, S volatile&>, "");
static_assert(!invocable<unqualified, S const volatile&>, "");
static_assert(invocable<unqualified, S&&>, "");
static_assert(!invocable<unqualified, S const&&>, "");
static_assert(!invocable<unqualified, S volatile&&>, "");
static_assert(!invocable<unqualified, S const volatile&&>, "");

static_assert(check_member_is_invocable<int (S::*)(double) const, S, double>(), "");
using const_qualified = void (S::*)() const;
static_assert(invocable<const_qualified, S&>, "");
static_assert(invocable<const_qualified, S const&>, "");
static_assert(!invocable<const_qualified, S volatile&>, "");
static_assert(!invocable<const_qualified, S const volatile&>, "");
static_assert(invocable<const_qualified, S&&>, "");
static_assert(invocable<const_qualified, S const&&>, "");
static_assert(!invocable<const_qualified, S volatile&&>, "");
static_assert(!invocable<const_qualified, S const volatile&&>, "");

static_assert(check_member_is_invocable<int (S::*)(double[]) volatile, S, double*>(), "");
using volatile_qualified = void (S::*)() volatile;
static_assert(invocable<volatile_qualified, S&>, "");
static_assert(!invocable<volatile_qualified, S const&>, "");
static_assert(invocable<volatile_qualified, S volatile&>, "");
static_assert(!invocable<volatile_qualified, S const volatile&>, "");
static_assert(invocable<volatile_qualified, S&&>, "");
static_assert(!invocable<volatile_qualified, S const&&>, "");
static_assert(invocable<volatile_qualified, S volatile&&>, "");
static_assert(!invocable<volatile_qualified, S const volatile&&>, "");

static_assert(check_member_is_invocable<int (S::*)(int, S&) const volatile, S, int, S&>(), "");
using cv_qualified = void (S::*)() const volatile;
static_assert(invocable<cv_qualified, S&>, "");
static_assert(invocable<cv_qualified, S const&>, "");
static_assert(invocable<cv_qualified, S volatile&>, "");
static_assert(invocable<cv_qualified, S const volatile&>, "");
static_assert(invocable<cv_qualified, S&&>, "");
static_assert(invocable<cv_qualified, S const&&>, "");
static_assert(invocable<cv_qualified, S volatile&&>, "");
static_assert(invocable<cv_qualified, S const volatile&&>, "");

static_assert(check_member_is_invocable<int (S::*)() &, S&>(), "");
using lvalue_qualified = void (S::*)() &;
static_assert(invocable<lvalue_qualified, S&>, "");
static_assert(!invocable<lvalue_qualified, S const&>, "");
static_assert(!invocable<lvalue_qualified, S volatile&>, "");
static_assert(!invocable<lvalue_qualified, S const volatile&>, "");
#if !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)
static_assert(!invocable<lvalue_qualified, S&&>, "");
#endif // !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)
static_assert(!invocable<lvalue_qualified, S const&&>, "");
static_assert(!invocable<lvalue_qualified, S volatile&&>, "");
static_assert(!invocable<lvalue_qualified, S const volatile&&>, "");

#if TEST_STD_VER > 2017
static_assert(check_member_is_invocable<int (S::*)() const&, S>(), "");
#endif // TEST_STD_VER > 2017
using lvalue_const_qualified = void (S::*)() const&;
static_assert(invocable<lvalue_const_qualified, S&>, "");
static_assert(invocable<lvalue_const_qualified, S const&>, "");
static_assert(!invocable<lvalue_const_qualified, S volatile&>, "");
static_assert(!invocable<lvalue_const_qualified, S const volatile&>, "");
#if TEST_STD_VER > 2017
static_assert(invocable<lvalue_const_qualified, S&&>, "");
static_assert(invocable<lvalue_const_qualified, S const&&>, "");
#endif // TEST_STD_VER > 2017
static_assert(!invocable<lvalue_const_qualified, S volatile&&>, "");
static_assert(!invocable<lvalue_const_qualified, S const volatile&&>, "");

static_assert(check_member_is_invocable<int (S::*)() volatile&, S&>(), "");
using lvalue_volatile_qualified = void (S::*)() volatile&;
static_assert(invocable<lvalue_volatile_qualified, S&>, "");
static_assert(!invocable<lvalue_volatile_qualified, S const&>, "");
static_assert(invocable<lvalue_volatile_qualified, S volatile&>, "");
static_assert(!invocable<lvalue_volatile_qualified, S const volatile&>, "");
#if !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)
static_assert(!invocable<lvalue_volatile_qualified, S&&>, "");
static_assert(!invocable<lvalue_volatile_qualified, S const&&>, "");
static_assert(!invocable<lvalue_volatile_qualified, S volatile&&>, "");
static_assert(!invocable<lvalue_volatile_qualified, S const volatile&&>, "");
#endif // !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)

static_assert(check_member_is_invocable<int (S::*)() const volatile&, S&>(), "");
using lvalue_cv_qualified = void (S::*)() const volatile&;
static_assert(invocable<lvalue_cv_qualified, S&>, "");
static_assert(invocable<lvalue_cv_qualified, S const&>, "");
static_assert(invocable<lvalue_cv_qualified, S volatile&>, "");
static_assert(invocable<lvalue_cv_qualified, S const volatile&>, "");
#if !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)
static_assert(!invocable<lvalue_cv_qualified, S&&>, "");
static_assert(!invocable<lvalue_cv_qualified, S const&&>, "");
static_assert(!invocable<lvalue_cv_qualified, S volatile&&>, "");
static_assert(!invocable<lvalue_cv_qualified, S const volatile&&>, "");
#endif // !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)

using rvalue_unqualified = void (S::*)() &&;
static_assert(!invocable<rvalue_unqualified, S&>, "");
static_assert(!invocable<rvalue_unqualified, S const&>, "");
static_assert(!invocable<rvalue_unqualified, S volatile&>, "");
static_assert(!invocable<rvalue_unqualified, S const volatile&>, "");
#if !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)
static_assert(invocable<rvalue_unqualified, S&&>, "");
static_assert(!invocable<rvalue_unqualified, S const&&>, "");
static_assert(!invocable<rvalue_unqualified, S volatile&&>, "");
static_assert(!invocable<rvalue_unqualified, S const volatile&&>, "");
#endif // !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)

using rvalue_const_unqualified = void (S::*)() const&&;
static_assert(!invocable<rvalue_const_unqualified, S&>, "");
static_assert(!invocable<rvalue_const_unqualified, S const&>, "");
static_assert(!invocable<rvalue_const_unqualified, S volatile&>, "");
static_assert(!invocable<rvalue_const_unqualified, S const volatile&>, "");
#if !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)
static_assert(invocable<rvalue_const_unqualified, S&&>, "");
static_assert(invocable<rvalue_const_unqualified, S const&&>, "");
#endif // !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)
static_assert(!invocable<rvalue_const_unqualified, S volatile&&>, "");
static_assert(!invocable<rvalue_const_unqualified, S const volatile&&>, "");

using rvalue_volatile_unqualified = void (S::*)() volatile&&;
static_assert(!invocable<rvalue_volatile_unqualified, S&>, "");
static_assert(!invocable<rvalue_volatile_unqualified, S const&>, "");
static_assert(!invocable<rvalue_volatile_unqualified, S volatile&>, "");
static_assert(!invocable<rvalue_volatile_unqualified, S const volatile&>, "");
#if !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)
static_assert(invocable<rvalue_volatile_unqualified, S&&>, "");
static_assert(!invocable<rvalue_volatile_unqualified, S const&&>, "");
static_assert(invocable<rvalue_volatile_unqualified, S volatile&&>, "");
static_assert(!invocable<rvalue_volatile_unqualified, S const volatile&&>, "");
#endif // !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)

using rvalue_cv_unqualified = void (S::*)() const volatile&&;
static_assert(!invocable<rvalue_cv_unqualified, S&>, "");
static_assert(!invocable<rvalue_cv_unqualified, S const&>, "");
static_assert(!invocable<rvalue_cv_unqualified, S volatile&>, "");
static_assert(!invocable<rvalue_cv_unqualified, S const volatile&>, "");
#if !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)
static_assert(invocable<rvalue_cv_unqualified, S&&>, "");
static_assert(invocable<rvalue_cv_unqualified, S const&&>, "");
static_assert(invocable<rvalue_cv_unqualified, S volatile&&>, "");
static_assert(invocable<rvalue_cv_unqualified, S const volatile&&>, "");
#endif // !defined(TEST_COMPILER_MSVC_2017) && !defined(TEST_COMPILER_MSVC_2019)
} // namespace pointer_to_member_functions

// Check the concept with closure types
template <class F, class... Args>
__host__ __device__ constexpr bool is_invocable(F, Args&&...)
{
  return invocable<F, Args...>;
}

// execution space annotations on lambda require --extended-lambda flag with nvrtc
#if TEST_STD_VER > 2014 && !defined(TEST_COMPILER_NVRTC)
static_assert(is_invocable([] {}), "");
static_assert(is_invocable([](int) {}, 0), "");
static_assert(is_invocable([](int) {}, 0L), "");
static_assert(!is_invocable([](int) {}, nullptr), "");
int i = 0;
static_assert(is_invocable([](int&) {}, i), "");
#endif // TEST_STD_VER > 2014

int main(int, char**)
{
  return 0;
}
