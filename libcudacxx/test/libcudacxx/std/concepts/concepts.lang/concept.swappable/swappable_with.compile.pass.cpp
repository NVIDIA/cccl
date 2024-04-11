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
// concept swappable_with = // see below

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>

#include "test_macros.h"
#include "type_classification/moveconstructible.h"
#include "type_classification/swappable.h"

using cuda::std::swappable;
using cuda::std::swappable_with;

template <class T, class U>
__host__ __device__ constexpr bool check_swappable_with_impl()
{
  static_assert(swappable_with<T, U> == swappable_with<U, T>, "");
  return swappable_with<T, U>;
}

template <class T, class U>
__host__ __device__ constexpr bool check_swappable_with()
{
  static_assert(!check_swappable_with_impl<T, U>(), "");
  static_assert(!check_swappable_with_impl<T, U const>(), "");
  static_assert(!check_swappable_with_impl<T const, U>(), "");
  static_assert(!check_swappable_with_impl<T const, U const>(), "");

  static_assert(!check_swappable_with_impl<T, U&>(), "");
  static_assert(!check_swappable_with_impl<T, U const&>(), "");
  static_assert(!check_swappable_with_impl<T, U volatile&>(), "");
  static_assert(!check_swappable_with_impl<T, U const volatile&>(), "");
  static_assert(!check_swappable_with_impl<T const, U&>(), "");
  static_assert(!check_swappable_with_impl<T const, U const&>(), "");
  static_assert(!check_swappable_with_impl<T const, U volatile&>(), "");
  static_assert(!check_swappable_with_impl<T const, U const volatile&>(), "");

  static_assert(!check_swappable_with_impl<T&, U>(), "");
  static_assert(!check_swappable_with_impl<T&, U const>(), "");
  static_assert(!check_swappable_with_impl<T const&, U>(), "");
  static_assert(!check_swappable_with_impl<T const&, U const>(), "");
  static_assert(!check_swappable_with_impl<T volatile&, U>(), "");
  static_assert(!check_swappable_with_impl<T volatile&, U const>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&, U>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&, U const>(), "");

  static_assert(!check_swappable_with_impl<T&, U const&>(), "");
  static_assert(!check_swappable_with_impl<T&, U volatile&>(), "");
  static_assert(!check_swappable_with_impl<T&, U const volatile&>(), "");
  static_assert(!check_swappable_with_impl<T const&, U&>(), "");
  static_assert(!check_swappable_with_impl<T const&, U const&>(), "");
  static_assert(!check_swappable_with_impl<T const&, U volatile&>(), "");
  static_assert(!check_swappable_with_impl<T const&, U const volatile&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&, U&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&, U const&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&, U const volatile&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&, U&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&, U const&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&, U volatile&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&, U const volatile&>(), "");

  static_assert(!check_swappable_with_impl<T, U&&>(), "");
  static_assert(!check_swappable_with_impl<T, U const&&>(), "");
  static_assert(!check_swappable_with_impl<T, U volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T, U const volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T const, U&&>(), "");
  static_assert(!check_swappable_with_impl<T const, U const&&>(), "");
  static_assert(!check_swappable_with_impl<T const, U volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T const, U const volatile&&>(), "");

  static_assert(!check_swappable_with_impl<T&&, U>(), "");
  static_assert(!check_swappable_with_impl<T&&, U const>(), "");
  static_assert(!check_swappable_with_impl<T const&&, U>(), "");
  static_assert(!check_swappable_with_impl<T const&&, U const>(), "");
  static_assert(!check_swappable_with_impl<T volatile&&, U>(), "");
  static_assert(!check_swappable_with_impl<T volatile&&, U const>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&&, U>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&&, U const>(), "");

  static_assert(!check_swappable_with_impl<T&, U&&>(), "");
  static_assert(!check_swappable_with_impl<T&, U const&&>(), "");
  static_assert(!check_swappable_with_impl<T&, U volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T&, U const volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T const&, U&&>(), "");
  static_assert(!check_swappable_with_impl<T const&, U const&&>(), "");
  static_assert(!check_swappable_with_impl<T const&, U volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T const&, U const volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&, U&&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&, U const&&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&, U volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&, U const volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&, U&&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&, U const&&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&, U volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&, U const volatile&&>(), "");

  static_assert(!check_swappable_with_impl<T&&, U&>(), "");
  static_assert(!check_swappable_with_impl<T&&, U const&>(), "");
  static_assert(!check_swappable_with_impl<T&&, U volatile&>(), "");
  static_assert(!check_swappable_with_impl<T&&, U const volatile&>(), "");
  static_assert(!check_swappable_with_impl<T const&&, U&>(), "");
  static_assert(!check_swappable_with_impl<T const&&, U const&>(), "");
  static_assert(!check_swappable_with_impl<T const&&, U volatile&>(), "");
  static_assert(!check_swappable_with_impl<T const&&, U const volatile&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&&, U&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&&, U const&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&&, U volatile&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&&, U const volatile&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&&, U&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&&, U const&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&&, U volatile&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&&, U const volatile&>(), "");

  static_assert(!check_swappable_with_impl<T&&, U&&>(), "");
  static_assert(!check_swappable_with_impl<T&&, U const&&>(), "");
  static_assert(!check_swappable_with_impl<T&&, U volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T&&, U const volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T const&&, U&&>(), "");
  static_assert(!check_swappable_with_impl<T const&&, U const&&>(), "");
  static_assert(!check_swappable_with_impl<T const&&, U volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T const&&, U const volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&&, U&&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&&, U const&&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&&, U volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T volatile&&, U const volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&&, U&&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&&, U const&&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&&, U volatile&&>(), "");
  static_assert(!check_swappable_with_impl<T const volatile&&, U const volatile&&>(), "");
  return check_swappable_with_impl<T&, U&>();
}

template <class T, class U>
__host__ __device__ constexpr bool check_swappable_with_including_lvalue_ref_to_volatile()
{
  constexpr auto result = check_swappable_with<T, U>();
  static_assert(check_swappable_with_impl<T volatile&, U volatile&>() == result, "");
  return result;
}

namespace fundamental
{
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<int, int>(), "");
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<double, double>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int, double>(), "");

static_assert(check_swappable_with_including_lvalue_ref_to_volatile<int*, int*>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int, int*>(), "");
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<int (*)(), int (*)()>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int, int (*)()>(), "");

struct S
{};
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int, S>(), "");
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<int S::*, int S::*>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int, int S::*>(), "");
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<int (S::*)(), int (S::*)()>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int, int (S::*)()>(), "");
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<int (S::*)() noexcept, int (S::*)() noexcept>(),
              "");
#if TEST_STD_VER > 2017
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int (S::*)() noexcept, int (S::*)()>(), "");
#endif // TEST_STD_VER > 2017
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<int (S::*)() const, int (S::*)() const>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int (S::*)() const, int (S::*)()>(), "");
static_assert(
  check_swappable_with_including_lvalue_ref_to_volatile<int (S::*)() const noexcept, int (S::*)() const noexcept>(),
  "");
#if TEST_STD_VER > 2017
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int (S::*)() const, int (S::*)() const noexcept>(),
              "");
#endif // TEST_STD_VER > 2017
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<int (S::*)() volatile, int (S::*)() volatile>(),
              "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int (S::*)() volatile, int (S::*)()>(), "");
static_assert(
  check_swappable_with_including_lvalue_ref_to_volatile<int (S::*)() const volatile, int (S::*)() const volatile>(),
  "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int (S::*)() const volatile, int (S::*)()>(), "");

#if TEST_STD_VER > 2017
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<int[5], int[5]>(), "");
#endif
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int[5], int[6]>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int[5], double[5]>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int[5], double[6]>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int[5][6], int[5]>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int[5][6], int[6]>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int[5][6], double[5]>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int[5][6], double[6]>(), "");
#if TEST_STD_VER > 2017
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<int[5][6], int[5][6]>(), "");
#endif
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int[5][6], int[5][4]>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int[5][6], int[6][5]>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int[5][6], double[5][6]>(), "");
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int[5][6], double[6][5]>(), "");

// always false
static_assert(!check_swappable_with_impl<void, void>(), "");
static_assert(!check_swappable_with_impl<int, void>(), "");
static_assert(!check_swappable_with_impl<int&, void>(), "");
static_assert(!check_swappable_with_impl<void, int>(), "");
static_assert(!check_swappable_with_impl<void, int&>(), "");
static_assert(!check_swappable_with_impl<int, int()>(), "");
static_assert(!check_swappable_with_impl<int, int (&)()>(), "");
} // namespace fundamental

namespace adl
{

#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC ignores the rvalue/lvalue distinction in the swap
                                                        // definitions
static_assert(check_swappable_with<lvalue_adl_swappable, lvalue_adl_swappable>(), "");
static_assert(check_swappable_with<lvalue_rvalue_adl_swappable, lvalue_rvalue_adl_swappable>(), "");
static_assert(check_swappable_with<rvalue_lvalue_adl_swappable, rvalue_lvalue_adl_swappable>(), "");
static_assert(check_swappable_with_impl<rvalue_adl_swappable, rvalue_adl_swappable>(), "");
static_assert(!check_swappable_with_impl<lvalue_rvalue_adl_swappable&, lvalue_rvalue_adl_swappable&&>(), "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017

struct s1
{};
struct no_common_reference_with_s1
{
  __host__ __device__ friend void swap(s1&, no_common_reference_with_s1&);
  __host__ __device__ friend void swap(no_common_reference_with_s1&, s1&);
};
static_assert(!check_swappable_with<s1, no_common_reference_with_s1>(), "");

struct one_way_swappable_with_s1
{
  __host__ __device__ friend void swap(s1&, one_way_swappable_with_s1&);
  __host__ __device__ operator s1();
};
static_assert(cuda::std::common_reference_with<one_way_swappable_with_s1, s1>, "");
static_assert(!check_swappable_with<one_way_swappable_with_s1, s1>(), "");

struct one_way_swappable_with_s1_other_way
{
  __host__ __device__ friend void swap(one_way_swappable_with_s1_other_way&, s1&);
  __host__ __device__ operator s1();
};
static_assert(cuda::std::common_reference_with<one_way_swappable_with_s1_other_way, s1>, "");
static_assert(!check_swappable_with<one_way_swappable_with_s1_other_way, s1>(), "");

struct can_swap_with_s1_but_not_swappable
{
  can_swap_with_s1_but_not_swappable(can_swap_with_s1_but_not_swappable&&) = delete;
  __host__ __device__ friend void swap(s1&, can_swap_with_s1_but_not_swappable&);
  __host__ __device__ friend void swap(can_swap_with_s1_but_not_swappable&, s1&);

  __host__ __device__ operator s1() const;
};
static_assert(cuda::std::common_reference_with<can_swap_with_s1_but_not_swappable, s1>, "");
static_assert(!swappable<can_swap_with_s1_but_not_swappable>, "");
static_assert(!check_swappable_with<can_swap_with_s1_but_not_swappable&, s1&>(), "");

#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC ignores the rvalue/lvalue distinction in the swap
                                                        // definitions
struct swappable_with_s1
{
  __host__ __device__ friend void swap(s1&, swappable_with_s1&);
  __host__ __device__ friend void swap(swappable_with_s1&, s1&);
  __host__ __device__ operator s1() const;
};
static_assert(check_swappable_with<swappable_with_s1, s1>(), "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017

struct swappable_with_const_s1_but_not_swappable
{
  __host__ __device__ swappable_with_const_s1_but_not_swappable(swappable_with_const_s1_but_not_swappable const&);
  __host__ __device__ swappable_with_const_s1_but_not_swappable(swappable_with_const_s1_but_not_swappable const&&);
  __host__ __device__ swappable_with_const_s1_but_not_swappable&
  operator=(swappable_with_const_s1_but_not_swappable const&);
  __host__ __device__ swappable_with_const_s1_but_not_swappable&
  operator=(swappable_with_const_s1_but_not_swappable const&&);

  __host__ __device__ friend void swap(s1 const&, swappable_with_const_s1_but_not_swappable const&);
  __host__ __device__ friend void swap(swappable_with_const_s1_but_not_swappable const&, s1 const&);

  __host__ __device__ operator s1 const&() const;
};
static_assert(!swappable<swappable_with_const_s1_but_not_swappable const&>, "");
static_assert(!swappable_with<swappable_with_const_s1_but_not_swappable const&, s1 const&>, "");

struct swappable_with_volatile_s1_but_not_swappable
{
  __host__ __device__
  swappable_with_volatile_s1_but_not_swappable(swappable_with_volatile_s1_but_not_swappable volatile&);
  __host__ __device__
  swappable_with_volatile_s1_but_not_swappable(swappable_with_volatile_s1_but_not_swappable volatile&&);
  __host__ __device__ swappable_with_volatile_s1_but_not_swappable&
  operator=(swappable_with_volatile_s1_but_not_swappable volatile&);
  __host__ __device__ swappable_with_volatile_s1_but_not_swappable&
  operator=(swappable_with_volatile_s1_but_not_swappable volatile&&);

  __host__ __device__ friend void swap(s1 volatile&, swappable_with_volatile_s1_but_not_swappable volatile&);
  __host__ __device__ friend void swap(swappable_with_volatile_s1_but_not_swappable volatile&, s1 volatile&);

  __host__ __device__ operator s1 volatile&() volatile;
};
static_assert(!swappable<swappable_with_volatile_s1_but_not_swappable volatile&>, "");
static_assert(!swappable_with<swappable_with_volatile_s1_but_not_swappable volatile&, s1 volatile&>, "");

struct swappable_with_cv_s1_but_not_swappable
{
  __host__ __device__ swappable_with_cv_s1_but_not_swappable(swappable_with_cv_s1_but_not_swappable const volatile&);
  __host__ __device__ swappable_with_cv_s1_but_not_swappable(swappable_with_cv_s1_but_not_swappable const volatile&&);
  __host__ __device__ swappable_with_cv_s1_but_not_swappable&
  operator=(swappable_with_cv_s1_but_not_swappable const volatile&);
  __host__ __device__ swappable_with_cv_s1_but_not_swappable&
  operator=(swappable_with_cv_s1_but_not_swappable const volatile&&);

  __host__ __device__ friend void swap(s1 const volatile&, swappable_with_cv_s1_but_not_swappable const volatile&);
  __host__ __device__ friend void swap(swappable_with_cv_s1_but_not_swappable const volatile&, s1 const volatile&);

  __host__ __device__ operator s1 const volatile&() const volatile;
};
static_assert(!swappable<swappable_with_cv_s1_but_not_swappable const volatile&>, "");
static_assert(!swappable_with<swappable_with_cv_s1_but_not_swappable const volatile&, s1 const volatile&>, "");

struct s2
{
  __host__ __device__ friend void swap(s2 const&, s2 const&);
  __host__ __device__ friend void swap(s2 volatile&, s2 volatile&);
  __host__ __device__ friend void swap(s2 const volatile&, s2 const volatile&);
};

#ifndef TEST_COMPILER_MSVC_2017
struct swappable_with_const_s2
{
  __host__ __device__ swappable_with_const_s2(swappable_with_const_s2 const&);
  __host__ __device__ swappable_with_const_s2(swappable_with_const_s2 const&&);
  __host__ __device__ swappable_with_const_s2& operator=(swappable_with_const_s2 const&);
  __host__ __device__ swappable_with_const_s2& operator=(swappable_with_const_s2 const&&);

  __host__ __device__ friend void swap(swappable_with_const_s2 const&, swappable_with_const_s2 const&);
  __host__ __device__ friend void swap(s2 const&, swappable_with_const_s2 const&);
  __host__ __device__ friend void swap(swappable_with_const_s2 const&, s2 const&);

  __host__ __device__ operator s2 const&() const;
};
static_assert(swappable_with<swappable_with_const_s2 const&, s2 const&>, "");

struct swappable_with_volatile_s2
{
  __host__ __device__ swappable_with_volatile_s2(swappable_with_volatile_s2 volatile&);
  __host__ __device__ swappable_with_volatile_s2(swappable_with_volatile_s2 volatile&&);
  __host__ __device__ swappable_with_volatile_s2& operator=(swappable_with_volatile_s2 volatile&);
  __host__ __device__ swappable_with_volatile_s2& operator=(swappable_with_volatile_s2 volatile&&);

  __host__ __device__ friend void swap(swappable_with_volatile_s2 volatile&, swappable_with_volatile_s2 volatile&);
  __host__ __device__ friend void swap(s2 volatile&, swappable_with_volatile_s2 volatile&);
  __host__ __device__ friend void swap(swappable_with_volatile_s2 volatile&, s2 volatile&);

  __host__ __device__ operator s2 volatile&() volatile;
};
using test  = cuda::std::common_reference_t<swappable_with_volatile_s2 volatile&, s2 volatile&>;
using test2 = cuda::std::common_reference_t<s2 volatile&, swappable_with_volatile_s2 volatile&>;
static_assert(swappable_with<swappable_with_volatile_s2 volatile&, s2 volatile&>, "");

struct swappable_with_cv_s2
{
  __host__ __device__ swappable_with_cv_s2(swappable_with_cv_s2 const volatile&);
  __host__ __device__ swappable_with_cv_s2(swappable_with_cv_s2 const volatile&&);
  __host__ __device__ swappable_with_cv_s2& operator=(swappable_with_cv_s2 const volatile&);
  __host__ __device__ swappable_with_cv_s2& operator=(swappable_with_cv_s2 const volatile&&);

  __host__ __device__ friend void swap(swappable_with_cv_s2 const volatile&, swappable_with_cv_s2 const volatile&);
  __host__ __device__ friend void swap(s2 const volatile&, swappable_with_cv_s2 const volatile&);
  __host__ __device__ friend void swap(swappable_with_cv_s2 const volatile&, s2 const volatile&);

  __host__ __device__ operator s2 const volatile&() const volatile;
};
static_assert(swappable_with<swappable_with_cv_s2 const volatile&, s2 const volatile&>, "");
#endif // !TEST_COMPILER_MSVC_2017

struct swappable_with_rvalue_ref_to_s1_but_not_swappable
{
  __host__ __device__ friend void
  swap(swappable_with_rvalue_ref_to_s1_but_not_swappable&&, swappable_with_rvalue_ref_to_s1_but_not_swappable&&);
  __host__ __device__ friend void swap(s1&&, swappable_with_rvalue_ref_to_s1_but_not_swappable&&);
  __host__ __device__ friend void swap(swappable_with_rvalue_ref_to_s1_but_not_swappable&&, s1&&);

  __host__ __device__ operator s1() const;
};
static_assert(!swappable<swappable_with_rvalue_ref_to_s1_but_not_swappable const&&>, "");
static_assert(!swappable_with<swappable_with_rvalue_ref_to_s1_but_not_swappable const&&, s1 const&&>, "");

struct swappable_with_rvalue_ref_to_const_s1_but_not_swappable
{
  __host__ __device__ friend void swap(s1 const&&, swappable_with_rvalue_ref_to_const_s1_but_not_swappable const&&);
  __host__ __device__ friend void swap(swappable_with_rvalue_ref_to_const_s1_but_not_swappable const&&, s1 const&&);

  __host__ __device__ operator s1 const() const;
};
static_assert(!swappable<swappable_with_rvalue_ref_to_const_s1_but_not_swappable const&&>, "");
static_assert(!swappable_with<swappable_with_rvalue_ref_to_const_s1_but_not_swappable const&&, s1 const&&>, "");

struct swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable
{
  __host__ __device__ swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable(
    swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&);
  __host__ __device__ swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable(
    swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&&);
  __host__ __device__ swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable&
  operator=(swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&);
  __host__ __device__ __host__ __device__ swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable&
  operator=(swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&&);

  __host__ __device__ friend void
  swap(s1 volatile&&, swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&&);
  __host__ __device__ friend void
  swap(swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&&, s1 volatile&&);

  __host__ __device__ operator s1 volatile&&() volatile&&;
};
static_assert(!swappable<swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&&>, "");
static_assert(!swappable_with<swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&&, s1 volatile&&>,
              "");

struct swappable_with_rvalue_ref_to_cv_s1_but_not_swappable
{
  __host__ __device__ swappable_with_rvalue_ref_to_cv_s1_but_not_swappable(
    swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&);
  __host__ __device__ swappable_with_rvalue_ref_to_cv_s1_but_not_swappable(
    swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&&);
  __host__ __device__ swappable_with_rvalue_ref_to_cv_s1_but_not_swappable&
  operator=(swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&);
  __host__ __device__ swappable_with_rvalue_ref_to_cv_s1_but_not_swappable&
  operator=(swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&&);

  __host__ __device__ friend void
  swap(s1 const volatile&&, swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&&);
  __host__ __device__ friend void
  swap(swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&&, s1 const volatile&&);

  __host__ __device__ operator s1 const volatile&&() const volatile&&;
};
static_assert(!swappable<swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&&>, "");
static_assert(
  !swappable_with<swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&&, s1 const volatile&&>, "");

struct s3
{
  __host__ __device__ friend void swap(s3&&, s3&&);
  __host__ __device__ friend void swap(s3 const&&, s3 const&&);
  __host__ __device__ friend void swap(s3 volatile&&, s3 volatile&&);
  __host__ __device__ friend void swap(s3 const volatile&&, s3 const volatile&&);
};

#ifndef TEST_COMPILER_MSVC_2017
struct swappable_with_rvalue_ref_to_s3
{
  __host__ __device__ friend void swap(swappable_with_rvalue_ref_to_s3&&, swappable_with_rvalue_ref_to_s3&&);
  __host__ __device__ friend void swap(s3&&, swappable_with_rvalue_ref_to_s3&&);
  __host__ __device__ friend void swap(swappable_with_rvalue_ref_to_s3&&, s3&&);

  __host__ __device__ operator s3() const;
};
static_assert(swappable_with<swappable_with_rvalue_ref_to_s3&&, s3&&>, "");

struct swappable_with_rvalue_ref_to_const_s3
{
  __host__ __device__ swappable_with_rvalue_ref_to_const_s3(swappable_with_rvalue_ref_to_const_s3 const&);
  __host__ __device__ swappable_with_rvalue_ref_to_const_s3(swappable_with_rvalue_ref_to_const_s3 const&&);
  __host__ __device__ swappable_with_rvalue_ref_to_const_s3& operator=(swappable_with_rvalue_ref_to_const_s3 const&);
  __host__ __device__ swappable_with_rvalue_ref_to_const_s3& operator=(swappable_with_rvalue_ref_to_const_s3 const&&);

  __host__ __device__ friend void
  swap(swappable_with_rvalue_ref_to_const_s3 const&&, swappable_with_rvalue_ref_to_const_s3 const&&);
  __host__ __device__ friend void swap(s3 const&&, swappable_with_rvalue_ref_to_const_s3 const&&);
  __host__ __device__ friend void swap(swappable_with_rvalue_ref_to_const_s3 const&&, s3 const&&);

  __host__ __device__ operator s3() const;
};
static_assert(swappable_with<swappable_with_rvalue_ref_to_const_s3 const&&, s3 const&&>, "");
#endif // !TEST_COMPILER_MSVC_2017

#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC ignores the rvalue/lvalue distinction in the swap
                                                        // definitions
struct swappable_with_rvalue_ref_to_volatile_s3
{
  __host__ __device__ swappable_with_rvalue_ref_to_volatile_s3(swappable_with_rvalue_ref_to_volatile_s3 volatile&);
  __host__ __device__ swappable_with_rvalue_ref_to_volatile_s3(swappable_with_rvalue_ref_to_volatile_s3 volatile&&);
  __host__ __device__ swappable_with_rvalue_ref_to_volatile_s3&
  operator=(swappable_with_rvalue_ref_to_volatile_s3 volatile&);
  __host__ __device__ swappable_with_rvalue_ref_to_volatile_s3&
  operator=(swappable_with_rvalue_ref_to_volatile_s3 volatile&&);

  __host__ __device__ friend void
  swap(swappable_with_rvalue_ref_to_volatile_s3 volatile&&, swappable_with_rvalue_ref_to_volatile_s3 volatile&&);
  __host__ __device__ friend void swap(s3 volatile&&, swappable_with_rvalue_ref_to_volatile_s3 volatile&&);
  __host__ __device__ friend void swap(swappable_with_rvalue_ref_to_volatile_s3 volatile&&, s3 volatile&&);

  __host__ __device__ operator s3 volatile&&() volatile;
};
static_assert(swappable_with<swappable_with_rvalue_ref_to_volatile_s3 volatile&&, s3 volatile&&>, "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017

struct swappable_with_rvalue_ref_to_cv_s3
{
  __host__ __device__ swappable_with_rvalue_ref_to_cv_s3(swappable_with_rvalue_ref_to_cv_s3 const volatile&);
  __host__ __device__ swappable_with_rvalue_ref_to_cv_s3(swappable_with_rvalue_ref_to_cv_s3 const volatile&&);
  __host__ __device__ swappable_with_rvalue_ref_to_cv_s3& operator=(swappable_with_rvalue_ref_to_cv_s3 const volatile&);
  __host__ __device__ swappable_with_rvalue_ref_to_cv_s3& operator=(swappable_with_rvalue_ref_to_cv_s3 const volatile&&);

  __host__ __device__ friend void
  swap(swappable_with_rvalue_ref_to_cv_s3 const volatile&&, swappable_with_rvalue_ref_to_cv_s3 const volatile&&);
  __host__ __device__ friend void swap(s3 const volatile&&, swappable_with_rvalue_ref_to_cv_s3 const volatile&&);
  __host__ __device__ friend void swap(swappable_with_rvalue_ref_to_cv_s3 const volatile&&, s3 const volatile&&);

  __host__ __device__ operator s3 const volatile&&() const volatile;
};
#if !defined(TEST_COMPILER_MSVC)
static_assert(
  cuda::std::common_reference_with<swappable_with_rvalue_ref_to_cv_s3 const volatile&&, s3 const volatile&&>, "");
static_assert(swappable_with<swappable_with_rvalue_ref_to_cv_s3 const volatile&&, s3 const volatile&&>, "");
#endif

namespace union_swap
{
union adl_swappable
{
  int x;
  double y;

  __host__ __device__ operator int() const;
};

__host__ __device__ void swap(adl_swappable&, adl_swappable&) noexcept;
__host__ __device__ void swap(adl_swappable&&, adl_swappable&&) noexcept;
__host__ __device__ void swap(adl_swappable&, int&) noexcept;
__host__ __device__ void swap(int&, adl_swappable&) noexcept;
} // namespace union_swap
#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC ignores the rvalue/lvalue distinction in the swap
                                                        // definitions
static_assert(swappable_with<union_swap::adl_swappable, union_swap::adl_swappable>, "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017
static_assert(swappable_with<union_swap::adl_swappable&, union_swap::adl_swappable&>, "");
static_assert(swappable_with<union_swap::adl_swappable&&, union_swap::adl_swappable&&>, "");
#ifndef TEST_COMPILER_MSVC_2017
static_assert(swappable_with<union_swap::adl_swappable&, int&>, "");
#endif // !TEST_COMPILER_MSVC_2017
} // namespace adl

namespace standard_types
{
#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC does not like to swap the arrays
static_assert(check_swappable_with<cuda::std::array<int, 10>, cuda::std::array<int, 10>>(), "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017
static_assert(!check_swappable_with<cuda::std::array<int, 10>, cuda::std::array<double, 10>>(), "");
} // namespace standard_types

namespace types_with_purpose
{
static_assert(!check_swappable_with<DeletedMoveCtor, DeletedMoveCtor>(), "");
static_assert(!check_swappable_with<ImplicitlyDeletedMoveCtor, ImplicitlyDeletedMoveCtor>(), "");
static_assert(!check_swappable_with<DeletedMoveAssign, DeletedMoveAssign>(), "");
static_assert(!check_swappable_with<ImplicitlyDeletedMoveAssign, ImplicitlyDeletedMoveAssign>(), "");
static_assert(!check_swappable_with<NonMovable, NonMovable>(), "");
static_assert(!check_swappable_with<DerivedFromNonMovable, DerivedFromNonMovable>(), "");
static_assert(!check_swappable_with<HasANonMovable, HasANonMovable>(), "");
} // namespace types_with_purpose

#ifndef TEST_COMPILER_MSVC_2017
namespace LWG3175
{
// Example taken directly from [concept.swappable]
_LIBCUDACXX_TEMPLATE(class T, class U)
_LIBCUDACXX_REQUIRES(swappable_with<T, U>)
__host__ __device__ constexpr void value_swap(T&& t, U&& u)
{
  cuda::std::ranges::swap(cuda::std::forward<T>(t), cuda::std::forward<U>(u));
}

_LIBCUDACXX_TEMPLATE(class T)
_LIBCUDACXX_REQUIRES(swappable<T>)
__host__ __device__ constexpr void lv_swap(T& t1, T& t2)
{
  cuda::std::ranges::swap(t1, t2);
}

namespace N
{
struct A
{
  int m;
};
struct Proxy
{
  A* a;
  __host__ __device__ constexpr Proxy(A& a_)
      : a{&a_}
  {}
  __host__ __device__ friend constexpr void swap(Proxy x, Proxy y)
  {
    cuda::std::ranges::swap(*x.a, *y.a);
  }
};
__host__ __device__ constexpr Proxy proxy(A& a)
{
  return Proxy{a};
}
} // namespace N
__host__ __device__ constexpr bool CheckRegression()
{
  int i = 1, j = 2;
  lv_swap(i, j);
  assert(i == 2 && j == 1);

  N::A a1 = {5}, a2 = {-5};
  value_swap(a1, proxy(a2));
  assert(a1.m == -5 && a2.m == 5);
  return true;
}

static_assert(CheckRegression(), "");
} // namespace LWG3175
#endif // !TEST_COMPILER_MSVC_2017

int main(int, char**)
{
  return 0;
}
