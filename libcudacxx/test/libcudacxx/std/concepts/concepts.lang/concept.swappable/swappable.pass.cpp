//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class T>
// concept swappable = // see below

#include "type_classification/swappable.h"

#include <cuda/std/cassert>
#include <cuda/std/concepts>

#include "test_macros.h"
#include "type_classification/moveconstructible.h"

#ifdef TEST_COMPILER_MSVC_2017
#  pragma warning(disable : 4239)
#endif // TEST_COMPILER_MSVC_2017

using cuda::std::swappable;

template <class T>
struct expected
{
  T x;
  T y;
};

// clang-format off
// Checks [concept.swappable]/2.1
_LIBCUDACXX_TEMPLATE(class T, class U)
  _LIBCUDACXX_REQUIRES( cuda::std::same_as<cuda::std::remove_cvref_t<T>, cuda::std::remove_cvref_t<U> > &&
         swappable<cuda::std::remove_cvref_t<T> >) //
__host__ __device__ constexpr bool check_swap_21(T&& x, U&& y) {
  expected<cuda::std::remove_cvref_t<T> > const e{y, x};
  cuda::std::ranges::swap(cuda::std::forward<T>(x), cuda::std::forward<U>(y));
  return x == e.x && y == e.y;
}

// Checks [concept.swappable]/2.2
_LIBCUDACXX_TEMPLATE(class T, cuda::std::size_t N)
  _LIBCUDACXX_REQUIRES( swappable<T>)
__host__ __device__ constexpr bool check_swap_22(T (&x)[N], T (&y)[N]) {
  expected<T[N]> e{};
  for (cuda::std::size_t i = 0; i < N; ++i) {
    e.x[i] = y[i];
    e.y[i] = x[i];
  }

  cuda::std::ranges::swap(x, y);
  for (cuda::std::size_t i = 0; i < N; ++i) {
    if (x[i] == e.x[i] && y[i] == e.y[i]) {
      continue;
    }
    return false;
  }
  return true;
}

// Checks [concept.swappable]/2.3
_LIBCUDACXX_TEMPLATE(class T)
  _LIBCUDACXX_REQUIRES( swappable<T> && cuda::std::copy_constructible<cuda::std::remove_cvref_t<T> >)
__host__ __device__ constexpr bool check_swap_23(T x, T y) {
  expected<cuda::std::remove_cvref_t<T> > const e{y, x};
  cuda::std::ranges::swap(x, y);
  return x == e.x && y == e.y;
}
// clang-format on
__host__ __device__ constexpr bool check_lvalue_adl_swappable()
{
  auto x = lvalue_adl_swappable(0);
  auto y = lvalue_adl_swappable(1);
  ASSERT_NOEXCEPT(cuda::std::ranges::swap(x, y));
  assert(check_swap_21(x, y));
  return true;
}

__host__ __device__ constexpr bool check_rvalue_adl_swappable()
{
  ASSERT_NOEXCEPT(cuda::std::ranges::swap(rvalue_adl_swappable(0), rvalue_adl_swappable(1)));
#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 10)
  assert(check_swap_21(rvalue_adl_swappable(0), rvalue_adl_swappable(1)));
#endif
  return true;
}

__host__ __device__ constexpr bool check_lvalue_rvalue_adl_swappable()
{
  auto x = lvalue_rvalue_adl_swappable(0);
  ASSERT_NOEXCEPT(cuda::std::ranges::swap(x, lvalue_rvalue_adl_swappable(1)));
  assert(check_swap_21(x, lvalue_rvalue_adl_swappable(1)));
  return true;
}

__host__ __device__ constexpr bool check_rvalue_lvalue_adl_swappable()
{
  auto x = rvalue_lvalue_adl_swappable(0);
  ASSERT_NOEXCEPT(cuda::std::ranges::swap(rvalue_lvalue_adl_swappable(1), x));
  assert(check_swap_21(rvalue_lvalue_adl_swappable(1), x));
  return true;
}

__host__ __device__ constexpr bool check_throwable_swappable()
{
  auto x = throwable_adl_swappable{0};
  auto y = throwable_adl_swappable{1};
#if !defined(TEST_COMPILER_BROKEN_SMF_NOEXCEPT) && !defined(TEST_COMPILER_MSVC_2017)
  ASSERT_NOT_NOEXCEPT(cuda::std::ranges::swap(x, y));
#endif // !TEST_COMPILER_BROKEN_SMF_NOEXCEPT && !TEST_COMPILER_MSVC_2017
  assert(check_swap_21(x, y));
  return true;
}

__host__ __device__ constexpr bool check_non_move_constructible_adl_swappable()
{
  auto x = non_move_constructible_adl_swappable{0};
  auto y = non_move_constructible_adl_swappable{1};
  ASSERT_NOEXCEPT(cuda::std::ranges::swap(x, y));
  assert(check_swap_21(x, y));
  return true;
}

#if TEST_STD_VER > 2014
#  ifndef TEST_COMPILER_MSVC_2017
__host__ __device__ constexpr bool check_non_move_assignable_adl_swappable()
{
  auto x = non_move_assignable_adl_swappable{0};
  auto y = non_move_assignable_adl_swappable{1};
  ASSERT_NOEXCEPT(cuda::std::ranges::swap(x, y));
  assert(check_swap_21(x, y));
  return true;
}
#  endif // !TEST_COMPILER_MSVC_2017
#endif // TEST_STD_VER > 2014

namespace swappable_namespace
{
enum unscoped
{
  hello,
  world
};
__host__ __device__ void swap(unscoped&, unscoped&);

enum class scoped
{
  hello,
  world
};
__host__ __device__ void swap(scoped&, scoped&);
} // namespace swappable_namespace

static_assert(swappable<swappable_namespace::unscoped>, "");
static_assert(swappable<swappable_namespace::scoped>, "");

__host__ __device__ constexpr bool check_swap_arrays()
{
  int x[] = {0, 1, 2, 3, 4};
  int y[] = {5, 6, 7, 8, 9};
  ASSERT_NOEXCEPT(cuda::std::ranges::swap(x, y));
  assert(check_swap_22(x, y));
  return true;
}

__host__ __device__ constexpr bool check_lvalue_adl_swappable_arrays()
{
  lvalue_adl_swappable x[] = {{0}, {1}, {2}, {3}};
  lvalue_adl_swappable y[] = {{4}, {5}, {6}, {7}};
  ASSERT_NOEXCEPT(cuda::std::ranges::swap(x, y));
  assert(check_swap_22(x, y));
  return true;
}

__host__ __device__ constexpr bool check_throwable_adl_swappable_arrays()
{
  throwable_adl_swappable x[] = {{0}, {1}, {2}, {3}};
  throwable_adl_swappable y[] = {{4}, {5}, {6}, {7}};
#if !defined(TEST_COMPILER_BROKEN_SMF_NOEXCEPT) && !defined(TEST_COMPILER_MSVC_2017)
  ASSERT_NOT_NOEXCEPT(cuda::std::ranges::swap(x, y));
#endif // !TEST_COMPILER_BROKEN_SMF_NOEXCEPT && !TEST_COMPILER_MSVC_2017
  assert(check_swap_22(x, y));
  return true;
}

__device__ auto global_x = 0;
ASSERT_NOEXCEPT(cuda::std::ranges::swap(global_x, global_x));
static_assert(check_swap_23(0, 0), "");
static_assert(check_swap_23(0, 1), "");
static_assert(check_swap_23(1, 0), "");

__host__ __device__ constexpr bool check_swappable_references()
{
  int x = 42;
  int y = 64;
  ASSERT_NOEXCEPT(cuda::std::ranges::swap(x, y));
  assert(check_swap_23(x, y));
  return true;
}

__host__ __device__ constexpr bool check_swappable_pointers()
{
  char const* x = "hello";
  ASSERT_NOEXCEPT(cuda::std::ranges::swap(x, x));
  assert(check_swap_23(x, {}));
  return true;
}
namespace union_swap
{
union adl_swappable
{
  int x;
  double y;
};

__host__ __device__ void swap(adl_swappable&, adl_swappable&);
__host__ __device__ void swap(adl_swappable&&, adl_swappable&&);
} // namespace union_swap
static_assert(swappable<union_swap::adl_swappable>, "");
static_assert(swappable<union_swap::adl_swappable&>, "");
static_assert(swappable<union_swap::adl_swappable&&>, "");

// All tests for swappable<T> are implicitly confirmed by `check_swap`, so we only need to
// sanity check for a few positive cases.
static_assert(swappable<int volatile&>, "");
static_assert(swappable<int&&>, "");
static_assert(swappable<int (*)()>, "");
static_assert(swappable<int rvalue_adl_swappable::*>, "");
static_assert(swappable<int (rvalue_adl_swappable::*)()>, "");

static_assert(!swappable<void>, "");
static_assert(!swappable<int const>, "");
static_assert(!swappable<int const&>, "");
static_assert(!swappable<int const&&>, "");
static_assert(!swappable<int const volatile>, "");
static_assert(!swappable<int const volatile&>, "");
static_assert(!swappable<int const volatile&&>, "");
static_assert(!swappable<int (&)()>, "");
static_assert(!swappable<DeletedMoveCtor>, "");
static_assert(!swappable<ImplicitlyDeletedMoveCtor>, "");
static_assert(!swappable<DeletedMoveAssign>, "");
static_assert(!swappable<ImplicitlyDeletedMoveAssign>, "");
static_assert(!swappable<NonMovable>, "");
static_assert(!swappable<DerivedFromNonMovable>, "");
static_assert(!swappable<HasANonMovable>, "");

using swap_type = cuda::std::remove_const_t<decltype(cuda::std::ranges::swap)>;
static_assert(cuda::std::default_initializable<swap_type>, "");
static_assert(cuda::std::move_constructible<swap_type>, "");
static_assert(cuda::std::copy_constructible<swap_type>, "");
static_assert(cuda::std::assignable_from<swap_type&, swap_type>, "");
static_assert(cuda::std::assignable_from<swap_type&, swap_type&>, "");
static_assert(cuda::std::assignable_from<swap_type&, swap_type const&>, "");
static_assert(cuda::std::assignable_from<swap_type&, swap_type const>, "");
static_assert(swappable<swap_type>, "");

int main(int, char**)
{
  assert(check_lvalue_adl_swappable());
#if (!defined(__GNUC__) || __GNUC__ >= 10)
  assert(check_rvalue_adl_swappable());
#endif
  assert(check_lvalue_rvalue_adl_swappable());
  assert(check_rvalue_lvalue_adl_swappable());
  assert(check_throwable_swappable());
  assert(check_non_move_constructible_adl_swappable());
#if TEST_STD_VER > 2014
#  ifndef TEST_COMPILER_MSVC_2017
  assert(check_non_move_assignable_adl_swappable());
#  endif // TEST_COMPILER_MSVC_2017
#endif // TEST_STD_VER > 2014
  assert(check_swap_arrays());
  assert(check_lvalue_adl_swappable_arrays());
  assert(check_throwable_adl_swappable_arrays());
  assert(check_swappable_references());
  assert(check_swappable_pointers());

#if (!defined(__GNUC__) || __GNUC__ >= 10)
  static_assert(check_lvalue_adl_swappable(), "");
  static_assert(check_rvalue_adl_swappable(), "");
  static_assert(check_lvalue_rvalue_adl_swappable(), "");
  static_assert(check_rvalue_lvalue_adl_swappable(), "");
  static_assert(check_throwable_swappable(), "");
  static_assert(check_non_move_constructible_adl_swappable(), "");
#  if TEST_STD_VER > 2014
#    ifndef TEST_COMPILER_MSVC_2017
  static_assert(check_non_move_assignable_adl_swappable(), "");
#    endif // TEST_COMPILER_MSVC_2017
#  endif // TEST_STD_VER > 2014
  static_assert(check_swap_arrays(), "");
  static_assert(check_lvalue_adl_swappable_arrays(), "");
  static_assert(check_throwable_adl_swappable_arrays(), "");
  static_assert(check_swappable_references(), "");
  static_assert(check_swappable_pointers(), "");
#endif

  return 0;
}
