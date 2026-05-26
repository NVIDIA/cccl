//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// gcc-10 segfaults with any use of constant_wrapper, gcc-11 fails to evaluate:
//   typename decltype(__cw_fixed_value(_Xp))::type
// UNSUPPORTED: gcc-10 || gcc-11

// nvcc 12.0 segfaults.
// UNSUPPORTED: nvcc-12.0

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
// UNSUPPORTED: nvrtc

// REQUIRES: !c++17

// constant_wrapper

// template<class... Args>
// static constexpr decltype(auto) operator[](Args&&... args) noexcept(see below);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/utility>

#include "helpers.h"
#include "MoveOnly.h"
#include "test_macros.h"

#if _CCCL_COMPILER(GCC, >=, 13)
TEST_DIAG_SUPPRESS_GCC("-Wdangling-reference")
#endif // _CCCL_COMPILER(GCC, >=, 13)

struct MoveOnlyIndex
{
  TEST_FUNC constexpr MoveOnly operator[](const MoveOnly& m1) const
  {
    return MoveOnly(m1.get());
  }

#if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
  TEST_FUNC constexpr MoveOnly operator[](const MoveOnly& m1, MoveOnly m2, MoveOnly&& m3) const
  {
    return MoveOnly(m1.get() + m2.get() + m3.get());
  }
#endif // !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
};

struct Nary
{
  TEST_FUNC constexpr int operator[](auto... args) const
  {
    return sizeof...(args);
  }
};

struct OverloadSet
{
  TEST_FUNC constexpr int operator[](int) const
  {
    return 1;
  }

  TEST_FUNC constexpr int operator[](cuda::std::__constant_wrapper<42>) const
  {
    return 2;
  }
};

struct ReturnNonStructural
{
  TEST_FUNC constexpr NonStructural operator[](int i) const
  {
    return NonStructural{i};
  }
};

struct CWOnly
{
  TEST_FUNC constexpr int operator[](cuda::std::__constant_wrapper<42>) const
  {
    return 42;
  }
};

struct ThrowingSubscript
{
  TEST_FUNC constexpr int operator[](int) const
  {
    return 42;
  }
};

struct NothrowSubscript
{
  TEST_FUNC constexpr int operator[](int) const noexcept
  {
    return 42;
  }
};

constexpr int arr[] = {1, 2, 3, 4};

// Let subscr-expr be constant_wrapper<value[remove_cvref_t<Args>::value...]>{} if all types in remove_cvref_t<Args>...
// satisfy constexpr-param and constant_wrapper<value[remove_cvref_t<Args>::value...]> is a valid type, otherwise let
// subscr-expr be value[cuda::std::forward<Args>(args)...].
// - Constraints: subscr-expr is a valid expression.
// - Remarks: The exception specification is equivalent to noexcept(subscr-expr).

#if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
template <class T, class... Args>
concept HasSubscript = requires(T t, Args&&... args) {
  { t[cuda::std::forward<Args>(args)...] };
};

template <class T, class... Args>
concept HasNothrowSubscript = requires(T t, Args&&... args) {
  { t[cuda::std::forward<Args>(args)...] } noexcept;
};
#else // ^^^ _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^ / vvv !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() vvv
template <class T, class Arg>
concept HasSubscript = requires(T t, Arg&& arg) {
  { t[cuda::std::forward<Arg>(arg)] };
};

template <class T, class Arg>
concept HasNothrowSubscript = requires(T t, Arg&& arg) {
  { t[cuda::std::forward<Arg>(arg)] } noexcept;
};
#endif // ^^^ !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^

static_assert(!HasSubscript<cuda::std::__constant_wrapper<4>, cuda::std::__constant_wrapper<1>>);

static_assert(HasSubscript<cuda::std::__constant_wrapper<arr>, int>);
static_assert(HasSubscript<cuda::std::__constant_wrapper<arr>, cuda::std::__constant_wrapper<1>>);

static_assert(HasNothrowSubscript<cuda::std::__constant_wrapper<arr>, int>);
static_assert(HasNothrowSubscript<cuda::std::__constant_wrapper<arr>, cuda::std::__constant_wrapper<1>>);

static_assert(HasSubscript<cuda::std::__constant_wrapper<NothrowSubscript{}>, int>);
static_assert(HasNothrowSubscript<cuda::std::__constant_wrapper<NothrowSubscript{}>, int>);

static_assert(HasSubscript<cuda::std::__constant_wrapper<ThrowingSubscript{}>, int>);
static_assert(!HasNothrowSubscript<cuda::std::__constant_wrapper<ThrowingSubscript{}>, int>);
static_assert(HasNothrowSubscript<cuda::std::__constant_wrapper<ThrowingSubscript{}>, cuda::std::__constant_wrapper<1>>,
              "the subscript expression is still nothrow because the constexpr path is taken");

template <class T>
struct MustBeInt
{
  static_assert(cuda::std::same_as<T, int>);
};

struct Poison
{
  template <class T>
  __host__ __device__ constexpr auto operator[](T) const noexcept -> MustBeInt<T>
  {
    return {};
  }
};

#if _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR()
#  define TEST_SUBSCRIPT(T, ...) T::operator[](__VA_ARGS__)
#else // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() vvv
#  define TEST_SUBSCRIPT(T, ...) T{}[__VA_ARGS__]
#endif // ^^^ !_CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^

TEST_FUNC constexpr bool test()
{
  {
    // with runtime param
    using T                                              = cuda::std::__constant_wrapper<arr>;
    cuda::std::same_as<const int&> decltype(auto) result = TEST_SUBSCRIPT(T, 1);
    assert(result == 2);
  }
  {
    // with constexpr param
    using T                                                                    = cuda::std::__constant_wrapper<arr>;
    cuda::std::same_as<cuda::std::__constant_wrapper<2>> decltype(auto) result = TEST_SUBSCRIPT(T, cuda::std::__cw<1>);
    static_assert(result == 2);
  }

#if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
  {
    // null-ary
    using T                                                                    = cuda::std::__constant_wrapper<Nary{}>;
    cuda::std::same_as<cuda::std::__constant_wrapper<0>> decltype(auto) result = TEST_SUBSCRIPT(T, );
    static_assert(result == 0);
  }

  {
    // n-ary
    using T = cuda::std::__constant_wrapper<Nary{}>;
    cuda::std::same_as<cuda::std::__constant_wrapper<3>> decltype(auto) result =
      TEST_SUBSCRIPT(T, cuda::std::__cw<1>, cuda::std::__cw<2>, cuda::std::__cw<3>);
    static_assert(result == 3);
  }

  {
    // mixing constexpr and runtime
    using T                                       = cuda::std::__constant_wrapper<Nary{}>;
    cuda::std::same_as<int> decltype(auto) result = TEST_SUBSCRIPT(T, cuda::std::__cw<1>, 2, cuda::std::__cw<3>);
    assert(result == 3);
  }
#endif // _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()

  {
    // move only
    using T = cuda::std::__constant_wrapper<MoveOnlyIndex{}>;
    MoveOnly m1(1);
    cuda::std::same_as<MoveOnly> decltype(auto) result = TEST_SUBSCRIPT(T, cuda::std::move(m1));
    assert(result.get() == 1);
  }
#if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
  {
    // move only n-ary
    using T = cuda::std::__constant_wrapper<MoveOnlyIndex{}>;
    MoveOnly m1(1), m2(2), m3(3);
    cuda::std::same_as<MoveOnly> decltype(auto) result =
      TEST_SUBSCRIPT(T, m1, cuda::std::move(m2), cuda::std::move(m3));
    assert(result.get() == 6);
  }
#endif // _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
  {
    // overload set
    // will always unwrap the constexpr params and call the non-constexpr overload
    using T                                        = cuda::std::__constant_wrapper<OverloadSet{}>;
    cuda::std::same_as<int> decltype(auto) result1 = TEST_SUBSCRIPT(T, 42);
    assert(result1 == 1);
    cuda::std::same_as<cuda::std::__constant_wrapper<1>> decltype(auto) result2 =
      TEST_SUBSCRIPT(T, cuda::std::__cw<42>);
    static_assert(result2 == 1);
  }

  {
    // return non-structural type
    using T                                                 = cuda::std::__constant_wrapper<ReturnNonStructural{}>;
    cuda::std::same_as<NonStructural> decltype(auto) result = TEST_SUBSCRIPT(T, 5);
    assert(result.get() == 5);
  }

  {
    // return non-structural type with constexpr param
    using T                                                 = cuda::std::__constant_wrapper<ReturnNonStructural{}>;
    cuda::std::same_as<NonStructural> decltype(auto) result = TEST_SUBSCRIPT(T, cuda::std::__cw<5>);
    assert(result.get() == 5);
  }

  {
    // cw only
    // the upwrapping case doesn't work so it falls back to the normal invoke path
    using T                                       = cuda::std::__constant_wrapper<CWOnly{}>;
    cuda::std::same_as<int> decltype(auto) result = TEST_SUBSCRIPT(T, cuda::std::__cw<42>);
    assert(result == 42);
  }

  {
    // just use the index operator
    assert(cuda::std::__cw<"abcd">[2] == 'c');
    assert(cuda::std::__cw<"abcd">[cuda::std::__cw<3>] == 'd');
  }

  {
    // integral_constant
    using T = cuda::std::__constant_wrapper<arr>;
    cuda::std::same_as<cuda::std::__constant_wrapper<2>> decltype(auto) result =
      TEST_SUBSCRIPT(T, cuda::std::integral_constant<int, 1>{});
    static_assert(result == 2);
  }

  {
    using T = cuda::std::__constant_wrapper<Poison{}>;
    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<MustBeInt<int>{}>> decltype(auto) result =
      TEST_SUBSCRIPT(T, cuda::std::__cw<5>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
