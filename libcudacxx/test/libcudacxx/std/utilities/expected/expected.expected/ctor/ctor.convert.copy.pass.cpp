//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

//  template<class U, class G>
//    constexpr explicit(see below) expected(const expected<U, G>&);
//
// Let:
// - UF be const U&
// - GF be const G&
//
// Constraints:
// - is_constructible_v<T, UF> is true; and
// - is_constructible_v<E, GF> is true; and
// - is_constructible_v<T, expected<U, G>&> is false; and
// - is_constructible_v<T, expected<U, G>> is false; and
// - is_constructible_v<T, const expected<U, G>&> is false; and
// - is_constructible_v<T, const expected<U, G>> is false; and
// - is_convertible_v<expected<U, G>&, T> is false; and
// - is_convertible_v<expected<U, G>&&, T> is false; and
// - is_convertible_v<const expected<U, G>&, T> is false; and
// - is_convertible_v<const expected<U, G>&&, T> is false; and
// - is_constructible_v<unexpected<E>, expected<U, G>&> is false; and
// - is_constructible_v<unexpected<E>, expected<U, G>> is false; and
// - is_constructible_v<unexpected<E>, const expected<U, G>&> is false; and
// - is_constructible_v<unexpected<E>, const expected<U, G>> is false.
//
// Effects: If rhs.has_value(), direct-non-list-initializes val with cuda::std::forward<UF>(*rhs). Otherwise,
// direct-non-list-initializes unex with cuda::std::forward<GF>(rhs.error()).
//
// Postconditions: rhs.has_value() is unchanged; rhs.has_value() == this->has_value() is true.
//
// Throws: Any exception thrown by the initialization of val or unex.
//
// Remarks: The expression inside explicit is equivalent to !is_convertible_v<UF, T> || !is_convertible_v<GF, E>.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

// Test Constraints:
template <class T1, class Err1, class T2, class Err2>
constexpr bool canCstrFromExpected =
  cuda::std::is_constructible<cuda::std::expected<T1, Err1>, const cuda::std::expected<T2, Err2>&>::value;

struct CtorFromInt
{
  __host__ __device__ CtorFromInt(int);
};

static_assert(canCstrFromExpected<CtorFromInt, int, int, int>, "");

struct NoCtorFromInt
{};

// !is_constructible_v<T, UF>
static_assert(!canCstrFromExpected<NoCtorFromInt, int, int, int>, "");

// !is_constructible_v<E, GF>
static_assert(!canCstrFromExpected<int, NoCtorFromInt, int, int>, "");

template <class T>
struct CtorFrom
{
  _LIBCUDACXX_TEMPLATE(class T2 = T)
  _LIBCUDACXX_REQUIRES((!cuda::std::same_as<T2, int>) )
  __host__ __device__ explicit CtorFrom(int);
  __host__ __device__ explicit CtorFrom(T);

  template <class U>
  __host__ __device__ explicit CtorFrom(U&&) = delete;
};

// is_constructible_v<T, expected<U, G>&>
static_assert(!canCstrFromExpected<CtorFrom<cuda::std::expected<int, int>&>, int, int, int>, "");

// is_constructible_v<T, expected<U, G>>
static_assert(!canCstrFromExpected<CtorFrom<cuda::std::expected<int, int>&&>, int, int, int>, "");

// is_constructible_v<T, expected<U, G>&>
// note that this is true because it is covered by the other overload
//   template<class U = T> constexpr explicit(see below) expected(U&& v);
// The fact that it is not ambiguous proves that the overload under testing is removed
static_assert(canCstrFromExpected<CtorFrom<cuda::std::expected<int, int> const&>, int, int, int>, "");

// is_constructible_v<T, expected<U, G>>
static_assert(!canCstrFromExpected<CtorFrom<cuda::std::expected<int, int> const&&>, int, int, int>, "");

template <class T>
struct ConvertFrom
{
  _LIBCUDACXX_TEMPLATE(class T2 = T)
  _LIBCUDACXX_REQUIRES((!cuda::std::same_as<T2, int>) )
  __host__ __device__ ConvertFrom(int);
  __host__ __device__ ConvertFrom(T);

  template <class U>
  __host__ __device__ ConvertFrom(U&&) = delete;
};

// is_convertible_v<expected<U, G>&, T>
static_assert(!canCstrFromExpected<ConvertFrom<cuda::std::expected<int, int>&>, int, int, int>, "");

// is_convertible_v<expected<U, G>&&, T>
static_assert(!canCstrFromExpected<ConvertFrom<cuda::std::expected<int, int>&&>, int, int, int>, "");

// is_convertible_v<const expected<U, G>&, T>
// note that this is true because it is covered by the other overload
//   template<class U = T> constexpr explicit(see below) expected(U&& v);
// The fact that it is not ambiguous proves that the overload under testing is removed
static_assert(canCstrFromExpected<ConvertFrom<cuda::std::expected<int, int> const&>, int, int, int>, "");

// is_convertible_v<const expected<U, G>&&, T>
static_assert(!canCstrFromExpected<ConvertFrom<cuda::std::expected<int, int> const&&>, int, int, int>, "");

// Note for below 4 tests, because their E is constructible from cvref of cuda::std::expected<int, int>,
// unexpected<E> will be constructible from cvref of cuda::std::expected<int, int>
// is_constructible_v<unexpected<E>, expected<U, G>&>
static_assert(!canCstrFromExpected<int, CtorFrom<cuda::std::expected<int, int>&>, int, int>, "");

// is_constructible_v<unexpected<E>, expected<U, G>>
static_assert(!canCstrFromExpected<int, CtorFrom<cuda::std::expected<int, int>&&>, int, int>, "");

// is_constructible_v<unexpected<E>, const expected<U, G>&> is false
static_assert(!canCstrFromExpected<int, CtorFrom<cuda::std::expected<int, int> const&>, int, int>, "");

// is_constructible_v<unexpected<E>, const expected<U, G>>
static_assert(!canCstrFromExpected<int, CtorFrom<cuda::std::expected<int, int> const&&>, int, int>, "");

// test explicit
static_assert(cuda::std::is_convertible_v<const cuda::std::expected<int, int>&, cuda::std::expected<short, long>>, "");

// !is_convertible_v<UF, T>
static_assert(
  cuda::std::is_constructible_v<cuda::std::expected<CtorFrom<int>, int>, const cuda::std::expected<int, int>&>, "");
static_assert(
  !cuda::std::is_convertible_v<const cuda::std::expected<int, int>&, cuda::std::expected<CtorFrom<int>, int>>, "");

// !is_convertible_v<GF, E>.
static_assert(
  cuda::std::is_constructible_v<cuda::std::expected<int, CtorFrom<int>>, const cuda::std::expected<int, int>&>, "");
static_assert(
  !cuda::std::is_convertible_v<const cuda::std::expected<int, int>&, cuda::std::expected<int, CtorFrom<int>>>, "");

struct Data
{
  int i;
  __host__ __device__ constexpr Data(int ii)
      : i(ii)
  {}
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // convert the value
  {
    const cuda::std::expected<int, int> e1(5);
    cuda::std::expected<Data, int> e2 = e1;
    assert(e2.has_value());
    assert(e2.value().i == 5);
    assert(e1.has_value());
    assert(e1.value() == 5);
  }

  // convert the error
  {
    const cuda::std::expected<int, int> e1(cuda::std::unexpect, 5);
    cuda::std::expected<int, Data> e2 = e1;
    assert(!e2.has_value());
    assert(e2.error().i == 5);
    assert(!e1.has_value());
    assert(e1.error() == 5);
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{
  struct Except
  {};

  struct ThrowingInt
  {
    __host__ __device__ ThrowingInt(int)
    {
      throw Except{};
    }
  };

  // throw on converting value
  {
    const cuda::std::expected<int, int> e1;
    try
    {
      cuda::std::expected<ThrowingInt, int> e2 = e1;
      unused(e2);
      assert(false);
    }
    catch (Except)
    {}
  }

  // throw on converting error
  {
    const cuda::std::expected<int, int> e1(cuda::std::unexpect);
    try
    {
      cuda::std::expected<int, ThrowingInt> e2 = e1;
      unused(e2);
      assert(false);
    }
    catch (Except)
    {}
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
