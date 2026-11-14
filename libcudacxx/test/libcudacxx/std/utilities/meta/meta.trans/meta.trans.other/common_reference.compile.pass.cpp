//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// type_traits

// common_reference

#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T, class = void>
inline constexpr bool has_type = false;

template <class T>
inline constexpr bool has_type<T, cuda::std::void_t<typename T::type>> = true;

// A slightly simplified variation of cuda::std::tuple
template <class...>
struct UserTuple
{};

template <class, class, class>
struct Tuple_helper
{};

template <class... Ts, class... Us>
struct Tuple_helper<cuda::std::void_t<cuda::std::common_reference_t<Ts, Us>...>, UserTuple<Ts...>, UserTuple<Us...>>
{
  using type = UserTuple<cuda::std::common_reference_t<Ts, Us>...>;
};

namespace cuda::std
{
template <class... Ts, class... Us, template <class> class TQual, template <class> class UQual>
struct basic_common_reference<::UserTuple<Ts...>, ::UserTuple<Us...>, TQual, UQual>
    : ::Tuple_helper<void, ::UserTuple<TQual<Ts>...>, ::UserTuple<UQual<Us>...>>
{};
} // namespace cuda::std

struct X2
{};
struct Y2
{};
struct Z2
{};

namespace cuda::std
{
template <>
struct common_type<X2, Y2>
{
  using type = Z2;
};

template <>
struct common_type<Y2, X2>
{
  using type = Z2;
};
} // namespace cuda::std

// (6.1)
//  -- If sizeof...(T) is zero, there shall be no member type.
static_assert(!has_type<cuda::std::common_reference<>>);

// (6.2)
//  -- Otherwise, if sizeof...(T) is one, let T0 denote the sole type in the
//     pack T. The member typedef type shall denote the same type as T0.
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<void>, void>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int>, int>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int&>, int&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int&&>, int&&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int const>, int const>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int const&>, int const&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int const&&>, int const&&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int volatile[]>, int volatile[]>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int volatile (&)[]>, int volatile (&)[]>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int volatile (&&)[]>, int volatile (&&)[]>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<void (&)()>, void (&)()>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<void (&&)()>, void (&&)()>);

// (6.3)
//  -- Otherwise, if sizeof...(T) is two, let T1 and T2 denote the two types in
//     the pack T. Then
// (6.3.1)
//    -- Let R be COMMON-REF(T1, T2). If T1 and T2 are reference types, R is well-formed,
//       and is_convertible_v<add_pointer_t<T1>, add_pointer_t<R>> && is_convertible_v<add_pointer_t<T2>,
//       add_pointer_t<R>> is true, then the member typedef type denotes R.

struct B
{};
struct D : B
{};
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B&, D&>, B&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B const&, D&>, B const&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B&, D const&>, B const&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B&, D const&, D&>, B const&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B&, D&, B&, D&>, B&>);

static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B&&, D&&>, B&&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B const&&, D&&>, B const&&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B&&, D const&&>, B const&&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B&, D&&>, B const&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B&, D const&&>, B const&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B const&, D&&>, B const&>);

static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B&&, D&>, B const&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B&&, D const&>, B const&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<B const&&, D&>, B const&>);

static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int const&, int volatile&>, int const volatile&>);
static_assert(
  cuda::std::is_same_v<cuda::std::common_reference_t<int const volatile&&, int volatile&&>, int const volatile&&>);

static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int (&)[10], int (&&)[10]>, int const (&)[10]>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int const (&)[10], int volatile (&)[10]>,
                                   int const volatile (&)[10]>);

// when conversion from pointers are not true
struct E
{};
struct F
{
  TEST_FUNC operator E&() const;
};

static_assert(!cuda::std::is_convertible_v<F*, E*>);

// The following should not use 6.3.1, but fallback to 6.3.3
// MSVC 19.44 does not consider the conversion operator when forming COND-RES(E&, F)
#if !TEST_COMPILER(MSVC, ==, 19, 44)
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<E&, F>, E&>);
#endif // !TEST_COMPILER(MSVC, ==, 19, 44)

// (6.3.2)
//    -- Otherwise, if basic_common_reference<remove_cvref_t<T1>,
//       remove_cvref_t<T2>, XREF(T1), XREF(T2)>::type is well-formed, then the
//       member typedef type denotes that type.
static_assert(
  cuda::std::is_same_v<cuda::std::common_reference_t<const UserTuple<int, short>&, UserTuple<int&, short volatile&>>,
                       UserTuple<const int&, const volatile short&>>);

static_assert(
  cuda::std::is_same_v<cuda::std::common_reference_t<volatile UserTuple<int, short>&, const UserTuple<int, short>&>,
                       const volatile UserTuple<int, short>&>);

// (6.3.3)
//    -- Otherwise, if COND_RES(T1, T2) is well-formed, then the member typedef
//       type denotes that type.
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<void, void>, void>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int, short>, int>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int, short&>, int>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int&, short&>, int>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int&, short>, int>);

// tricky volatile reference case
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int&&, int volatile&>, int>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int volatile&, int&&>, int>);

static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int (&)[10], int (&)[11]>, int*>);

// https://github.com/ericniebler/stl2/issues/338
struct MyIntRef
{
  TEST_FUNC MyIntRef(int&);
};
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int&, MyIntRef>, MyIntRef>);

// (6.3.4)
//    -- Otherwise, if common_type_t<T1, T2> is well-formed, then the member
//       typedef type denotes that type.
struct moveonly
{
  moveonly()                      = default;
  moveonly(moveonly&&)            = default;
  moveonly& operator=(moveonly&&) = default;
};
struct moveonly2 : moveonly
{};

static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<moveonly const&, moveonly>, moveonly>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<moveonly2 const&, moveonly>, moveonly>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<moveonly const&, moveonly2>, moveonly>);

static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<X2&, Y2 const&>, Z2>);

// (6.3.5)
//    -- Otherwise, there shall be no member type.
static_assert(!has_type<cuda::std::common_reference<volatile UserTuple<short>&, const UserTuple<int, short>&>>);

// (6.4)
//  -- Otherwise, if sizeof...(T) is greater than two, let T1, T2, and Rest,
//     respectively, denote the first, second, and (pack of) remaining types
//     comprising T. Let C be the type common_reference_t<T1, T2>. Then:
// (6.4.1)
//    -- If there is such a type C, the member typedef type shall denote the
//       same type, if any, as common_reference_t<C, Rest...>.
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int, int, int>, int>);
static_assert(
  cuda::std::is_same_v<cuda::std::common_reference_t<int&&, int const&, int volatile&>, int const volatile&>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<int&&, int const&, float&>, float>);

// (6.4.2)
//    -- Otherwise, there shall be no member type.
static_assert(!has_type<cuda::std::common_reference<int, short, int, char*>>);

static_assert(
  cuda::std::is_same_v<cuda::std::common_reference_t<cuda::std::tuple<int, int>>, cuda::std::tuple<int, int>>);
static_assert(
  cuda::std::is_same_v<cuda::std::common_reference_t<cuda::std::tuple<int, long>, cuda::std::tuple<long, int>>,
                       cuda::std::tuple<long, long>>);
static_assert(cuda::std::is_same_v<
              cuda::std::common_reference_t<cuda::std::tuple<int&, const int&>, cuda::std::tuple<const int&, int>>,
              cuda::std::tuple<const int&, int>>);
static_assert(cuda::std::is_same_v<
              cuda::std::common_reference_t<cuda::std::tuple<int&, volatile int&>, cuda::std::tuple<volatile int&, int>>,
              cuda::std::tuple<volatile int&, int>>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<cuda::std::tuple<int&, const volatile int&>,
                                                                 cuda::std::tuple<const volatile int&, int>>,
                                   cuda::std::tuple<const volatile int&, int>>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<cuda::std::tuple<const int&, volatile int&>,
                                                                 cuda::std::tuple<volatile int&, const int&>>,
                                   cuda::std::tuple<const volatile int&, const volatile int&>>);

static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<cuda::std::tuple<int, X2>, cuda::std::tuple<int, Y2>>,
                                   cuda::std::tuple<int, Z2>>);
static_assert(
  !has_type<cuda::std::common_reference<cuda::std::tuple<int, const X2>, cuda::std::tuple<float, const Z2>>>);
static_assert(!has_type<cuda::std::common_reference<cuda::std::tuple<int, X2>, cuda::std::tuple<float, Z2>>>);
static_assert(!has_type<cuda::std::common_reference<cuda::std::tuple<int, X2>, int, X2>>);

struct A
{};

namespace cuda::std
{
template <template <class> class TQual, template <class> class UQual>
struct basic_common_reference<A, tuple<B>, TQual, UQual>
{
  using type = tuple<UQual<B>>;
};
} // namespace cuda::std

static_assert(
  cuda::std::is_same_v<cuda::std::common_reference_t<A, cuda::std::tuple<B>, cuda::std::tuple<D>>, cuda::std::tuple<B>>);

static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<cuda::std::pair<int, int>>, cuda::std::pair<int, int>>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<cuda::std::pair<int, long>, cuda::std::pair<long, int>>,
                                   cuda::std::pair<long, long>>);
static_assert(cuda::std::is_same_v<
              cuda::std::common_reference_t<cuda::std::pair<int&, const int&>, cuda::std::pair<const int&, int>>,
              cuda::std::pair<const int&, int>>);
static_assert(cuda::std::is_same_v<
              cuda::std::common_reference_t<cuda::std::pair<int&, volatile int&>, cuda::std::pair<volatile int&, int>>,
              cuda::std::pair<volatile int&, int>>);
static_assert(
  cuda::std::is_same_v<
    cuda::std::common_reference_t<cuda::std::pair<int&, const volatile int&>, cuda::std::pair<const volatile int&, int>>,
    cuda::std::pair<const volatile int&, int>>);
static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<cuda::std::pair<const int&, volatile int&>,
                                                                 cuda::std::pair<volatile int&, const int&>>,
                                   cuda::std::pair<const volatile int&, const volatile int&>>);

static_assert(cuda::std::is_same_v<cuda::std::common_reference_t<cuda::std::pair<int, X2>, cuda::std::pair<int, Y2>>,
                                   cuda::std::pair<int, Z2>>);
static_assert(!has_type<cuda::std::common_reference<cuda::std::pair<int, const X2>, cuda::std::pair<float, const Z2>>>);
static_assert(!has_type<cuda::std::common_reference<cuda::std::pair<int, X2>, cuda::std::pair<float, Z2>>>);
static_assert(!has_type<cuda::std::common_reference<cuda::std::pair<int, X2>, int, X2>>);

int main(int, char**)
{
  return 0;
}
