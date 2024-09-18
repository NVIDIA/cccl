//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__type_traits/type_list.h>

// Test that the type_list header is self-contained.
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_identity.h>

#if defined(_CCCL_CUDACC_BELOW_12_2) || defined(_CCCL_COMPILER_ICC)
// These compilers have trouble making substitution failures during
// alias template instantiation non-fatal.
#  define SKIP_SFINAE_TESTS
#endif

struct Incomplete;

struct Empty
{};

template <class... Ts>
struct AlwaysFalse
{
  static constexpr bool value = false;
};

template <class... Ts>
struct DoNotInstantiate
{
  static_assert(AlwaysFalse<Ts...>::value, "");
};

template <class T>
struct Identity
{
  using type = T;
};

template <class T>
struct HasType : ::cuda::std::__type_callable<::cuda::std::__type_quote1<::cuda::std::__type>, T>
{};

static_assert(!HasType<Incomplete>::value, "");
static_assert(!HasType<Empty>::value, "");
static_assert(!HasType<int>::value, "");
static_assert(HasType<Identity<int>>::value, "");

template <class... Ts>
struct Types
{};

template <int I>
struct Int
{
  static constexpr int value = I;
};

struct Fn
{
  template <class... Ts>
  using __call = Types<Ts...>;
};

struct Fn1
{
  template <class T>
  using __call = Types<T>;
};

struct Fn2
{
  template <class T, class U>
  using __call = Types<T, U>;
};

// __type
static_assert(::cuda::std::is_same<::cuda::std::__type<::cuda::std::__type_identity<Incomplete>>, Incomplete>::value,
              "");

// __type_call
static_assert(::cuda::std::is_same<::cuda::std::__type_call<Fn, Incomplete>, Types<Incomplete>>::value, "");
static_assert(::cuda::std::is_same<::cuda::std::__type_call<Fn1, Incomplete>, Types<Incomplete>>::value, "");
static_assert(::cuda::std::is_same<::cuda::std::__type_call<Fn2, Incomplete, Empty>, Types<Incomplete, Empty>>::value,
              "");
static_assert(::cuda::std::is_same<::cuda::std::__type_call1<Fn1, Incomplete>, Types<Incomplete>>::value, "");
static_assert(::cuda::std::is_same<::cuda::std::__type_call2<Fn2, Incomplete, Empty>, Types<Incomplete, Empty>>::value,
              "");

// __type_call_indirect
template <class... Ts, class = ::cuda::std::__type_call_indirect<Fn2, Ts...>>
_CCCL_HOST_DEVICE constexpr bool test_call_indirect(int)
{
  return true;
}

template <class... Ts>
_CCCL_HOST_DEVICE constexpr bool test_call_indirect(long)
{
  return false;
}

static_assert(test_call_indirect<Incomplete, Empty>(0), "");
#if !defined(SKIP_SFINAE_TESTS)
static_assert(!test_call_indirect<Incomplete>(0), "");
static_assert(!test_call_indirect<Incomplete, Empty, int>(0), "");
#endif

template <class... Ts>
struct Template
{
  using type = Types<Ts...>;
};

template <class T>
struct Template1
{
  using type = Types<T>;
};

template <class T, class U>
struct Template2
{
  using type = Types<T, U>;
};

// __type_quote
static_assert(::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_quote<Template>, Incomplete, Empty>,
                                   Template<Incomplete, Empty>>::value,
              "");
static_assert(::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_quote1<Template1>, Incomplete>,
                                   Template1<Incomplete>>::value,
              "");
static_assert(::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_quote2<Template2>, Incomplete, Empty>,
                                   Template2<Incomplete, Empty>>::value,
              "");

// __type_quote_trait
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_quote_trait<Template>, Incomplete, Empty>,
                       Types<Incomplete, Empty>>::value,
  "");
static_assert(::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_quote_trait1<Template1>, Incomplete>,
                                   Types<Incomplete>>::value,
              "");
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_quote_trait2<Template2>, Incomplete, Empty>,
                       Types<Incomplete, Empty>>::value,
  "");

// __type_compose
static_assert(
  ::cuda::std::is_same<
    ::cuda::std::__type_call<
      ::cuda::std::__type_compose<::cuda::std::__type_quote1<Template1>, ::cuda::std::__type_quote<Template>>,
      Incomplete,
      Empty>,
    Template1<Template<Incomplete, Empty>>>::value,
  "");

// __type_bind_back
static_assert(::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_bind_back<Fn, Incomplete>, Empty>,
                                   Types<Empty, Incomplete>>::value,
              "");

// __type_bind_back
static_assert(::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_bind_front<Fn, Incomplete>, Empty>,
                                   Types<Incomplete, Empty>>::value,
              "");

// __type_always
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_always<Incomplete>, Empty>, Incomplete>::value, "");

// __type_list
static_assert(::cuda::std::__type_list<int, float, double>::__size == 3, "");
static_assert(::cuda::std::__type_list<>::__size == 0, "");
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_list<int, float, double>, Fn, Incomplete, Empty>,
                       Types<int, float, double, Incomplete, Empty>>::value,
  "");

// __type_list_size
static_assert(::cuda::std::__type_list_size<::cuda::std::__type_list<int, float, double>>::value == 3, "");
static_assert(::cuda::std::__type_list_size<::cuda::std::__type_list<>>::value == 0, "");

// __type_list_push_back
static_assert(::cuda::std::is_same<::cuda::std::__type_push_back<::cuda::std::__type_list<int, float>, double>,
                                   ::cuda::std::__type_list<int, float, double>>::value,
              "");

// __type_list_push_front
static_assert(::cuda::std::is_same<::cuda::std::__type_push_front<::cuda::std::__type_list<int, float>, double>,
                                   ::cuda::std::__type_list<double, int, float>>::value,
              "");

// __type_callable
static_assert(::cuda::std::__type_callable<Fn2, Incomplete, Empty>::value, "");
static_assert(!::cuda::std::__type_callable<Fn2, Incomplete>::value, "");

// __type_defer
using EnsureDeferIsLazy = ::cuda::std::__type_defer<DoNotInstantiate<>, Empty>;
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_defer<Fn2, Incomplete, Empty>::type, Types<Incomplete, Empty>>::value, "");
static_assert(!HasType<::cuda::std::__type_defer<Fn2, Incomplete>>::value, "");

// __type_index
// NOTE: __type_index has a fast path for indices 16 and below.
static_assert(::cuda::std::__type_index_c<0, Int<42>, double>::value == 42, "");
static_assert(::cuda::std::__type_index_c<1, int, Int<42>, double>::value == 42, "");
static_assert(
  ::cuda::std::__type_index<::cuda::std::integral_constant<::cuda::std::size_t, 1>, int, Int<42>, double>::value == 42,
  "");
static_assert(::cuda::std::__type_callable<::cuda::std::__type_quote_indirect<::cuda::std::__type_index>,
                                           ::cuda::std::integral_constant<::cuda::std::size_t, 1>,
                                           int,
                                           Int<42>>::value,
              "");

#if !defined(SKIP_SFINAE_TESTS)
static_assert(!::cuda::std::__type_callable<::cuda::std::__type_quote_indirect<::cuda::std::__type_index>,
                                            ::cuda::std::integral_constant<::cuda::std::size_t, 1>,
                                            int>::value,
              "");
#endif

static_assert(
  ::cuda::std::__type_index_c<
    14,
    Int<0>,
    Int<1>,
    Int<2>,
    Int<3>,
    Int<4>,
    Int<5>,
    Int<6>,
    Int<7>,
    Int<8>,
    Int<9>,
    Int<10>,
    Int<11>,
    Int<12>,
    Int<13>,
    Int<14>,
    Int<15>>::value
    == 14,
  "");
static_assert(
  ::cuda::std::__type_index_c<
    15,
    Int<0>,
    Int<1>,
    Int<2>,
    Int<3>,
    Int<4>,
    Int<5>,
    Int<6>,
    Int<7>,
    Int<8>,
    Int<9>,
    Int<10>,
    Int<11>,
    Int<12>,
    Int<13>,
    Int<14>,
    Int<15>>::value
    == 15,
  "");
static_assert(
  ::cuda::std::__type_index_c<
    15,
    Int<0>,
    Int<1>,
    Int<2>,
    Int<3>,
    Int<4>,
    Int<5>,
    Int<6>,
    Int<7>,
    Int<8>,
    Int<9>,
    Int<10>,
    Int<11>,
    Int<12>,
    Int<13>,
    Int<14>,
    Int<15>,
    Int<16>,
    Int<17>>::value
    == 15,
  "");
static_assert(
  ::cuda::std::__type_index_c<
    16,
    Int<0>,
    Int<1>,
    Int<2>,
    Int<3>,
    Int<4>,
    Int<5>,
    Int<6>,
    Int<7>,
    Int<8>,
    Int<9>,
    Int<10>,
    Int<11>,
    Int<12>,
    Int<13>,
    Int<14>,
    Int<15>,
    Int<16>,
    Int<17>>::value
    == 16,
  "");
static_assert(
  ::cuda::std::__type_index_c<
    17,
    Int<0>,
    Int<1>,
    Int<2>,
    Int<3>,
    Int<4>,
    Int<5>,
    Int<6>,
    Int<7>,
    Int<8>,
    Int<9>,
    Int<10>,
    Int<11>,
    Int<12>,
    Int<13>,
    Int<14>,
    Int<15>,
    Int<16>,
    Int<17>>::value
    == 17,
  "");
static_assert(
  ::cuda::std::__type_index_c<
    32,
    Int<0>,
    Int<1>,
    Int<2>,
    Int<3>,
    Int<4>,
    Int<5>,
    Int<6>,
    Int<7>,
    Int<8>,
    Int<9>,
    Int<10>,
    Int<11>,
    Int<12>,
    Int<13>,
    Int<14>,
    Int<15>,
    Int<16>,
    Int<17>,
    Int<18>,
    Int<19>,
    Int<20>,
    Int<21>,
    Int<22>,
    Int<23>,
    Int<24>,
    Int<25>,
    Int<26>,
    Int<27>,
    Int<28>,
    Int<29>,
    Int<30>,
    Int<31>,
    Int<32>,
    Int<33>,
    Int<34>,
    Int<35>>::value
    == 32,
  "");
static_assert(
  ::cuda::std::__type_index_c<
    33,
    Int<0>,
    Int<1>,
    Int<2>,
    Int<3>,
    Int<4>,
    Int<5>,
    Int<6>,
    Int<7>,
    Int<8>,
    Int<9>,
    Int<10>,
    Int<11>,
    Int<12>,
    Int<13>,
    Int<14>,
    Int<15>,
    Int<16>,
    Int<17>,
    Int<18>,
    Int<19>,
    Int<20>,
    Int<21>,
    Int<22>,
    Int<23>,
    Int<24>,
    Int<25>,
    Int<26>,
    Int<27>,
    Int<28>,
    Int<29>,
    Int<30>,
    Int<31>,
    Int<32>,
    Int<33>,
    Int<34>,
    Int<35>>::value
    == 33,
  "");

// __type_at
static_assert(::cuda::std::__type_at_c<0, ::cuda::std::__type_list<Int<42>, double>>::value == 42, "");
static_assert(::cuda::std::__type_at_c<1, ::cuda::std::__type_list<int, Int<42>, double>>::value == 42, "");
static_assert(::cuda::std::__type_at<::cuda::std::integral_constant<::cuda::std::size_t, 1>,
                                     ::cuda::std::__type_list<int, Int<42>, double>>::value
                == 42,
              "");
static_assert(::cuda::std::__type_callable<::cuda::std::__type_quote_indirect<::cuda::std::__type_at>,
                                           ::cuda::std::integral_constant<::cuda::std::size_t, 1>,
                                           ::cuda::std::__type_list<int, Int<42>>>::value,
              "");
#if !defined(SKIP_SFINAE_TESTS)
static_assert(!::cuda::std::__type_callable<::cuda::std::__type_quote_indirect<::cuda::std::__type_at>,
                                            ::cuda::std::integral_constant<::cuda::std::size_t, 1>,
                                            ::cuda::std::__type_list<int>>::value,
              "");
#endif

// __type_front
static_assert(::cuda::std::__type_front<::cuda::std::__type_list<Int<42>, double>>::value == 42, "");
#if !defined(SKIP_SFINAE_TESTS)
static_assert(!::cuda::std::__type_callable<::cuda::std::__type_quote1<::cuda::std::__type_front>,
                                            ::cuda::std::__type_list<>>::value,
              "");
#endif

// __type_back
static_assert(::cuda::std::__type_back<::cuda::std::__type_list<double, Int<42>>>::value == 42, "");
#if !defined(SKIP_SFINAE_TESTS)
static_assert(
  !::cuda::std::__type_callable<::cuda::std::__type_quote1<::cuda::std::__type_back>, ::cuda::std::__type_list<>>::value,
  "");
#endif

// __type_concat
static_assert(::cuda::std::is_same<::cuda::std::__type_concat<>, ::cuda::std::__type_list<>>::value, "");

static_assert(
  ::cuda::std::is_same<::cuda::std::__type_concat<::cuda::std::__type_list<int>, ::cuda::std::__type_list<float>>,
                       ::cuda::std::__type_list<int, float>>::value,
  "");

static_assert(
  ::cuda::std::is_same<
    ::cuda::std::__type_concat<::cuda::std::__type_list<Int<1>>,
                               ::cuda::std::__type_list<>,
                               ::cuda::std::__type_list<Int<3>, int>,
                               ::cuda::std::__type_list<Int<4>, int>,
                               ::cuda::std::__type_list<Int<5>, int>,
                               ::cuda::std::__type_list<Int<6>, int>,
                               ::cuda::std::__type_list<Int<7>, int>,
                               ::cuda::std::__type_list<Int<8>, int>,
                               ::cuda::std::__type_list<Int<9>, int>,
                               ::cuda::std::__type_list<Int<10>, int>,
                               ::cuda::std::__type_list<Int<11>, int, short, float>>,
    ::cuda::std::__type_list<
      Int<1>,
      Int<3>,
      int,
      Int<4>,
      int,
      Int<5>,
      int,
      Int<6>,
      int,
      Int<7>,
      int,
      Int<8>,
      int,
      Int<9>,
      int,
      Int<10>,
      int,
      Int<11>,
      int,
      short,
      float>>::value,
  "");

// __type_flatten
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_flatten<::cuda::std::__type_list<>>, ::cuda::std::__type_list<>>::value, "");

static_assert(
  ::cuda::std::is_same<
    ::cuda::std::__type_flatten<::cuda::std::__type_list<
      ::cuda::std::__type_list<Int<1>>,
      ::cuda::std::__type_list<>,
      ::cuda::std::__type_list<Int<3>, int>,
      ::cuda::std::__type_list<Int<4>, int>,
      ::cuda::std::__type_list<Int<5>, int>,
      ::cuda::std::__type_list<Int<6>, int>,
      ::cuda::std::__type_list<Int<7>, int>,
      ::cuda::std::__type_list<Int<8>, int>,
      ::cuda::std::__type_list<Int<9>, int>,
      ::cuda::std::__type_list<Int<10>, int>,
      ::cuda::std::__type_list<Int<11>, int, short, float>>>,
    ::cuda::std::__type_list<
      Int<1>,
      Int<3>,
      int,
      Int<4>,
      int,
      Int<5>,
      int,
      Int<6>,
      int,
      Int<7>,
      int,
      Int<8>,
      int,
      Int<9>,
      int,
      Int<10>,
      int,
      Int<11>,
      int,
      short,
      float>>::value,
  "");

struct BiggerThanFour
{
  template <class T>
  using __call = ::cuda::std::bool_constant<(sizeof(T) > 4)>;
};

// __type_find_if
static_assert(::cuda::std::is_same<::cuda::std::__type_find_if<::cuda::std::__type_list<>, BiggerThanFour>,
                                   ::cuda::std::__type_list<>>::value,
              "");
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_find_if<::cuda::std::__type_list<char, char[2], Empty>, BiggerThanFour>,
                       ::cuda::std::__type_list<>>::value,
  "");
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_find_if<::cuda::std::__type_list<char, char[5], Empty>, BiggerThanFour>,
                       ::cuda::std::__type_list<char[5], Empty>>::value,
  "");

// __type_transform
struct AddPointer
{
  template <class T>
  using __call = T*;
};

static_assert(::cuda::std::is_same<::cuda::std::__type_transform<::cuda::std::__type_list<>, AddPointer>,
                                   ::cuda::std::__type_list<>>::value,
              "");
static_assert(::cuda::std::is_same<::cuda::std::__type_transform<::cuda::std::__type_list<int, char const>, AddPointer>,
                                   ::cuda::std::__type_list<int*, char const*>>::value,
              "");
#if !defined(SKIP_SFINAE_TESTS)
static_assert(!::cuda::std::__type_callable<::cuda::std::__type_quote2<::cuda::std::__type_transform>,
                                            ::cuda::std::__type_list<int, char const, int&>,
                                            AddPointer>::value,
              "");
#endif

template <class A, class B>
struct Pair
{};

using MakePair = ::cuda::std::__type_quote2<Pair>;

// __type_fold_right
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_fold_right<::cuda::std::__type_list<>, int, MakePair>, int>::value, "");
static_assert(::cuda::std::is_same<::cuda::std::__type_fold_right<::cuda::std::__type_list<short>, int, MakePair>,
                                   Pair<int, short>>::value,
              "");
static_assert(
  ::cuda::std::is_same<
    ::cuda::std::__type_fold_right<
      ::cuda::std::__type_list<
        Int<0>,
        Int<1>,
        Int<2>,
        Int<3>,
        Int<4>,
        Int<5>,
        Int<6>,
        Int<7>,
        Int<8>,
        Int<9>,
        Int<10>,
        Int<11>,
        Int<12>,
        Int<13>,
        Int<14>,
        Int<15>,
        Int<16>,
        Int<17>,
        Int<18>,
        Int<19>,
        Int<20>>,
      int,
      MakePair>,
    Pair<
      Pair<
        Pair<
          Pair<
            Pair<
              Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<int, Int<0>>, Int<1>>, Int<2>>,
                                                                               Int<3>>,
                                                                          Int<4>>,
                                                                     Int<5>>,
                                                                Int<6>>,
                                                           Int<7>>,
                                                      Int<8>>,
                                                 Int<9>>,
                                            Int<10>>,
                                       Int<11>>,
                                  Int<12>>,
                             Int<13>>,
                        Int<14>>,
                   Int<15>>,
              Int<16>>,
            Int<17>>,
          Int<18>>,
        Int<19>>,
      Int<20>>>::value,
  "");

// __type_fold_left
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_fold_left<::cuda::std::__type_list<>, int, MakePair>, int>::value, "");
static_assert(::cuda::std::is_same<::cuda::std::__type_fold_left<::cuda::std::__type_list<short>, int, MakePair>,
                                   Pair<int, short>>::value,
              "");
static_assert(
  ::cuda::std::is_same<
    ::cuda::std::__type_fold_left<
      ::cuda::std::__type_list<
        Int<0>,
        Int<1>,
        Int<2>,
        Int<3>,
        Int<4>,
        Int<5>,
        Int<6>,
        Int<7>,
        Int<8>,
        Int<9>,
        Int<10>,
        Int<11>,
        Int<12>,
        Int<13>,
        Int<14>,
        Int<15>,
        Int<16>,
        Int<17>,
        Int<18>,
        Int<19>,
        Int<20>>,
      int,
      MakePair>,
    Pair<
      Pair<
        Pair<
          Pair<
            Pair<
              Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<Pair<int, Int<20>>, Int<19>>, Int<18>>,
                                                                               Int<17>>,
                                                                          Int<16>>,
                                                                     Int<15>>,
                                                                Int<14>>,
                                                           Int<13>>,
                                                      Int<12>>,
                                                 Int<11>>,
                                            Int<10>>,
                                       Int<9>>,
                                  Int<8>>,
                             Int<7>>,
                        Int<6>>,
                   Int<5>>,
              Int<4>>,
            Int<3>>,
          Int<2>>,
        Int<1>>,
      Int<0>>>::value,
  "");

// __type_remove
static_assert(::cuda::std::is_same<::cuda::std::__type_remove<::cuda::std::__type_list<>, ::cuda::std::__type_list<>>,
                                   ::cuda::std::__type_list<>>::value,
              "");
static_assert(::cuda::std::is_same<::cuda::std::__type_remove<::cuda::std::__type_list<int, int, int, int>, int>,
                                   ::cuda::std::__type_list<>>::value,
              "");
static_assert(::cuda::std::is_same<::cuda::std::__type_remove<::cuda::std::__type_list<int, short, int, int*>, int>,
                                   ::cuda::std::__type_list<short, int*>>::value,
              "");

// __type_cartesian_product
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_cartesian_product<::cuda::std::__type_list<>, ::cuda::std::__type_list<>>,
                       ::cuda::std::__type_list<>>::value,
  "");
static_assert(::cuda::std::is_same<
                ::cuda::std::__type_cartesian_product<::cuda::std::__type_list<>, ::cuda::std::__type_list<int, short>>,
                ::cuda::std::__type_list<>>::value,
              "");
static_assert(::cuda::std::is_same<
                ::cuda::std::__type_cartesian_product<::cuda::std::__type_list<>, ::cuda::std::__type_list<int, short>>,
                ::cuda::std::__type_list<>>::value,
              "");
static_assert(
  ::cuda::std::is_same<
    ::cuda::std::__type_cartesian_product<::cuda::std::__type_list<int*>, ::cuda::std::__type_list<int, short>>,
    ::cuda::std::__type_list<::cuda::std::__type_list<int*, int>, ::cuda::std::__type_list<int*, short>>>::value,
  "");
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_cartesian_product<::cuda::std::__type_list<int*>,
                                                             ::cuda::std::__type_list<int, short>,
                                                             ::cuda::std::__type_list<Empty, Incomplete>>,
                       ::cuda::std::__type_list<::cuda::std::__type_list<int*, int, Empty>,
                                                ::cuda::std::__type_list<int*, int, Incomplete>,
                                                ::cuda::std::__type_list<int*, short, Empty>,
                                                ::cuda::std::__type_list<int*, short, Incomplete>>>::value,
  "");

// __type_sizeof
static_assert(::cuda::std::__type_call<::cuda::std::__type_sizeof, char[42]>::value == 42, "");

// __type_strict_and
static_assert(::cuda::std::__type_call<::cuda::std::__type_strict_and>::value, "");
static_assert(::cuda::std::__type_call<::cuda::std::__type_strict_and, ::cuda::std::true_type>::value, "");
static_assert(!::cuda::std::__type_call<::cuda::std::__type_strict_and, ::cuda::std::false_type>::value, "");
static_assert(
  ::cuda::std::__type_call<::cuda::std::__type_strict_and, ::cuda::std::true_type, ::cuda::std::true_type>::value, "");
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_strict_and, ::cuda::std::true_type, ::cuda::std::false_type>::value,
  "");
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_strict_and, ::cuda::std::false_type, ::cuda::std::true_type>::value,
  "");
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_strict_and, ::cuda::std::false_type, ::cuda::std::false_type>::value,
  "");

// __type_strict_or
static_assert(!::cuda::std::__type_call<::cuda::std::__type_strict_or>::value, "");
static_assert(::cuda::std::__type_call<::cuda::std::__type_strict_or, ::cuda::std::true_type>::value, "");
static_assert(!::cuda::std::__type_call<::cuda::std::__type_strict_or, ::cuda::std::false_type>::value, "");
static_assert(
  ::cuda::std::__type_call<::cuda::std::__type_strict_or, ::cuda::std::true_type, ::cuda::std::true_type>::value, "");
static_assert(
  ::cuda::std::__type_call<::cuda::std::__type_strict_or, ::cuda::std::true_type, ::cuda::std::false_type>::value, "");
static_assert(
  ::cuda::std::__type_call<::cuda::std::__type_strict_or, ::cuda::std::false_type, ::cuda::std::true_type>::value, "");
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_strict_or, ::cuda::std::false_type, ::cuda::std::false_type>::value,
  "");

// __type_not
static_assert(!::cuda::std::__type_call<::cuda::std::__type_not, ::cuda::std::true_type>::value, "");

// __type_equal
static_assert(
  ::cuda::std::__type_call<::cuda::std::__type_equal, ::cuda::std::integral_constant<int, 42>, Int<42>>::value, "");
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_equal, ::cuda::std::integral_constant<int, 1>, Int<-1>>::value, "");

// __type_not_equal
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_not_equal, ::cuda::std::integral_constant<int, 42>, Int<42>>::value,
  "");
static_assert(
  ::cuda::std::__type_call<::cuda::std::__type_not_equal, ::cuda::std::integral_constant<int, 1>, Int<-1>>::value, "");

// __type_less
static_assert(
  ::cuda::std::__type_call<::cuda::std::__type_less, ::cuda::std::integral_constant<int, 1>, Int<42>>::value, "");
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_less, ::cuda::std::integral_constant<int, 42>, Int<1>>::value, "");
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_less, ::cuda::std::integral_constant<int, 1>, Int<1>>::value, "");

// __type_less_equal
static_assert(
  ::cuda::std::__type_call<::cuda::std::__type_less_equal, ::cuda::std::integral_constant<int, 1>, Int<42>>::value, "");
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_less_equal, ::cuda::std::integral_constant<int, 42>, Int<1>>::value,
  "");
static_assert(
  ::cuda::std::__type_call<::cuda::std::__type_less_equal, ::cuda::std::integral_constant<int, 1>, Int<1>>::value, "");

// __type_greater
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_greater, ::cuda::std::integral_constant<int, 1>, Int<42>>::value, "");
static_assert(
  ::cuda::std::__type_call<::cuda::std::__type_greater, ::cuda::std::integral_constant<int, 42>, Int<1>>::value, "");
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_greater, ::cuda::std::integral_constant<int, 1>, Int<1>>::value, "");

// __type_greater_equal
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_greater_equal, ::cuda::std::integral_constant<int, 1>, Int<42>>::value,
  "");
static_assert(
  ::cuda::std::__type_call<::cuda::std::__type_greater_equal, ::cuda::std::integral_constant<int, 42>, Int<1>>::value,
  "");
static_assert(
  ::cuda::std::__type_call<::cuda::std::__type_greater_equal, ::cuda::std::integral_constant<int, 1>, Int<1>>::value,
  "");

// __type_pair_first
static_assert(::cuda::std::is_same<::cuda::std::__type_pair_first<::cuda::std::__type_pair<int, short>>, int>::value,
              "");
static_assert(::cuda::std::is_same<::cuda::std::__type_pair_second<::cuda::std::__type_pair<int, short>>, short>::value,
              "");

// __type_value_list
static_assert(::cuda::std::__type_value_list<int>::__size == 0, "");
static_assert(::cuda::std::__type_value_list<int, 1>::__size == 1, "");
static_assert(::cuda::std::__type_value_list<int, 1, 2>::__size == 2, "");
static_assert(
  ::cuda::std::is_same<
    ::cuda::std::__type_call<::cuda::std::__type_value_list<int, 1, 2>, Fn, Incomplete, Empty>,
    Types<::cuda::std::integral_constant<int, 1>, ::cuda::std::integral_constant<int, 2>, Incomplete, Empty>>::value,
  "");

// __type_iota
static_assert(::cuda::std::__type_iota<int, 0, 0>::__size == 0, "");
static_assert(::cuda::std::is_same<::cuda::std::__type_iota<int, 0, 1>, ::cuda::std::__type_value_list<int, 0>>::value,
              "");
static_assert(::cuda::std::is_same<::cuda::std::__type_iota<int, 5, 10>,
                                   ::cuda::std::__type_value_list<int, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>>::value,
              "");
static_assert(::cuda::std::is_same<::cuda::std::__type_iota<int, 5, 10, 2>,
                                   ::cuda::std::__type_value_list<int, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23>>::value,
              "");

int main(int argc, char** argv)
{
  return 0;
}
