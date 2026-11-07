//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__type_traits/type_list_operations.h>

// Test that the type_list_operations header is self-contained.
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/pair.h>

#include "test_macros.h"

#if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_COMPILER(NVRTC) || _CCCL_CUDA_COMPILER(CLANG)
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
  static_assert(AlwaysFalse<Ts...>::value);
};

template <class T>
struct Identity
{
  using type = T;
};

template <class T>
struct HasType : ::cuda::std::__type_callable<::cuda::std::__type_quote1<::cuda::std::__type>, T>
{};

static_assert(!HasType<Incomplete>::value);
static_assert(!HasType<Empty>::value);
static_assert(!HasType<int>::value);
static_assert(HasType<Identity<int>>::value);

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

struct Fn2
{
  template <class T, class U>
  using __call = Types<T, U>;
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

// __type_quote1
static_assert(::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_quote1<Template1>, Incomplete>,
                                   Template1<Incomplete>>::value);

// __type_quote_trait
static_assert(::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_quote_trait1<Template1>, Incomplete>,
                                   Types<Incomplete>>::value);
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_quote_trait2<Template2>, Incomplete, Empty>,
                       Types<Incomplete, Empty>>::value);

// __type_always
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_always<Incomplete>, Empty>, Incomplete>::value);

// __type_list_size
static_assert(::cuda::std::__type_list_size<::cuda::std::__type_list<int, float, double>>::value == 3);
static_assert(::cuda::std::__type_list_size<::cuda::std::__type_list<>>::value == 0);

// __type_list_push_front
static_assert(::cuda::std::is_same<::cuda::std::__type_push_front<::cuda::std::__type_list<int, float>, double>,
                                   ::cuda::std::__type_list<double, int, float>>::value);

// __as_type_list
static_assert(::cuda::std::is_same<::cuda::std::__as_type_list<DoNotInstantiate<int, float, double>>,
                                   ::cuda::std::__type_list<int, float, double>>::value);
static_assert(::cuda::std::is_same<::cuda::std::__as_type_list<::cuda::std::pair<int, float>>,
                                   ::cuda::std::__type_list<int, float>>::value);
static_assert(::cuda::std::is_same<::cuda::std::__as_type_list<::cuda::std::index_sequence<0, 1>>,
                                   ::cuda::std::__type_list<::cuda::std::integral_constant<size_t, 0>,
                                                            ::cuda::std::integral_constant<size_t, 1>>>::value);
static_assert(::cuda::std::is_same<::cuda::std::__as_type_list<int(float&, double&&)>,
                                   ::cuda::std::__type_list<int, float&, double&&>>::value);

// __type_callable
static_assert(::cuda::std::__type_callable<Fn2, Incomplete, Empty>::value);
static_assert(!::cuda::std::__type_callable<Fn2, Incomplete>::value);

template <class Ip, class... Ts>
using TypeIndex = ::cuda::std::__type_index_c<Ip::value, Ts...>;

static_assert(::cuda::std::__type_callable<::cuda::std::__type_indirect_quote<TypeIndex>,
                                           ::cuda::std::integral_constant<::cuda::std::size_t, 1>,
                                           int,
                                           Int<42>>::value);
#if !defined(SKIP_SFINAE_TESTS)
static_assert(!::cuda::std::__type_callable<::cuda::std::__type_indirect_quote<TypeIndex>,
                                            ::cuda::std::integral_constant<::cuda::std::size_t, 1>,
                                            int>::value);
#endif

template <class Ip, class List>
using TypeAt = ::cuda::std::__type_at_c<Ip::value, List>;

static_assert(::cuda::std::__type_callable<::cuda::std::__type_indirect_quote<TypeAt>,
                                           ::cuda::std::integral_constant<::cuda::std::size_t, 1>,
                                           ::cuda::std::__type_list<int, Int<42>>>::value);
#if !defined(SKIP_SFINAE_TESTS)
static_assert(!::cuda::std::__type_callable<::cuda::std::__type_indirect_quote<TypeAt>,
                                            ::cuda::std::integral_constant<::cuda::std::size_t, 1>,
                                            ::cuda::std::__type_list<int>>::value);
static_assert(!::cuda::std::__type_callable<::cuda::std::__type_quote1<::cuda::std::__type_front>,
                                            ::cuda::std::__type_list<>>::value);
#endif

// __type_concat
static_assert(::cuda::std::is_same<::cuda::std::__type_concat<>, ::cuda::std::__type_list<>>::value);

static_assert(
  ::cuda::std::is_same<::cuda::std::__type_concat<::cuda::std::__type_list<int>, ::cuda::std::__type_list<float>>,
                       ::cuda::std::__type_list<int, float>>::value);

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
      float>>::value);

// __type_flatten
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_flatten<::cuda::std::__type_list<>>, ::cuda::std::__type_list<>>::value);

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
      float>>::value);

// __type_transform
struct AddPointer
{
  template <class T>
  using __call = T*;
};

static_assert(::cuda::std::is_same<::cuda::std::__type_transform<::cuda::std::__type_list<>, AddPointer>,
                                   ::cuda::std::__type_list<>>::value);
static_assert(::cuda::std::is_same<::cuda::std::__type_transform<::cuda::std::__type_list<int, char const>, AddPointer>,
                                   ::cuda::std::__type_list<int*, char const*>>::value);
#if !defined(SKIP_SFINAE_TESTS)
static_assert(!::cuda::std::__type_callable<::cuda::std::__type_indirect_quote<::cuda::std::__type_transform>,
                                            ::cuda::std::__type_list<int, char const, int&>,
                                            AddPointer>::value);
#endif

template <class A, class B>
struct Pair
{};

using MakePair = ::cuda::std::__type_quote<Pair>;

// __type_fold_left
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_fold_left<::cuda::std::__type_list<>, int, MakePair>, int>::value);
static_assert(::cuda::std::is_same<::cuda::std::__type_fold_left<::cuda::std::__type_list<short>, int, MakePair>,
                                   Pair<int, short>>::value);
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
      Int<0>>>::value);

// __type_remove
static_assert(::cuda::std::is_same<::cuda::std::__type_remove<::cuda::std::__type_list<>, ::cuda::std::__type_list<>>,
                                   ::cuda::std::__type_list<>>::value);
static_assert(::cuda::std::is_same<::cuda::std::__type_remove<::cuda::std::__type_list<int, int, int, int>, int>,
                                   ::cuda::std::__type_list<>>::value);
static_assert(::cuda::std::is_same<::cuda::std::__type_remove<::cuda::std::__type_list<int, short, int, int*>, int>,
                                   ::cuda::std::__type_list<short, int*>>::value);

// __type_cartesian_product
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_cartesian_product<::cuda::std::__type_list<>, ::cuda::std::__type_list<>>,
                       ::cuda::std::__type_list<>>::value);
static_assert(::cuda::std::is_same<
              ::cuda::std::__type_cartesian_product<::cuda::std::__type_list<>, ::cuda::std::__type_list<int, short>>,
              ::cuda::std::__type_list<>>::value);
static_assert(::cuda::std::is_same<
              ::cuda::std::__type_cartesian_product<::cuda::std::__type_list<>, ::cuda::std::__type_list<int, short>>,
              ::cuda::std::__type_list<>>::value);
static_assert(
  ::cuda::std::is_same<
    ::cuda::std::__type_cartesian_product<::cuda::std::__type_list<int*>, ::cuda::std::__type_list<int, short>>,
    ::cuda::std::__type_list<::cuda::std::__type_list<int*, int>, ::cuda::std::__type_list<int*, short>>>::value);
static_assert(::cuda::std::is_same<::cuda::std::__type_cartesian_product<::cuda::std::__type_list<int*>,
                                                                         ::cuda::std::__type_list<int, short>,
                                                                         ::cuda::std::__type_list<Empty, Incomplete>>,
                                   ::cuda::std::__type_list<::cuda::std::__type_list<int*, int, Empty>,
                                                            ::cuda::std::__type_list<int*, int, Incomplete>,
                                                            ::cuda::std::__type_list<int*, short, Empty>,
                                                            ::cuda::std::__type_list<int*, short, Incomplete>>>::value);

// __type_strict_and
static_assert(::cuda::std::__type_call<::cuda::std::__type_strict_and>::value);
static_assert(::cuda::std::__type_call<::cuda::std::__type_strict_and, ::cuda::std::true_type>::value);
static_assert(!::cuda::std::__type_call<::cuda::std::__type_strict_and, ::cuda::std::false_type>::value);
static_assert(
  ::cuda::std::__type_call<::cuda::std::__type_strict_and, ::cuda::std::true_type, ::cuda::std::true_type>::value);
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_strict_and, ::cuda::std::true_type, ::cuda::std::false_type>::value);
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_strict_and, ::cuda::std::false_type, ::cuda::std::true_type>::value);
static_assert(
  !::cuda::std::__type_call<::cuda::std::__type_strict_and, ::cuda::std::false_type, ::cuda::std::false_type>::value);

// __type_pair_first
static_assert(::cuda::std::is_same<::cuda::std::__type_pair_first<::cuda::std::__type_pair<int, short>>, int>::value);
static_assert(::cuda::std::is_same<::cuda::std::__type_pair_second<::cuda::std::__type_pair<int, short>>, short>::value);

// __type_value_list
static_assert(::cuda::std::__type_value_list<int>::__size == 0);
static_assert(::cuda::std::__type_value_list<int, 1>::__size == 1);
static_assert(::cuda::std::__type_value_list<int, 1, 2>::__size == 2);
static_assert(
  ::cuda::std::is_same<
    ::cuda::std::__type_call<::cuda::std::__type_value_list<int, 1, 2>, Fn, Incomplete, Empty>,
    Types<::cuda::std::integral_constant<int, 1>, ::cuda::std::integral_constant<int, 2>, Incomplete, Empty>>::value);

// __type_iota
static_assert(::cuda::std::__type_iota<int, 0, 0>::__size == 0);
static_assert(::cuda::std::is_same<::cuda::std::__type_iota<int, 0, 1>,
                                   ::cuda::std::__as_type_list<::cuda::std::integer_sequence<int, 0>>>::value);
static_assert(
  ::cuda::std::is_same<
    ::cuda::std::__type_iota<int, 5, 10>,
    ::cuda::std::__as_type_list<::cuda::std::integer_sequence<int, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>>>::value);
static_assert(
  ::cuda::std::is_same<
    ::cuda::std::__type_iota<int, 5, 10, 2>,
    ::cuda::std::__as_type_list<::cuda::std::integer_sequence<int, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23>>>::value);

int main(int argc, char** argv)
{
  return 0;
}
