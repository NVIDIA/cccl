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
static_assert(::cuda::std::is_same<::cuda::std::__type<::cuda::std::type_identity<Incomplete>>, Incomplete>::value);

// __type_call
static_assert(::cuda::std::is_same<::cuda::std::__type_call<Fn, Incomplete>, Types<Incomplete>>::value);
static_assert(::cuda::std::is_same<::cuda::std::__type_call<Fn1, Incomplete>, Types<Incomplete>>::value);
static_assert(::cuda::std::is_same<::cuda::std::__type_call<Fn2, Incomplete, Empty>, Types<Incomplete, Empty>>::value);
static_assert(::cuda::std::is_same<::cuda::std::__type_call1<Fn1, Incomplete>, Types<Incomplete>>::value);

// __type_call_indirect
template <class... Ts, class = ::cuda::std::__type_call_indirect<Fn2, Ts...>>
TEST_FUNC constexpr bool test_call_indirect(int)
{
  return true;
}

template <class... Ts>
TEST_FUNC constexpr bool test_call_indirect(long)
{
  return false;
}

static_assert(test_call_indirect<Incomplete, Empty>(0));
#if !defined(SKIP_SFINAE_TESTS)
static_assert(!test_call_indirect<Incomplete>(0));
static_assert(!test_call_indirect<Incomplete, Empty, int>(0));
#endif

template <class... Ts>
struct Template
{
  using type = Types<Ts...>;
};

// __type_quote
static_assert(::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_quote<Template>, Incomplete, Empty>,
                                   Template<Incomplete, Empty>>::value);

// __type_list
static_assert(::cuda::std::__type_list<int, float, double>::__size == 3);
static_assert(::cuda::std::__type_list<>::__size == 0);
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_call<::cuda::std::__type_list<int, float, double>, Fn, Incomplete, Empty>,
                       Types<int, float, double, Incomplete, Empty>>::value);

// __type_list_push_back
static_assert(::cuda::std::is_same<::cuda::std::__type_push_back<::cuda::std::__type_list<int, float>, double>,
                                   ::cuda::std::__type_list<int, float, double>>::value);

// __type_index_c
// NOTE: __type_index_c has a fast path for indices 16 and below.
static_assert(::cuda::std::__type_index_c<0, Int<42>, double>::value == 42);
static_assert(::cuda::std::__type_index_c<1, int, Int<42>, double>::value == 42);

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
  == 14);
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
  == 15);
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
  == 15);
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
  == 16);
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
  == 17);
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
  == 32);
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
  == 33);

// __type_at_c
static_assert(::cuda::std::__type_at_c<0, ::cuda::std::__type_list<Int<42>, double>>::value == 42);
static_assert(::cuda::std::__type_at_c<1, ::cuda::std::__type_list<int, Int<42>, double>>::value == 42);

// __type_front
static_assert(::cuda::std::__type_front<::cuda::std::__type_list<Int<42>, double>>::value == 42);

struct BiggerThanFour
{
  template <class T>
  using __call = ::cuda::std::bool_constant<(sizeof(T) > 4)>;
};

// __type_find_if
static_assert(::cuda::std::is_same<::cuda::std::__type_find_if<::cuda::std::__type_list<>, BiggerThanFour>,
                                   ::cuda::std::__type_list<>>::value);
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_find_if<::cuda::std::__type_list<char, char[2], Empty>, BiggerThanFour>,
                       ::cuda::std::__type_list<>>::value);
static_assert(
  ::cuda::std::is_same<::cuda::std::__type_find_if<::cuda::std::__type_list<char, char[5], Empty>, BiggerThanFour>,
                       ::cuda::std::__type_list<char[5], Empty>>::value);

int main(int argc, char** argv)
{
  return 0;
}
