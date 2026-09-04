// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/std/__type_traits/type_list.h>

#include <cstddef>

#include <c2h/catch2_main.h>
#include <c2h/catch2_nvtx.h>
#include <c2h/catch2_seed.h>
#include <c2h/catch2_test_macros.h>
#include <catch2/catch_template_test_macros.hpp>

#ifndef VAR_IDX
#  define VAR_IDX 0
#endif

namespace c2h
{
template <typename... Ts>
using type_list = ::cuda::std::__type_list<Ts...>;

template <typename TypeList>
using size = ::cuda::std::__type_list_size<TypeList>;

template <std::size_t Index, typename TypeList>
using get = ::cuda::std::__type_at_c<Index, TypeList>;

template <class... TypeLists>
using cartesian_product = ::cuda::std::__type_cartesian_product<TypeLists...>;

template <typename T, T... Ts>
using enum_type_list = ::cuda::std::__type_value_list<T, Ts...>;

template <typename T0, typename T1>
using pair = ::cuda::std::__type_pair<T0, T1>;

template <typename P>
using first = ::cuda::std::__type_pair_first<P>;

template <typename P>
using second = ::cuda::std::__type_pair_second<P>;

template <std::size_t Start, std::size_t Size, std::size_t Stride = 1>
using iota = ::cuda::std::__type_iota<std::size_t, Start, Size, Stride>;

template <typename TypeList, typename T>
using remove = ::cuda::std::__type_remove<TypeList, T>;
} // namespace c2h

#define C2H_TEST_NAME_IMPL(NAME, PARAM) C2H_TEST_STR(NAME) "(" C2H_TEST_STR(PARAM) ")"

#define C2H_TEST_NAME(NAME) C2H_TEST_NAME_IMPL(NAME, VAR_IDX)

#define C2H_TEST_CONCAT(A, B)       C2H_TEST_CONCAT_INNER(A, B)
#define C2H_TEST_CONCAT_INNER(A, B) A##B

#define C2H_TEST_IMPL(ID, NAME, TAG, ...)                                  \
  using C2H_TEST_CONCAT(types_, ID) = c2h::cartesian_product<__VA_ARGS__>; \
  CATCH_TEMPLATE_LIST_TEST_CASE_METHOD(::detail::nvtx_fixture, C2H_TEST_NAME(NAME), TAG, C2H_TEST_CONCAT(types_, ID))

#define C2H_TEST(NAME, TAG, ...) C2H_TEST_IMPL(__LINE__, NAME, TAG, __VA_ARGS__)

#define C2H_TEST_WITH_FIXTURE_IMPL(ID, FIXTURE, NAME, TAG, ...)            \
  using C2H_TEST_CONCAT(types_, ID) = c2h::cartesian_product<__VA_ARGS__>; \
  CATCH_TEMPLATE_LIST_TEST_CASE_METHOD(FIXTURE, C2H_TEST_NAME(NAME), TAG, C2H_TEST_CONCAT(types_, ID))

#define C2H_TEST_WITH_FIXTURE(FIXTURE, NAME, TAG, ...) \
  C2H_TEST_WITH_FIXTURE_IMPL(__LINE__, FIXTURE, NAME, TAG, __VA_ARGS__)

#define C2H_TEST_LIST_IMPL(ID, NAME, TAG, ...)                     \
  using C2H_TEST_CONCAT(types_, ID) = c2h::type_list<__VA_ARGS__>; \
  CATCH_TEMPLATE_LIST_TEST_CASE_METHOD(::detail::nvtx_fixture, C2H_TEST_NAME(NAME), TAG, C2H_TEST_CONCAT(types_, ID))

#define C2H_TEST_LIST(NAME, TAG, ...) C2H_TEST_LIST_IMPL(__LINE__, NAME, TAG, __VA_ARGS__)

#define C2H_TEST_LIST_WITH_FIXTURE_IMPL(ID, FIXTURE, NAME, TAG, ...) \
  using C2H_TEST_CONCAT(types_, ID) = c2h::type_list<__VA_ARGS__>;   \
  CATCH_TEMPLATE_LIST_TEST_CASE_METHOD(FIXTURE, C2H_TEST_NAME(NAME), TAG, C2H_TEST_CONCAT(types_, ID))

#define C2H_TEST_LIST_WITH_FIXTURE(FIXTURE, NAME, TAG, ...) \
  C2H_TEST_LIST_WITH_FIXTURE_IMPL(__LINE__, FIXTURE, NAME, TAG, __VA_ARGS__)

#define C2H_TEST_STR(a) #a

// Every CUB Catch2 test must specify CUB_SMALL or CUB_LARGE.
// Small tests may share a GPU; large tests run alone. Use CUB_LARGE when unsure.

#define CUB_TEST_MEMORY_TAG_CUB_SMALL "[small-mem]"
#define CUB_TEST_MEMORY_TAG_CUB_LARGE "[large-mem]"
#define CUB_TEST_MEMORY_TAG(MEMORY)   C2H_TEST_CONCAT(CUB_TEST_MEMORY_TAG_, MEMORY)

#define CUB_TEST(NAME, TAGS, MEMORY, ...) C2H_TEST(NAME, TAGS CUB_TEST_MEMORY_TAG(MEMORY), __VA_ARGS__)

#define CUB_TEST_CASE(NAME, TAGS, MEMORY) TEST_CASE(NAME, TAGS CUB_TEST_MEMORY_TAG(MEMORY))

#define CUB_TEST_LIST(NAME, TAGS, MEMORY, ...) C2H_TEST_LIST(NAME, TAGS CUB_TEST_MEMORY_TAG(MEMORY), __VA_ARGS__)
