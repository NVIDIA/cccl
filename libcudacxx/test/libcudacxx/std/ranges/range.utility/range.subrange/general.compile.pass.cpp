//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// class cuda::std::ranges::subrange;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER > 2017
template <cuda::std::ranges::subrange_kind K, class... Args>
concept ValidSubrangeKind = requires { typename cuda::std::ranges::subrange<Args..., K>; };

template <class... Args>
concept ValidSubrange = requires { typename cuda::std::ranges::subrange<Args...>; };
#else
// clang is really not helpfull here failing with concept emulation
template <class It, class = void>
constexpr bool ValidSubrange1 = false;

template <class It>
constexpr bool ValidSubrange1<It, cuda::std::void_t<cuda::std::ranges::subrange<It>>> = true;

template <class It, class Sent, class = void>
constexpr bool ValidSubrange2 = false;

template <class It, class Sent>
constexpr bool ValidSubrange2<It, Sent, cuda::std::void_t<cuda::std::ranges::subrange<It, Sent>>> = true;

template <class...>
constexpr bool ValidSubrange = false;

template <class It>
constexpr bool ValidSubrange<It> = ValidSubrange1<It>;

template <class It, class Sent>
constexpr bool ValidSubrange<It, Sent> = ValidSubrange2<It, Sent>;

template <cuda::std::ranges::subrange_kind Kind, class It, class Sent, class = void>
constexpr bool ValidSubrangeKind = false;

template <cuda::std::ranges::subrange_kind Kind, class It, class Sent>
constexpr bool ValidSubrangeKind<Kind, It, Sent, cuda::std::void_t<cuda::std::ranges::subrange<It, Sent, Kind>>> = true;
#endif

static_assert(ValidSubrange<forward_iterator<int*>>);
static_assert(ValidSubrange<forward_iterator<int*>, forward_iterator<int*>>);
static_assert(
  ValidSubrangeKind<cuda::std::ranges::subrange_kind::unsized, forward_iterator<int*>, forward_iterator<int*>>);
static_assert(
  ValidSubrangeKind<cuda::std::ranges::subrange_kind::sized, forward_iterator<int*>, forward_iterator<int*>>);
// Wrong sentinel type.
static_assert(!ValidSubrange<forward_iterator<int*>, int*>);
static_assert(ValidSubrange<int*>);
static_assert(ValidSubrange<int*, int*>);
// Must be sized.
static_assert(!ValidSubrangeKind<cuda::std::ranges::subrange_kind::unsized, int*, int*>);
static_assert(ValidSubrangeKind<cuda::std::ranges::subrange_kind::sized, int*, int*>);
// Wrong sentinel type.
static_assert(!ValidSubrange<int*, forward_iterator<int*>>);
// Not an iterator.
static_assert(!ValidSubrange<int>);

int main(int, char**)
{
  return 0;
}
