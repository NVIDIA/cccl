//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// indirect_result_t

#include <cuda/std/concepts>
#include <cuda/std/iterator>

static_assert(cuda::std::same_as<cuda::std::indirect_result_t<int (*)(int), int*>, int>);
static_assert(
  cuda::std::same_as<cuda::std::indirect_result_t<double (*)(int const&, float), int const*, float*>, double>);

struct S
{};
static_assert(cuda::std::same_as<cuda::std::indirect_result_t<S (&)(int), int*>, S>);
static_assert(cuda::std::same_as<cuda::std::indirect_result_t<long S::*, S*>, long&>);
static_assert(cuda::std::same_as<cuda::std::indirect_result_t<S && (S::*) (), S*>, S&&>);
static_assert(cuda::std::same_as<cuda::std::indirect_result_t<int S::* (S::*) (int) const, S*, int*>, int S::*>);

#if TEST_STD_VER > 2017
template <class F, class... Is>
constexpr bool has_indirect_result = requires { typename cuda::std::indirect_result_t<F, Is...>; };
#else
template <class F, class... Is>
_LIBCUDACXX_CONCEPT_FRAGMENT(has_indirect_result_, requires()(typename(cuda::std::indirect_result_t<F, Is...>)));

template <class F, class... Is>
_LIBCUDACXX_CONCEPT has_indirect_result = _LIBCUDACXX_FRAGMENT(has_indirect_result_, F, Is...);
#endif

static_assert(!has_indirect_result<int (*)(int), int>); // int isn't indirectly_readable
static_assert(!has_indirect_result<int, int*>); // int isn't invocable

int main(int, char**)
{
  return 0;
}
