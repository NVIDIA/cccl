//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// indirect_result_t

#include <cuda/std/concepts>
#include <cuda/std/iterator>

static_assert(cuda::std::same_as<cuda::std::indirect_result_t<int (*)(int), int*>, int>, "");
static_assert(
  cuda::std::same_as<cuda::std::indirect_result_t<double (*)(int const&, float), int const*, float*>, double>, "");

struct S
{};
static_assert(cuda::std::same_as<cuda::std::indirect_result_t<S (&)(int), int*>, S>, "");
static_assert(cuda::std::same_as<cuda::std::indirect_result_t<long S::*, S*>, long&>, "");
static_assert(cuda::std::same_as<cuda::std::indirect_result_t<S && (S::*) (), S*>, S&&>, "");
static_assert(cuda::std::same_as<cuda::std::indirect_result_t<int S::* (S::*) (int) const, S*, int*>, int S::*>, "");

template <class F, class... Is>
_CCCL_CONCEPT has_indirect_result =
  _CCCL_REQUIRES_EXPR((F, variadic Is))(typename(cuda::std::indirect_result_t<F, Is...>));

static_assert(!has_indirect_result<int (*)(int), int>, ""); // int isn't indirectly_readable
static_assert(!has_indirect_result<int, int*>, ""); // int isn't invocable

int main(int, char**)
{
  return 0;
}
