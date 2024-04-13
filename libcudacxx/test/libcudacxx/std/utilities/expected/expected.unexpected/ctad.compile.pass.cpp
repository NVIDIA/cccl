//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: gcc-6, gcc-7, gcc-8, gcc-9

// template<class E> unexpected(E) -> unexpected<E>;

#include <cuda/std/concepts>
#include <cuda/std/expected>

struct Foo
{};

static_assert(cuda::std::same_as<decltype(cuda::std::unexpected(5)), cuda::std::unexpected<int>>);
static_assert(cuda::std::same_as<decltype(cuda::std::unexpected(Foo{})), cuda::std::unexpected<Foo>>);
static_assert(
  cuda::std::same_as<decltype(cuda::std::unexpected(cuda::std::unexpected<int>(5))), cuda::std::unexpected<int>>);

int main(int, char**)
{
  return 0;
}
