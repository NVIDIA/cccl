//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <cuda/std/iterator>

// template<semiregular S>
//   class move_sentinel;

#include <cuda/std/iterator>

template <class T, class = void>
constexpr bool HasMoveSentinel = false;

template <class T>
constexpr bool HasMoveSentinel<T, cuda::std::void_t<typename cuda::std::move_sentinel<T>>> = true;

struct Semiregular
{};

struct NotSemiregular
{
  __host__ __device__ NotSemiregular(int);
};

static_assert(HasMoveSentinel<int>);
static_assert(HasMoveSentinel<int*>);
static_assert(HasMoveSentinel<Semiregular>);
static_assert(!HasMoveSentinel<NotSemiregular>);

int main(int, char**)
{
  return 0;
}
