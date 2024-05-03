//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// unique_ptr

#include <cuda/std/__memory_>
#include <cuda/std/iterator>

static_assert(cuda::std::indirectly_readable<cuda::std::unique_ptr<int>>);
static_assert(cuda::std::indirectly_writable<cuda::std::unique_ptr<int>, int>);
static_assert(!cuda::std::weakly_incrementable<cuda::std::unique_ptr<int>>);
static_assert(cuda::std::indirectly_movable<cuda::std::unique_ptr<int>, cuda::std::unique_ptr<int>>);
static_assert(cuda::std::indirectly_movable_storable<cuda::std::unique_ptr<int>, cuda::std::unique_ptr<int>>);
static_assert(cuda::std::indirectly_copyable<cuda::std::unique_ptr<int>, cuda::std::unique_ptr<int>>);
static_assert(cuda::std::indirectly_copyable_storable<cuda::std::unique_ptr<int>, cuda::std::unique_ptr<int>>);
static_assert(cuda::std::indirectly_swappable<cuda::std::unique_ptr<int>, cuda::std::unique_ptr<int>>);

static_assert(!cuda::std::indirectly_readable<cuda::std::unique_ptr<void>>);
static_assert(!cuda::std::indirectly_writable<cuda::std::unique_ptr<void>, void>);
static_assert(!cuda::std::weakly_incrementable<cuda::std::unique_ptr<void>>);
static_assert(!cuda::std::indirectly_movable<cuda::std::unique_ptr<void>, cuda::std::unique_ptr<void>>);
static_assert(!cuda::std::indirectly_movable_storable<cuda::std::unique_ptr<void>, cuda::std::unique_ptr<void>>);
static_assert(!cuda::std::indirectly_copyable<cuda::std::unique_ptr<void>, cuda::std::unique_ptr<void>>);
static_assert(!cuda::std::indirectly_copyable_storable<cuda::std::unique_ptr<void>, cuda::std::unique_ptr<void>>);

int main(int, char**)
{
  return 0;
}
