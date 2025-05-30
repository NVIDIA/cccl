//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class In, class Out>
// concept indirectly_copyable;

#include <cuda/std/iterator>

#include "MoveOnly.h"
#include "test_macros.h"

struct CopyOnly
{
  CopyOnly() = default;

  CopyOnly(CopyOnly const&)            = default;
  CopyOnly& operator=(CopyOnly const&) = default;

  CopyOnly(CopyOnly&&)            = delete;
  CopyOnly& operator=(CopyOnly&&) = delete;
};

// Can copy the underlying objects between pointers.
static_assert(cuda::std::indirectly_copyable<int*, int*>, "");
static_assert(cuda::std::indirectly_copyable<const int*, int*>, "");

// Can't copy if the output pointer is const.
static_assert(!cuda::std::indirectly_copyable<int*, const int*>, "");
static_assert(!cuda::std::indirectly_copyable<const int*, const int*>, "");

// Can copy from a pointer into an array but arrays aren't considered indirectly copyable-from.
#if !TEST_COMPILER(MSVC) || TEST_STD_VER != 2017
static_assert(cuda::std::indirectly_copyable<int*, int[2]>, "");
#endif // !TEST_COMPILER(MSVC) || TEST_STD_VER != 2017
static_assert(!cuda::std::indirectly_copyable<int[2], int*>, "");
static_assert(!cuda::std::indirectly_copyable<int[2], int[2]>, "");
static_assert(!cuda::std::indirectly_copyable<int (&)[2], int (&)[2]>, "");

// Can't copy between non-pointer types.
static_assert(!cuda::std::indirectly_copyable<int*, int>, "");
static_assert(!cuda::std::indirectly_copyable<int, int*>, "");
static_assert(!cuda::std::indirectly_copyable<int, int>, "");

// Check some less common types.
static_assert(!cuda::std::indirectly_movable<void*, void*>, "");
static_assert(!cuda::std::indirectly_movable<int*, void*>, "");
static_assert(!cuda::std::indirectly_movable<int(), int()>, "");
static_assert(!cuda::std::indirectly_movable<int*, int()>, "");
static_assert(!cuda::std::indirectly_movable<void, void>, "");

// Can't copy move-only objects.
static_assert(!cuda::std::indirectly_copyable<MoveOnly*, MoveOnly*>, "");
static_assert(!cuda::std::indirectly_copyable<MoveOnly*, const MoveOnly*>, "");
static_assert(!cuda::std::indirectly_copyable<const MoveOnly*, MoveOnly*>, "");
static_assert(!cuda::std::indirectly_copyable<const MoveOnly*, const MoveOnly*>, "");

// Can copy copy-only objects.
static_assert(cuda::std::indirectly_copyable<CopyOnly*, CopyOnly*>, "");
static_assert(!cuda::std::indirectly_copyable<CopyOnly*, const CopyOnly*>, "");
static_assert(cuda::std::indirectly_copyable<const CopyOnly*, CopyOnly*>, "");
static_assert(!cuda::std::indirectly_copyable<const CopyOnly*, const CopyOnly*>, "");

template <class T>
struct PointerTo
{
  using value_type = T;
  __host__ __device__ T& operator*() const;
};

// Can copy through a dereferenceable class.
static_assert(cuda::std::indirectly_copyable<int*, PointerTo<int>>, "");
static_assert(!cuda::std::indirectly_copyable<int*, PointerTo<const int>>, "");
static_assert(cuda::std::indirectly_copyable<PointerTo<int>, PointerTo<int>>, "");
static_assert(!cuda::std::indirectly_copyable<PointerTo<int>, PointerTo<const int>>, "");
static_assert(cuda::std::indirectly_copyable<CopyOnly*, PointerTo<CopyOnly>>, "");
static_assert(cuda::std::indirectly_copyable<PointerTo<CopyOnly>, CopyOnly*>, "");
static_assert(cuda::std::indirectly_copyable<PointerTo<CopyOnly>, PointerTo<CopyOnly>>, "");

int main(int, char**)
{
  return 0;
}
