//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... Types, class Alloc>
//   struct uses_allocator<tuple<Types...>, Alloc> : true_type { };

// UNSUPPORTED: c++98, c++03

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct NonTrivialEmpty
{
  __host__ __device__ NonTrivialEmpty() {}
};
static_assert(cuda::std::is_trivially_copyable<NonTrivialEmpty>::value, "");

struct NonTrivialNonEmpty
{
  int val_ = 0;
  __host__ __device__ NonTrivialNonEmpty() {}
};
static_assert(cuda::std::is_trivially_copyable<NonTrivialNonEmpty>::value, "");

struct NonTriviallyCopyAble
{
  int val_ = 0;
  __host__ __device__ NonTriviallyCopyAble& operator=(const NonTriviallyCopyAble)
  {
    return *this;
  }
};

int main(int, char**)
{
  static_assert(cuda::std::is_trivially_copyable<cuda::std::tuple<int, float>>::value, "");
  static_assert(cuda::std::is_trivially_copyable<cuda::std::tuple<int, NonTrivialEmpty>>::value, "");
  static_assert(cuda::std::is_trivially_copyable<cuda::std::tuple<int, NonTrivialNonEmpty>>::value, "");
  static_assert(!cuda::std::is_trivially_copyable<cuda::std::tuple<int, NonTriviallyCopyAble>>::value, "");

  return 0;
}
