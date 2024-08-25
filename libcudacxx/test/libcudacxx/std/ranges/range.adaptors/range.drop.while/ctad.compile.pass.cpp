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

//   template<class R, class Pred>
//     drop_while_view(R&&, Pred) -> drop_while_view<views::all_t<R>, Pred>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

struct Container
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct View : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct Pred
{
  __host__ __device__ bool operator()(int i) const;
};

__host__ __device__ bool pred(int);

// GCC really does not like local type defs...
using result_drop_while_view_owning =
  cuda::std::ranges::drop_while_view<cuda::std::ranges::owning_view<Container>, Pred>;
static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::drop_while_view(Container{}, Pred{})),
                                   result_drop_while_view_owning>);

using result_drop_while_view_function_pointer = cuda::std::ranges::drop_while_view<View, bool (*)(int)>;
static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::drop_while_view(View{}, pred)), //
                                   result_drop_while_view_function_pointer>);

using result_drop_while_view = cuda::std::ranges::drop_while_view<View, Pred>;
static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::drop_while_view(View{}, Pred{})), //
                                   result_drop_while_view>);

using result_drop_while_view_ref = cuda::std::ranges::drop_while_view<cuda::std::ranges::ref_view<Container>, Pred>;
__host__ __device__ void testRef()
{
  Container c{};
  Pred p{};
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::drop_while_view(c, p)), result_drop_while_view_ref>);
  unused(c);
  unused(p);
}

int main(int, char**)
{
  return 0;
}
