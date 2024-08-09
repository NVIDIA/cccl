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
//     take_while_view(R&&, Pred) -> take_while_view<views::all_t<R>, Pred>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "types.h"

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

using owning_result = cuda::std::ranges::take_while_view<cuda::std::ranges::owning_view<Container>, Pred>;
static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::take_while_view(Container{}, Pred{})), owning_result>);

using function_result = cuda::std::ranges::take_while_view<View, bool (*)(int)>;
static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::take_while_view(View{}, pred)), function_result>);

using view_result = cuda::std::ranges::take_while_view<View, Pred>;
static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::take_while_view(View{}, Pred{})), view_result>);

__host__ __device__ void testRef()
{
  Container c{};
  Pred p{};
  unused(c);
  unused(p);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::take_while_view(c, p)),
                                     cuda::std::ranges::take_while_view<cuda::std::ranges::ref_view<Container>, Pred>>);
}

int main(int, char**)
{
  return 0;
}
