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

// constexpr take_while_view(V base, Pred pred);

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

struct View : cuda::std::ranges::view_base
{
  MoveOnly mo;
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct Pred
{
  bool copied = false;
  bool moved  = false;
  Pred()      = default;
  __host__ __device__ constexpr Pred(Pred&&)
      : moved(true)
  {}
  __host__ __device__ constexpr Pred(const Pred&)
      : copied(true)
  {}
  __host__ __device__ bool operator()(int) const;
};

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::take_while_view<View, Pred> twv = {View{{}, MoveOnly{5}}, Pred{}};
    assert(twv.pred().moved);
    assert(!twv.pred().copied);
    assert(cuda::std::move(twv).base().mo.get() == 5);
  }
  return true;
}

int main(int, char**)
{
  test();
#ifndef TEST_COMPILER_CUDACC_BELOW_11_3
  static_assert(test(), "");
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3
  return 0;
}
