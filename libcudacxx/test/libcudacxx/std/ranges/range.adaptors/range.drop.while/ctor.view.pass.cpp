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

// constexpr drop_while_view(V base, Pred pred);

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"

struct View : cuda::std::ranges::view_base
{
  MoveOnly mo;
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct Pred
{
  bool copied      = false;
  bool moved       = false;
  constexpr Pred() = default;
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
    cuda::std::ranges::drop_while_view<View, Pred> dwv = {View{{}, MoveOnly{5}}, Pred{}};
    assert(dwv.pred().moved);
    assert(!dwv.pred().copied);
    assert(cuda::std::move(dwv).base().mo.get() == 5);
  }
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020
  return 0;
}
