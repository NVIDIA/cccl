//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <size_t I, class T, size_t N> T&& get(array<T, N>&& a);

// UNSUPPORTED: c++03

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

struct MoveOnly
{
  double val_ = 0.0;

  MoveOnly()                      = default;
  MoveOnly(MoveOnly&&)            = default;
  MoveOnly& operator=(MoveOnly&&) = default;

  MoveOnly(const MoveOnly&)            = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;

  __host__ __device__ constexpr MoveOnly(const double val) noexcept
      : val_(val)
  {}
};

int main(int, char**)
{
  {
    typedef cuda::std::array<MoveOnly, 1> C;
    C c = {3.5};
    static_assert(cuda::std::is_same<MoveOnly&&, decltype(cuda::std::get<0>(cuda::std::move(c)))>::value, "");
    MoveOnly t = cuda::std::get<0>(cuda::std::move(c));
    assert(t.val_ == 3.5);
  }

  return 0;
}
