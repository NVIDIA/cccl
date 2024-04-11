//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <size_t I, class MoveOnly, size_t N> const MoveOnly&& get(const array<MoveOnly, N>&& a);

// UNSUPPORTED: c++03

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct MoveOnly
{
  double val_ = 0.0;

  MoveOnly()                      = default;
  MoveOnly(MoveOnly&&)            = default;
  MoveOnly& operator=(MoveOnly&&) = default;

  // Not deleted because of non guaranteed copy elision in C++11/14
  __host__ __device__ MoveOnly(const MoveOnly&);
  __host__ __device__ MoveOnly& operator=(const MoveOnly&);

  __host__ __device__ constexpr MoveOnly(const double val) noexcept
      : val_(val)
  {}
};

int main(int, char**)
{
  {
    typedef cuda::std::array<MoveOnly, 1> C;
    const C c = {3.5};
    static_assert(cuda::std::is_same<const MoveOnly&&, decltype(cuda::std::get<0>(cuda::std::move(c)))>::value, "");
    static_assert(noexcept(cuda::std::get<0>(cuda::std::move(c))), "");
    const MoveOnly&& t = cuda::std::get<0>(cuda::std::move(c));
    assert(t.val_ == 3.5);
  }

#if TEST_STD_VER >= 2014
  {
    typedef double MoveOnly;
    typedef cuda::std::array<MoveOnly, 3> C;
    constexpr const C c = {1, 2, 3.5};
    static_assert(cuda::std::get<0>(cuda::std::move(c)) == 1, "");
    static_assert(cuda::std::get<1>(cuda::std::move(c)) == 2, "");
    static_assert(cuda::std::get<2>(cuda::std::move(c)) == 3.5, "");
  }
#endif

  return 0;
}
