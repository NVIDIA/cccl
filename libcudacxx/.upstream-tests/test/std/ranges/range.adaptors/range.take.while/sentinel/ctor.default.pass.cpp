//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// sentinel() = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

struct Sent {
  bool b; // deliberately uninitialised

  __host__ __device__ friend constexpr bool operator==(int*, const Sent& s) { return s.b; }
#if TEST_STD_VER < 20
  __host__ __device__ friend constexpr bool operator==(const Sent& s, int*) { return s.b; }
  __host__ __device__ friend constexpr bool operator!=(int*, const Sent& s) { return !s.b; }
  __host__ __device__ friend constexpr bool operator!=(const Sent& s, int*) { return !s.b; }
#endif
};

struct Range : cuda::std::ranges::view_base {
  __host__ __device__ int* begin() const;
  __host__ __device__ Sent end();
};

__host__ __device__ constexpr bool test() {
  {
    using R        = cuda::std::ranges::take_while_view<Range, bool (*)(int)>;
    using Sentinel = cuda::std::ranges::sentinel_t<R>;

    Sentinel s1 = {};
    assert(!s1.base().b);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
