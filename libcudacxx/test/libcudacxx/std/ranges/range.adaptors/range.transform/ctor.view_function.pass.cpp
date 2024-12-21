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

// constexpr transform_view(View, F);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

struct Range : cuda::std::ranges::view_base
{
  __host__ __device__ constexpr explicit Range(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  __host__ __device__ constexpr int* begin() const
  {
    return begin_;
  }
  __host__ __device__ constexpr int* end() const
  {
    return end_;
  }

private:
  int* begin_;
  int* end_;
};

struct F
{
  __host__ __device__ constexpr int operator()(int i) const
  {
    return i + 100;
  }
};

__host__ __device__ constexpr bool test()
{
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    Range range(buff, buff + 8);
    F f{};
    cuda::std::ranges::transform_view<Range, F> view(range, f);
    assert(view[0] == 101);
    assert(view[1] == 102);
    // ...
    assert(view[7] == 108);
  }

  {
    Range range(buff, buff + 8);
    F f{};
    cuda::std::ranges::transform_view<Range, F> view = {range, f};
    assert(view[0] == 101);
    assert(view[1] == 102);
    // ...
    assert(view[7] == 108);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF) && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  static_assert(test(), "");
#endif // _LIBCUDACXX_ADDRESSOF && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)

  return 0;
}
