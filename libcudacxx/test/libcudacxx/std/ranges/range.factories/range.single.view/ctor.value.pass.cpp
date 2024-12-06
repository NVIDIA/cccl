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

// constexpr explicit single_view(const T& t);
// constexpr explicit single_view(T&& t);

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

struct Empty
{};
struct BigType
{
  char buffer[64] = {10};
};

__host__ __device__ constexpr bool test()
{
  {
    BigType bt{};
    cuda::std::ranges::single_view<BigType> sv(bt);
    assert(sv.data()->buffer[0] == 10);
    assert(sv.size() == 1);
  }
  {
    const BigType bt{};
    const cuda::std::ranges::single_view<BigType> sv(bt);
    assert(sv.data()->buffer[0] == 10);
    assert(sv.size() == 1);
  }

  {
    BigType bt{};
    cuda::std::ranges::single_view<BigType> sv(cuda::std::move(bt));
    assert(sv.data()->buffer[0] == 10);
    assert(sv.size() == 1);
  }
  {
    const BigType bt{};
    const cuda::std::ranges::single_view<BigType> sv(cuda::std::move(bt));
    assert(sv.data()->buffer[0] == 10);
    assert(sv.size() == 1);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF) && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  static_assert(test());
#endif // _LIBCUDACXX_ADDRESSOF && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)

  return 0;
}
