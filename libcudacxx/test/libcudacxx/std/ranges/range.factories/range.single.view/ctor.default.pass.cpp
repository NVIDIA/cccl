//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// single_view() requires default_initializable<T> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

struct BigType
{
  char buffer[64] = {10};
};

template <bool DefaultCtorEnabled>
struct IsDefaultConstructible
{};

template <>
struct IsDefaultConstructible<false>
{
  IsDefaultConstructible() = delete;
};

__host__ __device__ constexpr bool test()
{
  static_assert(cuda::std::default_initializable<cuda::std::ranges::single_view<IsDefaultConstructible<true>>>);
  static_assert(!cuda::std::default_initializable<cuda::std::ranges::single_view<IsDefaultConstructible<false>>>);

  {
    cuda::std::ranges::single_view<BigType> sv;
    assert(sv.data()->buffer[0] == 10);
    assert(sv.size() == 1);
  }
  {
    const cuda::std::ranges::single_view<BigType> sv;
    assert(sv.data()->buffer[0] == 10);
    assert(sv.size() == 1);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // _CCCL_BUILTIN_ADDRESSOF

  return 0;
}
