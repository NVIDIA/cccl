//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// template <class T> constexpr bool operator>=(const optional<T>& x, nullopt_t) noexcept;
// template <class T> constexpr bool operator>=(nullopt_t, const optional<T>& x) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/optional>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test()
{
  using cuda::std::nullopt;
  using cuda::std::nullopt_t;
  using cuda::std::optional;

  {
    using O = optional<T>;
    cuda::std::remove_reference_t<T> val{1};

    O o1{}; // disengaged
    O o2{val}; // engaged

    assert((nullopt >= o1));
    assert(!(nullopt >= o2));
    assert((o1 >= nullopt));
    assert((o2 >= nullopt));

    static_assert(noexcept(nullopt >= o1), "");
    static_assert(noexcept(o1 >= nullopt), "");
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<int&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
