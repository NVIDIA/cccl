//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// constexpr optional(nullopt_t) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::nullopt;
using cuda::std::nullopt_t;
using cuda::std::optional;

template <class T>
__host__ __device__ constexpr void test()
{
  static_assert(cuda::std::is_nothrow_constructible<optional<T>, nullopt_t&>::value, "");
  static_assert(
    cuda::std::is_trivially_destructible<optional<T>>::value == cuda::std::is_trivially_destructible<T>::value, "");
  {
    optional<T> opt{nullopt};
    assert(static_cast<bool>(opt) == false);
  }
  {
    const optional<T> opt{nullopt};
    assert(static_cast<bool>(opt) == false);
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<int*>();
  test<const int>();

  test<ImplicitTypes::NoCtors>();
  test<NonTrivialTypes::NoCtors>();
  test<NonConstexprTypes::NoCtors>();

#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<int&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

__global__ void test_global_visibility()
{
  cuda::std::optional<int> meow{cuda::std::nullopt};
  unused(meow);
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
