//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>
//
// Test converting constructor:
//
// template<class OtherElementType>
//   constexpr default_accessor(default_accessor<OtherElementType>) noexcept {}
//
// Constraints: is_convertible_v<OtherElementType(*)[], element_type(*)[]> is true.

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "../MinimalElementType.h"
#include "test_macros.h"

struct Base
{};
struct Derived : public Base
{};

template <class FromT, class ToT>
__host__ __device__ constexpr void test_conversion()
{
  cuda::std::default_accessor<FromT> acc_from{};
  static_assert(noexcept(cuda::std::default_accessor<ToT>(acc_from)));
  cuda::std::default_accessor<ToT> acc_to(acc_from);
  unused(acc_to);
}

__host__ __device__ constexpr bool test()
{
  // default accessor conversion largely behaves like pointer conversion
  test_conversion<int, int>();
  test_conversion<int, const int>();
  test_conversion<const int, const int>();
  test_conversion<MinimalElementType, MinimalElementType>();
  test_conversion<MinimalElementType, const MinimalElementType>();
  test_conversion<const MinimalElementType, const MinimalElementType>();

  // char is convertible to int, but accessors are not
  static_assert(
    !cuda::std::is_constructible<cuda::std::default_accessor<int>, cuda::std::default_accessor<char>>::value, "");
  // don't allow conversion from const elements to non-const
  static_assert(
    !cuda::std::is_constructible<cuda::std::default_accessor<int>, cuda::std::default_accessor<const int>>::value, "");
  // MinimalElementType is constructible from int, but accessors should not be convertible
  static_assert(!cuda::std::is_constructible<cuda::std::default_accessor<MinimalElementType>,
                                             cuda::std::default_accessor<int>>::value,
                "");
  // don't allow conversion from const elements to non-const
  static_assert(!cuda::std::is_constructible<cuda::std::default_accessor<MinimalElementType>,
                                             cuda::std::default_accessor<const MinimalElementType>>::value,
                "");
  // don't allow conversion from Base to Derived
  static_assert(
    !cuda::std::is_constructible<cuda::std::default_accessor<Derived>, cuda::std::default_accessor<Base>>::value, "");
  // don't allow conversion from Derived to Base
  static_assert(
    !cuda::std::is_constructible<cuda::std::default_accessor<Base>, cuda::std::default_accessor<Derived>>::value, "");

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
