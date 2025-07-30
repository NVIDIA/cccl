//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

//  template<class ElementType>
//  struct default_accessor {
//    using offset_policy = default_accessor;
//    using element_type = ElementType;
//    using reference = ElementType&;
//    using data_handle_type = ElementType*;
//    ...
//  };
//
//  Each specialization of default_accessor is a trivially copyable type that models semiregular.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "../MinimalElementType.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test()
{
  using A = cuda::std::default_accessor<T>;
  static_assert(cuda::std::is_same_v<typename A::offset_policy, A>);
  static_assert(cuda::std::is_same_v<typename A::element_type, T>);
  static_assert(cuda::std::is_same_v<typename A::reference, T&>);
  static_assert(cuda::std::is_same_v<typename A::data_handle_type, T*>);

  static_assert(cuda::std::semiregular<A>, "");
  static_assert(cuda::std::is_trivially_copyable<A>::value, "");

  // libcu++ extension
  static_assert(cuda::std::is_empty<A>::value, "");
}

int main(int, char**)
{
  test<int>();
  test<const int>();
  test<MinimalElementType>();
  test<const MinimalElementType>();
  return 0;
}
