//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/execution>

// all other includes follow after <cuda/std/execution>
#include <cuda/std/__type_traits/is_aggregate.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_standard_layout.h>
#include <cuda/std/__type_traits/is_trivially_constructible.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>

#include "test_macros.h"

[[maybe_unused]] _CCCL_GLOBAL_CONSTANT struct a_query_t
{
} a_query{};

[[maybe_unused]] _CCCL_GLOBAL_CONSTANT struct none_such_t
{
} none_such{};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  [[maybe_unused]] cuda::std::execution::prop<a_query_t, int> prop1{a_query, 42};
  [[maybe_unused]] cuda::std::execution::prop prop2{a_query, 42};

  static_assert(cuda::std::is_same_v<decltype(prop1), decltype(prop2)>);
  static_assert(sizeof(prop1) == sizeof(int), "");

  assert(prop1.query(a_query) == 42);
  static_assert(cuda::std::is_same_v<decltype(prop1.query(a_query)), int const&>, "");
  static_assert(noexcept(prop1.query(a_query)), "");

  static_assert(cuda::std::is_aggregate_v<cuda::std::execution::prop<a_query_t, int>>, "");
  static_assert(cuda::std::is_standard_layout_v<cuda::std::execution::prop<a_query_t, int>>, "");
  static_assert(cuda::std::is_trivially_copyable_v<cuda::std::execution::prop<a_query_t, int>>, "");
  static_assert(cuda::std::is_trivially_constructible_v<cuda::std::execution::prop<a_query_t, int>>, "");
  static_assert(cuda::std::is_trivially_destructible_v<cuda::std::execution::prop<a_query_t, int>>, "");

  static_assert(!cuda::std::execution::__queryable_with<cuda::std::execution::prop<a_query_t, int>, none_such_t>, "");

  return true;
}

int main(int, char**)
{
  test();

#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020

  return 0;
}
