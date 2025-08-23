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

[[maybe_unused]] _CCCL_GLOBAL_CONSTANT struct query1_t
{
} query1{};

[[maybe_unused]] _CCCL_GLOBAL_CONSTANT struct query2_t
{
} query2{};

[[maybe_unused]] _CCCL_GLOBAL_CONSTANT struct query3_t
{
} query3{};

[[maybe_unused]] _CCCL_GLOBAL_CONSTANT struct none_such_t
{
} none_such{};

struct custom_env
{
  __host__ __device__ constexpr auto query(query1_t) const noexcept
  {
    return -1;
  }

  // A query that takes an extra argument:
  __host__ __device__ constexpr auto query(query3_t, int i) const noexcept
  {
    return i;
  }
};

template <class Ty>
__host__ __device__ constexpr bool is_trivial_aggregate()
{
  return cuda::std::is_aggregate_v<Ty> && cuda::std::is_standard_layout_v<Ty> && cuda::std::is_trivially_copyable_v<Ty>
      && cuda::std::is_trivially_constructible_v<Ty> && cuda::std::is_trivially_destructible_v<Ty>;
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  [[maybe_unused]] cuda::std::execution::env e1{};
  static_assert(cuda::std::is_same_v<decltype(e1), cuda::std::execution::env<>>);
  static_assert(is_trivial_aggregate<cuda::std::execution::env<>>(), "");
  static_assert(!cuda::std::execution::__queryable_with<cuda::std::execution::env<>, query1_t>, "");
  static_assert(sizeof(e1) == 1, "");

  cuda::std::execution::env e2{cuda::std::execution::prop{query1, 42}};
  assert(e2.query(query1) == 42);
  assert(cuda::std::execution::__query_or(e2, query1, 24) == 42);
  assert(cuda::std::execution::__query_or(e2, query2, 24) == 24);
  static_assert(cuda::std::is_same_v<
                cuda::std::remove_cvref_t<cuda::std::execution::__query_result_or_t<decltype(e2), query1_t, float>>,
                int>);
  static_assert(cuda::std::is_same_v<
                cuda::std::remove_cvref_t<cuda::std::execution::__query_result_or_t<decltype(e2), query2_t, float>>,
                float>);
  using expected_e2_t = cuda::std::execution::env<cuda::std::execution::prop<query1_t, int>>;
  static_assert(cuda::std::is_same_v<decltype(e2), expected_e2_t>);
  static_assert(is_trivial_aggregate<expected_e2_t>(), "");
  static_assert(cuda::std::is_same_v<decltype(e2.query(query1)), const int&>);
  static_assert(!cuda::std::execution::__queryable_with<expected_e2_t, query2_t>, "");
  static_assert(sizeof(e2) == sizeof(int), "");

  cuda::std::execution::env e3{cuda::std::execution::prop{query1, 42}, cuda::std::execution::prop{query2, 3.14}};
  assert(e3.query(query1) == 42);
  assert(e3.query(query2) == 3.14);
  using expected_e3_t =
    cuda::std::execution::env<cuda::std::execution::prop<query1_t, int>, cuda::std::execution::prop<query2_t, double>>;
  static_assert(cuda::std::is_same_v<decltype(e3), expected_e3_t>);
  static_assert(is_trivial_aggregate<expected_e3_t>(), "");
  static_assert(cuda::std::is_same_v<decltype(e3.query(query1)), const int&>);
  static_assert(cuda::std::is_same_v<decltype(e3.query(query2)), const double&>);

  cuda::std::execution::env e4{
    custom_env{}, cuda::std::execution::prop{query1, 42}, cuda::std::execution::prop{query2, 3.14}};
  assert(e4.query(query1) == -1);
  assert(e4.query(query2) == 3.14);
  assert(e4.query(query3, 42) == 42);
  using expected_e4_t = cuda::std::execution::
    env<custom_env, cuda::std::execution::prop<query1_t, int>, cuda::std::execution::prop<query2_t, double>>;
  static_assert(cuda::std::is_same_v<decltype(e4), expected_e4_t>);
  static_assert(is_trivial_aggregate<expected_e4_t>(), "");
  static_assert(cuda::std::is_same_v<decltype(e4.query(query1)), int>);
  static_assert(cuda::std::is_same_v<decltype(e4.query(query2)), const double&>);
  static_assert(cuda::std::is_same_v<decltype(e4.query(query3, 42)), int>);

  assert(cuda::std::execution::__query_or(e2, query1, 0) == 42);
  assert(cuda::std::execution::__query_or(e2, query2, &e2) == &e2);
  assert(cuda::std::execution::__query_or(e4, query3, 0) == 0);
  assert(cuda::std::execution::__query_or(e4, query3, 0, 42) == 42);

  // Test that env works with const references:
  cuda::std::execution::env<decltype(e2) const&> e5{e2};
  assert(e5.query(query1) == 42);

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
