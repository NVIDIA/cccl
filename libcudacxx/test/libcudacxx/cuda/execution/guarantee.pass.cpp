//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/execution.guarantee.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace exec = cuda::execution;

// Queries and guarantees standing in for real guarantees like an upper bound on the total number of items. One of them
// carries runtime state to exercise that guarantees, unlike requirements, are stored by value.
struct get_bound_t
{};

struct get_flag_t
{};

struct bound_guarantee : exec::__guarantee
{
  int __bound;

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto query(get_bound_t) const noexcept -> int
  {
    return __bound;
  }
};

struct flag_guarantee : exec::__guarantee
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto query(get_flag_t) const noexcept -> bool
  {
    return true;
  }
};

TEST_FUNC constexpr bool test()
{
  // A guarantee is only visible to an algorithm through the guarantees environment produced by guarantee(...),
  // mirroring how requirements are only visible through the requirements environment produced by require(...).
  const auto genv       = exec::guarantee(bound_guarantee{{}, 42});
  const auto guarantees = exec::__get_guarantees(genv);
  assert(guarantees.query(get_bound_t{}) == 42);

  // guarantee(...) accepts multiple guarantees; each is stored by value and stays individually queryable.
  const auto multi_genv       = exec::guarantee(bound_guarantee{{}, 7}, flag_guarantee{});
  const auto multi_guarantees = exec::__get_guarantees(multi_genv);
  assert(multi_guarantees.query(get_bound_t{}) == 7);
  assert(multi_guarantees.query(get_flag_t{}));

  // The guarantees environment only answers the queries of the guarantees it bundles.
  static_assert(!cuda::std::execution::__queryable_with<decltype(guarantees), get_flag_t>);

  // The guarantees query is a forwarding query, just like the requirements query.
  static_assert(cuda::std::execution::forwarding_query(exec::__get_guarantees_t{}));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
