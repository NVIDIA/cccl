//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/execution.max_total_num_items.h>

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_FUNC void test()
{
  namespace exec = cuda::execution;

  // (a) static upper bound: the element type is inferred from the non-type template parameter.
  {
    const auto guarantee = exec::max_total_num_items<1000>();
    using holder_t       = cuda::std::remove_cvref_t<decltype(guarantee)>;
    static_assert(cuda::std::is_base_of_v<exec::__guarantee, holder_t>);
    static_assert(cuda::std::is_same_v<holder_t::element_type, int>);
    static_assert(holder_t::static_highest == 1000);
    assert(guarantee.highest() == 1000);
  }

  // A bound that does not fit into int infers a wider type, distinguishing 32-bit from 64-bit bounds.
  {
    const auto guarantee = exec::max_total_num_items<5'000'000'000>();
    using holder_t       = cuda::std::remove_cvref_t<decltype(guarantee)>;
    static_assert(sizeof(holder_t::element_type) == 8);
    static_assert(holder_t::static_highest == 5'000'000'000);
  }

  // The element type can be selected explicitly through the literal type.
  {
    const auto guarantee = exec::max_total_num_items<cuda::std::int16_t{1000}>();
    using holder_t       = cuda::std::remove_cvref_t<decltype(guarantee)>;
    static_assert(cuda::std::is_same_v<holder_t::element_type, cuda::std::int16_t>);
  }

  // (b) runtime upper bound: the element type is inferred from the argument, the static bound spans the whole type.
  {
    const auto guarantee = exec::max_total_num_items(cuda::std::int32_t{1'000'000'000});
    using holder_t       = cuda::std::remove_cvref_t<decltype(guarantee)>;
    static_assert(cuda::std::is_same_v<holder_t::element_type, cuda::std::int32_t>);
    static_assert(holder_t::static_highest == (cuda::std::numeric_limits<cuda::std::int32_t>::max)());
    assert(guarantee.highest() == 1'000'000'000);
  }

  // (c) both static and runtime upper bounds; the runtime bound is narrower than the static one.
  {
    const auto guarantee = exec::max_total_num_items<1000>(500);
    using holder_t       = cuda::std::remove_cvref_t<decltype(guarantee)>;
    static_assert(holder_t::static_highest == 1000);
    assert(guarantee.highest() == 500);
  }

  // The query returns the guarantee itself, preserving both the compile-time and the runtime bounds.
  {
    const auto guarantee = exec::max_total_num_items<1000>(500);
    const auto resolved  = exec::__get_max_total_num_items(guarantee);
    using holder_t       = cuda::std::remove_cvref_t<decltype(resolved)>;
    static_assert(holder_t::static_highest == 1000);
    assert(resolved.highest() == 500);
  }

  // The query is a forwarding query, just like the requirement queries.
  static_assert(cuda::std::execution::forwarding_query(exec::__get_max_total_num_items_t{}));
}

int main(int, char**)
{
  test();

  return 0;
}
