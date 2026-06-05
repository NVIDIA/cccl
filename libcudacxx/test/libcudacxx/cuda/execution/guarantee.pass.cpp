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
#include <cuda/execution.max_total_num_items.h>

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_FUNC void test()
{
  namespace exec = cuda::execution;

  // A guarantee is only visible to an algorithm through the guarantees environment produced by guarantee(...), mirroring
  // how requirements are only visible through the requirements environment produced by require(...).
  const auto genv     = exec::guarantee(exec::max_total_num_items<1000>());
  const auto resolved = exec::__get_max_total_num_items(exec::__get_guarantees(genv));
  static_assert(cuda::std::is_base_of_v<exec::__guarantee, cuda::std::remove_cvref_t<decltype(resolved)>>);
  assert(resolved.highest() == 1000);

  // The guarantees query is a forwarding query, just like the requirements query.
  static_assert(cuda::std::execution::forwarding_query(exec::__get_guarantees_t{}));
}

int main(int, char**)
{
  test();

  return 0;
}
