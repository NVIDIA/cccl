//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/execution.tie_break.h>

#include "test_macros.h"

TEST_FUNC void test()
{
  namespace exec = cuda::execution;
  static_assert(cuda::std::is_base_of_v<exec::__requirement, exec::tie_break::unspecified_t>);
  static_assert(cuda::std::is_base_of_v<exec::__requirement, exec::tie_break::prefer_smaller_index_t>);
  static_assert(cuda::std::is_base_of_v<exec::__requirement, exec::tie_break::prefer_larger_index_t>);

  static_assert(cuda::std::is_same_v<decltype(exec::tie_break::__get_tie_break(exec::tie_break::unspecified)),
                                     exec::tie_break::unspecified_t>);
  static_assert(cuda::std::is_same_v<decltype(exec::tie_break::__get_tie_break(exec::tie_break::prefer_smaller_index)),
                                     exec::tie_break::prefer_smaller_index_t>);
  static_assert(cuda::std::is_same_v<decltype(exec::tie_break::__get_tie_break(exec::tie_break::prefer_larger_index)),
                                     exec::tie_break::prefer_larger_index_t>);
}

int main(int, char**)
{
  test();

  return 0;
}
