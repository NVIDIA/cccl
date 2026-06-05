//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/execution.determinism.h>

#include "test_macros.h"

TEST_FUNC void test()
{
  namespace exec = cuda::execution;
  static_assert(cuda::std::is_base_of_v<exec::__requirement, exec::determinism::run_to_run_t>);
  static_assert(cuda::std::is_base_of_v<exec::__requirement, exec::determinism::not_guaranteed_t>);
  static_assert(cuda::std::is_base_of_v<exec::__requirement, exec::determinism::gpu_to_gpu_t>);

  static_assert(cuda::std::is_same_v<decltype(exec::determinism::__get_determinism(exec::determinism::run_to_run)),
                                     exec::determinism::run_to_run_t>);
  static_assert(cuda::std::is_same_v<decltype(exec::determinism::__get_determinism(exec::determinism::not_guaranteed)),
                                     exec::determinism::not_guaranteed_t>);
  static_assert(cuda::std::is_same_v<decltype(exec::determinism::__get_determinism(exec::determinism::gpu_to_gpu)),
                                     exec::determinism::gpu_to_gpu_t>);

  // A tie-break folds into the deterministic guarantee: a guarantee carries an (optional) tie-break preference.
  using r2r_smaller = decltype(exec::determinism::run_to_run(exec::determinism::tie_break::prefer_smaller_index));
  static_assert(cuda::std::is_base_of_v<exec::__requirement, r2r_smaller>);
  static_assert(r2r_smaller::value == exec::determinism::__determinism_t::__run_to_run);
  static_assert(r2r_smaller::tie_break == exec::determinism::__tie_break_t::__prefer_smaller_index);
  static_assert(exec::determinism::run_to_run_t::tie_break == exec::determinism::__tie_break_t::__unspecified);
  using g2g_larger = decltype(exec::determinism::gpu_to_gpu(exec::determinism::tie_break::prefer_larger_index));
  static_assert(cuda::std::is_base_of_v<exec::__requirement, g2g_larger>);
  static_assert(g2g_larger::value == exec::determinism::__determinism_t::__gpu_to_gpu);
  static_assert(g2g_larger::tie_break == exec::determinism::__tie_break_t::__prefer_larger_index);
  static_assert(exec::determinism::gpu_to_gpu_t::tie_break == exec::determinism::__tie_break_t::__unspecified);
  static_assert(cuda::std::is_same_v<decltype(exec::determinism::__get_determinism(g2g_larger{})), g2g_larger>);

  // A tie-break tag is not a requirement on its own (it cannot be passed to require() without a determinism guarantee).
  static_assert(!cuda::std::is_base_of_v<exec::__requirement, exec::determinism::tie_break::prefer_smaller_index_t>);
}

int main(int, char**)
{
  test();

  return 0;
}
