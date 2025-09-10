//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>

__host__ __device__ void test()
{
  static_assert(
    cuda::std::is_same_v<decltype(cuda::execution::determinism::__get_determinism(cuda::execution::__get_requirements(
                           cuda::execution::require(cuda::execution::determinism::run_to_run)))),
                         cuda::execution::determinism::run_to_run_t>);

  static_assert(
    cuda::std::is_same_v<decltype(cuda::execution::determinism::__get_determinism(cuda::execution::__get_requirements(
                           cuda::execution::require(cuda::execution::determinism::not_guaranteed)))),
                         cuda::execution::determinism::not_guaranteed_t>);

  static_assert(
    cuda::std::is_same_v<
      decltype(cuda::execution::output_ordering::__get_output_ordering(
        cuda::execution::__get_requirements(cuda::execution::require(cuda::execution::output_ordering::sorted)))),
      cuda::execution::output_ordering::sorted_t>);

  static_assert(
    cuda::std::is_same_v<
      decltype(cuda::execution::output_ordering::__get_output_ordering(
        cuda::execution::__get_requirements(cuda::execution::require(cuda::execution::output_ordering::unsorted)))),
      cuda::execution::output_ordering::unsorted_t>);
}

int main(int, char**)
{
  test();

  return 0;
}
