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

__host__ __device__ void test()
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
}

int main(int, char**)
{
  test();

  return 0;
}
