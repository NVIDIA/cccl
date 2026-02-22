//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__execution/guarantee.h>
#include <cuda/__execution/max_segment_size.h>
#include <cuda/std/__execution/env.h>

__host__ __device__ void test(size_t dynamic_val)
{
  // Test for state-less guarantee
  auto static_genv           = cuda::execution::guarantee(cuda::execution::max_segment_size<42>{});
  auto static_env            = ::cuda::std::execution::env{static_genv};
  auto static_genv_extracted = cuda::execution::__get_guarantees(static_env);
  (void) static_genv_extracted;

  // Test for stateful guarantee
  auto dynamic_genv           = cuda::execution::guarantee(cuda::execution::max_segment_size<>{dynamic_val});
  auto dynamic_env            = ::cuda::std::execution::env{dynamic_genv};
  auto dynamic_genv_extracted = cuda::execution::__get_guarantees(dynamic_env);
  (void) dynamic_genv_extracted;

  // Test that max_segment_size is a guarantee
  static_assert(cuda::std::is_base_of_v<cuda::execution::__guarantee, cuda::execution::max_segment_size<42>>);
}

int main(int argc, char**)
{
  test(argc);

  return 0;
}
