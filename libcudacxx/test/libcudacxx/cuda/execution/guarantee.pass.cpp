//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__execution/max_segment_size.h>

__host__ __device__ void test()
{
  static_assert(
    cuda::std::is_same_v<decltype(cuda::execution::segment_size::__get_max_segment_size(cuda::execution::__get_guarantees(
                           cuda::execution::guarantee(cuda::execution::segment_size::max_segment_size<42>{})))),
                         cuda::execution::segment_size::max_segment_size<42>>);
}

int main(int, char**)
{
  test();

  return 0;
}
