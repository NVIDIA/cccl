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
  namespace exec = cuda::execution;
  static_assert(cuda::std::is_base_of_v<exec::__guarantee, exec::max_segment_size<42>>);

  static_assert(cuda::std::is_same_v<decltype(exec::__get_max_segment_size(exec::max_segment_size<42>{})),
                                     exec::max_segment_size<42>>);
}

int main(int, char**)
{
  test();

  return 0;
}
