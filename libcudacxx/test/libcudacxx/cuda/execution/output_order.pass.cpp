//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__execution/output_order.h>

__host__ __device__ void test()
{
  namespace exec = cuda::execution;
  static_assert(cuda::std::is_base_of_v<exec::__requirement, exec::output_order::sorted_t>);
  static_assert(cuda::std::is_base_of_v<exec::__requirement, exec::output_order::unsorted_t>);

  static_assert(cuda::std::is_same_v<decltype(exec::output_order::__get_output_order(exec::output_order::sorted)),
                                     exec::output_order::sorted>);
  static_assert(cuda::std::is_same_v<decltype(exec::output_order::__get_output_order(exec::output_order::unsorted)),
                                     exec::output_order::unsorted>);
}

int main(int, char**)
{
  test();

  return 0;
}
