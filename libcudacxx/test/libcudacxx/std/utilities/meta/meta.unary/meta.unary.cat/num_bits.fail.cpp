//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_num_bits()
{
  static_assert(cuda::std::__num_bits_v<T>);
}

struct likely_padded
{
  char c;
  int i;
};

int main(int, char**)
{
  test_num_bits<void>();
  test_num_bits<likely_padded>();
  return 0;
}
