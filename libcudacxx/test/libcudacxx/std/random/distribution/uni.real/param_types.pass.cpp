//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class uniform_real_distribution
// {
//     class param_type;

#include <cuda/std/__random_>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ void test()
{
  using D                 = cuda::std::uniform_real_distribution<float>;
  using param_type        = D::param_type;
  using distribution_type = param_type::distribution_type;
  static_assert((cuda::std::is_same<D, distribution_type>::value), "");
}

int main(int, char**)
{
  test();
  return 0;
}
