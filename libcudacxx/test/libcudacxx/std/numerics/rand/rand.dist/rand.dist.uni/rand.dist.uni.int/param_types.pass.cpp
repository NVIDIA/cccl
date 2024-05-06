//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class _IntType = int>
// class uniform_int_distribution
// {
//     class param_type;

#include <cuda/std/__algorithm_>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  {
    typedef cuda::std::uniform_int_distribution<long> D;
    typedef D::param_type param_type;
    typedef param_type::distribution_type distribution_type;
    static_assert((cuda::std::is_same<D, distribution_type>::value), "");
  }

  return 0;
}
