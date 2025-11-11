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
// class normal_distribution
// {
// public:
//     // types
//     typedef RealType result_type;

#include <cuda/std/__random_>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  {
    using D = cuda::std::normal_distribution<>;
    typedef D::result_type result_type;
    static_assert((std::is_same<result_type, double>::value), "");
  }
  {
    using D = cuda::std::normal_distribution<float>;
    typedef D::result_type result_type;
    static_assert((std::is_same<result_type, float>::value), "");
  }

  return 0;
}
