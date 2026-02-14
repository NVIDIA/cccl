//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/execution>
#include <cuda/std/type_traits>

#include <cuda/experimental/execution.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

template <class T, class U>
using is_same = cuda::std::is_same<cuda::std::remove_cvref_t<T>, U>;

C2H_TEST("Execution policies", "[execution][policies]")
{
  namespace execution = cuda::std::execution;
  SECTION("Individual options")
  {
    cudax::execution::any_execution_policy pol = execution::seq;
    pol                                        = execution::par;
    pol                                        = execution::par_unseq;
    pol                                        = execution::unseq;
    CHECK(pol == execution::unseq);
  }
}
