//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>

#include <cuda/experimental/execution.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

template <class T, class U>
using is_same = cuda::std::is_same<cuda::std::remove_cvref_t<T>, U>;

C2H_TEST("Execution policies", "[execution, policies]")
{
  namespace execution = cuda::experimental::execution;
  SECTION("Individual options")
  {
    execution::any_execution_policy pol = execution::seq;
    pol                                 = execution::par;
    pol                                 = execution::par_unseq;
    pol                                 = execution::unseq;
    CHECK(pol == execution::unseq);
  }

  SECTION("Global instances")
  {
    STATIC_CHECK(execution::seq == execution::seq);
    STATIC_CHECK(execution::par == execution::par);
    STATIC_CHECK(execution::par_unseq == execution::par_unseq);
    STATIC_CHECK(execution::unseq == execution::unseq);

    STATIC_CHECK_FALSE(execution::seq != execution::seq);
    STATIC_CHECK_FALSE(execution::par != execution::par);
    STATIC_CHECK_FALSE(execution::par_unseq != execution::par_unseq);
    STATIC_CHECK_FALSE(execution::unseq != execution::unseq);

    STATIC_CHECK_FALSE(execution::seq == execution::unseq);
    STATIC_CHECK_FALSE(execution::par == execution::seq);
    STATIC_CHECK_FALSE(execution::par_unseq == execution::par);
    STATIC_CHECK_FALSE(execution::unseq == execution::par_unseq);

    STATIC_CHECK(execution::seq != execution::unseq);
    STATIC_CHECK(execution::par != execution::seq);
    STATIC_CHECK(execution::par_unseq != execution::par);
    STATIC_CHECK(execution::unseq != execution::par_unseq);
  }

  SECTION("is_parallel_execution_policy")
  {
    using execution::__is_parallel_execution_policy;
    STATIC_CHECK(!__is_parallel_execution_policy<execution::seq>);
    STATIC_CHECK(__is_parallel_execution_policy<execution::par>);
    STATIC_CHECK(__is_parallel_execution_policy<execution::par_unseq>);
    STATIC_CHECK(!__is_parallel_execution_policy<execution::unseq>);
  }

  SECTION("is_unsequenced_execution_policy")
  {
    using execution::__is_unsequenced_execution_policy;
    STATIC_CHECK(!__is_unsequenced_execution_policy<execution::seq>);
    STATIC_CHECK(!__is_unsequenced_execution_policy<execution::par>);
    STATIC_CHECK(__is_unsequenced_execution_policy<execution::par_unseq>);
    STATIC_CHECK(__is_unsequenced_execution_policy<execution::unseq>);
  }
}
